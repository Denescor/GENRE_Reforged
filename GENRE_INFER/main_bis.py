#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import argparse
from tqdm import tqdm
from model import Model
#from concurrent.futures import ThreadPoolExecutor, as_completed

def predict_output(batch, model):
    articles = []
    paragraph_id = []
    to_predict = []
    list_text = []
    article_i, line = batch
    
    if args.article and article_i != args.article:
        continue
    article = json.loads(line)
    text = article["text"]
    if args.eval_span and "evaluation_span" in article:
        evaluation_span = article["evaluation_span"]
    else:
        evaluation_span = (0, len(text))

    before = text[:evaluation_span[0]]
    after = text[evaluation_span[1]:]
    text = text[evaluation_span[0]:evaluation_span[1]]
    
        list_text.append(([before, text, after], article))
    
        if args.split_iter:
            to_predict.append(text)
        else:
            paragraphs = text.split(PARAGRAPH_SEPARATOR)
            paragraph_id.append(len(paragraphs))
            to_predict.extend(paragraphs)
            
    if args.split_iter:
        prediction = model.predict_iteratively(texte) # no batching predictions
    else:
        paragraphs = text.split(PARAGRAPH_SEPARATOR)
        
        temp_prediction = [model.predict_paragraph(txt, args.split_sentences, args.split_long) for txt in to_predict] # batching predictions
        predictions = []

    genre_text = before + prediction + after
    article["GENRE"] = genre_text
    articles.append(article)  
    
    return articles

def main(args):
    print("Preparing Archive Map")
    if args.local_archive is None:
        archive_map = None
        local_archive = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe"
    else:
        archive_map = {
                "bart.base": args.local_archive,
                "bart.large" : args.local_archive
        }
        local_archive = "{}/gpt2".format(args.local_archive)
    print("choosen map : {}".format(local_archive))
    top = time.time()
    print("load model... ")
    model = Model(yago=args.yago,
                  type_model=args.type_model,
                  mention_trie=args.mention_trie,
                  mention_to_candidates_dict=args.mention_to_candidates_dict,
                  candidates_trie=args.candidates_trie,
                  spacy=args.spacy_model,
                  archive_map=archive_map, local=local_archive)
    print("... done in {}s".format(int(time.time()-top)))

    if args.split_iter: desc_tqdm = "Predict Iteratively"
    else: desc_tqdm = "Predict Paragraph"
    list_article = []
    list_predictions = []
    for article_i, line in enumerate(open(args.input_file)):
        list_article.append((article_i, line))
    
    print("execute with batch of size {}".format(args.batch))
    if args.batch == 1: #split the list in 1 part (no batch)
        for article in tqdm(list_article, total=len(list_article), desc=desc_tqdm):
            list_predictions = predict_output([article], model)
    
    else: #split the list in [batch] parts (> 1) and execute each part simultiany
        list_batch = [list_article[i:i+args.batch] for i in range(0, len(list_article), args.batch)] # batchs creation
        len_batch = sum([len(x) for x in list_batch])
        assert len_batch == len(list_article), "list article has {} items\nlist batch has {} items".format(len(list_article), len_batch) 
        list_predictions = []
        for batch in tqdm(list_batch, total=len(list_batch), desc=desc_tqdm):
            list_predictions.extend(predict_output(batch, model))
                
    with open(args.output_file, "w") as out_file: #write the prediction in a json file        
        for article in list_predictions:
            data = json.dumps(article)
            out_file.write(data + "\n")
    
    print("{}/{} predictions ({:.2f})".format(len(list_predictions), len(list_article), 100*(len(list_predictions)/len(list_article))))       
    print("last in : {}".format(list_predictions[-1]["text"]))
    print("last prediction : {}".format(list_predictions[-1]["GENRE"]))

if __name__ == "__main__":
    PARAGRAPH_SEPARATOR = "\n"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_file", type=str)
    parser.add_argument("-o", dest="output_file", type=str)
    parser.add_argument("--yago", type=str, default="wiki-abs", help="model name : 'yago', 'wiki-abs' or absolute path from the folder with the 'model.pt' file")
    parser.add_argument("--genre", dest="type_model", action="store_const", const="GENRE")
    parser.add_argument("--mgenre", dest="type_model", action="store_const", const="mGENRE")
    parser.add_argument("--batch", type=int, default=1, help="if > 1, use a multithread executor with [batch] workers")
    parser.add_argument("--local_archive", type=str, default=None)
    parser.add_argument("--sentences", "-s", dest="split_sentences", action="store_true")
    parser.add_argument("--split_long", action="store_true")
    parser.add_argument("--eval_span", action="store_true")
    parser.add_argument("--split_iter", action="store_true")
    parser.add_argument("--article", type=int, default=None, required=False)
    parser.add_argument("--mention_trie", type=str, default=None, required=False)
    parser.add_argument("--mention_to_candidates_dict", type=str, default=None, required=False)
    parser.add_argument("--candidates_trie", type=str, default=None, required=False)
    parser.add_argument("--spacy_model", type=str, default=None, required=False)
    parser.set_defaults(type_model="GENRE")
    args = parser.parse_args()
    main(args)


#        with ThreadPoolExecutor(max_workers=workers) as executor:
#            
#            list_batch = [list_article[i:i+args.batch] for i in range(0, len(list_article), args.batch)]
#            len_batch = sum([len(x) for x in list_batch])
#            assert len_batch == len(list_article), "list article has {} items\nlist batch has {} items".format(len(list_article), len_batch) 
#            
#            futures = {
#                executor.submit(predict_output, (item, model)): item for item in list_batch #définition des process a envoyer
#            }
#            
#            iter_ = tqdm(as_completed(futures), total=len(futures), desc=desc_tqdm) #préparation du tqdm centralisé et synchronisé sur les process parallélisés           
#            results = [future.result() for future in iter_] #exécution des process           
#            for articles in results: list_predictions.extend(articles)