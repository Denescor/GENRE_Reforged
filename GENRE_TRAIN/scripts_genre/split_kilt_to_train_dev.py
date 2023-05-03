#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:11:09 2022

@author: carpentier
"""

import argparse
import logging
import random
import pickle
import os

import jsonlines
import numpy as np
from tqdm import tqdm
from genre.utils import get_wikidata_ids
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_stats_mentions_doc(final_dataset):  
    lenght_list = []
    mentions_list = []
    for item in final_dataset:
        mentions_list.append(len(item["meta"]["mentions"]))
        lenght_list.append(len(item["input"]))
    #stat taille doc
    #stat nb mentions
    assert len(lenght_list) == len(mentions_list)
    ### Per list
    len_total = len(lenght_list)
    min_mentions = min(mentions_list)
    max_mentions = max(mentions_list)
    no_mention = len([1 for x in mentions_list if x == min_mentions])
    med_mentions = int(np.median(mentions_list))
    mean_mentions = np.mean(mentions_list)
    med_len = int(np.median(lenght_list))
    mean_len = np.mean(lenght_list)
    ### Quantiles
    first_mentions_quant = int(np.quantile(mentions_list, 0.25))
    last_len_quant = int(np.quantile(lenght_list, 0.75))
    first_len_quant = int(np.quantile(lenght_list, 0.25))
    ### Nb mentions per docs
    lot_mentions_short_doc = len([1 for i in range(len_total) if mentions_list[i] > med_mentions and lenght_list[i] <= med_len])
    lot_mentions_long_doc = len([1 for i in range(len_total) if mentions_list[i] > med_mentions and lenght_list[i] > med_len])
    low_mentions_text = len([1 for i in range(len_total) if mentions_list[i] <= first_mentions_quant and lenght_list[i] >= last_len_quant])
    few_mentions_text = len([1 for i in range(len_total) if mentions_list[i] <= first_mentions_quant and lenght_list[i] >= med_len])
    med_first_mentions_text = len([1 for i in range(len_total) if mentions_list[i] <= med_mentions and lenght_list[i] >= last_len_quant])
    med_second_mentions_text = len([1 for i in range(len_total) if mentions_list[i] <= med_mentions and lenght_list[i] >= med_len])
    few_text_mentions = len([1 for i in range(len_total) if mentions_list[i] <= med_mentions and lenght_list[i] <= med_len])
    low_text_mentions = len([1 for i in range(len_total) if mentions_list[i] <= med_mentions and lenght_list[i] <= first_len_quant])
    few_text_and_mentions = len([1 for i in range(len_total) if mentions_list[i] <= first_mentions_quant and lenght_list[i] <= med_len])
    no_text_mentions = len([1 for i in range(len_total) if mentions_list[i] <= first_mentions_quant and lenght_list[i] <= first_len_quant])
    mean_mentions_doc = np.mean(mentions_list)/np.mean(lenght_list)
    med_mentions_doc = med_mentions / med_len
    
    logging.info(100*"#")
    logging.info("#\tdoc lenght median : {} chars\t doc lenght mean : {:.2f} chars".format(med_len, mean_len))
    logging.info("#\tnb mentions per doc median : {}\t nb mentions per doc mean : {:.2f}".format(med_mentions, mean_mentions))
    logging.info("#\tmean of mentions per doc : {:.5f}".format(mean_mentions_doc))
    logging.info("#\tmedian of mentions per doc : {:.5f}".format(med_mentions_doc))
    logging.info("#\tmin nb mentions : {}\t max nb mentions : {}".format(min_mentions, max_mentions))
    logging.info("#\tdoc with min mentions (= {}) : {}/{} ({:.2f}%)".format(min_mentions, no_mention, len_total, 100*(no_mention/len_total)))
    logging.info("#\tshort documents (<= {} chars) with a lot of mentions (> {}) : {}/{} ({:.2f}%)".format(med_len, med_mentions, lot_mentions_short_doc, len_total, 100*(lot_mentions_short_doc/len_total)))
    logging.info("#\tlong documents (> {} chars) with a lot of mentions (> {}) : {}/{} ({:.2f}%)".format(med_len, med_mentions, lot_mentions_long_doc, len_total, 100*(lot_mentions_long_doc/len_total)))
    logging.info("#\tlarge documents (>= {} chars) with very few mentions (<= {}) : {}/{} ({:.2f}%)".format(last_len_quant, first_mentions_quant, low_mentions_text, len_total, 100*(low_mentions_text/len_total)))
    logging.info("#\tlong documents (>= {} chars) with very few mentions (<= {}) : {}/{} ({:.2f}%)".format(med_len, first_mentions_quant, few_mentions_text, len_total, 100*(few_mentions_text/len_total)))
    logging.info("#\tlarge documents (>= {} chars) with few mentions (<= {}) : {}/{} ({:.2f}%)".format(last_len_quant, med_mentions, med_first_mentions_text, len_total, 100*(med_first_mentions_text/len_total)))
    logging.info("#\tlong documents (>= {} chars) with few mentions (<= {}) : {}/{} ({:.2f}%)".format(med_len, med_mentions, med_second_mentions_text, len_total, 100*(med_second_mentions_text/len_total)))
    logging.info("#\tshort documents (<= {} chars) with few mentions (<= {}) : {}/{} ({:.2f}%)".format(med_len, med_mentions, few_text_mentions, len_total, 100*(few_text_mentions/len_total)))
    logging.info("#\tvery short documents (<= {} chars) with few mentions (<= {}) : {}/{} ({:.2f}%)".format(first_len_quant, med_mentions, low_text_mentions, len_total, 100*(low_text_mentions/len_total)))
    logging.info("#\tshort documents (<= {} chars) with very few mentions (<= {}) : {}/{} ({:.2f}%)".format(med_len, first_mentions_quant, few_text_and_mentions, len_total, 100*(few_text_and_mentions/len_total)))
    logging.info("#\tvery short documents (<= {} chars) with very few mentions (<= {}) : {}/{} ({:.2f}%)".format(first_len_quant, first_mentions_quant, no_text_mentions, len_total, 100*(no_text_mentions/len_total)))
    logging.info(100*"#")

def split_kilt(dataset, prop, dense):
    train_set = dict()
    dev_set = dict()
    list_doc = len(dataset) #for reproductibility
    max_pick = int(dense*list_doc) #max doc to pick from dataset
    random.seed(1158) #for reproductibility
    len_doc = int(prop*max_pick) #nombre de documents à déplacer
    pick_doc = set(random.sample([i for i in range(max_pick)], len_doc)) #liste des documents à déplacer
    for i, (idoc, doc) in tqdm(enumerate(dataset.items()), total=max_pick, desc="Split"):
        if i >= max_pick: break #end of picking
        if i in pick_doc: dev_set[idoc] = doc
        else: train_set[idoc] = doc
    logging.info("initial proportion keep from original wiki : {}%\ninitial dev proportion choose: {}%\nsize train : {} ({:.2f}%)\nsize dev : {} ({:.2f}%)".format(100*dense, 100*prop, len(train_set), 100*(len(train_set)/max_pick), len(dev_set), 100*(len(dev_set)/max_pick)))
    return train_set, dev_set

def write_dataset(filename, dataset, lang, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID):
    logging.info("Saving {}".format(filename))
    final_dataset = []
    ## Executor multithreads
    #with ThreadPoolExecutor(max_workers=32) as executor:
    #    futures = {
    #        executor.submit(collect_item_dataset, [title, doc]): [title, doc] for title, doc in dataset.items()
    #    }
    #    iter_ = tqdm(as_completed(futures), total=len(futures), smoothing=0, desc="Create items")
    #    results_temp = [future.result() for future in iter_]
    ## Merge results from threads
    #for item in results_temp:
    for title, doc in tqdm(dataset.items(), total=len(dataset), desc="Create items"):
        if args.mode == 1: final_dataset.extend(collect_item_dataset_1([title, doc]))
        elif (args.mode >= 2): 
            item = collect_item_dataset_2([title, doc, lang, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID])
            final_dataset.extend(item)
        else: continue
    logging.info("Number of items : {}".format(len(final_dataset)))
    generate_stats_mentions_doc(final_dataset)
    
    #logging.info("type dataset: {} of {}".format(type(final_dataset), type(final_dataset[0])))
    #logging.info("exemple : {}".format(final_dataset[:3]))
    ## write final dataset
    if args.format == "pkl":
        with open("{}.pkl".format(filename), "wb") as f:
            pickle.dump(final_dataset, f)        
    with jsonlines.open("{}.jsonl".format(filename), mode='w') as f:
        f.write_all(final_dataset)         

def collect_item_dataset_2(entry):
    title = entry[0]
    page = entry[1]
    lang = entry[2]
    lang_title2wikidataID = entry[3]
    lang_redirect2title = entry[4]
    label_or_alias2wikidataID = entry[5]
    doc = " ".join(page["paragraphs"]).strip()
    array_item = []
    meta = {
        "context": [],
        "mentions": [],
        "entities": []
    }
    start_array = []
    end_array = []
    hard_array = []
    prov_array = []
    i = 0
    for anchor in page["anchors"]:
        fr_title = anchor["href"]
        start = int(anchor["start"])
        end = int(anchor["end"])
        left = doc[i:start]
        mention = doc[start:end]
        entity = fr_title.replace("_", " ").strip()
        #right = doc[end:].strip()
        i = end #new start index for doc
        ## array item
        array_item.append(left)
        array_item.append((mention,entity))
        ## meta
        meta["context"].append(left)
        meta["mentions"].append(mention)
        meta["entities"].append(entity)
        ## start & end array
        start_array.append(start)
        end_array.append(end)
        ## hard array
        hard_array.append(entity != mention) #unused
        ## provenance array
        if (lang_title2wikidataID is not None) and (lang_redirect2title is not None) and (label_or_alias2wikidataID is not None):
            wikidataIDs, source = get_wikidata_ids(
                entity,
                lang,
                lang_title2wikidataID,
                lang_redirect2title,
                label_or_alias2wikidataID,
            )
            if type(wikidataIDs) == set: wikidataIDs = {"provenance" : source, "title": fr_title, "wikidataIDs" : [ID for ID in wikidataIDs]}
        else: wikidataIDs = {"title": fr_title}
        prov_array.append(wikidataIDs)
    ## last part of the text (from the last mention to then end of the doc)
    array_item.append(doc[i:]) 
    meta["context"].append(doc[i:])
    ## write input and output
    output_text = ""
    input_text = ""
    for text in array_item:
        if type(text) == str: 
            output_text += text
            input_text += text
        elif type(text) == tuple: 
            me_ent = "{ " + text[0] + " } " + "[ {} ] ".format(text[1])
            output_text += me_ent
            input_text += "{}".format(text[0])
        else: continue
    assert input_text == doc.strip(), "final input :\n{}\n{}\noriginal doc :\n{}\n{}\nliste mentions : \n{}".format(input_text, 20*"~", doc.strip(), 20*"~", meta["mentions"])
    item = {
        "id": "fr-{}".format(title),
        "input": input_text,
        "output": [
                {"answer": output_text, 
                 "provenance": prov_array,
                 "start": start_array,
                 "end": end_array
                }
        ],  #list(wikidataIDs)
        "meta": meta,
        "is_hard": hard_array,
    }
    return [item] 
        
        
def collect_item_dataset_1(entry):
    i = 1
    title = entry[0]
    page = entry[1]
    doc = " ".join(page["paragraphs"])
    final_dataset = []
    max_lenght = 192 #192
    for anchor in page["anchors"]:
        fr_title = anchor["href"]
        start = int(anchor["start"])
        end = int(anchor["end"])
        try: left = (doc[:start].strip())[-max_lenght:]
        except IndexError: left = doc[:start].strip() #left is less than 192 chars
        try: right = (doc[end:].strip())[:max_lenght]
        except IndexError: right = doc[end:].strip() #right is less than 192 chars
        meta = {
            "left_context": left,
            "mention": doc[start:end].strip(),
            "entity": fr_title,
            "right_context": right,
        }
        item = {
            "id": "fr-{}-{}".format(title, i),
            "input": (
                meta["left_context"]
                + " "
                + meta["mention"]
                + " "
                + meta["right_context"]
            ),
            "output": [
                    {"answer": meta["left_context"] + 
                               " { " + 
                               meta["mention"] + 
                               " } " + 
                               "[ " + 
                               meta["entity"] + 
                               " ] " + 
                               meta["right_context"], 
                     "provenance": [{"title": meta["entity"]}],
                     "start": start,
                     "end": end
                    }
            ],  #list(wikidataIDs)
            "meta": meta,
            "is_hard": 0,
        }
        final_dataset.append(item)
        i += 1
    return final_dataset

def load_wikidata_dicts(base_wikidata):
    """
        Load wikidata dicts for preprocess
    """
    filename = os.path.join(base_wikidata, "lang_title2wikidataID.pkl")
    logging.info("Loading {}".format(filename))
    try:
        with open(filename, "rb") as f:
            lang_title2wikidataID = pickle.load(f)
    except: 
        logging.info("Loading {} failed".format(filename))
        lang_title2wikidataID = None

    filename = os.path.join(base_wikidata, "lang_redirect2title.pkl")
    logging.info("Loading {}".format(filename))
    try:
        with open(filename, "rb") as f:
            lang_redirect2title = pickle.load(f)
    except: 
        logging.info("Loading {} failed".format(filename))
        lang_redirect2title = None

    filename = os.path.join(base_wikidata, "label_or_alias2wikidataID.pkl")
    logging.info("Loading {}".format(filename))
    try:
        with open(filename, "rb") as f:
            label_or_alias2wikidataID = pickle.load(f)
    except: 
        logging.info("Loading {} failed".format(filename))
        label_or_alias2wikidataID = None
    return lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_filename",
        type=str,
        help="Filename of the KILT dataset",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where to save the converted dataset",
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
        help="Base folder with Wikidata data.",
    )
    parser.add_argument(
        "--proportion_dev",
        default=0.15,
        type=float,
        help="proportion of the initial dataset to put in dev dataset. Remain will be drop to the train dataset"
    )
    parser.add_argument(
        "--proportion_wiki",
        default=1,
        type=float,
        help="proportion from the initial wiki dump keep for the final dataset (between 0 and 1)"        
    )
    parser.add_argument(
        "--format",
        default="jsonl",
        help="format of the kilt dataset ('pkl' or 'jsonl')"        
    )
    parser.add_argument( 
        "--mode",
        default=1,
        type=int,
        help="mode for preprocess.\n- 1 = 1 mention of 1 document per entry (as ED)\n- 2 & 3 = 1 document per entry"
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
 
    if "fr" in args.input_filename: lang = "fr"
    elif "en" in args.input_filename: lang = "en"
    else: 
        logging.info("unrecognized wiki lang")
        exit(0)
    logging.info("lang valid : {}".format(lang))
    
    lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID = load_wikidata_dicts(args.base_wikidata)

    logging.info("Loading {}".format(args.input_filename))
    if args.format == "pkl":
        with open(args.input_filename, "rb") as f:
            dataset = pickle.load(f)
    elif args.format == "jsonl":
        with jsonlines.open(args.input_filename) as f:
            dataset = [e for e in f]
    else:
        print("{} non pris en compte".format(args.format))
        exit(0)
    
    logging.info("initial wiki (size : {}): {}".format(len(dataset),type(dataset)))
    #logging.info("exemple frwiki : {}".format(dataset[list(dataset.keys())[10]]))
    
    train_set, dev_set = split_kilt(dataset, args.proportion_dev, args.proportion_wiki)
    dataset = None
    
    logging.info("new train wiki (size : {}): {}".format(len(train_set), type(train_set)))
    logging.info("new dev wiki (size : {}): {}".format(len(dev_set), type(dev_set)))

    filename = "{}/wiki-dev-kilt".format(args.output_path)
    write_dataset(filename, dev_set, lang, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID)
    dev_set = None
    
    filename = "{}/wiki-train-kilt".format(args.output_path)
    write_dataset(filename, train_set, lang, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID)
    train_set = None
    
    lang_redirect2title = None
    lang_title2wikidataID = None
    label_or_alias2wikidataID = None
    exit(0)