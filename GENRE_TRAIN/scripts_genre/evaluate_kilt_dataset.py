# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
import json
import logging
import os
import pickle
from copy import deepcopy

import jsonlines
from kilt.eval_retrieval import compute
from prettytable import PrettyTable
from tqdm.auto import tqdm

from genre.base_model import GENRE
from genre.trie import Trie
from genre.utils import batch_it, create_input
from genre.utils import get_entity_spans_fairseq as get_entity_spans
#from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from genre.utils import (
    get_micro_precision,
    get_micro_recall,
    get_micro_f1,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
)


def evaluate_kilt_dataset(
    model,
    dataset,
    batch_size=4,
    beams=10,
    max_len_a=384,
    max_len_b=15,
    candidates=False,
    trie=None,
    mention_trie=None,
    mention_candidates=None,
    candidates_set=None,
    title2id={},
    free_generation=False,
    test=False,
):

    dataset_original = deepcopy(dataset)

    gold = []
    pred = []
    document = dict()
    gold_entities = []
    guess_entities = []

    iter_ = tqdm(dataset, desc="Evaluating")
    for docs in batch_it(iter_, batch_size):

        if not free_generation:
            batch_trie = {
                i: (
                    (
                        Trie(
                            [
                                [2] + model.encode(e).tolist()[1:]
                                for e in doc["candidates"]
                            ]
                        )
                        if doc["candidates"]
                        else Trie([[2] + model.encode("NIL").tolist()[1:]])
                    )
                    if candidates
                    else trie
                )
                for i, doc in enumerate(docs)
            }
    
        final_docs = [
            create_input(
                doc,
                max_len_a,
                start_delimiter="",#"[START_ENT]",
                end_delimiter=""#"[END_ENT]",
            )
            for doc in docs
        ]
        
        if not free_generation and args.mode == "ed":
            def prefix_allowed_tokens_fn(batch_id, sent):
                return batch_trie[batch_id].get(sent.tolist())
            outputs = model.sample(
                final_docs,
                beam=beams,
                max_len_b=max_len_b,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )
        elif args.mode == "el":
            logging.info(50*"#")
            logging.info("docs {} ({}) : {}".format(type(final_docs), len(final_docs), final_docs))
            outputs = get_entity_spans(model, docs, 
                                       mention_trie=mention_trie, 
                                       candidates_trie=mention_candidates,
                                       mention_to_candidates_dict=candidates_set
                                       )               
        else: # free_generation
            outputs = model.sample(
                final_docs,
                beam=beams,
                max_len_b=max_len_b,
                prefix_allowed_tokens_fn=None
            )
            
            
        if args.mode == "el":
            #TODO calcul des scores
            logging.info(50*"#")
            logging.info("outs {} ({}) : {}".format(type(outputs), len(outputs), outputs))
            logging.info(50*"#")
            for doc, out in zip(docs, outputs):
                
                document[doc["id"]] = doc["input"]
                gold_entities.append( (doc["id"],0,0,doc["output"]["answer"]) )
                guess_entities.extend([(doc["id"],) + x for x in out])
            logging.info("gold_entities : {}".format(gold_entities))
            logging.info("guess_entities : {}".format(guess_entities))
            if not test:
                micro_p = get_micro_precision(guess_entities, gold_entities)
                micro_r = get_micro_recall(guess_entities, gold_entities)
                micro_f1 = get_micro_f1(guess_entities, gold_entities)
                macro_p = get_macro_precision(guess_entities, gold_entities)
                macro_r = get_macro_recall(guess_entities, gold_entities)
                macro_f1 = get_macro_f1(guess_entities, gold_entities)
                f1 = macro_f1
                precision = macro_p
                recall = macro_r
                iter_.set_postfix(f1=f1, prec=precision, rec=recall)
        elif args.mode == "ed":
            for doc, out in zip(final_docs, outputs):
                
                if not test:
                    gold.append(doc["output"][0]["answer"])
                    try:
                        pred.append(out[0]["text"])
                    except Exception as e:
                        pred.append("NIL")
                        print(doc)
                        print(e)
    
                doc["output"] = [
                    {
                        "answer": "",
                        "provenance": [
                            {
                                "wikipedia_id": title2id.get(prov["text"], None),
                                "title": prov["text"],
                                "score": prov["score"].item(),
                            }
                            for prov in out
                        ],
                    }
                ]
            if not test:
                true_pos = 0
                for g, p in zip(gold, pred):
                    if g == p and p != "NIL":
                        true_pos += 1
    
                precision = (
                    (true_pos / len([p for p in pred if p != "NIL"]))
                    if len([p for p in pred if p != "NIL"])
                    else 0
                )
                recall = (true_pos / len(gold)) if len(gold) else 0
                f1 = (
                    (2 * precision * recall / (precision + recall))
                    if precision + recall
                    else 0
                )
    
                iter_.set_postfix(f1=f1, prec=precision, rec=recall)
        else:
            print("no scores computed")
            exit()
       
        logging.info("outputs : \n{}\n{}\n{}".format(50*"#", outputs, 50*"#"))
        


    if not test:
        kilt_dict = compute(dataset_original, dataset, ks=[1, 5], rank_keys=["title"])
        return dataset, f1, precision, recall, kilt_dict["Rprec"], kilt_dict["recall@5"], kilt_dict["success_rate@5"]
    else:
        return dataset, 0, 0, 0, 0, 0, 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        type=str,
        help="Model path",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="model.pt",
        help="Checkpoint file",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path where to load the dataset(s)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where to save the prediction(s)",
    )
    parser.add_argument(
        "--mode",
        default="ed",
        type=str,
        help="evaluation mode. 'ed' or 'el'"        
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--beams",
        default=10,
        type=int,
        help="Number of beams",
    )
    parser.add_argument(
        "--max_len_a",
        default=384,
        type=int,
        help="Max input length",
    )
    parser.add_argument(
        "--max_len_b",
        default=15,
        type=int,
        help="Max output length",
    )
    parser.add_argument(
        "--local_archive",        
        default=None,
        help="path where find the language models archive (targz)"
    )
    parser.add_argument(
        "--trie",
        type=str,
        help="Trie pickle file (ed mode)",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Enables the use of provided candidates (ed mode)",
    )
    parser.add_argument(
        "--free_generation",
        action="store_true",
        help="Disables constrained decoding (ed and el mode)",
    )
    parser.add_argument(
        "--mention_trie",
        type=str,
        help="Trie pickle file to constrain the mention (el mode) "       
    )
    parser.add_argument(
        "--candidates_trie",
        type=str,
        help="Trie pickle file to constrain the candidates (el mode) "       
    )
    parser.add_argument(
        "--candidates_set",
        type=str,
        help="Dict pickle file of candidates set for each mentions (el mode)"        
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="CPU/GPU device",
    )
    parser.add_argument(
        "--id_title",
        type=str,
        help="ID to title map json file",
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
    parser.add_argument(
        "--test",
        help="Run tests (no evaluation)",
        action="store_true",
    )

    args = parser.parse_args()

    assert (os.path.isdir(args.input_path) and os.path.isdir(args.output_path)) or (
        not os.path.isdir(args.input_path) and not os.path.isdir(args.output_path)
    ), "`input_path` and `output_path` have either to be both files or folders"

    logging.basicConfig(level=args.loglevel)

    logging.info("Loading model")
    if "cuda" not in args.device and torch.cuda.is_available():
        logging.warning(
            "CUDA is available but running on CPU. Set --device cuda:<ID> for running on GPU."
        )

    logging.info("Preparing Archive Map")
    if args.local_archive is None:
        logging.info("choosen map : default from 'http://dl.fbaipublicfiles.com'")
        archive_map = None
        model = (
            GENRE.from_pretrained(args.model_path, checkpoint_file=args.checkpoint_file, archive_map=archive_map)
            .eval()
            .to(args.device)
        )
    else:
        archive_map = {
                "bart.base": args.local_archive,
                "bart.large" : args.local_archive
        }
        logging.info("choosen map : {}".format(args.local_archive))
        model = (
            GENRE.from_pretrained(args.model_path, checkpoint_file=args.checkpoint_file, archive_map=archive_map, local="{}/gpt2".format(args.local_archive))
            .eval()
            .to(args.device)
        )



    logging.info("candidates is {}\nfree_generation is {}".format(args.candidates, args.free_generation))

    if not args.free_generation: #and not args.candidates:
        logging.info("Loading Trie from {}".format(args.trie))
        with open(args.trie, "rb") as f:
            trie = Trie.load_from_dict(pickle.load(f))
    else:
        trie = None

    if args.id_title is not None:
        logging.info("Loading ID to title map from {}".format(args.id_title))
        with open(args.id_title) as f:
            id2title = json.load(f)
            title2id = {v: k for k, v in id2title.items()}
    else:
        title2id = {}
        
    if args.mention_trie is not None:
        logging.info("Loading mentions Trie from {}".format(args.mention_trie))
        with open(args.mention_trie, "rb") as f:
            mention_trie = Trie.load_from_dict(pickle.load(f))
    else:
        mention_trie = None
        
    if args.candidates_trie is not None:
        logging.info("Loading candidates Trie from {}".format(args.candidates_trie))
        with open(args.candidates_trie, "rb") as f:
            candidates_trie = Trie.load_from_dict(pickle.load(f))
    else:
        candidates_trie = None
        
    if args.candidates_set is not None:
        logging.info("Loading candidates set dict from {}".format(args.candidates_set))
        with open(args.candidates_set, "rb") as f:
            candidates_trie = pickle.load(f)
    else:
        candidates_set = None

    results = PrettyTable()
    results.field_names = [
        "Dataset",
        "F1",
        "Precision",
        "Recall",
        "R-precision",
        "Recall@5",
        "Accuracy@5" #success_rate@5
    ]

    datasets_filenames = (
        [os.path.join(args.input_path, fname) for fname in os.listdir(args.input_path)]
        if os.path.isdir(args.input_path)
        else [args.input_path]
    )

    for dataset_filename in datasets_filenames:

        logging.info("Loading {}".format(dataset_filename))
        with jsonlines.open(dataset_filename) as f:
            dataset = [e for e in f]

        dataset, f1, precision, recall, rprec, recall_at_5, accuracy = evaluate_kilt_dataset(
            model,
            dataset,
            args.batch_size,
            args.beams,
            args.max_len_a,
            args.max_len_b,
            args.candidates,
            trie,
            mention_trie,
            candidates_trie,
            candidates_set,
            title2id,
            args.free_generation,
            args.test,
        )

        results.add_row(
            [
                os.path.splitext(os.path.basename(dataset_filename))[0],
            ]
            + [
                "{:.2f}".format(100 * e)
                for e in (f1, precision, recall, rprec, recall_at_5, accuracy)
            ]
        )

        output_filename = (
            os.path.join(args.output_path, os.path.basename(dataset_filename))
            if os.path.isdir(args.output_path)
            else args.output_path
        )
        logging.info("Saving dataset in {}".format(output_filename))
        with jsonlines.open(output_filename, "w") as f:
            f.write_all(dataset)

    print(results)
