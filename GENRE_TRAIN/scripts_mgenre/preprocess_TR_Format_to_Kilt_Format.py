#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:46:38 2022

@author: carpentier
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pickle

import jsonlines
#import pandas
import numpy as np
from genre.utils import get_wikidata_ids #, chunk_it
from tqdm.auto import tqdm

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

def process_TR_1(doc, mentions):
    item = []
    for i, mention in enumerate(mentions):
        start, end, fr_title, title, is_hard = mention.strip().split("\t")
        start, end, is_hard = int(start), int(end), bool(int(is_hard))
        if (lang_title2wikidataID is not None) and (lang_redirect2title is not None) and (label_or_alias2wikidataID is not None):
            wikidataIDs = list(get_wikidata_ids(
                fr_title.replace("_", " "),
                lang,
                lang_title2wikidataID,
                lang_redirect2title,
                label_or_alias2wikidataID,
            )[0])
        else: wikidataIDs = [{"title": fr_title}]

        meta = {
            "left_context": doc[:start].strip(),
            "mention": doc[start:end].strip(),
            "entity": fr_title,
            "right_context": doc[end:].strip(),
        }
        base_item = {
            "id": "{}-{}-{}".format(lang, filename, i),
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
                               fr_title + 
                               " ] " + 
                               meta["right_context"], 
                     "provenance": wikidataIDs,
                     "start": start,
                     "end": end
                    }
            ],
            "meta": meta,
            "is_hard": is_hard,
        }
        item.append(base_item)
    return item

def process_TR_2(doc, mentions):
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
    mentions_sort = []
    j = 0
    for i, mention in enumerate(mentions):
        start, end, title, fr_title, is_hard = mention.strip().split("\t")
        start, end, is_hard = int(start), int(end), bool(int(is_hard))
        entity = fr_title.replace("_", " ").strip()
        mentions_sort.append((start, end, entity, is_hard)) # sort the mention to prevent massive errors
    mentions_sort.sort(key=lambda x: x[0])
    for i, mention in enumerate(mentions_sort):
        start, end, entity, is_hard = mention
        if start < j: continue
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
        
        left = doc[j:start]
        mention = doc[start:end]
        #right = doc[end:].strip()
        j = end #new start index for doc
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
        hard_array.append(is_hard)
        ## provenance array
        prov_array.append(wikidataIDs)
    ## last part of the text (from the last mention to then end of the doc)
    array_item.append(doc[j:]) 
    meta["context"].append(doc[j:])
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
    assert input_text.strip() == doc.strip(), "final input :\n{}\n{}\noriginal doc :\n{}\n{}\nliste mentions : \n{}\n{}\nliste entités : \n{}\n{}\n{}".format(input_text.strip(), 20*"~", doc.strip(), 20*"~", meta["mentions"], 20*"~", meta["entities"], 20*"~", array_item)
    item = {
        "id": "{}-{}-{}".format(lang, filename, i),
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
    return item

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
        "--input_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
        help="Base folder with Wikidata data.",
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

    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.DEBUG)

    lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID = load_wikidata_dicts(args.base_wikidata)

    for lang in os.listdir(args.input_dir):
        if lang != "fr": continue #seulement en français
        logging.info("Converting {}".format(lang))
        for split in  os.listdir(os.path.join(args.input_dir, lang)): #("test", "train"):

            kilt_dataset = []
            for filename in tqdm(
                set(
                    ".".join(e.split(".")[:-1])
                    for e in os.listdir(os.path.join(args.input_dir, lang, split))
                )
            ):
                with open(
                    os.path.join(args.input_dir, lang, split, filename + ".txt")
                ) as f:
                    doc = f.read()

                with open(
                    os.path.join(args.input_dir, lang, split, filename + ".mentions")
                ) as f:
                    mentions = f.readlines()

                if args.mode == 1: kilt_dataset.append(process_TR_1(doc, mentions))
                elif (args.mode >= 2): kilt_dataset.append(process_TR_2(doc, mentions))
                else: continue

            if len(kilt_dataset) > 0: 
                generate_stats_mentions_doc(kilt_dataset)
            
                filename = os.path.join(
                    args.output_dir, "{}-kilt-{}.jsonl".format(lang, split)
                )
                logging.info("Saving {}".format(filename))
                with jsonlines.open(filename, mode='w') as f:
                    f.write_all(kilt_dataset)
    
                kilt_dataset = [e for e in kilt_dataset if e["is_hard"]] 
    
                filename = os.path.join(
                    args.output_dir, "{}-hard.jsonl".format(filename.split(".")[0])
                )
                logging.info("Saving {}".format(filename))
                with jsonlines.open(filename, mode='w') as f:
                    f.write_all(kilt_dataset)
            else: print("empty dataset")
    label_or_alias2wikidataID = None
    lang_title2wikidataID = None
    lang_redirect2title = None
    exit(0)