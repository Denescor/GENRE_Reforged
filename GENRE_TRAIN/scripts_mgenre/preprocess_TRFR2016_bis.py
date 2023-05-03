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
import re

import jsonlines
import pandas
from genre.utils import chunk_it, get_wikidata_ids
from tqdm.auto import tqdm, trange

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

    filename = os.path.join(args.base_wikidata, "lang_title2wikidataID.pkl")
    logging.info("Loading {}".format(filename))
    #with open(filename, "rb") as f:
    #    lang_title2wikidataID = pickle.load(f)

    filename = os.path.join(args.base_wikidata, "lang_redirect2title.pkl")
    logging.info("Loading {}".format(filename))
    #with open(filename, "rb") as f:
    #    lang_redirect2title = pickle.load(f)

    filename = os.path.join(args.base_wikidata, "label_or_alias2wikidataID.pkl")
    logging.info("Loading {}".format(filename))
    #with open(filename, "rb") as f:
    #    label_or_alias2wikidataID = pickle.load(f)

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
                j = 0
                for i, mention in enumerate(mentions):
                    start, end, fr_title, title, is_hard = mention.strip().split("\t")
                    start, end, is_hard = int(start), int(end), bool(int(is_hard))
                    #wikidataIDs = get_wikidata_ids(
                    #    title.replace("_", " "),
                    #    lang,
                    #    lang_title2wikidataID,
                    #    lang_redirect2title,
                    #    label_or_alias2wikidataID,
                    #)[0]
                    
                    left = doc[j:start].strip()
                    mention = doc[start:end].strip()
                    entity = fr_title
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
                    prov_array.append({"title": fr_title})
                ## last part of the text (from the last mention to then end of the doc)
                array_item.append(doc[j:].strip()) 
                meta["context"].append(doc[j:].strip())
                ## write input and output
                output_text = ""
                input_text = ""
                for text in array_item:
                    if type(text) == str: 
                        output_text += text
                        input_text += text
                    elif type(text) == tuple: 
                        me_ent = " { " + text[0] + " } " + "[ {} ] ".format(text[1])
                        output_text += me_ent
                        input_text += " {} ".format(text[0])
                    else: continue
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
                kilt_dataset.append(item)

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
