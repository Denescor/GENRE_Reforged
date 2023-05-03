#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:07:01 2022

@author: carpentier
"""

import argparse
import logging
import os
import pickle
import re

import pandas
import jsonlines
import numpy as np
from genre.utils import create_input_el, get_wikidata_ids#, chunk_it, 
from tqdm.auto import tqdm, trange

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

def load_wikidata_dicts(base_wikidata):
    """
        Load wikidata dicts only for ED preprocess
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

def from_aida_txt_to_el_json(aida_name, aida_set, delimiter_mnt, delimiter_ent):
    new_dataset = []
    idtoent = dict()
    
    input_text, output_text, title = "", "", ""
    prov_array, start_array, end_array = [], [], []
    context, mentions, entities = [], [], []
    total_doc = 0
    doc_error = 0
    nb_universe = 0
    total_men_flag = 0
    assertion_error = 0
    
    with open(args.input_dir+args.wiki_name_id_map, "r") as f:
        for line in f:
            try: ent, i = line.strip().split("\t")
            except ValueError: pass
            else: idtoent[i] = ent
            finally: nb_universe += 1
    logging.info("{}/{} ({:.2f}%) entities in universe".format(len(idtoent), nb_universe,  100*(len(idtoent)/nb_universe) ))
    #logging.debug("exemple idtoent : {}".format(list(idtoent.keys())[:10]))
    
    with open(aida_set, "r") as aida:
        mention = ""
        entity = ""
        current_context = ""
        men_flag = False
        doc_flag = False
        incrementor = 0
        for word in tqdm(aida, desc="Format 2 & 3 : process {}".format(aida_name)):
            if word.startswith("DOCSTART_"):  #start of a doc == begin a new entry
                assert not doc_flag, "new doc without closing older\n\t- flag : {}\n\t- incrementor : {}".format(doc_flag, incrementor)
                doc_flag = True
                total_doc += 1
                title = "{}-{}".format(aida_name, word[9:].strip())
                input_text, output_text = "", ""
                prov_array, start_array, end_array = [], [], []
                context, mentions, entities = [], [], []
            elif word.startswith("DOCEND"): #end of a doc == end of an entry == save the final entry
                assert doc_flag, "close a doc twice\n\t- flag : {}\n\t- doc ({}) : {}\n\t- incrementor : {}".format(doc_flag, len(input_text), input_text, incrementor)
                meta = {
                    "context": context,
                    "mentions": mentions,
                    "entities": entities
                }
                entry = {
                    "id": title,
                    "input": input_text,
                    "output": [
                        {
                            "answer": output_text,
                            "provenance": prov_array,
                            "start": start_array,
                            "end": end_array
                        }
                    ],
                    "meta": meta
                }
                #logging.debug("{} completed:\n\t- incrementor : {}\n\t- nb mentions : {}".format(title, incrementor, len(mentions)))
                incrementor = 0
                doc_flag = False
                men_flag = False #in case of previous men_flag error
                mention, current_context = "", "" #in case of previous men_flag error
                if len(input_text) == 0 or len(entities) == 0 or (len(entities) != len(mentions)): 
                    doc_error += 1
                    continue
                else: new_dataset.append(entry)
            elif word.startswith("*NL*"): continue
            elif word.startswith("MMSTART_"): # start of a mention == end of current left context
                total_men_flag += 1
                if men_flag: #men_flag error --> previous MMEND missing --> canceling current mention
                    men_flag = False
                    input_text += mention
                    output_text += mention
                    current_context += mention
                    assertion_error += 1
                    assert incrementor == len(input_text), "overflow incrementor at men_flag error correction : \n\t- doc ({}) : {}\n\t- mention ({}) : {}\n\t- incrementor : {}".format(len(input_text), input_text, len(mention), mention, incrementor)
                try: entity = idtoent[word[8:].strip()]
                except KeyError: continue #logging.debug("{} not in idtoent".format(word[8:].strip()))
                else:
                    men_flag = True
                    entities.append(entity)
                    context.append(current_context)
                    start_array.append(incrementor)
                    mention, current_context = "", ""
                    assert incrementor == len(input_text), "overflow incrementor at mention start : \n\t- doc ({}) : {}\n\t- incrementor : {}".format(len(input_text), input_text, incrementor)
            elif word.startswith("MMEND"): # end of a mention == save the mention + entities in output_text
                if men_flag:
                    men_flag = False
                    input_text += mention
                    mentions.append(mention[:-1])
                    end_array.append(incrementor-1)
                    incrementor_mention = input_text[start_array[-1]:end_array[-1]]
                    # output = { mention } [ entity ]
                    output_text += "{} {} {} {} {} {} ".format(delimiter_mnt[0], mention[:-1], delimiter_mnt[1], delimiter_ent[0], entity, delimiter_ent[1])
                    assert incrementor == len(input_text), "overflow incrementor at mention end : \n\t- doc ({}) : {}\n\t- incrementor : {}".format(len(input_text), input_text, incrementor)
                    assert incrementor_mention == mention[:-1], "incrementor error :\n\t- doc ({}): '{}'\n\t- incrementor mention : '{}' ({}:{})\n\t- true mention : '{}'".format(len(input_text), input_text, incrementor_mention, start_array[-1], end_array[-1], mention[:-1])
                else: continue # entity not found in universe
            else: # == a simple word
                word_to_add = word.strip()
                if men_flag:
                    mention += "{} ".format(word_to_add)
                else:
                    input_text += "{} ".format(word_to_add)
                    output_text += "{} ".format(word_to_add)
                    current_context += "{} ".format(word_to_add)
                incrementor += len(word_to_add)+1
                assert (incrementor == len(input_text) + len(mention)) or (incrementor == len(input_text)), "incrementor error:\n\t- doc_flag : {}\n\t- men_flag : {}\n\t- doc ({}) : '{}'\n\t- mention ({}) : {}\n\t- word ({}) : {}\n\t- incrementor : {}".format(doc_flag, men_flag, len(input_text), input_text, len(mention), mention, len(word.strip()), word_to_add, incrementor)
            assert len(input_text) == 0 or men_flag or incrementor <= len(input_text), "overflow incrementor:\n\t- doc ({}) : {}\n\t- incrementor : {}\n\t- process mention : {}".format(len(input_text), input_text, incrementor, men_flag)
    if assertion_error > 0:
        logging.info("canceling non compliant mentions : {} ({:.2f}%)".format(assertion_error, 100*(assertion_error/total_men_flag)))
    if doc_error > 0:
        logging.info("passed empty doc or doc without mentions : {} ({:.2f}%)".format(doc_error, 100*(doc_error/total_doc)))
    return new_dataset

def from_aida_txt_to_ed_json(aida_name, aida_set, delimiter, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID):
    new_dataset = []
    nb_universe = 0
    idtoent=dict()
    with open(args.input_dir+args.wiki_name_id_map, "r") as f:
        for line in f:
            try: ent, i = line.strip().split("\t")
            except ValueError: pass
            else: idtoent[i] = ent
            finally: nb_universe += 1
    logging.info("{}/{} ({:.2f}%) entities in universe".format(len(idtoent), nb_universe,  100*(len(idtoent)/nb_universe) ))
    with open(aida_set, "r") as aida:
        mention = ""
        entity = ""
        men_flag = False
        incrementor = 0
        for word in tqdm(aida, desc="Format 4 : process {}".format(aida_name)):
            if word.startswith("DOCSTART_"):  #start of a doc == begin a new entry
                title = "{}-{}".format(aida_name, word[9:].strip())
                doc = ""
                temp_dataset = []
            elif word.startswith("DOCEND"): #end of a doc == end of an entry == save the final entry
                for entry in temp_dataset:
                    total_left = doc[:entry["output"][0]["start"]]
                    total_right = doc[entry["output"][0]["end"]:]
                    if len(total_left) >= 384: left_context = total_left[(len(total_left)-384):]
                    else: left_context = total_left
                    if len(total_right) >= 384: right_context = total_right[(len(total_right)-384):]
                    else: right_context = total_right
                    entry["meta"]["left-context"] = left_context
                    entry["meta"]["right-context"] = right_context
                    entry["input"] = "{} {} {} {} {}".format(left_context, delimiter[0], entry["meta"]["mention"], delimiter[1], right_context)
                new_dataset.extend(temp_dataset)
                #logging.debug("{} completed:\n\t- incrementor : {}\n\t- nb mentions : {}".format(title, incrementor, len(mentions)))
                incrementor = 0
            elif word.startswith("*NL*"): continue
            elif word.startswith("MMSTART_"): # start of a mention == end of current left context
                try: entity = idtoent[word[8:].strip()]
                except KeyError: continue #logging.debug("{} not in idtoent".format(word[8:].strip()))
                else:
                    men_flag = True
                    start = incrementor
                    mention = ""
            elif word.startswith("MMEND"): # end of a mention == save the mention + entities in output_text
                if men_flag:
                    men_flag = False
                    end = incrementor-1
                    incrementor_mention = "".join(doc[start:end])
                    if (lang_title2wikidataID is not None) and (lang_redirect2title is not None) and (label_or_alias2wikidataID is not None):
                        wikidataIDs = list(get_wikidata_ids(
                            entity.replace("_", " "),
                            "en",
                            lang_title2wikidataID,
                            lang_redirect2title,
                            label_or_alias2wikidataID,
                        )[0])
                    else: wikidataIDs = [{"title": entity}]
                    assert incrementor_mention == mention[:-1], "incrementor error :\n\t- doc : '{}'\n\t- incrementor mention : '{}' ({}:{})\n\t- true mention : '{}'".format(doc, incrementor_mention, start, end, mention[:-1])
                    meta = {
                        "left-context": None,
                        "mention": mention[-1],
                        "right-context": None
                    }
                    entry = {
                        "id": "{}-{}-{}".format(title, aida_name, start),
                        "input": None,
                        "output": [
                            {
                                "answer": entity,
                                "provenance": wikidataIDs,
                                "start": start,
                                "end": end
                            }
                        ],
                        "meta": meta
                    }
                    temp_dataset.append(entry)
                else: continue # entity not found in universe
            else: # == a simple word
                if men_flag:
                    mention += "{} ".format(word.strip())
                doc += "{} ".format(word.strip())
                incrementor += len(word.strip())+1
                assert men_flag or (len(doc) == incrementor), "incrementor error:\n\t- doc : '{}'\n\t- len input : {}\n\t- word : {} ({})\n\t- incrementor : {}".format(doc, len(doc), word.strip(), len(word.strip()), incrementor)
    for entry in new_dataset:
        left = entry["meta"]["left-context"]
        right = entry["meta"]["right-context"]
        assert entry["input"] is not None, "error during preprocess"
        assert (left is not None) and (len(left) <= 384), "error during preprocess"
        assert (right is not None) and (len(right) <= 384), "error during preprocess"
    return new_dataset

def from_ed_json_to_el_json(aida_name, aida_set, delimiter_mnt, delimiter_ent):
    new_dataset = []
    with jsonlines.open(aida_set) as f:
        dataset = [e for e in f]
    for entry in tqdm(dataset, total=len(dataset), desc="Format 1 : process {}".format(aida_name)):
        # change meta of the entry
        meta = entry["meta"]
        try: meta["entity"] = entry["output"][0]["answer"]
        except KeyError: 
            logging.debug(entry["output"])
            continue
        entry["meta"] = meta
        # the document without annotations
        input_e = " ".join([meta["left_context"], meta["mention"], meta["right_context"]]) 
        entry["input"] = input_e
        # the document with annotations
        entry["output"][0]["answer"] = create_input_el(entry, max_length=384, span_delimiter=delimiter_mnt, entity_delimiter=delimiter_ent)
        # save new entry
        new_dataset.append(entry)
    return new_dataset

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
        "--wiki_name_id_map",
        type=str,
        help="wiki_name_id_map name (in imput_folder).",
    )
    parser.add_argument(
        "--format",
        type=int,
        default=1,
        help="mode for preprocess.\n- 1 = 1 mention of 1 document per entry (as ED)\n- 2 & 3 = 1 document per entry\n- 4 = 1 mention of 1 document per entry (for ED)"
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
    
    #logging.info(os.listdir(args.input_dir))
    #logging.info([os.path.splitext(x) for x in os.listdir(args.input_dir)])
    #logging.info([os.path.isfile(x) for x in os.listdir(args.input_dir)])
    
    delimiter_mnt = "{}"
    delimiter_ent = "[]"
    delimiter_ed = ["[START_ENT]","[END_ENT]"]
    
    if args.format == 1:    
        logging.info("Format choosen : 1 mention of 1 document per entry (as ED)")
        aida_list = [(os.path.splitext(x)[0], args.input_dir+x) for x in os.listdir(args.input_dir) if os.path.splitext(x)[1]==".jsonl"]
        logging.info("datasets : {}".format([x[0] for x in aida_list]))
    elif (args.format == 4):
        logging.info("Format choosen : 1 mention of 1 document per entry (for ED)")
        lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID = load_wikidata_dicts(args.base_wikidata)
        aida_list = [(os.path.splitext(x)[0], args.input_dir+x) for x in os.listdir(args.input_dir) if os.path.splitext(x)[1]==".txt"]
        logging.info("datasets : {}".format([x[0] for x in aida_list]))
    elif (args.format >= 2):
        logging.info("Format choosen : 1 document per entry")
        aida_list = [(os.path.splitext(x)[0], args.input_dir+x) for x in os.listdir(args.input_dir) if os.path.splitext(x)[1]==".txt"]
        logging.info("datasets : {}".format([x[0] for x in aida_list]))
    else:
        logging.info("Format not recognized")
        exit(1)

    for aida_name, aida_set in aida_list:
        
        logging.info("Loading {}".format(aida_set))
        
        if args.format == 1: new_dataset = from_ed_json_to_el_json(aida_name, aida_set, delimiter_mnt, delimiter_ent)
        elif (args.format == 4): new_dataset = from_aida_txt_to_ed_json(aida_name, aida_set, delimiter_ed, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID)
        elif (args.format >= 2) : new_dataset = from_aida_txt_to_el_json(aida_name, aida_set, delimiter_mnt, delimiter_ent)
        else: new_dataset = []
        
        logging.info("{} entries in {}".format(len(new_dataset), aida_name))
        generate_stats_mentions_doc(new_dataset)
        
        filename = "{}{}-EL.jsonl".format(args.output_dir,aida_name)
        logging.info("Saving {}".format(filename))
        with jsonlines.open(filename, mode='w') as f:
            f.write_all(new_dataset)
            