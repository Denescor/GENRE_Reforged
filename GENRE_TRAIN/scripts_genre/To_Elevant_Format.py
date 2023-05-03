#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:36:13 2022

@author: carpentier
"""

import os
import tqdm
import random
import pickle
import logging
import argparse
import jsonlines
from genre.utils import get_wikidata_ids

######################################################################################################################################
def unify_entity_name(entity):
    """
    Doit :
        - retirer les écrits entre parenthèses
        - remplacer les "_", "-", ":" par des espaces
        - retire les espaces superflus qui resteraient
        - passe tout en minuscule
    """
    try: entity = entity.replace("_", " ")
    except Exception as e: logging.info(e)
    return entity

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
######################################################################################################################################


############################################################ CONVERT TR & DB ############################################################
def extract_text_TR(txt_file, doc, i_id):
    json_doc = {"id" : i_id, "title" : doc, "text" : None, "evaluation_span": None}
    text = ""
    with open(txt_file, "r", encoding='utf8') as txt:
        for line in txt:
            text += line
    json_doc["text"] = text
    json_doc["evaluation_span"] = [0, len(text)] #the whole document
    return json_doc

def extract_mentions_TR(mention_file, doc, i_id, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID):
    labels = []
    nb_entities = 0
    empty_entities = 0
    with open(mention_file, "r") as txt:
        for line in txt:
            bg, nd, ent_en, ent_fr, hard = line.strip().split("\t")
            entity = unify_entity_name(ent_fr)
            id_label = int("{}{}".format(i_id,bg))
            if (lang_title2wikidataID is not None) and (lang_redirect2title is not None) and (label_or_alias2wikidataID is not None):
                wikidataIDs = get_wikidata_ids(
                    entity.replace("_", " "),
                    args.entity_language,
                    lang_title2wikidataID,
                    lang_redirect2title,
                    label_or_alias2wikidataID,
                )[0]
                state = not( (wikidataIDs == entity.replace("_", " ")) or (len(wikidataIDs) == 0) )
                if state:
                    if type(wikidataIDs) == set: wikidataIDs = list(wikidataIDs)
                    elif type(wikidataIDs) != list: wikidataIDs = [wikidataIDs]
                    assert type(wikidataIDs) == list, "wikidataIDs is a {}".format(type(wikidataIDs))
                    assert len(wikidataIDs) > 0, "len wikidataIDs = {}".format(len(wikidataIDs))
                    assert type(wikidataIDs[0]) != list, "wikidataIDs items are lists"
                else: 
                    wikidataIDs = ["Unknown"]
                    empty_entities += 1
            else: state, wikidataIDs = True, [{"title": entity}]
            if state:
                nb_entities += 1
                label = {
                    "id": id_label,
                    "span": [int(bg),int(nd)],
                    "entity_id": wikidataIDs[0],
                    "name": entity,
                    "parent": None,
                    "children": [],
                    "optional": False,
                    "type": None #"|".join()
                }
                labels.append(label)
    return labels, nb_entities, empty_entities

def process_TR(folder, out_filepath, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID):
    i_id = 1
    os.chdir(folder)
    all_json = []
    nb_entities = 0
    empty_entities = 0
    list_doc = [os.path.splitext(x)[0] for x in os.listdir() if os.path.isfile(x) and os.path.splitext(x)[1]==".mentions"]
    for doc in tqdm.tqdm(list_doc, total=len(list_doc), desc="Convert TR to Elevant"):
        mention_file = "{}.mentions".format(doc)
        txt_file = "{}.txt".format(doc)
        json_doc = extract_text_TR(txt_file, doc, i_id) #generate {id : ..., title : ..., text : ...}
        labels, nb_temp, empty_temp = extract_mentions_TR(mention_file, doc, i_id, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID) #generate {labels : [...]}
        nb_entities += nb_temp
        empty_entities += empty_temp
        ## Ajout Info JSON
        # fusion {id : ..., title : ..., text : ...} and {labels : [...]}
        json_doc["labels"] = labels
        all_json.append(json_doc)
        i_id += 1
    prop_entities = empty_entities / (nb_entities + empty_entities)
    logging.info("nombre d'entités : {}\n\t- entités vide (skipped) : {} ({:.2f}%)".format(nb_entities, empty_entities, 100*prop_entities))
    # écriture JSON
    with jsonlines.open(out_filepath, mode="w") as fout:
        fout.write_all(all_json)  
    logging.info("file save at '{}'".format(out_filepath))
######################################################################################################################################
######################################################################################################################################
############################################################ CONVERT AIDA ############################################################
def extract_text_AIDA(file_reader, file_name, idtoent, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID):
    all_json = []
    men_flag = False
    i = 0
    nb_entities = 0
    empty_entities = 0
    for word in file_reader:
        if word.startswith("DOCSTART_"):  #start of a doc == begin a new entry
            title = word[9:].strip()
            id_doc = i #"{}-{}".format(file_name, title)
            doc = ""
            labels = []
            incrementor = 0
            i += 1
        elif word.startswith("DOCEND"): #end of a doc == end of an entry == save the final entry
            entry = {
                "id": id_doc,
                "title": title,
                "text": doc,
                "labels": labels,
                "evaluation_span": [0, incrementor] #the whole document
            }
            all_json.append(entry)
        elif word.startswith("*NL*"): continue
        elif word.startswith("MMSTART_"): # start of a mention == end of current left context
            try: entity = idtoent[word[8:].strip()]
            except KeyError: continue #logging.debug("{} not in idtoent".format(word[8:].strip()))
            else:
                if (lang_title2wikidataID is not None) and (lang_redirect2title is not None) and (label_or_alias2wikidataID is not None):
                    wikidataIDs = get_wikidata_ids(
                        entity.replace("_", " "),
                        args.entity_language,
                        lang_title2wikidataID,
                        lang_redirect2title,
                        label_or_alias2wikidataID,
                    )[0]
                    state = not( (wikidataIDs == entity.replace("_", " ")) or (len(wikidataIDs) == 0) )
                    if state:
                        if type(wikidataIDs) == set: wikidataIDs = list(wikidataIDs)
                        elif type(wikidataIDs) != list: wikidataIDs = [wikidataIDs]
                        assert type(wikidataIDs) == list, "wikidataIDs is a {}".format(type(wikidataIDs))
                        assert len(wikidataIDs) > 0, "len wikidataIDs = {}".format(len(wikidataIDs))
                        assert type(wikidataIDs[0]) != list, "wikidataIDs items are lists"
                    else: 
                        wikidataIDs = ["Unknown"]
                        empty_entities += 1
                else: state, wikidataIDs = True, [{"title": entity}]
                if state:
                    nb_entities += 1
                    men_flag = True
                    start = incrementor
                    mention = ""
                else: continue
        elif word.startswith("MMEND"): # end of a mention == save the mention + entities in output_text
            if men_flag:
                men_flag = False
                end = incrementor-1
                incrementor_mention = doc[start:end]
                id_label = int("{}{}".format(start,end))
                label = {
                    "id": id_label,
                    "span": [start, end],
                    "entity_id": wikidataIDs[0],
                    "name": entity,
                    "parent": None,
                    "children": [],
                    "optional": False,
                    "type": None #"|".join()
                }
                labels.append(label)
                # output = { mention } [ entity ]
                assert incrementor_mention == mention[:-1], "incrementor error :\n\t- doc : '{}'\n\t- incrementor mention : '{}' ({}:{})\n\t- true mention : '{}'".format(doc, incrementor_mention, start, end, mention[:-1])
            else: continue # entity not found in universe
        else: # == a simple word
            if men_flag: mention += "{} ".format(word.strip())
            doc += "{} ".format(word.strip())
            incrementor += len(word.strip())+1
            assert men_flag or (len(doc) == incrementor), "incrementor error:\n\t- doc : '{}'\n\t- len input : {}\n\t- word : '{}' ({})\n\t- incrementor : {}".format(doc, len(doc), word.strip(), len(word.strip()), incrementor)
    prop_entities = empty_entities / (nb_entities + empty_entities)
    logging.info("nombre d'entités : {}\n\t- entités vide (skipped) : {} ({:.2f}%)".format(nb_entities, empty_entities, 100*prop_entities))
    return all_json
        
def process_AIDA(file, out_filepath, aida_name, idtoent, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID):
    all_json = []
    with open(file, "r") as txt:
        #generate {id : ..., title : ..., text : ..., labels : [...]}
        all_json = extract_text_AIDA(txt, aida_name, idtoent, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID)
    with jsonlines.open(out_filepath, mode="w") as fout:
        # écriture JSON
        fout.write_all(all_json)  
    logging.info("file save at '{}'".format(out_filepath))
######################################################################################################################################
######################################################################################################################################
############################################################ CONVERT WIKI ############################################################
def extract_item_WIKI(item, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID):
    id_doc = int(item["id"].split("-")[-1])
    doc = item["input"]
    title = doc[:10].strip() #generated title
    labels = []
    empty_entities = 0
    for output in item["output"]:
        entities = item["meta"]["entities"]
        assert len(output["provenance"]) == len(entities), "KILT inconsistent"
        assert len(output["provenance"]) == len(output["start"]) and len(output["start"]) == len(output["end"]), "KILT inconsistent"
        for ent in range(len(output["provenance"])):
            start = output["start"][ent]
            end = output["end"][ent]
            entity = entities[ent]
            if (lang_title2wikidataID is not None) and (lang_redirect2title is not None) and (label_or_alias2wikidataID is not None):
                wikidataIDs = get_wikidata_ids(
                    entity.replace("_", " "),
                    args.entity_language,
                    lang_title2wikidataID,
                    lang_redirect2title,
                    label_or_alias2wikidataID,
                )[0]
                state = not( (wikidataIDs == entity.replace("_", " ")) or (len(wikidataIDs) == 0) )
                if state:
                    if type(wikidataIDs) == set: wikidataIDs = list(wikidataIDs)
                    elif type(wikidataIDs) != list: wikidataIDs = [wikidataIDs]
                    assert type(wikidataIDs) == list, "wikidataIDs is a {}".format(type(wikidataIDs))
                    assert len(wikidataIDs) > 0, "len wikidataIDs = {}".format(len(wikidataIDs))
                    assert type(wikidataIDs[0]) != list, "wikidataIDs items are lists"
                else: 
                    wikidataIDs = ["Unknown"]
                    empty_entities += 1
            else: 
                state, wikidataIDs = True, [{"title": entity}]
            id_label = "{}-{}-{}".format(entity, start, end)
            label = {
                "id": id_label,
                "span": [start, end],
                "entity_id": wikidataIDs[0],
                "name": entity,
                "parent": None,
                "children": [],
                "optional": False,
                "type": None #"|".join()
            }
            labels.append(label)
    entry = {
        "id": id_doc,
        "title": title,
        "text": doc,
        "labels": labels,
        "evaluation_span": [0, len(doc)] #the whole document
    }
    return entry, len(labels), empty_entities

def process_WIKI(file, out_filename, out_filename2, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID):
    all_json, mini_json = [], []
    random.seed(1158) #for reproductibility
    nb_entities = 0
    empty_entities = 0
    with jsonlines.open(file) as f:
        for entry in tqdm.tqdm(f, desc="Convert KILT to Elevant"):
            item, nbe, ee = extract_item_WIKI(entry, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID)
            nb_entities += nbe
            empty_entities += ee #+0 or +1
            all_json.append(item)
    try: pick_item = set(random.sample([i for i in range(len(all_json))], 5000))
    except ValueError: logging.info("cannot generating mini corpus") #if an other dataset than WIKI is processed
    else: 
        mini_json = [all_json[i] for i in range(len(all_json)) if i in pick_item]
        logging.info("mini set of {} items".format(len(pick_item)))
    prop_entities = empty_entities / nb_entities
    logging.info("nombre d'entités : {}\n\t- entités vide (skipped) : {} ({:.2f}%)".format(nb_entities, empty_entities, 100*prop_entities))
    with jsonlines.open(out_filename, mode='w') as f:
        f.write_all(all_json)
    logging.info("file save at '{}' ({} items)".format(out_filename, len(all_json)))
    if len(mini_json) > 0:
        with jsonlines.open(out_filename2, mode='w') as f:
            f.write_all(mini_json)
        logging.info("file save at '{}' ({} items)".format(out_filename2, len(mini_json)))
######################################################################################################################################
    
def create_necessary_folders():
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

def _parse_args():
    # exemple using : 
    #python -m scripts_genre.To_Elevant_Format 
    #   --dataset_folder="$WORK/TR/fr/" 
    #   --output_folder="$WORK/Reforged_GENRE/data/benchmarks/"
    #   --entity_language="fr"
    #   --wiki_path="wiki_name_id_map_FR.txt"
    #   --base_wikidata="$STORE/wikidata_dump"
    #   --type_dataset="TR"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", default="../data/basic_data/test_datasets/AIDA/")
    parser.add_argument("--output_folder", default="../data/new_datasets/")
    parser.add_argument("--entity_language", default="fr", help="'fr' or 'en'")
    parser.add_argument("--wiki_path", default="wiki_name_id_map.txt")
    parser.add_argument("--base_wikidata", default="")
    parser.add_argument("--type_dataset", default="TR", help="format of the original dataset : 'TR'/'DB' or 'AIDA'")
    parser.add_argument("-v", action="store_const", dest="loglevel", const=logging.INFO, help="Be verbose")
    #parser.add_argument("-d", help="Debugging Mode", action="store_const", dest="loglevel", const=logging.DEBUG)
    parser.set_defaults(loglevel=logging.WARNING)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    print("log level : {}".format(args.loglevel))
    create_necessary_folders()
    
    logging.info("START {} {}".format(args.type_dataset, args.entity_language))
    current_dir = os.getcwd()
    (current_dir)
    
    lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID = load_wikidata_dicts(args.base_wikidata)
    
    ## preprocess datasets
    if (args.type_dataset == "TR") or (args.type_dataset == "DB"):
        for data in ["train", "dev", "test"]:
            os.chdir(current_dir)
            logging.info("current file : {}{}".format(args.dataset_folder, data))
            process_TR("{}{}".format(args.dataset_folder, data), "{}{}_{}.jsonl".format(args.output_folder, args.type_dataset, data), lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID)
    #####################################################################
    elif args.type_dataset == "AIDA":
        aida_list = [(os.path.splitext(x)[0], args.dataset_folder+x) for x in os.listdir(args.dataset_folder) if os.path.splitext(x)[1]==".txt"]
        nb_universe = 0
        idtoent = dict()
        with open(args.dataset_folder+args.wiki_path, "r") as f:
            for line in f:
                try: ent, i = line.strip().split("\t")
                except ValueError: pass
                else: idtoent[i] = ent
                finally: nb_universe += 1
        logging.info("{}/{} ({:.2f}%) entities in universe".format(len(idtoent), nb_universe,  100*(len(idtoent)/nb_universe) ))

        for aida_name, aida_file in aida_list:
            logging.info("current file : {}".format(aida_file))
            process_AIDA(aida_file, "{}{}.jsonl".format(args.output_folder, aida_name), aida_name, idtoent, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID)
    #####################################################################
    elif args.type_dataset == "KILT":
        wiki_list = [(os.path.splitext(x)[0], args.dataset_folder+x) for x in os.listdir(args.dataset_folder) if os.path.splitext(x)[1]==".jsonl"]
        for wiki_name, wiki_file in wiki_list:
            output_name = "-".join(wiki_name.split("-")[0:2])
            process_WIKI(wiki_file, "{}{}.jsonl".format(args.output_folder, output_name), "{}{}_mini.jsonl".format(args.output_folder, output_name), lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID)
    else:
        logging.info("format '{}' not recognized. Only 'KILT', 'TR'/'DB' or 'AIDA'".format(args.type_dataset))
    
    lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID = None, None, None
    logging.info("DONE")
    exit(0)