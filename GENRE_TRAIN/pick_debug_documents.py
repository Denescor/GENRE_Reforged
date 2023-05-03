#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:14:09 2022

@author: carpentier
"""

import os
import sys
import locale
import shutil
import random
import argparse
import numpy as np
import tqdm.auto as tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="name of the folder where pick documents")
parser.add_argument("--save", help="name of the folder where put the documents picked")
parser.add_argument("--lendoc", default="20", type=int, help="number of documents to pick")
parser.add_argument("--max_lenght", default="5000", type=int, help="max character per doc")
parser.add_argument("--min_lenght", default="500", type=int, help="min character per doc")
args = parser.parse_args()

#print("start héhé")
#print("encoding stdout : {}".format(sys.stdout.encoding))
#print("prefer encoding : {}".format(locale.getpreferredencoding()))
#print("file encoding : {}".format(sys.getfilesystemencoding()))
######################## LIST OF ALL DOCS ########################
path_1 = os.path.join(args.folder, "TR/fr/test")
path_2 = os.path.join(args.folder, "DB/fr/test")
list_doc_brut_1 = [os.path.join(path_1, os.path.splitext(x)[0]) for x in os.listdir(path_1) if os.path.isfile(os.path.join(path_1, x)) and os.path.splitext(x)[1]==".mentions"]
list_doc_brut_2 = [os.path.join(path_2, os.path.splitext(x)[0]) for x in os.listdir(path_2) if os.path.isfile(os.path.join(path_2, x)) and os.path.splitext(x)[1]==".mentions"]
assert len(list_doc_brut_1) > 0, path_1
assert len(list_doc_brut_2) > 0, path_2
######################## FILTER DOC BY LIMIT SIZE ########################
def filter_doc(list_doc_brut):
    list_doc = []
    all_size_final = []
    for doc in tqdm.tqdm(list_doc_brut, total=len(list_doc_brut), desc="Filter docs"):
        txt_file = "{}.txt".format(doc)
        with open(txt_file, "r") as txt:
            size = sum([len(x) for x in txt.readlines()])
            if size >= args.min_lenght and size <= args.max_lenght:
                list_doc.append(doc)
                all_size_final.append(size)
    print("total docs contenant entre {} et {} caractères : {}/{} ({:.2f}%)".format(args.min_lenght, args.max_lenght, len(list_doc), len(list_doc_brut), 100*(len(list_doc)/len(list_doc_brut))))
    print("taille moyenne des documents : {:.1f}".format(np.mean(all_size_final)))
    return list_doc
list_doc_TR = filter_doc(list_doc_brut_1)
list_doc_DB = filter_doc(list_doc_brut_2)
######################## SELECT LIMITED NUMBER OF DOC ########################
def pick_doc(list_doc, lendoc):
    error = 0
    list_error = []
    if len(list_doc) > lendoc: # on prend seulement le nombre nécessaire
        list_doc.sort() #for reproductibility
        random.seed(1158) #for reproductibility
        pick_doc = random.sample([i for i in range(len(list_doc))], lendoc) #liste des documents à déplacer
    else: pick_doc = list_doc #on prend tout
    print("pick doc name : {}".format(list_doc[:5]))
    for i in tqdm.tqdm(pick_doc, total=len(pick_doc), desc="pick docs"): #On déplace uniformément 10% des documents dans le dev
        mention_file = "{}.mentions".format(list_doc[i])
        txt_file = "{}.txt".format(list_doc[i])
        if (not os.path.isfile(mention_file)) or (not os.path.isfile(txt_file)):
            error += 1
            list_error.append((os.path.isfile(mention_file),os.path.isfile(txt_file)))
            continue
        shutil.copy(mention_file, args.save)
        shutil.copy(txt_file, args.save)
    print("{}/{} documents found and copy ({:.2f}%)\nDONE".format(len(pick_doc)-error, len(pick_doc), 100*((len(pick_doc)-error)/len(pick_doc))))
    if error > 0: print("exemple erreurs : {}".format(list_error[:5]))
pick_doc(list_doc_TR, int(args.lendoc//2))
pick_doc(list_doc_DB, int(args.lendoc//2))