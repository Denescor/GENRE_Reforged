#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:07:48 2022

@author: carpentier
"""

import argparse
from tqdm import tqdm

def read_source(doc):
    dico = dict()
    with open(doc, "r") as f:
        for line in f:
            index, item = line.strip().split("\t")
            dico[int(index)] = item
    return dico

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="path to [doc].spm.source")
    parser.add_argument("--target", type=str, required=True, help="path to [doc].spm.target")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    source = read_source(args.source) #dico
    target = read_source(args.target) #dico
    print("len source : {}".format(len(source)))
    print("len target : {}".format(len(target)))
    max_len = max(len(source), len(target))
    min_len = min(len(source), len(target))
    if max_len == min_len:
        print("{} & {} sont cohérents".format(args.source, args.target))
        exit(0)
    else:
        diff = max_len-min_len
        print("{}/{} ({:.2f}%) entrées en trop\n\t==> suppression en cours...".format(diff, max_len, 100*(diff/max_len)))
    
        new_source = []
        new_target = []
        new_len = 0
        
        for i in tqdm(range(max_len), total=max_len, desc="re-align source & target"):
            try: index_source, item_source = i, source[i]
            except KeyError: index_source, item_source = -1, None
            try: index_target, item_target = i, target[i]
            except KeyError: index_target, item_target = -1, None
#            index_source = int(index_source)
#            index_target = int(index_target)
            if index_source == index_target:
                new_source.append(item_source)
                new_target.append(item_target)
                new_len += 1
        print("\t==> new entries count : {}".format(new_len))
        with open(args.source, "w") as s, open(args.target, "w") as t:
            for i in tqdm(range(new_len), total=new_len, desc="read new source & target"):
                s.write("{}\n".format(new_source[i]))
                t.write("{}\n".format(new_target[i]))
        print("\t==> DONE")
    
        