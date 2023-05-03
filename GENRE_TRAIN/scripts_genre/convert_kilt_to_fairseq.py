# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import jsonlines
from tqdm import tqdm

from genre.utils import create_input, create_input_el


def convert_kilt_to_fairseq_ed(dataset, start_delimiter="[START_ENT]", end_delimiter="[END_ENT]"):

    source = []
    target = []
    for doc in tqdm(dataset, desc="Processing"):
        for title in set(
            prov["title"]
            for out in doc["output"]
            if "provenance" in out
            for prov in out["provenance"]
            if prov.get("bleu_score", 1) > 0.5
        ):
            source.append(create_input(doc, max_length=384, start_delimiter=start_delimiter, end_delimiter=end_delimiter))
            target.append(title)
            if "meta" in doc and "template_questions" in doc["meta"]:
                for template_question in doc["meta"]["template_questions"]:
                    source.append(template_question)
                    target.append(title)

    return source, target

def convert_kilt_to_fairseq_el(dataset, delimiter_mention="[]", delimiter_entity="{}"):
    
    source = []
    target = []
    i = 0
    j = 0
    k = 0
    l = 0
    for doc in tqdm(dataset, desc="Processing"):
        new_input = create_input_el(doc, max_length=384, span_delimiter=delimiter_mention, entity_delimiter=delimiter_entity)
        if doc["output"][0]["answer"] != new_input: 
            j+=1
            if doc["input"].startswith("__NOINDEX__"): k+=1
            if "(CEST)" in doc["input"]: l+=1
            continue
        if doc["input"].startswith("__NOINDEX__") or "(CEST)" in doc["input"]: 
            i+=1
            continue
        #assert doc["output"][0]["answer"] == new_input, "final output :\n{}\n{}\noriginal output :\n{}\n{}\nliste mentions : \n{}".format(new_input, 20*"~", doc["output"][0]["answer"], 20*"~", doc["meta"]["mentions"])
        source.append( doc["input"] )
        target.append( new_input )
    if i+k > 0:
        print("NOINDEX : {}/{} ({:.2f}%)".format(i+k, len(dataset), 100*((i+k)/len(dataset))))
        print("valid NOINDEX : {}/{} ({:.2f}%)".format(i, i+k, 100*(i/(i+k))))
    print("invalid outputs : {}/{} ({:.2f}%)".format(j, len(dataset), 100*(j/len(dataset))))
    if k > 0: print("\t- because of NOINDEX : {}/{} ({:.2f}%)".format(k, j, 100*(k/j)))
    if l > 0: print("\t- because of CEST discuss : {}/{} ({:.2f}%)".format(l, j, 100*(l/j)))
    print("Final Outputs : {}/{} ({:.2f}%)".format(len(target), len(dataset), 100*(len(target)/len(dataset))))
    assert len(target) == len(source), "invalid source ({}) & target ({})".format(len(source), len(target))
    
    return source, target


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
        "--mode",
        type=str,
        default="ed",
        help="'ed' or 'el'"        
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
    
    logging.info("mode : {}".format(args.mode))

    logging.info("Loading {}".format(args.input_filename))
    with jsonlines.open(args.input_filename) as f:
        dataset = [e for e in f]
    split_name = os.path.basename(args.input_filename).split("-")[1]

    
    logging.info("type dataset: {} of {}".format(type(dataset), type(dataset[0])))
    #logging.info("exemple : {}".format(dataset[:3]))
    
    if args.mode == "ed":
        delimiter_bg = "[START]"
        delimiter_nd = "[END]"
        source, target = convert_kilt_to_fairseq_ed(
                dataset, delimiter_bg, delimiter_nd
        )
    else:
        delimiter_mnt = "{}"
        delimiter_ent = "[]"
        source, target = convert_kilt_to_fairseq_el(
                dataset, delimiter_mnt, delimiter_ent
        )    


    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    for type_name, data in (("source", source), ("target", target)):

        with open(
            os.path.join(
                args.output_path,
                "{}.{}".format(split_name, type_name),
            ),
            "w",
        ) as f:
            f.writelines(
                [doc.replace("\r", ">>").replace("\n", ">>") + "\n" for doc in data]
            )
