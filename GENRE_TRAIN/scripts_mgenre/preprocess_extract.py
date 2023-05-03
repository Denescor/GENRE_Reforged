# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

from genre.utils import chunk_it, extract_pages, process_jsonNEL_entry
from tqdm.auto import tqdm, trange

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_wikipedia",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        help="Language ID.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
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
        "-e",
        "--extract_mode",
        help="extract with 'bs4' or using the json from 'nel' extraction",
        default="bs4"
    )
    
    parser.add_argument(
        "--total_nel",
        "--t",
        type=float,
        help="proportion from the nel jsonl extracted",
        default=1
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)

    filenames = [
        os.path.join(args.base_wikipedia, args.lang, sub_folder, file_name)
        for sub_folder in os.listdir(os.path.join(args.base_wikipedia, args.lang))
        if os.path.isdir(os.path.join(args.base_wikipedia, args.lang, sub_folder))
        for file_name in os.listdir(
            os.path.join(args.base_wikipedia, args.lang, sub_folder)
        )
    ]

    if args.rank < min(len(filenames), 32):
        filenames = chunk_it(filenames, min(len(filenames), 32))[args.rank]
    else:
        quit()

    num_threads = 32
    if args.extract_mode == "bs4":
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
    
            futures = {
                executor.submit(extract_pages, filename): filename for filename in filenames
            }
    
            if args.rank == 0:
                iter_ = tqdm(as_completed(futures), total=len(futures), smoothing=0)
            else:
                iter_ = as_completed(futures)
    
            results = {futures[future]: future.result() for future in iter_}
    
        results = {k: v for sub_results in results.values() for k, v in sub_results.items()}
    else:
        with open(os.path.join(args.base_wikipedia, args.lang,"gold_entities.jsonl"), "r") as jf: 
            temp_ligns = jf.readlines()
            json_doc = [json.loads(content) for content in temp_ligns]
            len_original = len(json_doc)
            len_final = int(args.total_nel*len(json_doc))
            json_doc = json_doc[:len_final]
            logging.info("{}/{} ({:.2f}%) docs in final".format(len_original, len_final, 100*(len_final/len_original)))
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {
                    executor.submit(process_jsonNEL_entry, entry): entry for entry in json_doc
                }
                iter_ = tqdm(as_completed(futures), total=len(futures), smoothing=0)
                results_temp = [future.result() for future in iter_]
            
            results = {}
            total_error = 0
            total_ers = 0
            total_sample = 0
            for dico in results_temp: 
                results[dico["id"]] = dico
                total_error += dico["error"]
                total_ers += dico["error_sample"]
                total_sample += dico["total_sample"]
        logging.info("total sample unprocessed : {}/{} ({:.2f}%)".format(total_error,total_sample,100*(total_error/total_sample)))
        logging.info("total sample error : {}/{} ({:.2f}%)".format(total_ers,total_sample,100*(total_ers/total_sample)))

    filename = os.path.join(
        args.base_wikipedia,
        args.lang,
        "{}wiki{}.pkl".format(args.lang, args.rank),
    )
    logging.info("Saving {}".format(filename))
    with open(filename, "wb") as f:
        pickle.dump(results, f)
