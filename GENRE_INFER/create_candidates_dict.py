import re
import tqdm
import string
import logging
import argparse
import itertools
import pickle
import sys


def read_dalab_candidates(args):
    for line in open("data/dalab/{}".format(args.prob_path)):
        line = line[:-1]
        columns = line.split("\t")
        mention = columns[0]
        for column in columns[2:]:
            if len(column.strip()) == 0:
                continue
            values = column.split(",")
            candidate = ",".join(values[2:])
            candidate = candidate.replace("_", " ")
            yield mention, candidate


def hex2int(hexa: str) -> int:
    return int(hexa, 16)


def replace_unicode(u_str):
    matches = set(re.findall("\\\\u....", u_str))
    for match in matches:
        u_str = u_str.replace(match, chr(hex2int(match[2:])))
    return u_str


PUNCTUATION_CHARS = set(string.punctuation)


def filter_mention(mention):
    #if mention[0].islower():
    #    return True
    if mention[0] in PUNCTUATION_CHARS:
        return True
    return False

def read_dataline_candidates(data):
    for line in open("data/{}_means.tsv".format(data)):
        values = line.strip().split("\t")
        if len(values) != 2: yield None, None #, "line : '{}'".format(line)
        mention = replace_unicode("".join(values[0][1:-1]))
        candidate = replace_unicode(values[1]).replace("_", " ")
        assert (len(mention) > 0) and (len(candidate) > 0), "\nsplit : '{}'\nmention : '{}'\ncandidate : '{}'".format(values, mention, candidate)
        yield mention, candidate
        
def read_entities_universe(entities_universe):
    entities = set()
    for line in open("data/dalab/{}".format(entities_universe)):
        entity = line.split("\t")[-1][:-1].replace("_", " ")
        entities.add(entity)
    return entities

def fill_mention_candidates(reader_candidates, arg, mention_candidates_dict, dalab_entities, desc="Generate Candidates Dict"):
    total_mentions = 0
    total_ent = 0
    passed = 0
    filtering = 0
    unk_ent = 0
    for mention, candidate in tqdm.tqdm(reader_candidates(arg), desc=desc):
        total_ent += 1
        if mention is None and candidate is None:
            unk_ent += 1
            continue
        if filter_mention(mention): 
            filtering += 1
            continue
        if dalab_entities and mention not in entities: 
            unk_ent += 1
            continue
        if mention not in mention_candidates_dict:
            total_mentions += 1
            mention_candidates_dict[mention] = set()
        passed += 1
        mention_candidates_dict[mention].add(candidate)
    logging.info(50*"#")
    logging.info("### unique mentions : {}/{} ({:.2f}%) ###".format(total_mentions, total_ent, 100*(total_mentions/total_ent)))
    logging.info("### passed mentions : {}/{} ({:.2f}%) ###".format(passed, total_ent, 100*(passed/total_ent)))
    logging.info("### skiped mentions : {}/{} ({:.2f}%) ###".format(unk_ent, total_ent, 100*(unk_ent/total_ent)))
    logging.info("### filter mentions : {}/{} ({:.2f}%) ###".format(filtering, total_ent, 100*(filtering/total_ent)))
    logging.info(50*"#")
    return mention_candidates_dict

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
    parser.add_argument("--dalab", type=bool, default=False, help="use 'entities_universe' file")
    parser.add_argument("--dalab_name", type=str, default="entities_universe.txt", help="entities universe file name")
    parser.add_argument("--dataset", default="aida", help="dataset between 'aida', 'TR'/'DB'")
    parser.add_argument("--entity_language", default="fr", help="'fr' or 'en'")
    parser.add_argument("--prob_path", default="prob_yago_crosswikis_wikipedia_p_e_m.txt", help="pem prior file")
    parser.add_argument("-v", action="store_const", dest="loglevel", const=logging.INFO, help="Be verbose")
    parser.set_defaults(loglevel=logging.WARNING)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=args.loglevel)
    dalab_entities = args.dalab
    if dalab_entities:
        print("read {}...".format(args.dalab_name))
        entities = read_entities_universe(args.dalab_name)
    mention_candidates_dict = {}
    print("read mention - candidate pairs...")
    n = 0
    mention_candidates_dict = fill_mention_candidates(read_dalab_candidates, args, mention_candidates_dict, dalab_entities, "Generate Candidates Dict - PEM  File")
    if args.entity_language == "fr":
        mention_candidates_dict = fill_mention_candidates(read_dataline_candidates, "DB", mention_candidates_dict, dalab_entities, "Generate Candidates Dict - DB   File")
        mention_candidates_dict = fill_mention_candidates(read_dataline_candidates, "TR", mention_candidates_dict, dalab_entities, "Generate Candidates Dict - TR   File")
        mention_candidates_dict = fill_mention_candidates(read_dataline_candidates, "WIKI-FR", mention_candidates_dict, dalab_entities, "Generate Candidates Dict - WIKI File")
    else:
        mention_candidates_dict = fill_mention_candidates(read_dataline_candidates, "AIDA", mention_candidates_dict, dalab_entities, "Generate Candidates Dict - aida File")
        mention_candidates_dict = fill_mention_candidates(read_dataline_candidates, "WIKI-EN", mention_candidates_dict, dalab_entities, "Generate Candidates Dict - WIKI File")
    for mention in mention_candidates_dict:
        mention_candidates_dict[mention] = sorted(mention_candidates_dict[mention])
    out_file = "data/mention_to_candidates_dict_{}{}.pkl".format(args.entity_language, ".dalab-entities" if dalab_entities else "")
    print("write to {}...".format(out_file))
    with open(out_file, "wb") as f:
        pickle.dump(mention_candidates_dict, f)
