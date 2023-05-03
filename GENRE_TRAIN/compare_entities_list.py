#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:14:30 2023

@author: carpentier
"""

import os
import time
import pickle
import argparse
import jsonlines
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

################################################################################################################
################################################################################################################

def unify_entity_name(entity):
    """
    Doit :
        - remplacer les "_", "-", ":" par des espaces
        - retire les espaces superflus qui resteraient
        - retire tous les espaces entre les mots
    """
    try:
        entity = entity.strip()
        entity = entity.replace("_", " ")
        entity = entity.replace("-", " ")
        entity = entity.replace(":", " ")
        entity = entity.replace("'", "")
        entity = "".join(entity.split(" "))
        #entity = clear_parenthesis.sub("", entity)
        #assert "(" not in entity1, "old : {} | new : {}".format(entity, entity1)
        #entity = " ".join([x for x in entity1.split(" ") if x != ''])
        #entity = entity.lower()
    except Exception as e: print(e)
    return entity

def make_list_Entities(datasets):
    """
    Give the list of entities for the file_path
    Param:
        - file_path : string (valid path of the KILT corpus)
    Return
        - list of string (entities with occurences)
        - dict : key is string ; values is list of string (list of mentions by entities)
    TO UPDATE
    """
    densite_list = []
    entities_list = []
    mentions_dict = dict()
    for dataset in datasets:
        with jsonlines.open(dataset) as f:
            for item in tqdm(f, desc="extract entities from dataset"):
                size = len(item["input"]) #Nb Char on doc
                item_entities = item["meta"]["entities"] #Nb unique entities on doc
                item_mentions = item["meta"]["mentions"] #Nb mentions on doc
                entities_list.extend([unify_entity_name(ent) for ent in item_entities])
                densite_list.append(100*len(item_mentions)/size)
                for ent, ment in zip(item_entities, item_mentions):
                    ent = unify_entity_name(ent)
                    try: mentions_dict[ent].append(ment)
                    except KeyError: mentions_dict[ent] = [ment]
    return entities_list, mentions_dict, densite_list

################################################################################################################
################################################################################################################
    
def common_Entities(entities_set, entities_name):
    """
    Give number of common entities between all the entities set
    Param :
        - entities_set : list of set of string (the set of entities)
        - entities_name : list of string (the name of the set)
        Both lists must have the same lenght
    Print the output
    Return:
        - common : list of list of int 
    """
    common = []
    for index, entities in enumerate(entities_set):
        common_index = [0 for i in range(len(entities_set))]
        for index_compare, other_entities in enumerate(entities_set):
            if index_compare == index: continue #common_index[index_compare] = None #cas inutile à traiter
            else:
                for ent in entities:
                    if ent in other_entities: common_index[index_compare] += 1
        common.append(common_index)
    for index, (name1, common_count) in enumerate(zip(entities_name, common)):
        len_ent_index = len(entities_set[index])
        for name2, count in zip(entities_name, common_count):
            if count > 0: 
                prop = 100*(count/len_ent_index)
                print("entities of {} in {} : {} ({:.2f}%)".format(name1, name2, count, prop))
    return common

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////
def check_mentions(mentions_dicts, entities_name):
    diff_mentions = dict()
    same_mentions = dict()
    all_mentions = []
    unq_mentions = []
    tot_mentions = []
    occ_mentions = []
    cap_mentions = []
    com_mentions = []
    for name, mentions_dict in zip(entities_name, mentions_dicts):
        nb_corpus = []
        total_corpus = []
        med_corpus = []
        std_corpus = []
        capi_corpus = []
        comm_corpus = []
        for ent, list_mentions in tqdm(mentions_dict.items(), total=len(mentions_dict), desc="Analyse mentions for {}".format(name)):
            mentions_counter = Counter(list_mentions)
            mentions_varia = mentions_counter.values()
            mentions_name = mentions_counter.keys()
            nb_corpus.append(len(mentions_varia))
            total_corpus.append(sum(mentions_varia))
            med_corpus.append(np.median(list(mentions_varia)))
            std_corpus.append(np.std(list(mentions_varia)))
            if ent.islower(): capi_corpus.append((len([x for x in mentions_name if not x.islower()]), len(mentions_name)))
            else: comm_corpus.append((len([x for x in mentions_name if x.islower()]), len(mentions_name)))
            for namother, dict_other in zip(entities_name, mentions_dicts):
                if name == namother: continue #Don't compare entities between the same corpus
                if ent in dict_other: #If the entity is in an other corpus, we can make a mentions comparaison
                    list_mentions_other = dict_other[ent]
                    othername = set(Counter(list_mentions_other).keys())
                    #Keep the mentions refering to the entity only in one of the corpus (so, it's a symetric difference)
                    if ent in diff_mentions: 
                        diff_mentions[ent] = diff_mentions[ent] ^ othername
                        same_mentions[ent] = same_mentions[ent] & othername
                    else: 
                        diff_mentions[ent] = set(mentions_name) ^ othername
                        same_mentions[ent] = set(mentions_name) & othername
        all_mentions.append(nb_corpus)
        tot_mentions.append(total_corpus)
        occ_mentions.append(med_corpus)
        cap_mentions.append(capi_corpus)
        com_mentions.append(comm_corpus)
        print_mentions(nb_corpus, total_corpus, med_corpus, std_corpus, capi_corpus, comm_corpus)
    # Nombre de mentions différentes pour une même entités entre deux corpus
    diff_mentions_list = [len(x) for x in list(diff_mentions.values())]
    print("Mean differents mentions between the {} corpus for a same entity : {:.2f} (med : {} | min : {} | max : {})".format(
        len(entities_name),
        np.mean(diff_mentions_list),
        np.median(diff_mentions_list),
        min(diff_mentions_list),
        max(diff_mentions_list)
    ))
    # Save the diff_mentions dict (pkl & txt)
    write_mentions(diff_mentions)
    return tot_mentions, all_mentions, occ_mentions, cap_mentions, com_mentions, [len(x) for x in list(same_mentions.values())]

def print_mentions(nb_corpus, total_corpus, med_corpus, std_corpus, capi_corpus, comm_corpus):
    print("{}\nCorpus {}".format(5*"~",name))
    # Nombre de mentions différentes pour une même entité
    print("Mean number of different mentions for 1 entity : {:.2f} (med : {} | min : {} | max : {})".format(
            np.mean(nb_corpus), 
            np.median(nb_corpus),
            min(nb_corpus),
            max(nb_corpus)
    ))
    # Nombre de mentions qui sont des noms communs quand l'entités est un nom propre
    if len(capi_corpus) > 0:
        print("\t- {:.2f} common word for Named Entities ({:.2f}% of total mentions) (med : {} | min : {} | max : {})".format(
            np.mean([x for (x, _) in capi_corpus]),
            100*(np.mean([x for (x, _) in capi_corpus])/ np.mean([x for (_, x) in capi_corpus])),
            np.median([x for (x, _) in capi_corpus]),
            min([x for (x, _) in capi_corpus]),
            max([x for (x, _) in capi_corpus])
        ))
    else: print("\t- no Named Entities")
    # Nombre de mentions qui sont des noms propres quand l'entités est un nom commun
    if len(comm_corpus) > 0:
        print("\t- {:.2f} named word for Common Entities ({:.2f}% of total mentions) (med : {} | min : {} | max : {})".format(
            np.mean([x for (x, _) in comm_corpus]),
            100*(np.mean([x for (x, _) in comm_corpus])/ np.mean([x for (_, x) in comm_corpus])),
            np.median([x for (x, _) in comm_corpus]),
            min([x for (x, _) in comm_corpus]),
            max([x for (x, _) in comm_corpus])
        ))
    else: print("\t- no Common Entities")
    print("Mean total number of mentions for 1 entity : {:.2f} (med : {} | min : {} | max : {})".format(
            np.mean(total_corpus),
            np.median(total_corpus),
            min(total_corpus),
            max(total_corpus)
    ))
    # Nombre d'occurences médiane de chaque mentions pour une même entité
    print("Mean of median occurences of each mentions : {:.2f} (med : {} | min : {} | max : {})".format(
            np.mean(med_corpus),
            np.median(med_corpus),
            min(med_corpus),
            max(med_corpus)
    ))
    # Ecart-type de l'occurences de chaque mentions pour une même entité
    print("\t- Standard Deviation : {:.2f} (med : {} | min : {:.2f} | max : {:.2f})".format(
            np.mean(std_corpus),
            np.median(std_corpus),
            min(std_corpus),
            max(std_corpus)
    ))
    print(5*"~")

def write_mentions(diff_mentions):
    corpus_list = ""
    for name in entities_name:
        corpus_list += "_{}".format(name)
    with open("diff_mentions{}.pkl".format(corpus_list), "wb") as sld:
        pickle.dump(diff_mentions, sld)
    print("diff_mentions{}.pkl wrote".format(corpus_list))
    with open("diff_mentions_detailed{}.txt".format(corpus_list), "w") as std:
        std.write(50*"#"+"\n")
        std.write("entités communes (avec mentions communes) entre {}\n".format(entities_name))
        std.write(50*"#"+"\n")
        for ent, mentions in diff_mentions.items():
            std.write("{} {} {}\n".format(15*"/", ent, 15*"/"))
            for mention in mentions:
                std.write("\t- {}\n".format(mention))
        std.write(50*"#"+"\n")
    print("diff_mentions_detailed{}.txt wrote".format(corpus_list))
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
def is_a_evenment(entities_set, entities_name):
    """
    Give number of common entities between all the entities set
    Param :
        - entities_set : list of set of string (the set of entities)
        - entities_name : list of string (the name of the set)
        Both lists must have the same lenght
    Print the output
    Return:
        - evenement : dict (string, (int, int))
    """
    evenement = dict()
    for name, entities in zip(entities_name, entities_set):
        count = 0
        for ent in entities:
            #Une entités est un évènement si :
            #   - long texte : 20 caractères
            #   - présence de plusieurs mots
            #   - 
            if len(ent) >= 20:
                count += 1
        evenement[name] = (count, len(entities))
    for name, (count, total) in evenement.items():
        print("Nb evènements in {} : {} ({:.2f}%)".format(name, count, 100*(count/total)))
    return evenement

def common_name_Entities(entities_set, entities_name):
    """
    Give number of entities which are common name in all the entities set
    Param :
        - entities_set : list of set of string (the set of entities)
        - entities_name : list of string (the name of the set)
        Both lists must have the same lenght
    Print the output
    """
    errors = 0
    for name, entities in zip(entities_name, entities_set):
        count = 0
        for ent in entities:
            entity = list(ent)
            if len(entity) == 0: 
                errors += 1
                continue
            if ent.islower(): count += 1
        prop = 100*(count/len(entities))
        print("{} has {} common word has entities ({:.2f}%)".format(name, count, prop))
        print("{} errors (empty entities)".format(errors))

def occurence_Entities(entities_list, entities_name):
    """
    Give the number of mentions refering to a common word
    and the common entities between all the entities set for the 10% most common entities
    Param
        - entities_list : list of list of string (the list of entities with occurences)
        - entities_name : list of string (the name of the set)
    Return
        - most_common_set : list of set of string
        - ten_common : list of list of int
    Print the output
    """
    most_common_set = []
    for name, entities in zip(entities_name, entities_list):
        occurences = Counter(entities)
        ten_most_common = int(0.1*len(occurences))
        common_mentions = sum([occurences[ent] for ent in occurences if len(ent) > 0 and ent[0].islower()])
        occ_common = 100*(common_mentions/sum(occurences.values()))
        most_common_set.append(set(occurences.most_common(ten_most_common)))
        print("{} has {} mentions refering to a common word ({:.2f}% of mentions)".format(name, common_mentions, occ_common))
    print("{}\ncommon entities between the most common :".format(15*"#"))
    ten_common = common_Entities(most_common_set, entities_name)
    print(15*"#")
    return most_common_set, ten_common

################################################################################################################
################################################################################################################

def print_bar(entities_name, entities_set, entities_list, most_common_set, ten_common, all_mentions, same_mentions, common_list, evenement_list):
    """
    Param:
        - entities_name : list of string (the name of the set)
        - entities_set : list of set of string (the set of entities)
        - entities_list : list of list of string (the list of entities with occurences)
        - common_list : list of list of int
        - evenement_list : dict (string, (int, int))
    Affiche:
        - Nombre d'entités pour les deux corpus + nombre d'entités communes en filigrane
        - Nombre de mentions
        - Nombre de mentions pour les entités communes + nombre mentions communes en filigrane
        - Nombre d'évènements
    """
    names = ""
    for name in entities_name:
        names += "_{}".format(name)
    title = "Nombres d'entités et mentions entre corpus de même langue"
    plt.figure(figsize=(24,11.75))
    width = 0.35  # épaisseur de chaque bâton
    textsize = 6
    textfont = 15
    all_data = [[len(x) for x in entities_set], 
                [max(x) for x in common_list], 
                [len(x) for x in most_common_set],
                [max(x) for x in ten_common],
                [len(x) for x in entities_set],
                [x[0] for x in evenement_list.values()],
                [np.mean(x) for x in all_mentions],
                [np.mean(same_mentions) for i in range(len(all_mentions))], 
                [len(x) for x in entities_list],
    ]
    #print(all_data)
    all_pos = []
    for x in np.arange((len(all_data)//2)+1):
        all_pos.append(x)
        all_pos.append(x)
    if len(all_pos) > len(all_data): del all_pos[-1]
    #print(all_pos)
    legend = ["Nb Entités", "Nb Entités Freq", "Entités as Evènements", "Nb Mentions Communes", "Nb Mentions"]
    patches = [mpatches.Patch(color=c, label=data) for data,c in zip(entities_name, ["tab:olive", "tab:cyan"])] + [mpatches.Patch(color="red", label="Commun")]
    #print(legend)
    
    i = 0
    for pos, data in zip(all_pos, all_data):
        if i % 2 == 0:
            color0 = "tab:olive"
            color1 = "tab:cyan"
            edgecolor = "black"
            alpha = True
            plt.annotate("{:.1f}".format(data[0]), alpha=0.5, xy=(pos - width/4, data[0]), 
                         xytext=(pos - width/4, 5), 
                         textcoords='offset points', ha='right', va='bottom', size=textsize, fontsize=textfont)
            plt.annotate("{:.1f}".format(data[1]), alpha=0.5, xy=(pos + 3*width/4, data[1]), 
                         xytext=(pos + 3*width/4, 5), 
                         textcoords='offset points', ha='right', va='bottom', size=textsize, fontsize=textfont)
        else:
            color0 = "tab:gray"
            color1 = "tab:gray"
            edgecolor = "red"
            alpha = False
            plt.annotate("{:.1f}".format(data[0]), alpha=0.5, xy=(pos - width/4, data[0]), 
                         xytext=(pos - width/4, -5), 
                         textcoords='offset points', ha='right', va='bottom', size=textsize, fontsize=textfont)
            plt.annotate("{:.1f}".format(data[1]), alpha=0.5, xy=(pos + 3*width/4, data[1]), 
                         xytext=(pos + 3*width/4, -5), 
                         textcoords='offset points', ha='right', va='bottom', size=textsize, fontsize=textfont)
        plt.bar(pos - width/2, data[0], width, color=color0, fill=alpha, linewidth=4, edgecolor=edgecolor, label=entities_name[0])
        plt.bar(pos + width/2, data[1], width, color=color1, fill=alpha, linewidth=4, edgecolor=edgecolor, label=entities_name[1])
        i+=1
    plt.title(title, fontsize=20)
    plt.yscale('log')
    plt.xticks(np.arange(len(legend)), legend)
    plt.legend(handles = patches, fancybox=True, ncol=3, loc="best", fontsize=10)
    plt.savefig("bar_plot{}.png".format(names), format='png', dpi=150, bbox_inches='tight')
    plt.close()
    print("bar_plot{}.png generated".format(names))

def print_boxplot(entities_name, tot_mentions, all_mentions, occ_mentions, cap_mentions, com_mentions, densite_list):
    names = ""
    for name in entities_name: names += "_{}".format(name)
    textsize = 6
    textfont = 15
    plt.figure(figsize=(24,11.75))
    legend1 = ["Nb Mentions / Entités", 
              "Nb Mentions Unq / Entité", 
              "Occ Médiane Mentions / Entité",   
              ]
    legend2 = ["Common Mentions / Named Entity", 
              "Named Mentions / Common Entity",
    ]
    legend3 = ["densité de mentions",
    ]
    k = 1
    for i in range(len(entities_name)):
        for j in [1,2,3]:
            index_subplot = "{}{}{}".format(len(entities_name), 3, k)
            plt.subplot(int(index_subplot))
            k += 1
            if j == 1:
                bp = plt.boxplot(
                        [tot_mentions[i],
                        all_mentions[i],
                        occ_mentions[i],
                        ],
                        labels=legend1,
                        vert = False
                )
                plt.annotate("{:.1f}".format(np.median(tot_mentions[i])), 
                            xy=(bp['medians'][0].get_xdata()[0], bp['medians'][0].get_ydata()[0]),
                            xytext=(5,5), textcoords='offset points', size=textsize, fontsize=textfont)
                plt.annotate("{:.1f}".format(np.median(all_mentions[i])), 
                            xy=(bp['medians'][1].get_xdata()[0], bp['medians'][1].get_ydata()[0]),
                            xytext=(5,5), textcoords='offset points', size=textsize, fontsize=textfont)
                plt.annotate("{:.1f}".format(np.median(occ_mentions[i])), 
                            xy=(bp['medians'][2].get_xdata()[0], bp['medians'][2].get_ydata()[0]),
                            xytext=(5,5), textcoords='offset points', size=textsize, fontsize=textfont)
            elif j == 2:
                tab1 = cap_mentions[i] if len(cap_mentions[i]) > 0 else [0]
                tab2 = com_mentions[i] if len(com_mentions[i]) > 0 else [0]
                bp = plt.boxplot(
                        [tab1,
                         tab2,
                        ],
                        labels=legend2,
                        vert = False
                )
                if tab1 != [0]: plt.annotate("{:.1f}".format(np.median(tab1)),
                            xy=(bp['medians'][0].get_xdata()[0], bp['medians'][0].get_ydata()[0]),
                            xytext=(5,5), textcoords='offset points', size=textsize, fontsize=textfont)
                if tab2 != [0]: plt.annotate("{:.1f}".format(np.median(tab2)), 
                            xy=(bp['medians'][1].get_xdata()[0], bp['medians'][1].get_ydata()[0]),
                            xytext=(5,5), textcoords='offset points', size=textsize, fontsize=textfont)
            else: #j == 3
                bp = plt.boxplot(
                        [densite_list[i],
                        ],
                        labels=legend3,
                        vert = False
                )
                plt.annotate("{:.1f}".format(np.median(densite_list[i]), np.mean(densite_list[i])),
                            xy=(bp['medians'][0].get_xdata()[0], bp['medians'][0].get_ydata()[0]),
                            xytext=(5,5), textcoords='offset points', size=textsize, fontsize=textfont)
            plt.grid(True)
            plt.xscale('log')
            plt.yticks(rotation = 68.5)
            plt.title("Répartition sur {}".format(entities_name[i]), fontsize=20)
    plt.savefig("box_plot{}.png".format(names), format='png', dpi=150, bbox_inches='tight')
    plt.close()
    print("box_plot{}.png generated".format(names))
    
################################################################################################################
################################################################################################################
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folders with the KILT datasets")
    parser.add_argument("--corpus_name", help="Corpus name")
    parser.add_argument("--input_separator", type=str, default="|", help="separator between the folders")    
    parser.add_argument("--entities_universe", default="")
    parser.add_argument("--entity_language", default="fr", help="'fr' or 'en'")
    parser.add_argument("--unify_entity_name", dest="unify", action='store_true')
    parser.set_defaults(unify=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    current_dir = os.getcwd()
    print("START {}".format(args.entity_language))
    entities_list = []
    entities_set = []
    mentions_dicts = []
    densite_list = []
    entities_name = [name for name in args.corpus_name.split(args.input_separator)]
    dataset_list = [
        [
            os.path.join(folder, x) for x in os.listdir(folder) 
            if os.path.isfile(os.path.join(folder, x)) 
            and os.path.splitext(x)[1]==".jsonl"
        ] for folder in args.folder.split(args.input_separator)
    ]
    for name, corpus in zip(entities_name, dataset_list):
        print("corpus :\n{}".format(corpus))
        print("corpus name : {}".format(name))
        walltime = time.time()
        print("MAKE list_entities.txt ...",end="")
        entities, mentions, densites = make_list_Entities(corpus)
        entitie_s = set(entities)
        assert len(entitie_s) == len(mentions)
        print(" {} mentions & {} entities in list".format(len(entities), len(entitie_s)))
        entities_list.append(entities)
        entities_set.append(entitie_s)
        mentions_dicts.append(mentions)
        densite_list.append(densites)
        print("exemples entities : {}".format(list(entitie_s)[:10]))
    print("Common Entities :")
    common_list = common_Entities(entities_set, entities_name)
    print(15*"#")
    print("Common Word as Entities :")
    common_name_Entities(entities_set, entities_name)
    print(15*"#")
    print("Entities as an evenment")
    evenement_list = is_a_evenment(entities_set, entities_name)
    print(15*"#")
    print("Mentions refering to an entity :")
    most_common_set, ten_common = occurence_Entities(entities_list, entities_name)
    print("Mentions Ananlysis :")
    tot_mentions, all_mentions, occ_mentions, cap_mentions, com_mentions, same_mentions = check_mentions(mentions_dicts, entities_name)
    print(15*"#")
    print_bar(entities_name, entities_set, entities_list, most_common_set, ten_common, all_mentions, same_mentions, common_list, evenement_list)
    print_boxplot(entities_name, tot_mentions, all_mentions, occ_mentions, cap_mentions, com_mentions, densite_list)
################################################################################################################
################################################################################################################