#!/bin/bash
# script bash to preprocess WikiFR and TRFR2016 for GENRE and launch training

# Jean Zay Environement
#$STORE="/gpfsstore/rech/emf/ujs45li"
#$HOME="/linkhome/rech/genrqo01/ujs45li"
#$WORK="/gpfswork/rech/emf/ujs45li"

# usage exemple : ./preprocess_To_Elevant.sh "TR" "-v"

# Load Environnement
module load python/3.7.5
export PYTHONUSERBASE=$WORK/.local_GENRE
export PATH=$PYTHONUSERBASE/bin:$PATH

# ARGUMENTS
#################################################################################################################
CORPUS=$1
VERBOSE=$2
#################################################################################################################

# VERIF ARGUMENTS   
#################################################################################################################
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
if [ -z "$CORPUS" ]; then
    echo "CORPUS have to be set"
    echo "Available CORPUS : 'AIDA', 'DB', 'TR'"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    exit
else
    echo "Selected Corpus : $CORPUS"
fi

if [ "$VERBOSE" == "v" ] || [ "$VERBOSE" == "-v" ]; then
    VERBOSE="-v" #"-v" is verbose mode in all scripts
    echo "verbose mode activated"
elif [ "$VERBOSE" == "d" ] || [ "$VERBOSE" == "-d" ]; then
    VERBOSE="-d" #"-d" is debug mode in all scripts
    echo "debug mode activated"
else
    VERBOSE="" #empty argument is no argument
    echo "verbose mode desactivated"
fi
#################################################################################################################

# COMMON VARIABLES
#################################################################################################################
##### FOLDERS
CURRENT_F=$(pwd) #current script folder : "elevant/"
SCRIPT_E="$WORK/elevant" #folder where are the scripts
SCRIPT_G="$WORK/Reforged_GENRE" #folder where the scripts from Reforged_GENRE
BENCHMARKS_F="$SCRIPT_G/data/benchmarks/"
##### WIKIPEDIA DATA
WIKIDATA_F="$STORE/wikidata_dump" #folder where is the file "wikidata-all.json" (1To)
##### PRINT VARIABLES
echo "CURRENT FOLDER : '$CURRENT_F'"
echo "Final Benchmarks Outputs : '$BENCHMARKS_F'"
#################################################################################################################

# CONFIGURABLE VARIABLES
#################################################################################################################
##### FOLDERS
if [ "$CORPUS" == "AIDA" ]; then
    DATASET_F="$WORK/AIDACONLL/ED/"
    CORPUS_TYPE="AIDA"
elif [ "$CORPUS" == "TR" ] || [ "$CORPUS" == "DB" ]; then
    DATASET_F="$WORK/$CORPUS/fr/"
    CORPUS_TYPE="TR"
elif [ "$CORPUS" == "ACE" ] || [ "$CORPUS" == "AQUAINT" ] || [ "$CORPUS" == "CLUEWEB" ] || [ "$CORPUS" == "MSNBC" ]; then
    DATASET_F="$WORK/OTHEREN/"
    CORPUS_TYPE="AIDA"
elif [ "$CORPUS" == "DEBUG" ]; then
    DATASET_F="$WORK/$CORPUS/fr/"
    CORPUS_TYPE="TR"
elif [ "$CORPUS" == "WIKI" ]; then
    DATASET_F="$WORK/GENRE/Transfert_FR_WIKI/"
    CORPUS_TYPE="KILT"
elif [ "$CORPUS" == "DBMINI" ] || [ "$CORPUS" == "DBMINIPOOR" ]; then
    DATASET_F="$WORK/GENRE/Transfert_FR_${CORPUS}/"
    CORPUS_TYPE="KILT"
elif [ "$CORPUS" == "WIKIMINI" ] || [ "$CORPUS" == "WIKIMINIPOOR" ]; then
    DATASET_F="$WORK/GENRE/Transfert_FR_${CORPUS}/"
    CORPUS_TYPE="KILT"
else
    echo "CORPUS not recognized"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    exit
fi
##### WIKIPEDIA DATA
if [ "$CORPUS" == "AIDA" ] || [ "$CORPUS" == "ACE" ] || [ "$CORPUS" == "AQUAINT" ] || [ "$CORPUS" == "CLUEWEB" ] || [ "$CORPUS" == "MSNBC" ]; then
    LANG="en"
    WIKI_NAME_ID_MAP="wiki_name_id_map_EN19.wiki"
else
    LANG="fr"
    WIKI_NAME_ID_MAP="wiki_name_id_map_FR.wiki"
fi
##### PRINT OPTIONS
echo "$CORPUS FOLDER : '$DATASET_F'"
echo "$CORPUS LANG : '$LANG'"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#################################################################################################################

# EXECUTE SCRIPT
#################################################################################################################
date
python -m scripts_genre.To_Elevant_Format --dataset_folder="$DATASET_F" --output_folder="$BENCHMARKS_F" --entity_language="$LANG" --wiki_path="$WIKI_NAME_ID_MAP" --base_wikidata="$WIKIDATA_F" --type_dataset="$CORPUS_TYPE"
ls -l $BENCHMARKS_F
echo "DONE"
date
#################################################################################################################
