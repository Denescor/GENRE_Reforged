#!/bin/bash
# script bash to preprocess WikiFR and TRFR2016 for GENRE and launch training

# Jean Zay Environement
#$STORE="/gpfsstore/rech/emf/ujs45li"
#$HOME="/linkhome/rech/genrqo01/ujs45li"
#$WORK="/gpfswork/rech/emf/ujs45li"

# usage exemple : ./preprocess_GENRE.sh "AIDACONLL" "EN" "-v" "2"

# Load Environnement
module load python/3.7.5
export PYTHONUSERBASE=$emf_CCFRWORK/.local_GENRE
export PATH=$PYTHONUSERBASE/bin:$PATH

# ARGUMENTS
#################################################################################################################
CORPUS=$1 #corpus name ("DB", "TR", "WIKI", "AIDACONLL")
LANG=$2 #lang of the corpus ("EN" or "FR")
VERBOSE=$3 #set to "v" to use verbose mode to all scripts
EXPERIMENT_NAME="Transfert_${LANG}_${CORPUS}" #experiment name (depends of LANG and CORPUS)
FORMAT=$4 # Type of Preprocessing (use for data folder) : default : 1 (like ED) / 2 (1 doc per entry) / hez (use BART_HEZ) / hard (use TR_hard only - always active in addition with CORPUS TR)
#################################################################################################################

# VERIF ARGUMENTS   
#################################################################################################################
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
if [ -z "$CORPUS" ]; then
    echo "CORPUS have to be set"
    exit
else
    echo "Selected Corpus : $CORPUS"
fi

if [ "$LANG" == "EN" ] || [ "$LANG" == "FR" ]; then
    echo "Selected Lang : $LANG"
else
    echo "LANG must be 'FR' or 'EN'"
    exit
fi

echo "Experiment Name : $EXPERIMENT_NAME" # Transfert_EN_WIKI / Transfert_FR_WIKI / Transfert_FR_DB / ...

if [ -n "$FORMAT" ]; then
    echo "Preprocess Type : $FORMAT"
else
    FORMAT="2"
    echo "Preprocess Type : $FORMAT (default)"
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

if [ "$LANG" == "FR" ]; then
    LANG_LOWER="fr"
else
    LANG_LOWER="en"
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#################################################################################################################

# COMMON VARIABLES
#################################################################################################################
##### FOLDERS
CURRENT_F=$(pwd) #current script folder : "GENRE/"
SCRIPT_F="$CURRENT_F" #folder where are the scripts
SCRIPT_E="" #folder where the scripts from Reforged_GENRE #TODO définir
EL_F="$WORK/$CORPUS" #folder where are the TR or DB data
DATASET_F="$WORK/GENRE/$EXPERIMENT_NAME" #folder where are the fairseq's preprocessed datasets
DATA_TYPE="Preprocess" # "WIKIFR" / "WIKIEN" / "DBFR" / "TRFR" / "AIDACONLLEN"
KOLITSAS_DATA="$WORK/end2end/data"
KOLITSAS_CODE="$WORK/end2end/code"
##### FILES
WIKI_NAME_ID_MAP="wiki_name_id_map_${LANG}.txt" 
PEM_FILE="$KOLITSAS_DATA/basic_data/crosswikis_wikidump${LANG}_p_e_m.txt" 
ENTITIES_UNIVERSE="$KOLITSAS_DATA/entities/entities_universe_${LANG_LOWER}.txt" #TODO importer les bons fichiers / revoir les scripts
##### MODELS & ENCODERS
TOKEN_T="spm" # "bpe" or "spm"
DICT_BART="$WORK/models/mbarthez.large"
# bpe : $WORK/models/bart.large
# spm : $WORK/models/mbart.cc25 || $WORK/models/mbart.cc100 || $WORK/models/barthez.base || $WORK/models/mbarthez.large
TOKEN_F="$WORK/models/mbarthez.large/sentence.bpe.model" #folder where is the tokeniser model for the preprocessed_fairseq (SentencePiece - spm or BinaryPairEncoder - bpe)
# bpe : $WORK/models/gpt2/
# spm : $WORK/models/mbart.cc25/sentence.bpe.model
# spm : $WORK/models/mbart.cc100/spm_256000.model
# spm : $WORK/models/barthez.base/sentence.bpe.model
# spm : $WORK/models/mbarthez.large/sentence.bpe.model
REDIRECT_WIKI="${WORK}/target"
##### WIKIPEDIA DATA
WIKIDATA_F="$STORE/wikidata_dump" #folder where is the file "wikidata-all.json" (1To)
##### OPTIONS
KILT="NO"
PROCESS_WIKIDATA="NO"
#################################################################################################################

# CONFIGURABLE VARIABLES
#################################################################################################################
##### WIKIPEDIA DATA
if [ "$LANG" == "FR" ]; then
    WIKIPEDIA_F="$STORE/wikipedia/fr" #folder where are the wikipedia data (and where the generated wikidata files will move)
else # LANG == "EN"
    WIKIPEDIA_F="$STORE/wikipedia/en" #folder where are the wikipedia data (and where the generated wikidata files will move) #TODO vérifier l'existence du dossier
fi
##### OPTIONS
if [ "$LANG" == "FR" ]; then
    PROP_WIKI=1 #0.1
    PROP_NEL=1
    ENTLANG="--entity_language=fr"
else # LANG == "EN"
    PROP_WIKI=1 #0.001
    PROP_NEL=1 #0.15
    ENTLANG="--entity_language=en"
fi
#################################################################################################################

# BEGIN PREPROCESS
echo "scripts files : $SCRIPT_F"
cd $SCRIPT_F
mkdir -p $DATASET_F/${DATA_TYPE}_${FORMAT}

#create pem + entities universe
#   - cf scripts kolitsas
#   - cf $PEM_FILE
echo ">>>>>>>>>>> find PEM file"
echo "exemple PEM final : $PEM_FILE"
head -n 3 $PEM_FILE

#create cand entities
#   - create_candidates_dict.py
#create mentions tries
#   - create_mentions_trie.py
#   - il faut un modèle pré-appris
#TODO vérifier cette étape -- UTILISER GENRE ELEVANT
#cd SCRIPT_E
#python -m create_candidates_dict 
#python -m create_mentions_trie
#cd $SCRIPT_F

if [ "$CORPUS" == "WIKI" ]; then
####################################### PREPROCESS WIKI #######################################
    echo ">>>>>>>>>>> verif mentions integrity ----------------- ABORTED"
    echo ">>>>>>>>>>> extract and process wikipedia and wikidata"
    if [ "$KILT" == "Yes" ]; then
        echo "create kilt document --- slower"
        python -m create_kilt_data_paragraphs --step preprocess --folder "./kilt_data" --threads 32
        python -m create_kilt_data_paragraphs --step main --chunk_size 100 --folder "./kilt_data" --rank 0
        python -m create_kilt_data_paragraphs --step merge --folder "./kilt_data" --threads 32
        echo "kilt document created"
    else
        echo "kilt document skipped --- faster"
    fi
    if [ "$PROCESS_WIKIDATA" == "YES" ]; then
        echo "process wikidata --- slower"
        python -m scripts_mgenre.preprocess_extract --base_wikipedia="$WIKIPEDIA_F" --lang="$LANG_LOWER" --extract_mode="nel" --total_nel=$PROP_NEL $VERBOSE
        echo "process wikipedia done"
        python -m scripts_mgenre.preprocess_anchors "prepare" --base_wikipedia="$WIKIPEDIA_F" --base_wikidata="$WIKIDATA_F" --langs="$LANG_LOWER" $VERBOSE
        python -m scripts_mgenre.preprocess_anchors "solve" --base_wikipedia="$WIKIPEDIA_F" --base_wikidata="$WIKIDATA_F" --langs="$LANG_LOWER" $VERBOSE
        python -m scripts_mgenre.preprocess_anchors "fill" --base_wikipedia="$WIKIPEDIA_F" --base_wikidata="$WIKIDATA_F" --langs="$LANG_LOWER" $VERBOSE
        echo "process anchors done"
    else
        echo "process wikidata skipped --- faster"
    fi
    echo "process Wikipedia ${LANG} done"
    echo ">>>>>>>>>>> preprocess dataset"
    python -m scripts_genre.split_kilt_to_train_dev "$WIKIPEDIA_F/${LANG_LOWER}wiki.pkl" "$EL_F" --base_wikidata="$WIKIDATA_F" --mode=$FORMAT --proportion_dev=0.15 --proportion_wiki=$PROP_WIKI --format="pkl" $VERBOSE
    mv $EL_F/wiki-train-kilt.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-train-kilt.jsonl
    mv $EL_F/wiki-dev-kilt.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-dev-kilt.jsonl
elif [ "$CORPUS" == "DB" ] || [ "$CORPUS" == "TR" ]; then
####################################### PREPROCESS DB/TR #######################################
    #aligner dataset sur les entités/mentions du pem/cand
    #   - vérifier la proportion de mentions du dataset dans PEM
    #   - vérifier la proportion d'entités du dataset dans cand
    #   - cf count_mentions.py
    cd $KOLITSAS_CODE
    echo ">>>>>>>>>>> verif mentions integrity"
    python -m count_mentions --p_e_m_file="$PEM_FILE" --TR_folder="$EL_F/fr/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name $ENTLANG
    python -m count_entities --entities_universe="$ENTITIES_UNIVERSE" --TR_folder="$EL_F/fr/" --unify_entity_name $ENTLANG
    cd $SCRIPT_F
    echo ">>>>>>>>>>> preprocess dataset"
    python -m scripts_mgenre.preprocess_TR_Format_to_Kilt_Format --input_dir="$EL_F" --output_dir="$DATASET_F" --base_wikidata="$WIKIDATA_F" --mode=$FORMAT $VERBOSE
    mv $DATASET_F/fr-kilt-train.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-train-kilt.jsonl
    mv $DATASET_F/fr-kilt-test.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-test-kilt.jsonl
    mv $DATASET_F/fr-kilt-dev.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-dev-kilt.jsonl
    if [ "$CORPUS" == "TR" ]; then
        mkdir -p $DATASET_F/${DATA_TYPE}_hard
        mv $DATASET_F/fr-kilt-train-hard.jsonl $DATASET_F/${DATA_TYPE}_hard/${CORPUS}-train_hard-kilt.jsonl
        mv $DATASET_F/fr-kilt-test-hard.jsonl $DATASET_F/${DATA_TYPE}_hard/${CORPUS}-test_hard-kilt.jsonl
        mv $DATASET_F/fr-kilt-dev-hard.jsonl $DATASET_F/${DATA_TYPE}_hard/${CORPUS}-dev_hard-kilt.jsonl
    fi
elif [ "$CORPUS" == "DEBUG" ]; then
    echo ">>>>>>>>>>> Create Debug Test Corpus"
    python -m pick_debug_documents --folder="$EL_F/.." --save="$WORK/DEBUG/fr/dev" --lendoc=20 --min_lenght=500 --max_lenght=2000 #Select 10 docs from TR/test & 10 docs from DB/test
    cp $WORK/DEBUG/fr/dev/* $WORK/DEBUG/fr/train
    echo ">>>>>>>>>>> verif mentions integrity"
    cd $KOLITSAS_CODE
    python -m count_mentions --p_e_m_file="$PEM_FILE" --TR_folder="$EL_F/fr/" --wiki_path="$WIKI_NAME_ID_MAP" --unify_entity_name $ENTLANG
    python -m count_entities --entities_universe="$ENTITIES_UNIVERSE" --TR_folder="$EL_F/fr/" --unify_entity_name $ENTLANG
    cd $SCRIPT_F
    echo ">>>>>>>>>>> preprocess dataset"
    python -m scripts_mgenre.preprocess_TR_Format_to_Kilt_Format --input_dir="$EL_F" --output_dir="$DATASET_F" --base_wikidata="$WIKIDATA_F" --mode=$FORMAT $VERBOSE
    mv $DATASET_F/fr-kilt-train.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-train-kilt.jsonl
    mv $DATASET_F/fr-kilt-dev.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-dev-kilt.jsonl
    rm $DATASET_F/fr-kilt-test.jsonl
elif [ "$CORPUS" == "AIDACONLL" ]; then
####################################### PREPROCESS AIDA #######################################
    #spécifique à AIDA : convertir les entrées ED en entrées EL
    echo ">>>>>>>>>>> verif mentions integrity ----------------- ABORTED"
    echo ">>>>>>>>>>> preprocess dataset"
    cp $KOLITSAS_DATA/basic_data/$WIKI_NAME_ID_MAP $EL_F/ED/wiki_name_id_map_${LANG}.wiki
    python -m scripts_genre.convert_aidaED_to_aidaEL --input_dir=$EL_F/ED/ --output_dir=$EL_F/EL/ --wiki_name_id_map=wiki_name_id_map_${LANG}.wiki --base_wikidata="$WIKIDATA_F" --format=$FORMAT $VERBOSE
    mv $WORK/AIDACONLL/EL/aida-dev-kilt-EL.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-dev-kilt.jsonl
    mv $WORK/AIDACONLL/EL/aida-test-kilt-EL.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-test-kilt.jsonl
    mv $WORK/AIDACONLL/EL/aida-train-kilt-EL.jsonl $DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-train-kilt.jsonl
else
    echo "unrecognize corpus '$CORPUS'"
fi

#create dataset (source : non annoté || target : annoté)
#   - refaire la fonction "create_input"
#   - fonctions : 
#       - scripts_mgenre.preprocess_TRFR2016
#       - scripts_genre.convert_kilt_to_fairseq
#       - preprocess_fairseq.sh

# PREPROCESS TO FAIRSEQ
echo ">>>>>>>>>>> convert to fairseq"
python -m scripts_genre.convert_kilt_to_fairseq "$DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-train-kilt.jsonl" "$DATASET_F/${DATA_TYPE}_${FORMAT}" --mode='el' $VERBOSE
python -m scripts_genre.convert_kilt_to_fairseq "$DATASET_F/${DATA_TYPE}_${FORMAT}/${CORPUS}-dev-kilt.jsonl" "$DATASET_F/${DATA_TYPE}_${FORMAT}" --mode='el' $VERBOSE
if [ "$CORPUS" == "TR" ]; then
    python -m scripts_genre.convert_kilt_to_fairseq "$DATASET_F/${DATA_TYPE}_hard/${CORPUS}-train_hard-kilt.jsonl" "$DATASET_F/${DATA_TYPE}_hard" --mode='el' $VERBOSE
    python -m scripts_genre.convert_kilt_to_fairseq "$DATASET_F/${DATA_TYPE}_hard/${CORPUS}-test_hard-kilt.jsonl" "$DATASET_F/${DATA_TYPE}_hard" --mode='el' $VERBOSE
fi
ls -lah $DATASET_F/${DATA_TYPE}_${FORMAT}

#preprocess for fairseq format
# with
#   - dataset folder where are the .source and .target files
#   - the Tokeniser's model folder for encoding the .source and .target data
#       - choice between the SentencePiece tokeniser or BPE Tokeniser from RoBERTa. 
#       - for Transfert Learning, requirement is to choice the same as the pretrained model
#   - no reuse dict because we assume that the data are new
#       - Give a 4th argument to reuse a dict (from BART or BARThez for exemple)
echo ">>>>>>>>>>> preprocess fairseq"
cd $SCRIPT_F/scripts_genre 
./preprocess_fairseq.sh "$DATASET_F/${DATA_TYPE}_${FORMAT}" "$TOKEN_T" "$TOKEN_F" "$DICT_BART"
if [ "$CORPUS" == "TR" ]; then
    ./preprocess_fairseq.sh "$DATASET_F/${DATA_TYPE}_hard" "$TOKEN_T" "$TOKEN_F" "$DICT_BART"
fi
echo ">>>>>>>>>>> Preprocess Done, training is ready <<<<<<<<<<<<<<<"
