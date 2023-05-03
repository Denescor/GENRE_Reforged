#!/bin/bash

# Sub-Scripts for long corpus
# Merge splitting predictions from a same corpus and for a same model and transform prediction for the final evaluation

# Load Environnement
module load python/3.7.5
export PYTHONUSERBASE=$WORK/.local_GENRE
export PATH=$PYTHONUSERBASE/bin:$PATH

DATASET=$1       # AIDA / DB_[train/dev/test] / TR_[train/dev/test]
if [ "$2" == "" ]; then
    EXPERIMENT="predictions"
else
    EXPERIMENT=$2 # name of output file in elevant folder
fi

# OPTION
ELEVANT_FOLDER="$WORK/elevant/evaluation-results/$EXPERIMENT"
RESULTS="./extract_results"
SPLIT_FOLDER="${RESULTS}/${DATASET}_${EXPERIMENT}"
if [ "$DATASET" == "AIDA_dev" ] || [ "$DATASET" == "AIDA_test" ]; then
    SPLIT_SIZE=150
    LG="en"
elif [ "$DATASET" == "DB_dev" ] || [ "$DATASET" == "DB_test" ] || [ "$DATASET" == "DB_train" ]; then
    SPLIT_SIZE=3000
    LG="fr"
elif [ "$DATASET" == "TR_dev" ] || [ "$DATASET" == "TR_test" ] || [ "$DATASET" == "TR_train" ]; then
    SPLIT_SIZE=500 #1000 ?
    LG="fr"
elif [ "$DATASET" == "WIKI_dev" ] || [ "$DATASET" == "WIKI-dev_mini" ] || [ "$DATASET" == "WIKI-train_mini" ]; then
    SPLIT_SIZE=1000
    LG="fr"
else
    echo "'$DATASET' unknowed"
    exit
fi
mkdir -p $ELEVANT_FOLDER
cd .. #$WORK/Reforged_GENRE
date

echo "merge $DATASET predictions"
python split_merge_large_corpus.py --mode="merge" --output_folder=${SPLIT_FOLDER} --corpus="./data/benchmarks/${DATASET}.jsonl" --split_size=${SPLIT_SIZE} --model_name=${EXPERIMENT}

echo "transform $DATASET predictions"
python transform_predictions.py ${SPLIT_FOLDER}/${DATASET}_${EXPERIMENT}.jsonl -o ${SPLIT_FOLDER}/${DATASET}_${EXPERIMENT}.qids.jsonl -l $LG
echo "move generated predictions to $ELEVANT_FOLDER"
rm ${SPLIT_FOLDER}/${DATASET}_${EXPERIMENT}.jsonl
mv ${SPLIT_FOLDER}/${DATASET}_${EXPERIMENT}.qids.jsonl ${ELEVANT_FOLDER}/${EXPERIMENT}.${DATASET}.linked_articles.jsonl

ls -l ${ELEVANT_FOLDER}

echo "Predictions Done"
date
