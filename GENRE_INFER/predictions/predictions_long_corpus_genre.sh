#!/bin/bash

# Sub-Scripts for long corpus
# Split Long Corpus and make predictions for each spitting corpus

# Load Environnement
module load cpuarch/amd # Config GPU A100
module load pytorch-gpu/py3/1.12.1 # Config GPU A100
# module load python/3.7.5 # Config GPU V100
export PYTHONUSERBASE=$WORK/.local_GENRE_AMD
export PATH=$PYTHONUSERBASE/bin:$PATH

LANG=$1          # FR or EN
MODEL_PATH=$2    # 'yago' / 'wiki-abs' / custom path (where there is the folder "models" or "models_[m/t]")
DATASET=$3       # AIDA_[dev/test] / DB_[train/dev/test] / TR_[train/dev/test]
MODE=$4          # 'Base' / 'Multi' / 'Trans' / 'MTrans'
PREDICT_STRAT=$5 #prediction_genre.sh will process this arg
ARCHITECTURE=$6 #prediction_genre.sh will process this arg

if [ "$7" == "" ]; then
    EXPERIMENT="predictions"
else
    EXPERIMENT=$7 # name of output file in elevant folder
fi  

# OPTION
ELEVANT_FOLDER="$WORK/elevant/evaluation-results/$EXPERIMENT"
RESULTS="./extract_results"
SPLIT_FOLDER="./data/benchmarks/${DATASET}_split"
if [ "$DATASET" == "AIDA_dev" ] || [ "$DATASET" == "AIDA_test" ]; then
    SPLIT_SIZE=150
elif [ "$DATASET" == "DB_dev" ] || [ "$DATASET" == "DB_test" ] || [ "$DATASET" == "DB_train" ]; then
    SPLIT_SIZE=3000
elif [ "$DATASET" == "WIKI_dev" ] || [ "$DATASET" == "WIKI-dev_mini" ] || [ "$DATASET" == "WIKI-train_mini" ]; then
    SPLIT_SIZE=1000
elif [ "$DATASET" == "TR_dev" ] || [ "$DATASET" == "TR_test" ] || [ "$DATASET" == "TR_train" ]; then
    SPLIT_SIZE=500 #1000 ?
else
    echo "'$DATASET' unknowed"
    exit
fi

cd .. #$WORK/Reforged_GENRE
mkdir -p $ELEVANT_FOLDER
mkdir -p ${RESULTS}/${DATASET}_${EXPERIMENT}
date

if [ "$7" == "" ] || [ "$7" == "DO_SPLIT" ]; then
    echo "make new split for $DATASET"
    rm -rf ${SPLIT_FOLDER}
    python split_merge_large_corpus.py --mode="split" --output_folder=${SPLIT_FOLDER} --corpus="./data/benchmarks/${DATASET}.jsonl" --split_size=${SPLIT_SIZE}
else
    echo "use existing split of $DATASET"
fi

TOTAL_DOC=$(find ${SPLIT_FOLDER} -maxdepth 1 -type f | wc -l)
echo "Total Doc : ${TOTAL_DOC}"
#exit

cd predictions
if [ "${TOTAL_DOC}" == 0 ]; then exit; fi 
for (( i=0; i<${TOTAL_DOC}; i++ ))
do
    PRED_ARGS="${LANG} ${MODEL_PATH} ${SPLIT_FOLDER}/${DATASET}-${i}.jsonl ${MODE} $PREDICT_STRAT $ARCHITECTURE ${DATASET}-${i}-${EXPERIMENT} ${RESULTS}/${DATASET}_${EXPERIMENT}"
    echo "make prediction for $PRED_ARGS"
    sbatch ./predictions_genre.sh $PRED_ARGS
done

echo "Predictions launched"
date
