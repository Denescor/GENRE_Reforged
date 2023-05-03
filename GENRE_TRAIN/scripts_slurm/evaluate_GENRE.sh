#!/bin/bash

#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH -A gcp@v100

#SBATCH --time=98:00:00

#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --job-name=EVAL_GENRE
#SBATCH --signal=B:USR1@120

CURRENT=$(pwd)/..
DATA_PATH="$WORK/GENRE/$1"
if [ "$2" == "t" ] || [ "$2" == "T" ] || [ "$2" == "transfert" ]; then
    TRANSFERT="with fine-tune"
    MODEL_PATH="$DATA_PATH/models_t/ --mode=el --beams=6 --checkpoint_file=checkpoint_best.pt"
    cp $DATA_PATH/bin/dict* $DATA_PATH/models_t/
else # $2 == B || b || baseline
    TRANSFERT="baselined"
    MODEL_PATH="$DATA_PATH/models/ --mode=el --beams=6 --checkpoint_file=checkpoint_best.pt"
    cp $DATA_PATH/bin/dict* $DATA_PATH/models/
fi
INPUT_PATH="$WORK/GENRE/KILT/$3/"
OUTPUT_PATH="$DATA_PATH/eval/"
BART_PATH="--local_archive=$WORK/models"
TRIE=$4
VERBOSE=$5

echo "prepare eval $1"
mkdir -p $OUTPUT_PATH
#mkdir -p $INPUT_PATH
#cp ${DATA_PATH}/*kilt.jsonl "$INPUT_PATH"
cd $CURRENT
echo "current folder : $(pwd)"
echo "datasets :"
ls $INPUT_PATH

if [ -n "$TRIE" ]; then
    TRIE="--trie=$TRIE --candidates"
else
    TRIE="--candidates --free_generation"
fi

if [ -z "$VERBOSE" ]; then
    echo "verbose mode desactivated"
else
    echo "verbose mode activated"
fi

ARGS="$MODEL_PATH $INPUT_PATH $OUTPUT_PATH $TRIE $BART_PATH $VERBOSE"

date
echo ">>>>>>>>>>>>>>> evaluate $1 $TRANSFERT on $3 <<<<<<<<<<<<<<<"
echo ">>>>> args : $ARGS"
python -m scripts_genre.evaluate_kilt_dataset $ARGS
echo "done"
date
