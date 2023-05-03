#!/bin/bash

# exemple : ./train_continue.sh [Folder with data] [Name for Tensorboard] "$WORK/models/bart.large" t

CURRENT=$(pwd)
DATASET=$1 # Dataset folder
NAME=$2 # Experiment Name
# Model to restore (folder with "model.pt" file)
CHECKPOINT_ML=$3 # unused for continue training
# Mode between "Transfer" ('t' / 'T'), "Multilingue" ('m' / 'M') or "Monolingue" (default / empty)
if [ -z "$4" ]; then
    MODELS="models"
    FINE_TUNE="Monolingual BART"
elif [ "$4" == "t" ] || [ "$4" == "T" ]; then
    MODELS="models_t"
    FINE_TUNE="Fine Tuning from training model"
elif [ "$4" == "m" ] || [ "$4" == "M" ]; then
    MODELS="models_m"
    FINE_TUNE="Multilingual BART"
else
    MODELS="models"
    FINE_TUNE="Monolingual BART"
fi

# Recap settings
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Experiment : $NAME"
echo "Restore '$MODELS/checkpoint_last.pt' for '$DATASET'"
echo "Mode : $FINE_TUNE"
date
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# Define Arguments
GENERAL_ARGS="    --save-dir $DATASET/$MODELS/ \
    --tensorboard-logdir $WORK/tensorboard_logs/$NAME \
    --restore-file $DATASET/$MODELS/checkpoint_last.pt \
    --task translation  \
    --no-epoch-checkpoints \
    --criterion label_smoothed_cross_entropy  \
    --source-lang source  \
    --target-lang target  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --max-tokens 1024  \
    --update-freq 1  \
    --max-update 2000000  \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.01 \
    --optimizer adam  \
    --adam-eps 1e-08 \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 3e-05  \
    --total-num-update 2000000  \
    --warmup-updates 500  \
    --ddp-backend no_c10d  \
    --num-workers 20  \
    --layernorm-embedding \
    --share-decoder-input-output-embed  \
    --share-all-embeddings \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --log-interval 10  \
    --patience 200"

#    --task translation_from_pretrained_bart \
#    --langs en_XX,fr_XX \
#    --reset-dataloader \
#    --reset-lr-scheduler"
#     --adam-betas '(0.9, 0.999)'  \
if [ "$MODELS" == "models_m" ]; then
    ARGS_MODEL="    --arch mbart_large  \
    --encoder-normalize-before \
    --decoder-normalize-before"
else # [ "$MODELS" == "models" ] || [ "$MODELS" == "models_t" ]
    ARGS_MODEL="    --arch bart_large"
fi

if [ "$FINE_TUNE" == "Fine Tuning from training model" ]; then
    ARGS_FINE_TUNE="    --lr 3e-05 "
else
    ARGS_FINE_TUNE="    --lr 3e-04 "
fi

cd "fairseq"

# Launch Training
FINAL_ARGS="$DATASET/bin/ $GENERAL_ARGS $ARGS_MODEL $ARGS_FINE_TUNE"
echo "fairseq_cli.train args :"
echo "$FINAL_ARGS"
python -m fairseq_cli.train $FINAL_ARGS
# python -m fairseq_cli.train $DATASET/bin/ $ARGS --finetune-from-model $FINE_TUNE # Old fine tuning method

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "copy and rename model and dicts"
cp $DATASET/$MODELS/checkpoint_best.pt $DATASET/$MODELS/model.pt #final model (used for predict outputs)
cp $DATASET/bin/dict.* $DATASET/$MODEL #copy dict files 
echo "DONE"
date
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
