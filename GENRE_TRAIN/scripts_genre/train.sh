#!/bin/bash

# exemple : ./train.sh [Folder with data] [Name for Tensorboard] "$WORK/models/bart.large" t

CURRENT=$(pwd)
SEED=14058
DATASET=$1 # Dataset folder
NAME=$2 # Experiment Name
# Mode between "Transfer" ('t' / 'T'), "Multilingue" ('m' / 'M') or "Monolingue" (default / empty)
if [ -z "$4" ]; then
    MODELS="models"
    FINE_TUNE="Monolingual BART"
    BIN_DATASET="$DATASET/bin_bpe/"
elif [ "$4" == "t" ] || [ "$4" == "T" ]; then
    MODELS="models_t"
    FINE_TUNE="Fine Tuning from training model"
    BIN_DATASET="$DATASET/bin_bpe/"
elif [ "$4" == "m" ] || [ "$4" == "M" ]; then
    MODELS="models_m"
    FINE_TUNE="Multilingual BART"
    BIN_DATASET="$DATASET/bin_spm/"
elif [ "$4" == "h" ] || [ "$4" == "H" ]; then
    MODELS="models_hez"
    FINE_TUNE="Monolingual BARThez"
    BIN_DATASET="$DATASET/bin_spm/"
else
    MODELS="models"
    FINE_TUNE="Monolingual BART"
    BIN_DATASET="$DATASET/bin_bpe/"
fi
# Model to restore (folder with "model.pt" file)
if [ -n "$3" ]; then
    CHECKPOINT_ML=$3
elif [ "$3" == "LAST" ]; then
    CHECKPOINT_ML=$DATASET/$MODELS/checkpoint_last.pt
else # [ -z "$3" ]
    CHECKPOINT_ML="$WORK/models/bart.large" # model for BART / BARThez or other Language Model to use
fi 

# Recap settings
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Experiment : $NAME"
echo "Restore '$CHECKPOINT_ML' for '$DATASET'"
echo "Mode : $FINE_TUNE"
date
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# Define Arguments
GENERAL_ARGS="    --save-dir $DATASET/$MODELS \
    --tensorboard-logdir $WORK/tensorboard_logs/$NAME \
    --restore-file $CHECKPOINT_ML/model.pt \
    --task translation  \
    --no-epoch-checkpoints \
    --criterion label_smoothed_cross_entropy  \
    --source-lang source  \
    --target-lang target  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --update-freq 1  \
    --max-update 200000  \
    --total-num-update 200000  \
    --max-epoch 200 \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.01 \
    --optimizer adam  \
    --adam-eps 1e-08 \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --warmup-updates 500  \
    --ddp-backend no_c10d  \
    --num-workers 20  \
    --layernorm-embedding \
    --share-decoder-input-output-embed  \
    --share-all-embeddings \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --log-interval 10  \
    --patience 200 \
    --fp16 \
    --seed $SEED"

#    --task translation_from_pretrained_bart \
#    --langs en_XX,fr_XX \
#    --reset-lr-scheduler"
#    --adam-betas '(0.9, 0.999)'  \
#    --eval-scorer eval-precision-recall \
#    --best-checkpoint-metric f-1 --maximize-best-checkpoint-metric \
#    --eval-scorer-args '{\"beam\": 2}' \
if [ "$MODELS" == "models_m" ]; then
    ARGS_MODEL="    --arch mbart_large  \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --max-tokens 1024"
elif [ "$MODELS" == "models_hez" ]; then
    ARGS_MODEL="    --arch bart_base \
    --decoder-normalize-before \
    --encoder-normalize-before
    --max-tokens 4096  "
else # [ "$MODELS" == "models" ] || [ "$MODELS" == "models_t" ]
    ARGS_MODEL="    --arch bart_large \
    --max-tokens 1024"
fi

if [ "$FINE_TUNE" == "Fine Tuning from training model" ]; then
    ARGS_FINE_TUNE="    --lr 3e-05  "
else
    if [ "$3" == "LAST" ]; then
        ARGS_FINE_TUNE="    --lr 3e-05"
    else
        ARGS_FINE_TUNE="    --lr 3e-05 \
        --reset-meters \
        --reset-optimizer \
        --reset-dataloader"
    fi
fi

cd "fairseq"

# Launch Training
FINAL_ARGS="$BIN_DATASET $GENERAL_ARGS $ARGS_MODEL $ARGS_FINE_TUNE"
echo "fairseq_cli.train args :"
echo "$FINAL_ARGS"
python -m fairseq_cli.train $FINAL_ARGS
# python -m fairseq_cli.train $DATASET/bin/ $ARGS --finetune-from-model $FINE_TUNE # Old fine tuning method

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "copy and rename model and dicts"
cp ${DATASET}/${MODELS}/checkpoint_best.pt ${DATASET}/${MODELS}/model.pt #final model (used for predict outputs)
cp ${BIN_DATASET}/dict.* ${DATASET}/${MODELS} #copy dict files 
echo "DONE"
date
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

