#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


DATASET=$1 # Dataset folder
TOKENISER=$2 # "bpe" or "spm"
MODEL=$3 # tokeniser Model (SentencePiece or vocab.bpe)
DICT=$4 # dict.source/target.txt model. Optionnal, recompute if empty

echo "Processing ${DATASET}"

cd ..

for SPLIT in train dev; do
    for LANG in "source" "target"; do
        if [ "$TOKENISER" == "spm" ]; then
            python -m scripts_mgenre.preprocess_sentencepiece --m ${MODEL} \
            --inputs ${DATASET}/${SPLIT}.${LANG} \
            --outputs ${DATASET}/${SPLIT}.spm.${LANG} \
            --workers 40
        else # $TOKENISER = "bpe"
            cd fairseq
            python -m examples.roberta.multiprocessing_bpe_encoder\
            --encoder-json "../${MODEL}/encoder.json" \
            --vocab-bpe "../${MODEL}/vocab.bpe" \
            --inputs "../${DATASET}/${SPLIT}.${LANG}" \
            --outputs "../${DATASET}/${SPLIT}.bpe.${LANG}" \
            --workers 60 \
            --keep-empty;
            cd ..
        fi
    done
done

if [ -z "$DICT" ]; then

    fairseq-preprocess \
      --source-lang "source" \
      --target-lang "target" \
      --trainpref ${DATASET}/train.spm \
      --validpref ${DATASET}/dev.spm \
      --destdir ${DATASET}/bin \
      --workers 40 ; 
      
else

    fairseq-preprocess \
      --source-lang "source" \
      --target-lang "target" \
      --trainpref ${DATASET}/train.spm \
      --validpref ${DATASET}/dev.spm \
      --destdir ${DATASET}/bin \
      --srcdict "${DICT}/dict.source.txt" \
      --tgtdict "${DICT}/dict.source.txt" \
      --workers 40 ; 

fi
