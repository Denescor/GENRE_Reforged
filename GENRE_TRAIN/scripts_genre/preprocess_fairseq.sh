#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


DATASET=$1 # Dataset folder
TOKENISER=$2 # "bpe" or "spm"
MODEL=$3 # tokeniser Model (SentencePiece or vocab.bpe)
DICT_DATASET=$4 # dict.source/target.txt model. Optionnal, recompute if empty

if [ "$DICT_DATASET" == "" ]; then
    DICT="${DATASET}/bin_${TOKENISER}" # dict.source/target.txt model. Optionnal, recompute if empty
else
    DICT="${DICT_DATASET}/bin_${TOKENISER}" # dict.source/target.txt model. Optionnal, recompute if empty
    mkdir -p $DICT
    cp "${DICT_DATASET}/dict.txt" "$DICT/dict.source.txt"
    cp "${DICT_DATASET}/dict.txt" "$DICT/dict.target.txt"
    echo "use dict from $DICT_DATASET"
fi

echo "Processing ${DATASET}"

cd .. #folder GENRE/

echo "----------- TOKENISE DATASETS -----------"
for SPLIT in train dev; do
    for LANG in "source" "target"; do
        if [ "$TOKENISER" == "spm" ]; then
            python -m scripts_mgenre.preprocess_sentencepiece --m ${MODEL} \
            --inputs ${DATASET}/${SPLIT}.${LANG} \
            --outputs ${DATASET}/${SPLIT}.spm.${LANG} \
            --workers 40
        else # $TOKENISER = "bpe"
            cd fairseq #folder GENRE/fairseq
            python -m examples.roberta.multiprocessing_bpe_encoder\
            --encoder-json "${MODEL}/encoder.json" \
            --vocab-bpe "${MODEL}/vocab.bpe" \
            --inputs "${DATASET}/${SPLIT}.${LANG}" \
            --outputs "${DATASET}/${SPLIT}.bpe.${LANG}" \
            --workers 60 \
            --keep-empty;
            cd .. #folder GENRE/
        fi
    done
    if [ "$TOKENISER" == "spm" ]; then
        python -m scripts_mgenre.align_sentencepiece \
            --source ${DATASET}/${SPLIT}.spm.source \
            --target ${DATASET}/${SPLIT}.spm.target
    fi
done

cd fairseq

echo "----------- BINARIZE DATASETS -----------"
if [ -d $DICT ]; then

    #fairseq-preprocess \
    python -m fairseq_cli.preprocess \
      --source-lang "source" \
      --target-lang "target" \
      --trainpref ${DATASET}/train.${TOKENISER} \
      --validpref ${DATASET}/dev.${TOKENISER} \
      --destdir ${DATASET}/bin_${TOKENISER} \
      --srcdict "${DICT}/dict.source.txt" \
      --tgtdict "${DICT}/dict.target.txt" \
      --workers 40 ; 
      
else

    #fairseq-preprocess \
    python -m fairseq_cli.preprocess \
      --source-lang "source" \
      --target-lang "target" \
      --trainpref ${DATASET}/train.${TOKENISER} \
      --validpref ${DATASET}/dev.${TOKENISER} \
      --destdir ${DATASET}/bin_${TOKENISER} \
      --workers 40 ;

fi

echo "----------- ----------- -----------"
