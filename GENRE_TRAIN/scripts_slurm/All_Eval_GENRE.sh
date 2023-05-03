#!/bin/bash

CURRENT=$(eval)

#sbatch ./evaluate_GENRE.sh "Transfert_FR_TR/TRFR" "B" "TR_KILT" "" "-v"

LANG=$1

if [ "$LANG" = "fr" ] || [ "$LANG" = "FR" ]; then
    echo ">>>> eval on FR models and datasets"
    for dataset in "TR_KILT" "DB_KILT_dev_test" "WK_KILT_dev" "DB_KILT_Train" "WK_KILT_Train"
    do
        for folder in "TR/TRFR" "TR/TRFR_hard" "DB/DBFR" "WIKI/WIKIFR"
        do
            echo "evaluate $folder on $dataset"
            sbatch ./evaluate_GENRE.sh "Transfert_FR_${folder}" "B" $dataset "" "-v"
            sbatch ./evaluate_GENRE.sh "Transfert_FR_${folder}" "T" $dataset "" "-v"
        done
    done
    echo "all eval launched"
elif [ "$LANG" = "en" ] || [ "$LANG" = "EN" ]; then
    echo ">>> eval on EN models and datasets"
    for dataset in "AIDA_KILT" #"WK_EN_KILT_dev_test" "WK_EN_Train"
    do
        for folder in "AIDA" #"WIKI"
        do
            echo "evaluate $folder on $dataset"
            sbatch ./evaluate_GENRE.sh "Init_EN_${folder}" "B" $dataset "" "-v" #"$WORK/GENRE/TRIE/trie_dict_en.pkl" "-v"
        done
    done
    echo "all eval launched"
else
    echo "ARG must be 'EN' or 'FR'"
fi
