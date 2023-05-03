#!/bin/bash
# All Preprocess or Training
# Fast Script

# exemple (Preprocess EN) : ./Fast_All_Script.sh "PREPROCESS" "EN" "2"
# exemple (Preprocess FR) : ./Fast_All_Script.sh "PREPROCESS" "FR" "2"

# exemple (Train FR with prepared data) : ./Fast_All_Script.sh "TRAIN" "FR" "2"
# exemple (Continue last FR training) : ./Fast_All_Script.sh "CONTINUE" "FR" "2"

MODE=$1
LANG=$2
FORM=$3

if [ -z "$FORM" ]; then
    FORM="2"
    echo "Default Preprocess Type : 'Preprocess_2'"
else
    echo "Selected Preprocess Type : 'Preprocess_${FORM}'"
fi

if [ "$MODE" == "PREPROCESS" ]; then
    if [ "$LANG" == "FR" ]; then
        ./preprocess_GENRE.sh "TR" $LANG "-v" $FORM
        ./preprocess_GENRE.sh "DB" $LANG "-v" $FORM
        ./preprocess_GENRE.sh "WIKI" $LANG "-v" $FORM
    elif [ "$LANG" == "EN" ]; then
        ./preprocess_GENRE.sh "AIDACONLL" $LANG "-v" $FORM
        ./preprocess_GENRE.sh "WIKI" $LANG "-v" $FORM
    else
        echo "LANG must be 'FR' or 'EN'"
	fi
elif [ "$MODE" == "TRAIN" ]; then
    cd scripts_slurm
    if [ "$LANG" == "FR" ]; then
        sbatch ./launch_exp_Baseline.slurm $LANG "WIKI DB TR" $FORM "START"
        sbatch ./launch_exp_Transfert.slurm $LANG "DB TR" $FORM "START" "AIDACONLL"
        sbatch ./launch_exp_Multilingue.slurm $LANG "WIKI DB TR" $FORM "START"
    elif [ "$LANG" == "EN" ]; then
        sbatch ./launch_exp_Baseline.slurm $LANG "WIKI AIDACONLL" $FORM "START"
        sbatch ./launch_exp_Transfert.slurm $LANG "AIDACONLL" $FORM "START" "WIKIEN"
    else
        echo "LANG must be 'FR' or 'EN'"
	fi
elif [ "$MODE" == "CONTINUE" ]; then
    cd scripts_slurm
    if [ "$LANG" == "FR" ]; then
        sbatch ./launch_exp_Baseline.slurm $LANG "WIKI DB TR" $FORM "CONTINUE"
        sbatch ./launch_exp_Transfert.slurm $LANG "DB TR" $FORM "CONTINUE" "AIDACONLL"
        sbatch ./launch_exp_Multilingue.slurm $LANG "WIKI DB TR" $FORM "CONTINUE"
    elif [ "$LANG" == "EN" ]; then
        sbatch ./launch_exp_Baseline.slurm $LANG "WIKI AIDACONLL" $FORM "CONTINUE"
        sbatch ./launch_exp_Transfert.slurm $LANG "AIDACONLL" $FORM "CONTINUE" "WIKIEN"
    else
        echo "LANG must be 'FR' or 'EN'"
	fi
else
    echo "Choose mode between 'PREPROCESS', 'TRAIN' or 'CONTINUE'"
fi
