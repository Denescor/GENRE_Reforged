#!/bin/bash
# script bash expérience : expériences baseline learning GENRE + TR / DB / WIKI

#SBATCH --qos=qos_gpu-t3
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:4
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH -A gcp@v100

#SBATCH --time=19:00:00

#SBATCH --array=0-3

#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --job-name=TS_ALL
#SBATCH --signal=B:USR1@120

# Load Environnement
module load python/3.7.5 #GPU V100
export PYTHONUSERBASE=$WORK/.local_GENRE
export PATH=$PYTHONUSERBASE/bin:$PATH

CURRENT_F=$(pwd)/.. #current script folder : "GENRE/"
LANG=$1 # Lang : "FR" or "EN"
DATASETS=$2 # Datasets, exemple : "WIKI DB TR" / "WIKI AIDACONLL"
PREPROCESS=$3 # Type of Preprocess : 1 (default) / 2 / hez
MODE=$4 # Begin training or continue existing training
TRANSFER_MODEL=$5
TRANSFER_SIGN="t"
if [ "$TRANSFER_MODEL" == "" ]; then
    echo "default base model : WIKI EN"
    MODEL_F="$WORK/GENRE/Init_EN_WIKI" # Initial pretrained model (EN on WIKI)
elif [ "$TRANSFER_MODEL" == "BARTHEZ" ]; then
    echo "Selected base model : BARThez"
    MODEL_F="$WORK/models/barthez.base" # BARThez
elif [ "$TRANSFER_MODEL" == "WIKIEN" ]; then
    echo "Selected base model : WIKI EN"
    MODEL_F="$WORK/GENRE/Init_EN_WIKI" # Initial pretrained model (EN on WIKI)
elif [ "$TRANSFER_MODEL" == "AIDACONLL" ]; then
    echo "Selected base model : AIDA/CoNLL"
    MODEL_F="$WORK/GENRE/Init_EN_AIDA" # Initial pretrained model (EN on AIDA/CoNLL)
elif [ "$TRANSFER_MODEL" == "WIKIFR" ]; then
    echo "Selected base model : WIKI FR"
    echo "WIKI FR unavailable" # Initial pretrained model (FR on WIKI)
    exit
else
    echo "TRANSFER_MODEL have to be set ('WIKIEN' / 'WIKIFR' / 'AIDACONLL')"
    exit
fi
LOGSTDOUT="$WORK/GENRE/transfer_exp_$LANG.stdout"

if [ "$LANG" == "FR" ] || [ "$LANG" == "EN" ]; then
    echo "Selected LANG $LANG"
else
    echo "LANG must be 'FR' or 'EN'"
    exit
fi

if [ "$DATASETS" == "" ]; then
    echo "DATASETS have to be set with at least one corpus"
    exit
fi

if [ -n "$PREPROCESS" ]; then
    echo "Use Preprocess_${PREPROCESS}"
    PREPROCESS="Preprocess_${PREPROCESS}"
else
    echo "Use default Preprocess : Preprocess_1"
    PREPROCESS="Preprocess_1"
fi


if [ "$MODE" == "" ]; then
    MODE="START"
    echo "Default mode : START"
elif [ "$MODE" == "START" ] || [ "$MODE" == "CONTINUE" ]; then
    echo "Selected mode : $MODE"
else
    echo "Mode must be 'START' or 'CONTINUE'"
    exit
fi

function restart { echo "Restarting required" >> $LOGSTDOUT ; scontrol requeue $SLURM_JOB_ID ; echo "Scheduled job" >> $LOGSTDOUT ; }

function ignore { echo "SIGTERM ignored" >> $LOG_STDOUT ; }
trap restart USR1
trap ignore TERM

date
echo "#### START RUN ####"
echo "name : EXP TRANSFERT $LANG ON : $DATASETS"
echo "Job ID : $SLURM_JOB_ID"
cd ..

args=()

for TYPE in $DATASETS
do
    EXPERIMENT_NAME="Transfert_${LANG}_${TYPE}"
    DATASET_F="$WORK/GENRE/$EXPERIMENT_NAME/${PREPROCESS}"
    DIR_OPTION="$DATASET_F $EXPERIMENT_NAME $MODEL_F $TRANSFER_SIGN" # Transfer
    args+=("${DIR_OPTION}")
    if [ "$TYPE" == "TR" ]; then
        args+=("$WORK/GENRE/Transfert_FR_TR/Preprocess_hard Transfert_FR_TR $MODEL_F $TRANSFER_SIGN") # TR hard
    fi
done

set -x

echo "${#args[@]}"
printf "%s\n" "${args[@]}"
if [ "${#args[@]}" == 0 ]; then
    echo "erreur de configuration des arguments"
    echo "$(DATASETS)"
    exit
fi

echo ">>>>>>>>>>> $MODE TRAINING <<<<<<<<<<<<<<<"
echo ">>>>>>>>>>> current : ${args[${SLURM_ARRAY_TASK_ID}]} <<<<<<<<<<<<<<<"
if [ "$MODE" == "START" ]; then
    $CURRENT_F/scripts_genre/train.sh ${args[${SLURM_ARRAY_TASK_ID}]}
else # MODE == "CONTINUE"
    $CURRENT_F/scripts_genre/train_continue.sh ${args[${SLURM_ARRAY_TASK_ID}]}
fi
    
date
echo "#### END ####"
