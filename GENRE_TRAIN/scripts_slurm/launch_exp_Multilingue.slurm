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

#SBATCH --job-name=ML_ALL
#SBATCH --signal=B:USR1@120

# Load Environnement
module load python/3.7.5
export PYTHONUSERBASE=$WORK/.local_GENRE
export PATH=$PYTHONUSERBASE/bin:$PATH

CURRENT_F=$(pwd) #current script folder : "GENRE/"
LANG=$1 # Lang : "FR" or "EN"
DATASETS=$2 # Datasets, exemple : "WIKI DB TR" / "WIKI AIDACONLL"
PREPROCESS=$3 # Type of preprocess : 1 (defaut) / 2 / hez
MODE=$4 # Begin training or continue existing training
MULTI_MODEL="$WORK/models/mbart.cc100"
MULTI_SIGN="m"
LOGSTDOUT="$WORK/GENRE/multi_exp_$LANG.stdout"

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
echo "name : EXP MULTILINGUAL BART FINE TUNE FOR $LANG ON : $DATASETS"
echo "Job ID : $SLURM_JOB_ID"
cd ..

args=()

for TYPE in $DATASETS
do
    EXPERIMENT_NAME="Transfert_${LANG}_${TYPE}"
    DATASET_F="$WORK/GENRE/$EXPERIMENT_NAME/${PREPROCESS}"
    DIR_OPTION="$DATASET_F $EXPERIMENT_NAME $MULTI_MODEL $MULTI_SIGN" # Baseline
    args+=("${DIR_OPTION}")
    if [ "$TYPE" == "TR" ]; then
        args+=("$WORK/GENRE/Transfert_FR_TR/Preprocess_hard Transfert_FR_TR $MULTI_MODEL $MULTI_SIGN") # TR hard
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
    scripts_genre/train.sh ${args[${SLURM_ARRAY_TASK_ID}]}
else # MODE == "CONTINUE"
    scripts_genre/train_continue.sh ${args[${SLURM_ARRAY_TASK_ID}]}
fi
    
date
echo "#### END ####"
