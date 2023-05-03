#!/bin/bash
# Extract GENRE predictions and transform to elevant format
# Move the prediction to the Elevant folder on the proper model folder

# exemple command : 

# TEST ANGLAIS : ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/ AIDA Trans --long GENRE_EN_Test

# FR (eval DB) : ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ DB_dev Base --long GENRE_FR_BASELINE

# FR (eval TR) : ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ TR_dev Base --long GENRE_FR_BASELINE

# FR (multi DB) : ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_3/ DB_dev Multi --long GENRE_FR_MULTIL

#SBATCH --qos=qos_gpu-t3
#SBATCH --cpus-per-task=25
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH -A gcp@a100

#SBATCH --time=15:00:00

#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --job-name=EVAL_GENRE
#SBATCH --signal=B:USR1@120

# Load Environnement
module load cpuarch/amd # Config GPU A100
module load pytorch-gpu/py3/1.12.1 # Config GPU A100
# module load python/3.7.5 # Config GPU V100
export PYTHONUSERBASE=$WORK/.local_GENRE_AMD
export PATH=$PYTHONUSERBASE/bin:$PATH

LANG=$1          # FR or EN
MODEL_PATH=$2    # 'yago' / 'wiki-abs' / custom path (where there is the folder "models" or "models_[m/t]")
DATASET=$3       # AIDA / DB_[train/dev/test] / TR_[train/dev/test]
# if DATASET is specific (for french only), give the path with the file. exemple : DATASET="./DB_split/DB_dev-1.jsonl"
MODE=$4          # 'Base' / 'Multi' / 'Trans' / 'MTrans'

if [ "$5" == "" ] || [ "$5" == "--iter" ]; then
    PREDICT_STRAT="--split_iter"
elif [ "$5" == "--long" ]; then
    PREDICT_STRAT="--split_long"
elif [ "$5" == "--para" ]; then
    PREDICT_STRAT=""
else
    echo "PREDICT_STRAT must be '--iter' / '--long' or 'para'"
    exit
fi

if [ "$6" == "" ] || [ "$6" == "fairseq" ]; then
    echo "Fairseq Architecture choosen"
    ARCHITECTURE="fairseq"
    ARCH_NAME="fairseq"
elif [ "$6" == "huggingface" ] || [ "$6" == "HF" ]; then
    echo "HuggingFace Architecture choosen"
    ARCHITECTURE="huggingface"
    ARCH_NAME="hf"
else
    echo "Architecture must be 'fairseq' or 'huggingface'"
fi

if [ "$7" == "" ]; then
    EXPERIMENT="predictions"
else
    EXPERIMENT=$7 # name of output file in elevant folder
fi

PRED_OUTPUT=$8

if [ "$MODE" == "Base" ]; then
    LOCAL_ARCHIVE="$WORK/models" # default GPT encoder local position
    MODEL_NAME="models"
    GENRE_TYPE="--genre"
    MOD_NAME="BART"
elif [ "$MODE" == "Trans" ]; then
    LOCAL_ARCHIVE="$WORK/models" # default GPT encoder local position
    MODEL_NAME="models_t"
    GENRE_TYPE="--genre"
elif [ "$MODE" == "Multi" ]; then
    LOCAL_ARCHIVE="$WORK/models" # default mBART encoder local position
    MODEL_NAME="models_m"
    GENRE_TYPE="--mgenre"
    MOD_NAME="MBART"
elif [ "$MODE" == "MTrans" ]; then
    LOCAL_ARCHIVE="$WORK/models" # default mBART encoder local position
    MODEL_NAME="models_t"
    GENRE_TYPE="--mgenre"
elif [ "$MODE" == "Barthez" ]; then
    LOCAL_ARCHIVE="$WORK/models"
    MODEL_NAME="models_hez"
    GENRE_TYPE="--barthez"
    MOD_NAME="BARThez"
elif [ "$MODE" == "MBarthez" ]; then
    LOCAL_ARCHIVE="$WORK/models"
    MODEL_NAME="models_m"
    GENRE_TYPE="--mbarthez"
    MOD_NAME="MBARThez"
else
    echo "MODE must be 'Base' / '[M]Trans' / '[M]Barthez' or 'Multi'"
    exit
fi

if [ "$ARCHITECTURE" == "huggingface" ]; then
    MODEL_NAME="HF" #change model name to HF ==> where the HF model is store
fi

if [ "$MODEL_PATH" == "yago" ] || [ "$MODEL_PATH" == "wiki-abs" ]; then
    MODEL=$MODEL_PATH
else
    MODEL=${MODEL_PATH}${MODEL_NAME}  
fi

if [ "$LANG" == "FR" ]; then
    SPACY="$WORK/elevant/nlp_model/fr_core_news_sm/fr_core_news_sm-3.4.0/" # FR Spacy Model
elif [ "$LANG" == "EN" ]; then
    SPACY="$WORK/elevant/nlp_model/en_core_web_sm/en_core_web_sm-3.4.0/" # EN Spacy Model
else
    echo "$LANG not recognized"
    exit
fi

# OPTION
ELEVANT_FOLDER="$WORK/elevant/evaluation-results/$EXPERIMENT"
KOLITSAS_DATA="$WORK/end2end/data/basic_data"
PEM_FILE="crosswikis_wikidump${LANG}_p_e_m.txt"
RESULTS="./extract_results"
if [ "$PRED_OUTPUT" == "" ]; then
    PRED_OUTPUT=$RESULTS
fi
mkdir -p $ELEVANT_FOLDER
cd .. #$WORK/Reforged_GENRE

echo "make prediction for the $LANG model : $EXPERIMENT ($MODEL)"
date

if [ "$LANG" == "fr" ] || [ "$LANG" == "FR" ]; then
    LOWER_LANG="fr"
    MENTION_TRIE="mention_trie_${LOWER_LANG}_${MOD_NAME}_${ARCH_NAME}.pkl"
    CANDIDATES_DICT="mention_to_candidates_dict_${LOWER_LANG}.pkl"
elif [ "$LANG" == "en" ] || [ "$LANG" == "EN" ]; then
    LOWER_LANG="en"
    MENTION_TRIE="mention_trie_${LOWER_LANG}_${MOD_NAME}_${ARCH_NAME}.pkl"
    CANDIDATES_DICT="mention_to_candidates_dict_${LOWER_LANG}.pkl"
else
    echo "LANG must be 'FR' or 'EN'"
    exit
fi

if [ ! -f data/${CANDIDATES_DICT} ]; then
    if [ "$LOWER_LANG" == "fr" ]; then
        echo "extract DB & TR to mention candidates tsv file"
        python To_TSV_file.py --in_dir="$WORK/DB/fr" --out_file="data/DB_means.tsv" --format="TR" -v
        python To_TSV_file.py --in_dir="$WORK/TR/fr" --out_file="data/TR_means.tsv" --format="TR" -v
        python To_TSV_file.py --in_dir="$WORK/WIKI/fr" --out_file="data/WIKI-FR_means.tsv" --format="KILT" -v
    else #LOWER_LANG == "en"
        python To_TSV_file.py --in_dir="$WORK/AIDACONLL/EL" --out_file="data/AIDA_means.tsv" --format="KILT" -v
        python To_TSV_file.py --in_dir="$WORK/WIKI/en" --out_file="data/WIKI-EN_means.tsv" --format="KILT" -v
    fi
    python create_candidates_dict.py --entity_language=${LOWER_LANG} --prob_path="$PEM_FILE" -v
else
    echo "found ${CANDIDATES_DICT}"
fi
if [ ! -f data/${MENTION_TRIE} ]; then
    echo "create candidates dict & mention trie for ${LANG} ${MOD_NAME} ${ARCH_NAME}"
    mkdir -p data/dalab
    cp ${KOLITSAS_DATA}/${PEM_FILE} data/dalab
    python create_mentions_trie.py --model_path=$MODEL --local_archive $LOCAL_ARCHIVE --entity_language=${LOWER_LANG} --model_type=${MOD_NAME} --architecture=${ARCH_NAME} -v
else
    echo "found ${MENTION_TRIE}"
fi
#   /////////////////////////////////////////////////////////// START INFERENCE /////////////////////////////////////////////////////////////////////////////////////
if [ -a $DATASET ]; then #adapted prediction for long corpus (only prediction, no transform)
    echo "prediction for $DATASET IN $PRED_OUTPUT"
    python main.py --yago=$MODEL ${PREDICT_STRAT} ${GENRE_TYPE} \
        -i ${DATASET} -o ${PRED_OUTPUT}/${EXPERIMENT}.jsonl \
        --mention_trie data/${MENTION_TRIE} \
        --mention_to_candidates_dict data/${CANDIDATES_DICT} \
        --local_archive $LOCAL_ARCHIVE --spacy=$SPACY\
        --architecture=$ARCHITECTURE
    ls -l ${PRED_OUTPUT}
else #classic prediction + transform prediction
    echo "prediction for $DATASET"
    python main.py --yago=$MODEL ${PREDICT_STRAT} ${GENRE_TYPE} \
        -i ./data/benchmarks/$DATASET.jsonl -o ${PRED_OUTPUT}/$DATASET_${EXPERIMENT}.jsonl \
        --mention_trie data/${MENTION_TRIE} \
        --mention_to_candidates_dict data/${CANDIDATES_DICT} \
        --local_archive $LOCAL_ARCHIVE --spacy=$SPACY\
        --architecture=$ARCHITECTURE
    echo "transform predictions"
    python transform_predictions.py ${PRED_OUTPUT}/$DATASET_${EXPERIMENT}.jsonl -o ${PRED_OUTPUT}/$DATASET_${EXPERIMENT}.qids.jsonl -l "fr"
    echo "move generated predictions to $ELEVANT_FOLDER"
    mv ${PRED_OUTPUT}/$DATASET_${EXPERIMENT}.qids.jsonl ${ELEVANT_FOLDER}/${EXPERIMENT}.$DATASET.linked_articles.jsonl
    if [ "$DATASET" == "AIDA" ]; then
        echo "WARNING: this is AIDA dataset. the name for elevant is aida-conll. Don't forget to change"
    fi
fi

echo "FIN"
date
