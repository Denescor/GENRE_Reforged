#!/bin/bash
# script bash to preprocess WikiFR and TRFR2016 for GENRE and launch training

# ARGUMENTS
#################################################################################################################
DOWNLOAD=$1 #set to "Yes" to download data, therefore use already downloaded data
VERBOSE=$2 #set to "v" to use verbose mode to all scripts
EXPERIMENT_NAME=$3 #experiment name if training (require $4)
#################################################################################################################

# Jean Zay Environement
#$STORE="/gpfsstore/rech/emf/ujs45li"
#$HOME="/linkhome/rech/genrqo01/ujs45li"
#$WORK="/gpfswork/rech/emf/ujs45li"

# GLOBAL VARIABLES : FOLDERS
#################################################################################################################
CURRENT_F=$(pwd) #current script folder : "GENRE/"
SCRIPT_F="$CURRENT_F/scripts_mgenre" #folder where are the scripts
TR_F="$WORK/TR" #folder where are the TR data
DATASET_F="$WORK/GENRE/Transfert_FR_GENRE" #folder where are the fairseq's preprocessed datasets
DATA_TYPE="WIKI"
MODEL_F="$CURRENT_F/$4" #folder where is the pretrained model
TOKEN_T="bpe" # "bpe" or "spm"
TOKEN_F="$SCRIPT_F/fairseq_multilingual_entity_disambiguation" #folder where is the tokeniser model for the preprocessed_fairseq (SentencePiece - spm or BinaryPairEncoder - bpe)
WIKIDATA_F="$STORE/wikidata_dump" #folder where is the file "wikidata-all.json" (1To)
WIKIPEDIA_F="$STORE/wikipedia" #folder where are the wikipedia data (and where the generated wikidata files will move)
WIKI_DUMP="20211120"
REDIRECT_WIKI="${WORK}/target"
MBART_PATH="${WORK}/models/mbart.cc25"
#################################################################################################################

# APPLY ARGUMENTS

echo "scripts files : $SCRIPT_F"

if [ "$DOWNLOAD" == "YES" ]; then
    echo "download wikipedia"
    cd $SCRIPT_F
    if [ -f "${WIKIPEDIA_F}/frwiki-${WIKI_DUMP}-pages-articles-multistream.xml.bz2" ]; then
    	echo "\t wikipedia already downloaded"	
    else
	for LANG in en fr
    	do
        	wget -P ${WIKIPEDIA_F} http://wikipedia.c3sl.ufpr.br/${LANG}wiki/${WIKI_DUMP}/${LANG}wiki-${WIKI_DUMP}-pages-articles-multistream.xml.bz2
    	done
    fi
    for LANG in fr en
    do
        ~/.local/bin/wikiextractor ${WIKIPEDIA_F}/${LANG}wiki-${WIKI_DUMP}-pages-articles-multistream.xml.bz2 -o ${WIKIPEDIA_F}/${LANG} -l #--lists --s
    done
    mv "wikipedia" "$WIKIPEDIA_F"
    echo "download wikidata json"
    if [ -f "$WIKIDATA_F/latest-all.json.gz" ]; then
        echo "\t latest-all.json.gz already exist"
    else
        wget -P "$WIKIDATA_F" "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz"
    fi
    echo "decompress wikidata json"
    if [ -f "$WIKIDATA_F/wikidata-all.json" ]; then
        echo "\t wikidata-all.json already exist"
    else
        cd "$WIKIDATA_F"
        gzip -d "./latest-all.json.gz"
        mv "./latest-all.json" "./wikidata-all.json"
    fi
    cd "$CURRENT_F"    
    echo "all download done"
fi

if [ "$VERBOSE" == "v" ] || [ "$VERBOSE" == "-v" ]; then
    VERBOSE="-v" #"-v" is verbose mode in all scripts
    echo "verbose mode activated"
else
    VERBOSE="" #empty argument is no argument
    echo "verbose mode desactivated"
fi

# move to the script folder
cd $CURRENT_F
echo $(pwd)

#preprocess wikidata
# python -m scripts_mgenre.preprocess_wikidata "compress" --base_wikidata="$WIKIDATA_F" $VERBOSE
# python -m scripts_mgenre.preprocess_wikidata "normalize" --base_wikidata="$WIKIDATA_F" --mgenre_path="$MBART_PATH" $VERBOSE
# python -m scripts_mgenre.preprocess_wikidata "dicts" --base_wikidata="$WIKIDATA_F" $VERBOSE
# python -m scripts_mgenre.preprocess_wikidata "dicts" --normalized --base_wikidata="$WIKIDATA_F" $VERBOSE
# python -m scripts_mgenre.preprocess_wikidata "redirects" --base_wikidata="$WIKIDATA_F" --redirect_path="$REDIRECT_WIKI" $VERBOSE
echo "compress & create dicts done"
# mv "${WIKIDATA_F}/*.pkl" $WIKIPEDIA_F
# mv "$WIKIDATA_F/wikidata-all-compressed.jsonl" $WIKIPEDIA_F
# WIKIDATA_F=$WIKIPEDIA_F ## now all the generated files are in the same folder than the wikipedia files
echo "moving wikidata files done"

#preprocess wikipedia
# python -m scripts_mgenre.preprocess_extract --base_wikipedia="$WIKIPEDIA_F" --lang="fr" --extract_mode="nel" $VERBOSE
# python -m scripts_mgenre.preprocess_anchors "prepare" --base_wikipedia="$WIKIPEDIA_F" --base_wikidata="$WIKIDATA_F" --langs="fr" $VERBOSE
echo "process wikipedia done"
# python -m scripts_mgenre.preprocess_anchors "solve" --base_wikipedia="$WIKIPEDIA_F" --base_wikidata="$WIKIDATA_F" --langs="fr" $VERBOSE
# python -m scripts_mgenre.preprocess_anchors "fill" --base_wikipedia="$WIKIPEDIA_F" --base_wikidata="$WIKIDATA_F" --langs="fr" $VERBOSE 

# python -m scripts_mgenre.preprocess_tries "" --base_wikidata="$WIKIDATA_F" --allowed_langs="fr" $VERBOSE


#preprocess TR
mkdir -p $DATASET_F/$DATA_TYPE
python -m scripts_mgenre.preprocess_TRFR2016 --input_dir="$TR_F" --output_dir="$DATASET_F/$DATA_TYPE" --base_wikidata="$WIKIDATA_F" $VERBOSE
python -m scripts_genre.split_kilt_to_train_dev "$WIKIPEDIA_F/fr/frwiki.pkl" "$WIKIPEDIA_F" --proportion_dev=0.15 --format="pkl" $VERBOSE
mkdir -p $WIKIPEDIA_F/train/fr
mkdir -p $WIKIPEDIA_F/dev/fr
cp $WIKIPEDIA_F/wiki-train-kilt.pkl $WIKIPEDIA_F/train/fr/frwiki.pkl
cp $WIKIPEDIA_F/wiki-dev-kilt.pkl $WIKIPEDIA_F/dev/fr/frwiki.pkl

python -m scripts_mgenre.preprocess_mgenre "lang_titles" --base_wikidata="$WIKIDATA_F" --base_wikipedia="$WIKIPEDIA_F/train" --base_tr2016="$DATASET_F/$DATA_TYPE" --langs="fr" --monolingual $VERBOSE
python -m scripts_mgenre.preprocess_mgenre "lang_titles" --base_wikidata="$WIKIDATA_F" --base_wikipedia="$WIKIPEDIA_F/dev" --base_tr2016="$DATASET_F/$DATA_TYPE" --langs="fr" --monolingual $VERBOSE
echo "process $DATA_TYPE done"
mv $WIKIPEDIA_F/train/*.source "$DATASET_F/$DATA_TYPE/train.source" ## move from wikipedia folder to the datasets folder. Easier to the next part of the preprocess
mv $WIKIPEDIA_F/train/*.target "$DATASET_F/$DATA_TYPE/train.target" ## move from wikipedia folder to the datasets folder. Easier to the next part of the preprocess
mv $WIKIPEDIA_F/dev/*.source "$DATASET_F/$DATA_TYPE/dev.source" ## move from wikipedia folder to the datasets folder. Easier to the next part of the preprocess
mv $WIKIPEDIA_F/dev/*.target "$DATASET_F/$DATA_TYPE/dev.target" ## move from wikipedia folder to the datasets folder. Easier to the next part of the preprocess
echo "all files are ready to fairseq"

# preprocess for fairseq format
# with
#   - dataset folder where are the .source and .target files
#   - the Tokeniser's model folder for encoding the .source and .target data
#       - choice between the SentencePiece tokeniser or BPE Tokeniser from RoBERTa. 
#       - for Transfert Learning, requirement is to choice the same as the pretrained model
#   - no reuse dict because we assume that the data are new
# ${CURRENT_F}/scripts_genre/preprocess_fairseq.sh "$DATASET_F/$DATA_TYPE" "$TOKEN_T" "$TOKEN_F/" ""
echo "process fairseq is done. Training model is ready"

### Ready to train or fine-tune on WikiFR ###

# Train with new data ==> fine tuning pretrained model
# with
#   - Wiki dataset for the fine-tuning
#   - The name of the experiment
#   - The pretrained model's folder
if [ -n "$4" ]; then
    ${CURRENT_F}/scripts_mgenre/train.sh "$DATASET_F/$DATA_TYPE" "$EXPERIMENT_NAME" "$MODEL_F/"
fi
