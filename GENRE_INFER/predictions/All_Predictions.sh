#!/bin/bash

LANG=$1
MODEL=$2
CORPUS=$3

ARCHITECTURE="fairseq" #fairseq #HF
echo "Choosen Architecture : $ARCHITECTURE"

### DB dev : 5 parts ==> 5 runs
### DB test : 3 parts ==> 3 runs
### DB train : ~30 parts ==> ? runs
## TR dev : ? parts ==> ? parts
## TR train : ? parts ==> ? parts
## TR test : ? parts ==> ? parts

if grep -q "FR" <<< "$LANG"; then
    if grep -q " BART " <<< "$MODEL"; then
        if grep -q "WIKI" <<< "$CORPUS"; then
            echo "FR BART FT with WIKI (eval with WIKI-MINI)"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ WIKI-dev_mini Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKI DO_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ WIKI-train_mini Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKI DO_SPLIT
        fi
        if grep -q "DB" <<< "$CORPUS"; then
            echo "FR BART FT with WIKI (eval with DB)"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ DB_dev Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKI DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ DB_test Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKI DONT_SPLIT     
        fi
        if grep -q "TR" <<< "$CORPUS"; then
            echo "FR BART FT with WIKI (eval with TR)"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ TR_dev Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKI DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ TR_test Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKI DONT_SPLIT
        fi
        if grep -q "MINID" <<< "$CORPUS"; then
            echo "FR BART FT with DBMINI (eval with DBMINI)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ DBMINI-train Base --long $ARCHITECTURE GENRE_FR_BASELINE_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ DBMINI-dev Base --long $ARCHITECTURE GENRE_FR_BASELINE_DBMINI
	    sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ DBMINI-test Base --long $ARCHITECTURE GENRE_FR_BASELINE_DBMINI
            echo "FR BART FT with DBMINIPOOR (eval with DBMINIPOOR)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ DBMINIPOOR-train Base --long $ARCHITECTURE GENRE_FR_BASELINE_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ DBMINIPOOR-dev Base --long $ARCHITECTURE GENRE_FR_BASELINE_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ DBMINIPOOR-test Base --long $ARCHITECTURE GENRE_FR_BASELINE_DBMINI
        fi
        if grep -q "MINIW" <<< "$CORPUS"; then
            echo "FR BART FT with WIKIMINI (eval with WIKIMINI)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINI/Preprocess_2/ WIKIMINI-train Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKIMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINI/Preprocess_2/ WIKIMINI-dev Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKIMINI
            echo "FR BART FT with WIKIMINIPOOR (eval with WIKIMINIPOOR)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_2/ WIKIMINIPOOR-train Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKIMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_2/ WIKIMINIPOOR-dev Base --long $ARCHITECTURE GENRE_FR_BASELINE_WIKIMINI
        fi
        if grep -q "DEBUG" <<< "$CORPUS"; then
            echo "FR BART FT with WIKI (eval with DEBUG corpus)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_2/ Debug Base --iter $ARCHITECTURE Debug
        fi
    fi
    if grep -q " MBART " <<< "$MODEL"; then
        if grep -q "WIKI" <<< "$CORPUS"; then
            echo "FR M-BART FT with WIKI"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_6/ WIKI-dev_mini Multi --long $ARCHITECTURE GENRE_FR_MULTIL DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_6/ WIKI-train_mini Multi --long $ARCHITECTURE GENRE_FR_MULTIL DONT_SPLIT
        fi
        if grep -q "DB" <<< "$CORPUS"; then
            echo "FR M-BART FT with DB"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_6/ DB_dev Multi --long $ARCHITECTURE GENRE_FR_MULTIL DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_6/ DB_test Multi --long $ARCHITECTURE GENRE_FR_MULTIL DONT_SPLIT        
        fi      
        if grep -q "TR" <<< "$CORPUS"; then
            echo "FR M-BART FT with TR"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_6/ TR_dev Multi --long $ARCHITECTURE GENRE_FR_MULTIL DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_6/ TR_test Multi --long $ARCHITECTURE GENRE_FR_MULTIL DONT_SPLIT
        fi 
        if grep -q "MINID" <<< "$CORPUS"; then
            echo "FR MBART FT with DBMINI (eval with DBMINI)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_3/ DBMINI-train Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_3/ DBMINI-dev Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_3/ DBMINI-test Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_DBMINI
            echo "FR MBART FT with DBMINIPOOR (eval with DBMINIPOOR)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_3/ DBMINIPOOR-train Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_3/ DBMINIPOOR-dev Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_3/ DBMINIPOOR-test Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_DBMINI
        fi
        if grep -q "MINIW" <<< "$CORPUS"; then
            echo "FR MBART FT with WIKIMINI (eval with WIKIMINI)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINI/Preprocess_3/ WIKIMINI-train Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_WIKIMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINI/Preprocess_3/ WIKIMINI-dev Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_WIKIMINI
            echo "FR MBART FT with WIKIMINIPOOR (eval with WIKIMINIPOOR)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_3/ WIKIMINIPOOR-train Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_WIKIMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_3/ WIKIMINIPOOR-dev Multi --iter $ARCHITECTURE GENRE_FR_MULTIL_WIKIMINI
        fi
    fi    
    if grep -q " BARThez " <<< "$MODEL"; then
        if grep -q "WIKI" <<< "$CORPUS"; then
            echo "FR BARThez FT with WIKI"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_4/ WIKI-dev_mini Base --long $ARCHITECTURE GENRE_FR_BARTHEZ_WIKI DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_4/ WIKI-train_mini Base --long $ARCHITECTURE GENRE_FR_BARTHEZ_WIKI DONT_SPLIT
        fi
        if grep -q "DB" <<< "$CORPUS"; then
            echo "FR BARThez FT with WIKI"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_4/ DB_dev Barthez --long $ARCHITECTURE GENRE_FR_BARTHEZ_WIKI DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_4/ DB_test Barthez --long $ARCHITECTURE GENRE_FR_BARTHEZ_WIKI DONT_SPLIT
        fi
        if grep -q "TR" <<< "$CORPUS"; then
            echo "FR BARThez FT with WIKI"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_4/ DB_dev Barthez --long $ARCHITECTURE GENRE_FR_BARTHEZ_WIKI DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_4/ DB_test Barthez --long $ARCHITECTURE GENRE_FR_BARTHEZ_WIKI DONT_SPLIT
        fi    
    fi
    if grep -q " MBARThez " <<< "$MODEL"; then
        if grep -q "WIKI" <<< "$CORPUS"; then
            echo "FR MBARThez FT with WIKI"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_5/ WIKI-dev_mini Base --long $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKI DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_5/ WIKI-train_mini Base --long $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKI DONT_SPLIT
        fi
        if grep -q "DB" <<< "$CORPUS"; then
            echo "FR MBARThez FT with WIKI"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_5/ DB_dev Barthez --long $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKI DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_5/ DB_test Barthez --long $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKI DONT_SPLIT
        fi
        if grep -q "TR" <<< "$CORPUS"; then
            echo "FR MBARThez FT with WIKI"
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_5/ DB_dev Barthez --long $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKI DONT_SPLIT
            ./predictions_long_corpus_genre.sh FR ../GENRE/Transfert_FR_WIKI/Preprocess_5/ DB_test Barthez --long $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKI DONT_SPLIT
        fi
        if grep -q "MINID" <<< "$CORPUS"; then
            echo "FR MBARThez FT with DBMINI (eval with DBMINI)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_5/ DBMINI-train MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_5/ DBMINI-dev MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_5/ DBMINI-test MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_DBMINI
            echo "FR MBARThez FT with DBMINIPOOR (eval with DBMINIPOOR)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_5/ DBMINIPOOR-train MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_5/ DBMINIPOOR-dev MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_DBMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_5/ DBMINIPOOR-test MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_DBMINI
        fi
        if grep -q "MINIW" <<< "$CORPUS"; then
            echo "FR MBARThez FT with WIKIMINI (eval with WIKIMINI)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINI/Preprocess_5/ WIKIMINI-train MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKIMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINI/Preprocess_5/ WIKIMINI-dev MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKIMINI
            echo "FR MBARThez FT with WIKIMINIPOOR (eval with WIKIMINIPOOR)"
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_5/ WIKIMINIPOOR-train MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKIMINI
            sbatch ./predictions_genre.sh FR ../GENRE/Transfert_FR_WIKIMINIPOOR/Preprocess_5/ WIKIMINIPOOR-dev MBarthez --iter $ARCHITECTURE GENRE_FR_MBARTHEZ_WIKIMINI
        fi	
    fi
fi
if grep -q "EN" <<< "$LANG"; then
    if grep -q " BART " <<< "$MODEL"; then
        if grep -q "AIDA" <<< "$CORPUS"; then
            echo "EN BART FT with AIDA"
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/ AIDA_dev Base --iter $ARCHITECTURE GENRE_EN_wFT_P2
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/ AIDA_test Base --iter $ARCHITECTURE GENRE_EN_wFT_P2
        fi
        if grep -q "ACE2004" <<< "$CORPUS"; then
            echo "EN BART ACE2004"
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_base/models_lr5/ ace2004 Base --long $ARCHITECTURE GENRE_EN_wFT_P2
            sbatch ./predictions_genre.sh EN ../GENRE/Init_EN_AIDA/fairseq_e2e_entity_linking_aidayago/ ace2004 Base --long $ARCHITECTURE genre.yago
        fi
        if grep -q "AQUAINT" <<< "$CORPUS"; then
            echo "EN BART AQUAINT"
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_base/models_lr5/ aquaint Base --long $ARCHITECTURE GENRE_EN_wFT_P2
            sbatch ./predictions_genre.sh EN ../GENRE/Init_EN_AIDA/fairseq_e2e_entity_linking_aidayago/ aquaint Base --long $ARCHITECTURE genre.yago
        fi
        if grep -q "MSNBC" <<< "$CORPUS"; then
            echo "EN BART MSNBC"
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_base/models_lr5/ msnbc Base --iter $ARCHITECTURE GENRE_EN_wFT_P2
            sbatch ./predictions_genre.sh EN ../GENRE/Init_EN_AIDA/fairseq_e2e_entity_linking_aidayago/ msnbc Base --iter $ARCHITECTURE genre.yago
        fi
        if grep -q "KORE50" <<< "$CORPUS"; then
            echo "EN BART KORE50"
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_base/models_lr5/ kore50 Base --long $ARCHITECTURE GENRE_EN_wFT_P2
            sbatch ./predictions_genre.sh EN ../GENRE/Init_EN_AIDA/fairseq_e2e_entity_linking_aidayago/ kore50 Base --long $ARCHITECTURE genre.yago
        fi
        if grep -q "DB" <<< "$CORPUS"; then
            echo "EN BART DBSpotlight"
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_base/models_lr5/ spotlight Base --long $ARCHITECTURE GENRE_EN_wFT_P2
            sbatch ./predictions_genre.sh EN ../GENRE/Init_EN_AIDA/fairseq_e2e_entity_linking_aidayago/ spotlight Base --long $ARCHITECTURE genre.yago
        fi
        if grep -q "CLUEWEB" <<< "$CORPUS"; then
            echo "EN BART CLUEWEB"
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_base/models_lr5/ clueweb Base --long $ARCHITECTURE GENRE_EN_wFT_P2
            sbatch ./predictions_genre.sh EN ../GENRE/Init_EN_AIDA/fairseq_e2e_entity_linking_aidayago/ clueweb Base --long $ARCHITECTURE genre.yago
        fi
    fi
    if grep -q " MBART " <<< "$MODEL"; then
        if grep -q "AIDA" <<< "$CORPUS"; then
            echo "EN M-BART FT with AIDA"
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_3/ AIDA_dev Multi --iter $ARCHITECTURE mGENRE_EN
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_3/ AIDA_test Multi --iter $ARCHITECTURE mGENRE_EN
        fi 
    fi
    if grep -q " MBARThez " <<< "$MODEL"; then
        if grep -q "AIDA" <<< "$CORPUS"; then
            echo "EN M-BART FT with AIDA"
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_5/ AIDA_dev MBarthez --iter $ARCHITECTURE mGENREZ_EN
            sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_5/ AIDA_test MBarthez --iter $ARCHITECTURE mGENREZ_EN
        fi 
    fi
fi


### AIDA_dev : 1 part
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_base/models_lr4/ AIDA_dev Base --long GENRE_EN_AIDA-BASE-4
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_base/models_lr5/ AIDA_dev Base --long GENRE_EN_AIDA-BASE-5
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_from_aida/models_lr2/ AIDA_dev Trans --long GENRE_EN_AIDA-AIDA-2
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_from_wiki/models_lr2/ AIDA_dev Trans --long GENRE_EN_AIDA-WIKI-2
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_from_aida/models_lr4/ AIDA_dev Trans --long GENRE_EN_AIDA-AIDA-4
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_from_wiki/models_lr4/ AIDA_dev Trans --long GENRE_EN_AIDA-WIKI-4
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_from_aida/models_lr5/ AIDA_dev Trans --long GENRE_EN_AIDA-AIDA-5
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_from_wiki/models_lr5/ AIDA_dev Trans --long GENRE_EN_AIDA-WIKI-5
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_from_aida/models_lr6/ AIDA_dev Trans --long GENRE_EN_AIDA-AIDA-6
#sbatch ./predictions_genre.sh EN ../GENRE/Transfert_EN_AIDACONLL/Preprocess_2/models_from_aida/models_lr8/ AIDA_dev Trans --long GENRE_EN_AIDA-AIDA-8
