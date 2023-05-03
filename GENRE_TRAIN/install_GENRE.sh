#!/bin/bash
# script to install genre

PATH=$1 #path where install GENRE. Better if it's an absolute path. Could be ignore to install in the current folder

if [ -z "$PATH" ]; then
    PATH="." #Current folder
fi

echo "$PATH"

# copy git of GENRE
git clone https://github.com/facebookresearch/GENRE.git $PATH # into the choose folder
cd $PATH/GENRE
cp -r "./genre" "./mgenre" #duplicate the codes for the "mgenre_scripts"

# install requirement
pip install --user --no-cache-dir -r requirements.txt

# install fairseq
git clone https://github.com/nicola-decao/fairseq.git # into the folder GENRE
cd fairseq
pip install --user --no-cache-dir ./

# install kilt
cd .. # folder GENRE
pip install --user --no-cache-dir -e git+https://github.com/facebookresearch/DPR.git#egg=DPR # into the folder GENRE
