#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

PATH="$STORE/wikipedia_data"
FOLDER="20211120"

mkdir $PATH
echo $(pwd)
echo $PATH

#for LANG in af am ar as az be bg bm bn br bs ca cs cy da de el en eo es et eu fa ff fi fr fy ga gd gl gn gu ha he hi hr ht hu hy id ig is it ja jv ka kg kk km kn ko ku ky la lg ln lo lt lv mg mk ml mn mr ms my ne nl no om or pa pl ps pt qu ro ru sa sd si sk sl so sq sr ss su sv sw ta te th ti tl tn tr uk ur uz vi wo xh yo zh
for LANG in en fr
do
    wget -P ${PATH} http://wikipedia.c3sl.ufpr.br/${LANG}wiki/${FOLDER}/${LANG}wiki-${FOLDER}-pages-articles-multistream.xml.bz2
done

#for LANG in af am ar as az be bg bm bn br bs ca cs cy da de el en eo es et eu fa ff fi fr fy ga gd gl gn gu ha he hi hr ht hu hy id ig is it ja jv ka kg kk km kn ko ku ky la lg ln lo lt lv mg mk ml mn mr ms my ne nl no om or pa pl ps pt qu ro ru sa sd si sk sl so sq sr ss su sv sw ta te th ti tl tn tr uk ur uz vi wo xh yo zh
for LANG in en fr
do
    wikiextractor ${PATH}/${LANG}wikinews-${FOLDER}-pages-articles-multistream.xml.bz2 -o ${LANG} --links --lists --sections
done
