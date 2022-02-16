#!/bin/bash
fileid="1unN_IdEs_3I_Epv___NkTwKef-xo-l6G"
filename="weights-20220215Feb02.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip weights-20220215Feb02.zip 