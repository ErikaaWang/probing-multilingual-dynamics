#!/bin/bash

BLOOM=$1
LAYER=$2
CHECKPOINT=$3
ROOT_DIR="$(pwd)"
DATA_DIR=${ROOT_DIR}/data/ud/ud-treebanks-v2.1

# loop over available languages: 
for CORPUS in $(cat scripts/languages_bloom.lst); do
echo "python preprocess_treebank.py $CORPUS --experiment-name inter-layer-$LAYER --treebanks-root $DATA_DIR --bloom $BLOOM --checkpoint $CHECKPOINT --inter-layer $LAYER --use-gpu"
python preprocess_treebank.py $CORPUS --experiment-name inter-layer-$LAYER --treebanks-root $DATA_DIR --bloom $BLOOM --checkpoint $CHECKPOINT --inter-layer $LAYER --use-gpu
done