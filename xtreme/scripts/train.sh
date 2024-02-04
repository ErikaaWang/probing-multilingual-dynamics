#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ERIKA: Notation for ADDING BLOOM:
# MODEL= bloom-560m, bloom-1b1, bloom-1b7
# ADDING BLOOM checkpoints:
# MODEL= bloom-560m-intermediate-global_stepxxxx, bloom-1b1-intermediate-global_stepxxxx, bloom-1b7-intermediate-global_stepxxxx

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
TASK=${2:-pawsx}
CACHE_IN_WORK_DIR=${3:-false}
GPU=${4:-0}
DATA_DIR=${5:-"$REPO/download/"}
OUT_DIR=${6:-"$REPO/outputs-temp/"}
echo "Fine-tuning $MODEL on $TASK using GPU $GPU"
echo "Load data from $DATA_DIR, and save models to $OUT_DIR"

if [ $TASK == 'pawsx' ]; then
  bash $REPO/scripts/train_pawsx.sh $MODEL $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'xnli' ]; then
  if [[ $MODEL == *"bloom"* ]]; then
    bash $REPO/scripts/train_xnli_bloom.sh $MODEL $GPU $DATA_DIR $OUT_DIR $CACHE_IN_WORK_DIR
  else
    bash $REPO/scripts/train_xnli.sh $MODEL $GPU $DATA_DIR $OUT_DIR
  fi
elif [ $TASK == 'xcopa' ]; then
  if [[ $MODEL == *"bloom"* ]]; then
    bash $REPO/scripts/train_xcopa_bloom.sh $MODEL $GPU $DATA_DIR $OUT_DIR $CACHE_IN_WORK_DIR
  else
    bash $REPO/scripts/train_xcopa.sh $MODEL $GPU $DATA_DIR $OUT_DIR
  fi
elif [ $TASK == 'udpos' ]; then
  bash $REPO/scripts/preprocess_udpos.sh $MODEL $DATA_DIR $CACHE_IN_WORK_DIR
  if [[ $MODEL == *"bloom"* ]]; then
    bash $REPO/scripts/train_udpos_bloom.sh $MODEL $GPU $DATA_DIR $OUT_DIR $CACHE_IN_WORK_DIR
  else
    bash $REPO/scripts/train_udpos.sh $MODEL $GPU $DATA_DIR $OUT_DIR
  fi
elif [ $TASK == 'panx' ]; then
  bash $REPO/scripts/preprocess_panx.sh $MODEL $DATA_DIR
  bash $REPO/scripts/train_panx.sh $MODEL $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'xquad' ]; then
  bash $REPO/scripts/train_qa.sh $MODEL squad $TASK $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'mlqa' ]; then
  bash $REPO/scripts/train_qa.sh $MODEL squad $TASK $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'tydiqa' ]; then
  bash $REPO/scripts/train_qa.sh $MODEL tydiqa $TASK $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'bucc2018' ]; then
  bash $REPO/scripts/run_bucc2018.sh $MODEL $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'tatoeba' ]; then
  bash $REPO/scripts/run_tatoeba.sh $MODEL $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'mewslix' ]; then
  bash $REPO/scripts/train_mewslix.sh $MODEL $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'lareqa' ]; then
  bash $REPO/scripts/train_lareqa.sh $MODEL $GPU $DATA_DIR $OUT_DIR
  bash $REPO/scripts/run_eval_lareqa.sh $MODEL $GPU $DATA_DIR $OUT_DIR
fi

