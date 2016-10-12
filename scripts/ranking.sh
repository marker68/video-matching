#!/usr/bin/env bash

if [ "$#" -ne 6 ] ; then
    echo "Usage: $0 model_path model_type lm_model test_data refine_size" >&2
    echo "we only have $# params"
    exit
fi

echo "TRECVID 2016: Video-to-Text matching and ranking"

MODEL=$1
MODEL_TYPE=$2
LM_MODEL=$3
TEST_DATA=$4
REFINE=$5
INPUT_DIM=$6
DATE=$(date +'%Y-%m-%d_%H-%M-%S')
OUTPUT="data/$MODEL_TYPE-$REFINE-$DATE.np"

YAML="
video_data: $TEST_DATA\n
language_model: $LM_MODEL\n
v_dim: 300\n
video_model: $MODEL\n
gt: False\n
output_ranks: $OUTPUT\n
n_captions: 1\n
pool_type: concat\n
compute_d2v: False\n
refine: $REFINE\n
model_name: $MODEL_TYPE
in_dim: $INPUT_DIM"

echo -e "$YAML" > log/test-$DATE.yaml

python keras/test.py log/test-$DATE.yaml 2>&1 | tee log/test-$DATE.log
