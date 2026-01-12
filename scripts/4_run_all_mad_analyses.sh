#! /bin/bash

model_names=("qwen2.5-14b" "qwen2.5-32b" "qwen2.5-72b")

for model in "${model_names[@]}"; do
    ./scripts/mad_analysis.sh $model
done