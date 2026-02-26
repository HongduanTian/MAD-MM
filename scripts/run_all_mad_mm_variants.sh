#! /bin/bash

model_names=("qwen2.5-14b" "qwen2.5-32b" "qwen2.5-72b")
save_path="new_results"

for model in "${model_names[@]}"; do
    ./scripts/mad_reasoning.sh $model $save_path
done