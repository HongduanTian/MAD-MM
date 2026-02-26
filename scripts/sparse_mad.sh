#! /bin/bash
model_names=("qwen2.5-7b" "qwen2.5-math-7b" "deepseek-math-7b")
datasets=("gsm8k" "math" "mmlu_pro")
seeds=(41 42 43 44 45)
gpus=0,1
save_path="new_results"

for model in "${model_names[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            python sparse_mad.py --model_name $model --dataset $dataset --seed $seed --gpu_id $gpus --parallel --save_path $save_path --exp_name "sparse_mad"
        done
    done
    python sparse_mad.py --model_name $model --dataset "aime24" --seed 42 --gpu_id $gpus --parallel --save_path $save_path --exp_name "sparse_mad"

    python sparse_mad.py --model_name $model --dataset "aime25" --seed 42 --gpu_id $gpus --parallel --save_path $save_path --exp_name "sparse_mad"
done