#! /bin/bash
model_name=$1
datasets=("gsm8k" "math" "mmlu_pro")
seeds=(41 42 43 44 45)
gpus=0,1
save_path=$2

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        python chain_of_thoughts.py --model_name $model_name --dataset $dataset --seed $seed --gpu_id $gpus --parallel --save_path $save_path
    done
done

python chain_of_thoughts.py --model_name $model_name --dataset "aime24" --seed 42 --gpu_id $gpus --parallel --save_path $save_path

python chain_of_thoughts.py --model_name $model_name --dataset "aime25" --seed 42 --gpu_id $gpus --parallel --save_path $save_path