#! /bin/bash
MODEL_NAME=$1
SAVE_PATH=$2
datasets=("gsm8k" "math" "mmlu_pro")
seeds=(41 42 43 44 45)
gpus=0,1

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        python multi_agent_debate.py --model_name $MODEL_NAME --dataset $dataset --seed $seed --num_agents 3 --max_round 2 --prune_strategy "mixed" --gpu_id $gpus --parallel --save_path $SAVE_PATH
    done
done

python multi_agent_debate.py --model_name $MODEL_NAME --dataset "aime24" --seed 42 --num_agents 3 --max_round 2 --prune_strategy "mixed" --gpu_id $gpus --parallel --save_path $SAVE_PATH

python multi_agent_debate.py --model_name $MODEL_NAME --dataset "aime25" --seed 42 --num_agents 3 --max_round 2 --prune_strategy "mixed" --gpu_id $gpus --parallel --save_path $SAVE_PATH