#! /bin/bash
model_name=$1
save_path="new_results"
exp_name="analysis_num_rounds"
datasets=("gsm8k" "math" "mmlu_pro")
seeds=(41 42 43 44 45)
gpus=0,1

num_agents=3
num_rounds_arr=(3 5 6 8 10)

for seed in "${seeds[@]}"; do
    for ds in "${datasets[@]}"; do
        for num_rounds in "${num_rounds_arr[@]}"; do
            python multi_agent_debate.py --model_name $model_name --dataset $ds --seed $seed --num_agents $num_agents --max_round $num_rounds --prune_strategy "naive" --gpu_id $gpus --parallel --exp_name $exp_name --save_path $save_path
            python multi_agent_debate.py --model_name $model_name --dataset $ds --seed $seed --num_agents $num_agents --max_round $num_rounds --prune_strategy "subjective" --gpu_id $gpus --parallel --exp_name $exp_name --save_path $save_path
            python multi_agent_debate.py --model_name $model_name --dataset $ds --seed $seed --num_agents $num_agents --max_round $num_rounds --prune_strategy "objective" --gpu_id $gpus --parallel --exp_name $exp_name --save_path $save_path
        done
    done
done

for num_rounds in "${num_rounds_arr[@]}"; do
    python multi_agent_debate.py --model_name $model_name --dataset "aime24" --seed 42 --num_agents $num_agents --max_round $num_rounds --prune_strategy "naive" --gpu_id $gpus --parallel --exp_name $exp_name --save_path $save_path
    python multi_agent_debate.py --model_name $model_name --dataset "aime24" --seed 42 --num_agents $num_agents --max_round $num_rounds --prune_strategy "subjective" --gpu_id $gpus --parallel --exp_name $exp_name --save_path $save_path
    python multi_agent_debate.py --model_name $model_name --dataset "aime24" --seed 42 --num_agents $num_agents --max_round $num_rounds --prune_strategy "objective" --gpu_id $gpus --parallel --exp_name $exp_name --save_path $save_path
done

for num_rounds in "${num_rounds_arr[@]}"; do
    python multi_agent_debate.py --model_name $model_name --dataset "aime25" --seed 42 --num_agents $num_agents --max_round $num_rounds --prune_strategy "naive" --gpu_id $gpus --parallel --exp_name $exp_name --save_path $save_path
    python multi_agent_debate.py --model_name $model_name --dataset "aime25" --seed 42 --num_agents $num_agents --max_round $num_rounds --prune_strategy "subjective" --gpu_id $gpus --parallel --exp_name $exp_name --save_path $save_path
    python multi_agent_debate.py --model_name $model_name --dataset "aime25" --seed 42 --num_agents $num_agents --max_round $num_rounds --prune_strategy "objective" --gpu_id $gpus --parallel --exp_name $exp_name --save_path $save_path
done