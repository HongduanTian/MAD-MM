<h1 align='center'> Multi-Agent Debate with Memory Masking</h1>

<p align='center'>
<a href=""><img src="https://img.shields.io/badge/arXiv-2602.00000-b31b1b.svg" alt="Paper"></a> <a href="https://iclr.cc/"><img src="https://img.shields.io/badge/Pub-ICLR'26-blue" alt="Conf"></a> <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Liscence"></a> <a href=""><img src="https://img.shields.io/badge/Slides%20-D76364" alt="Slides"></a> <a href=""><img src="https://img.shields.io/badge/Poster%20-Ffa500" alt="Poster"></a> <a href=""><img src="https://img.shields.io/badge/CN_Video%20-54b345" alt="CN_Video"></a> <a href=""><img src="https://img.shields.io/badge/EN_Video%20-54b345" alt="EN_Video"></a>

</p>

This repository contains the source codes for reproducing the results of ICLR'26 paper: [**Multi-Agent Debate with Memory Masking**]().

**Author List**: Hongduan Tian, Feng Liu, Zhanke Zhou, Tongliang Liu, Chengqi Zhang, Bo Han. 

## Introduction

<p align='center'>
<img src=./assets/mad_mm.png height=600 width=800/>
</p>

Large language models (LLMs) have recently demonstrated impressive capabilities in reasoning tasks. Currently, mainstream LLM reasoning frameworks predominantly focus on scaling up inference-time sampling to enhance performance. In particular, among all LLM reasoning frameworks, *multi-agent debate* (MAD), which employs multiple LLMs as agents to perform reasoning in the way of multi-round debate, has emerged as a powerful reasoning paradigm since it allows agents to access previous memories to alleviate fallacious content and refine their reasoning iteratively in each debate round. However, although MAD significantly improves the reasoning capabilities of LLMs, in this paper, we observe that there remain erroneous memories, and LLM agents are vulnerable to these erroneous memories. To explore this phenomenon, we provide a theoretical insight that the performance of MAD is highly dependent on the quality of memories derived from the previous debate, indicating that the existence of erroneous memories poses a threat to the performance of MAD. To address this problem, we introduce a simple yet effective multi-agent debate framework, *multi-agent debate with memory masking* (MAD-M$^2$), to improve the robustness of MAD by allowing LLM agents to mask erroneous memories from the previous debate round at the beginning of each debate round. In this way, MAD-M$^2$ can polish the contextual information before each debate round by preserving informative and meaningful memories while discarding the erroneous memories. Extensive experiments and analyses on mainstream mathematical and logical reasoning benchmarks demonstrate that MAD-M$^2$ can identify the erroneous memories and achieve better performance in reasoning than MAD. 

## Install Environments

The dependencies required to run MAD-M$^2$ includde:
```
datasets==3.1.0
latex2sympy2==1.9.1
numpy
PyYAML
regex==2024.11.6
sympy==1.13.1
torch
tqdm==4.67.0
vllm==0.6.3
huggingface_hub>=0.27.0
transformers==4.46.2
```
Or you can directly install the environments via our prepared ```requirements.txt``` file.
```
pip install -r requirements.txt
```

## Prepare Models and Datasets
In this paper, we evaluate our proposed MAD-M$^2$ equipped with four mainstream open-source benchmarks: `Qwen2.5-7B-Instruct`, `Qwen2.5-Math-7B-Instruct`, `DeepSeek-Math-7B-Instruct`, and `QwQ-32B` on both mathematical reasoning (`GSM8K`, `MATH`, `AIME24`, and `AIME25`) and language understanding (`MMLU_Pro`) benchmarks.

#### Data Preparation

**All datasets** have been prepared in the `processed_data` folder. Meanwhile, you can also download these datasets by running:

`python download_datasets.py --dataset_name=$DATASET_NAME --dataset_dir=./data`

The `--dataset_dir` can be reiplaced with any other path. Here, we recommend you to use `./data` as the path.

Then, you can further process the raw data by running:

`python prepare_data.py --dataset_name=$DATASET_NAME`

#### Model Preparation

You can download the LLMs we adopted in this paper via [huggingface](https://huggingface.co/). In the case that huggingface version > 1.3, you can download models with the command:

`hf download $MODEL_NAME --local_dir $SAVE_PATH`.

Then, you can copy the path to modify the `model_path` variable in the `config.yaml` file so that the model can be loaded when running the code.

## Run Experiments

After installing the dependencies and preparing the datasets and models, you can run the experiments.

To run **CoT** baseline, you can run:
`
python chain_of_thoughts.py --model_name $model_name --dataset $dataset --seed $seed --gpu_id $gpus --parallel --save_path $save_path
`

To Run **CoT-SC** baseline, you can run:
`
python chain_of_thoughts.py --model_name $model_name --dataset $dataset --seed $seed --self_consistency --num_reasoning_paths num_reasoning_paths --gpu_id $gpus --parallel --save_path $save_path
`

To Run **MAD** baseline, you can run:
`
python multi_agent_debate.py --model_name $model_name --dataset $dataset --seed $seed --num_agents 3 --max_round 2 --prune_strategy "naive" --gpu_id $gpus --parallel --save_path $save_path
`

To run **MAD-M$^2$ (S)** (i.e., MAD-M$^2$ with the subjective masking strategy), you can run:
`
python multi_agent_debate.py --model_name $model_name --dataset $dataset --seed $seed --num_agents 3 --max_round 2 --prune_strategy "subjective" --gpu_id $gpus --parallel --save_path $save_path
`

To run **MAD-M$^2$ (O)** (i.e., MAD-M$^2$ with the objective masking strategy), you can run:
`
python multi_agent_debate.py --model_name $model_name --dataset $dataset --seed $seed --num_agents 3 --max_round 2 --prune_strategy "objective" --gpu_id $gpus --parallel --save_path $save_path
`


Here, you can modify the hyperparameters for different cases:
- `--model_name`: The model you choose as the agents in MAD frameworks;
- `--dataset`: The dataset for evaluation;
- `--seed`: The random seed for the experiments
- `--gpu_id`: The index of GPUs;
- `--parallel`: Whether to use multiple GPUs in the experiments;
- `--save_path`: The path to save the final results and the debate log;
- `--num_reasoning_paths`: **For CoT-SC only**, the number of sampled answers;
- `--num_agents`: **For MAD/MAD-M$^2$ only**, the number of agents in each debate round;
- `--max_rounds`: **For MAD/MAD-M$^2$ only**, the maximum number of debate rounds.


For simplicity you can also reproduces the results reported in our work with the script files we provided in the folder `./scirpts`.

To reproduce the baselines in our paper (i.e., CoT, CoT-SC, MAD), you can directly run:
`
./scripts/run_all_model_baselines.sh
`

To reproduce all cases of MAD-M$^2$ (i.e., MAD-M$^2$ (S), MAD-M$^2$ (O)), you can directly run:
`
./scripts/run_all_mad_mm_variants.sh
`

To reproduce the scaling experiments in our paper, you can directly run:
`
./scripts/run_all_mad_analyses.sh
`

## Citation
```
@inproceedings{tian2025multi,
    title={Multi-agent debate with memory masking},
    author={Hongduan Tian and Xiao Feng and Ziyuan Zhao and Xiangyu Zhu and Rolan Yan and Bo Han},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2026}
}
```

