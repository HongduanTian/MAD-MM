import os
import yaml
import asyncio
import sys
import torch

import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.worker").setLevel(logging.ERROR)
logging.getLogger("vllm.logger").setLevel(logging.ERROR)

from src.args import parse_args
from src.config_utils import LLMConfig, load_configs_from_yaml
from src.models import LanguageModel
from src.reasoning_models import SparseMultiAgentDebate
from src.evaluator import MATHEval, MMLUProEval, AIMEEval, GSM8KEval
from src.utils import extract_answers

from functools import partial


def main():
    args = parse_args()
    
    # GPU settings
    if args.parallel:
        gpus = args.gpu_id.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        tensor_parallel_size = len(gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        tensor_parallel_size = 1
    
    # load configurations
    configs = load_configs_from_yaml("configs.yaml")   
    llm_configs = LLMConfig(configs["llm_configs"]["general_configs"], configs["llm_configs"][args.model_name])
    llm_configs.tensor_parallel_size = tensor_parallel_size
    
    # dataset
    dataset_path = f"./processed_data/{args.dataset}/{args.dataset}_test.jsonl"
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
    print(f">>>>>> Loading {args.dataset} dataset from {dataset_path}...")
    
    save_path = os.path.join(args.save_path, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    if args.dataset == "math":
        samples = 100
        evaluator = MATHEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "gsm8k":
        samples = 100
        evaluator = GSM8KEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "mmlu_pro":
        samples = 100
        evaluator = MMLUProEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "aime24":
        samples = None 
        evaluator = AIMEEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "aime25":
        samples = None 
        evaluator = AIMEEval(dataset_path, save_path, samples, args.seed)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    # Initialize an LLM as an agent
    extract_fn = partial(extract_answers, dataset_name=args.dataset)
    agent = LanguageModel(llm_configs, extract_fn=extract_fn)
    print(f"strict mode: {args.strict}")
    mad = SparseMultiAgentDebate(agent, dataset_name=args.dataset, num_agents=6, max_round=2)
    
    print(f"================================= Task Info =================================")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Num Agents: {args.num_agents}")
    print(f"Max Round: {args.max_round}")
    print(f"Prune Strategy: {args.prune_strategy}")
    print(f"Strict: {args.strict}")
    print(f"================================= Task Info =================================")
    
    evaluator.eval(mad, args)
    
if __name__ == "__main__":
    main()