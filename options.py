import argparse
import os
from pathlib import Path
import logging
from ragllama import RAGModel
logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="EleutherAI/pythia-6.9b", help="the model to attack")
        self.parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help="the base model for LLMJudger/reference (default: llama-7b-hf)")
        self.parser.add_argument('--ref_model', type=str, default="EleutherAI/pythia-160m")
        self.parser.add_argument('--output_dir', type=str, default="out")
        self.parser.add_argument('--dataset', type=str, help="dataset name")
        self.parser.add_argument('--sub_dataset', type=int, default=128, help="the length of the input text to evaluate. Choose from 32, 64, 128, 256")
        self.parser.add_argument('--num_shots', type=str, default="12", help="number of shots to evaluate.")
        self.parser.add_argument('--pass_window', type=bool, default=True, help="whether to pass the window to the model.")
        self.parser.add_argument("--synehtic_prefix", type=bool, default=False, help="whether to use synehtic prefix.")
        self.parser.add_argument("--api_key_path", type=str, default=None, help="path to the api key file for OpenAI API if using synehtic prefix.")
        self.parser.add_argument("--train_sta_idx", type=int, default=0, help="the start index of the training data.")
        self.parser.add_argument("--train_end_idx", type=int, default=10000, help="the end index of the training data.")
        self.parser.add_argument("--eval_sta_idx", type=int, default=0, help="the start index of the evaluation data.")
        self.parser.add_argument("--eval_end_idx", type=int, default=1000, help="the end index of the evaluation data.")
        self.parser.add_argument("--maximum_samples", type=int, default=200, help="the maximum number of samples to evaluate.")
        self.parser.add_argument("--validation_split_percentage", type=float, default=0.1, help="the percentage of the train set used as validation set in case there's no validation split")
        self.parser.add_argument("--dataset_config_name", type=str, default=None, help="the config name of the dataset")
        self.parser.add_argument("--rag", action="store_true", help="use RAG mode")
        self.parser.add_argument("--packing", type=bool, default=True, help="packing")
        self.parser.add_argument("--block_size", type=int, default=128, help="the block size of the packing")
        self.parser.add_argument("--cache_path", type=str, default="/home/haoweifu/rag_mia/attack_models/ANeurIPS2024_SPV-MIA/ft_llms/cache", help="the path to the cache")
        self.parser.add_argument("--preprocessing_num_workers", type=int, default=1, help="the number of preprocessing workers")
        self.parser.add_argument("--use_dataset_cache", type=bool, default=True, help="whether to use the dataset cache")
        self.parser.add_argument("--llm_judge", action="store_true", default=False, help="If set, use LLMJudger for model loading and inference.")
        self.parser.add_argument("--llm_judge_noise", action="store_true", default=False, help="If set, use LLMJudgerNoise with MIA defense capabilities.")
        self.parser.add_argument("--judge_model", type=str, default=None, help="Path to judge model for LLMJudger.")
        
        # LLM-HAMP related arguments
        self.parser.add_argument("--llm_hamp", action="store_true", default=False, help="If set, use LLMHAMP for MIA defense.")
        self.parser.add_argument("--entropy_percentile", type=float, default=0.95, help="HAMP entropy percentile (0.8-0.99)")
        self.parser.add_argument("--modification_strength", type=float, default=0.5, help="Output modification strength (0.0-1.0)")
        self.parser.add_argument("--non_member_samples", type=int, default=3, help="Number of non-member samples to generate")

        # LLM-Defense related arguments
        self.parser.add_argument("--llm_defense", action="store_true", default=False, help="If set, use LLMDefender for inference-time defense.")
        self.parser.add_argument("--defense_type", type=str, default="gaussian", 
                                choices=['gaussian', 'laplace', 'temperature', 'confidence', 'ensemble'], 
                                help="Type of inference-time defense to apply")
        
        # Defense parameters
        self.parser.add_argument("--noise_std", type=float, default=0.1, help="Standard deviation for gaussian noise defense")
        self.parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon for differential privacy (laplace defense)")
        self.parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for temperature scaling defense")
        self.parser.add_argument("--confidence_threshold", type=float, default=0.8, help="Confidence threshold for confidence-based defense")
        self.parser.add_argument("--n_models", type=int, default=5, help="Number of models for ensemble defense")
        self.parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for ensemble defense")
        
        # RAG-Squad related arguments
        self.parser.add_argument("--top_k_contexts", type=int, default=7, help="Number of top-k contexts to retrieve for RAG-Squad")




