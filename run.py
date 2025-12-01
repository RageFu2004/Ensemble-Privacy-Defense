# This implementation is adapted from Min-K% and WikiMIA: https://github.com/swj0419/detect-pretrain-code 
import os
from tqdm import tqdm
from process_data import create_dataset
import torch 
from options import Options
import numpy as np
import torch
import zlib
from eval import *
import os
import torch.nn.functional as F
from visualization import analyze_final_results
from transformers import set_seed
import torch 
import random
import openai 
from accelerate import Accelerator
import torch 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MambaForCausalLM,
    LlamaTokenizer, BitsAndBytesConfig
)
from peft import PeftConfig, PeftModel
from ragllama import RAGModel
from ragllama_squad import RAGModelSquad
import math

def load_model(name1, name2, use_float16=True):    
    accelerator = Accelerator()

    def load_specific_model(name, use_float16=False):
        if "mamba" in name:
            model = MambaForCausalLM.from_pretrained(
                name, return_dict=True, trust_remote_code=True,
                torch_dtype=torch.float16 if use_float16 else torch.float32,
                device_map="cuda:0"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                name, return_dict=True, trust_remote_code=True,
                torch_dtype=torch.float16 if use_float16 else torch.float32,
                device_map="auto"
            )
        return model

    # Load the first model
    model1 = load_specific_model(name1, use_float16)
    tokenizer1 = AutoTokenizer.from_pretrained(name1)
    # Load the second model with the same float16 setting as model1
    model2 = load_specific_model(name2, use_float16)
    tokenizer2 = AutoTokenizer.from_pretrained(name2)

    model1.eval()
    model2.eval()

    model1, model2 = accelerator.prepare(model1, model2)

    return model1, model2, tokenizer1, tokenizer2, accelerator

def api_key_setup(key_path):
    openai.api_key = open(key_path, "r").read()

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def get_ll(sentence, model, tokenizer, device):
    # If model is LLMJudger, directly use target_model to calculate log-likelihood
    if hasattr(model, 'target_model') and hasattr(model, 'target_tokenizer'):
        print(f"[DEBUG get_ll] Detected LLMJudger, using target_model for log-likelihood")
        target_model = model.target_model
        target_tokenizer = model.target_tokenizer
        input_ids = torch.tensor(target_tokenizer.encode(sentence)).unsqueeze(0)
        input_ids = input_ids.to(device)
        print(f"[DEBUG get_ll] sentence length: {len(sentence)}, input_ids shape: {input_ids.shape}")
        print(f"[DEBUG get_ll] input_ids token_id range: min={input_ids.min().item()}, max={input_ids.max().item()}")
        with torch.no_grad():
            outputs = target_model(input_ids, labels=input_ids)
    else:
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
        input_ids = input_ids.to(device)
        print(f"[DEBUG get_ll] sentence length: {len(sentence)}, input_ids shape: {input_ids.shape}")
        print(f"[DEBUG get_ll] input_ids token_id range: min={input_ids.min().item()}, max={input_ids.max().item()}")
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
    
    # Compatible with RAGModelSquad output format
    if hasattr(outputs, 'loss') and hasattr(outputs, 'logits'):
        # Object returned by RAGModelSquad
        loss = outputs.loss
        logits = outputs.logits
    else:
        # Standard model output
        loss, logits = outputs[0], outputs[1]
    
    #print("logits:", logits)
    #print("loss:", loss)
    # Compatible with LLMJudger returning list/dict
    if isinstance(logits, list) and len(logits) > 0:
        logits_item = logits[0]
        if isinstance(logits_item, dict) and 'target' in logits_item:
            logits_tensor = logits_item['target']
        else:
            logits_tensor = logits_item
    elif isinstance(logits, dict) and 'target' in logits:
        logits_tensor = logits['target']
    else:
        logits_tensor = logits
    # Compatible with loss
    if isinstance(loss, list) and len(loss) > 0:
        loss_item = loss[0]
        if isinstance(loss_item, dict) and 'target' in loss_item:
            loss_tensor = loss_item['target']
        else:
            loss_tensor = loss_item
    elif isinstance(loss, dict) and 'target' in loss:
        loss_tensor = loss['target']
    else:
        loss_tensor = loss
    
    # Now directly use input_ids (from target_tokenizer) since logits are also from target_model
    input_ids_to_use = input_ids
    print(f"[DEBUG get_ll] Using input_ids (target_tokenizer), shape: {input_ids_to_use.shape}")
    print(f"[DEBUG get_ll] logits_tensor shape: {logits_tensor.shape}, vocab_size: {logits_tensor.shape[-1]}")
    return get_all_prob(input_ids_to_use, loss_tensor, logits_tensor)

def get_conditional_ll(input_text, target_text, model, tokenizer, device):
    # If model is LLMJudger, directly use target_model to calculate log-likelihood
    if hasattr(model, 'target_model') and hasattr(model, 'target_tokenizer'):
        print(f"[DEBUG get_conditional_ll] Detected LLMJudger, using target_model for log-likelihood")
        target_model = model.target_model
        target_tokenizer = model.target_tokenizer
        print(f"[DEBUG get_conditional_ll] input_text length: {len(input_text)}, target_text length: {len(target_text)}")
        input_encodings = target_tokenizer(input_text, return_tensors="pt")
        target_encodings = target_tokenizer(target_text, return_tensors="pt")
        print(f"[DEBUG get_conditional_ll] input_encodings shape: {input_encodings.input_ids.shape}, target_encodings shape: {target_encodings.input_ids.shape}")
        concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1)
        labels = concat_ids.clone()
        labels[:, : input_encodings.input_ids.size(1)] = -100
        print(f"[DEBUG get_conditional_ll] concat_ids shape: {concat_ids.shape}, labels shape: {labels.shape}")
        print(f"[DEBUG get_conditional_ll] labels token_id range: min={labels.min().item()}, max={labels.max().item()}")
        with torch.no_grad():
            outputs = target_model(concat_ids, labels=labels)
    else:
        print(f"[DEBUG get_conditional_ll] input_text length: {len(input_text)}, target_text length: {len(target_text)}")
        input_encodings = tokenizer(input_text, return_tensors="pt")
        target_encodings = tokenizer(target_text, return_tensors="pt")
        print(f"[DEBUG get_conditional_ll] input_encodings shape: {input_encodings.input_ids.shape}, target_encodings shape: {target_encodings.input_ids.shape}")
        concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1)
        labels = concat_ids.clone()
        labels[:, : input_encodings.input_ids.size(1)] = -100
        print(f"[DEBUG get_conditional_ll] concat_ids shape: {concat_ids.shape}, labels shape: {labels.shape}")
        print(f"[DEBUG get_conditional_ll] labels token_id range: min={labels.min().item()}, max={labels.max().item()}")
        with torch.no_grad():
            outputs = model(concat_ids, labels=labels)
    
    # Compatible with RAGModelSquad output format
    if hasattr(outputs, 'loss') and hasattr(outputs, 'logits'):
        # Object returned by RAGModelSquad
        loss = outputs.loss
        logits = outputs.logits
    else:
        # Standard model output
        loss, logits = outputs[0], outputs[1]
    
    # Compatible with LLMJudger returning list/dict
    if isinstance(logits, list) and len(logits) > 0:
        logits_item = logits[0]
        if isinstance(logits_item, dict) and 'target' in logits_item:
            logits_tensor = logits_item['target']
        else:
            logits_tensor = logits_item
    elif isinstance(logits, dict) and 'target' in logits:
        logits_tensor = logits['target']
    else:
        logits_tensor = logits
    # Compatible with loss
    if isinstance(loss, list) and len(loss) > 0:
        loss_item = loss[0]
        if isinstance(loss_item, dict) and 'target' in loss_item:
            loss_tensor = loss_item['target']
        else:
            loss_tensor = loss_item
    elif isinstance(loss, dict) and 'target' in loss:
        loss_tensor = loss['target']
    else:
        loss_tensor = loss
    
    print(f"[DEBUG get_conditional_ll] logits_tensor shape: {logits_tensor.shape}, vocab_size: {logits_tensor.shape[-1]}")
    
    # Now directly use labels (from target_tokenizer) since logits are also from target_model
    labels_to_use = labels
    print(f"[DEBUG get_conditional_ll] Using labels (target_tokenizer), shape: {labels_to_use.shape}")
    
    return get_all_prob(labels_to_use, loss_tensor, logits_tensor)

def get_all_prob(input_ids, loss, logits):
    print(f"[DEBUG get_all_prob] input_ids shape: {input_ids.shape}, logits shape: {logits.shape}")
    vocab_size = logits.shape[-1]
    print(f"[DEBUG get_all_prob] vocab_size: {vocab_size}")
    
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    print(f"[DEBUG get_all_prob] input_ids_processed shape: {input_ids_processed.shape}")
    print(f"[DEBUG get_all_prob] input_ids_processed token_id range: min={input_ids_processed.min().item()}, max={input_ids_processed.max().item()}")
    
    for i, token_id in enumerate(input_ids_processed):
        #print("probabilities shape:", probabilities.shape)
        #print("token_id:", token_id)
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    if hasattr(loss, 'item'):
        loss_value = loss.item()
    else:
        loss_value = float(loss)
    ll = -loss_value  # log-likelihood
    ppl = math.exp(loss_value)
    prob = math.exp(-loss_value)
    return prob, ll , ppl, all_prob, loss_value

def inference(model1, model2, tokenizer1, tokenizer2, target_data, prefix, accelerator, num_shots, ex):
    pred = {}
    
    # unconditional log-likelihood
    ll = get_ll(target_data, model1, tokenizer1,accelerator.device)[1]
     
    # ReCaLL
    if int(num_shots) != 0:   
        # conditional log-likelihood with prefix     
        ll_nonmember = get_conditional_ll("".join(prefix), target_data, model1, tokenizer1, accelerator.device)[1]
        pred["recall"] = ll_nonmember / ll 

    # baselines 
    with torch.no_grad():
        outputs = model1(torch.tensor(tokenizer1.encode(target_data)).unsqueeze(0).to(accelerator.device), labels=torch.tensor(tokenizer1.encode(target_data)).unsqueeze(0).to(accelerator.device))
        # If it's RAGModel or RAGModelSquad, outputs will have input_ids_list
        if hasattr(outputs, 'input_ids_list'):
            input_ids = outputs.input_ids_list
            candid = torch.tensor(tokenizer1.encode(target_data)).unsqueeze(0).to(accelerator.device)
            print("input_ids:-----------------", input_ids[0].shape)
            print("candid:_________________", candid.shape)
            logits = outputs.logits[0]
            #print("input_ids:", input_ids)
            print("logits:", logits.shape)
            # Compatible with dict/list/tensor
            if isinstance(input_ids, dict):
                if 'target' in input_ids and isinstance(input_ids['target'], torch.Tensor):
                    input_ids_tensor = input_ids['target']
                else:
                    raise ValueError("input_ids dict does not contain a valid 'target' tensor.")
            elif isinstance(input_ids, list) and len(input_ids) > 0:
                input_ids_tensor = input_ids[0]
            elif isinstance(input_ids, torch.Tensor):
                input_ids_tensor = input_ids
            else:
                raise ValueError(f"Unsupported input_ids type: {type(input_ids)}")
            #print("input_ids_tensor:", input_ids_tensor)
            input_ids = input_ids_tensor.to(logits.device)
        else:
            input_ids = torch.tensor(tokenizer1.encode(target_data)).unsqueeze(0).to(accelerator.device)
            logits = outputs[1]
    ll_ref = get_ll(target_data, model2, tokenizer2, accelerator.device)[1]

    # loss and zlib
    pred["ll"] = ll
    pred["ref"] = ll - ll_ref
    pred["zlib"] = ll / len(zlib.compress(bytes(target_data, "utf-8")))

    # For mink and mink++
    '''
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    '''

    # Handle logits with different dimensions
    print(f"Debug - logits shape: {logits.shape}, input_ids shape: {input_ids.shape}")
    
    if len(logits.shape) == 3:  # LLM-HAMP returns 3D logits [batch, seq_len, vocab_size]
        logits = logits[0]  # Take the first batch
        input_ids = input_ids[0] if len(input_ids.shape) > 1 else input_ids
    elif len(logits.shape) == 2:  # Standard models return 2D logits [seq_len, vocab_size]
        logits = logits
        input_ids = input_ids[0] if len(input_ids.shape) > 1 else input_ids
    
    print(f"Debug - after processing: logits shape: {logits.shape}, input_ids shape: {input_ids.shape}")
    
    probs = F.softmax(logits[:-1], dim=-1)         # [seq_len, vocab_size]
    log_probs = F.log_softmax(logits[:-1], dim=-1) # [seq_len, vocab_size]
    target_ids = input_ids[1:]                     # [seq_len]
    
    # Ensure target_ids has correct dimensions
    if len(target_ids.shape) == 1:
        target_ids = target_ids.unsqueeze(-1)  # [seq_len, 1]
    
    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)  # [seq_len]
    mu = (probs * log_probs).sum(-1)               # [seq_len]
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)  # [seq_len]

    print("token_log_probs shape:", token_log_probs.shape)
    print("mu shape:", mu.shape)
    print("sigma shape:", sigma.shape)


    ## mink
    for ratio in [0.2]:
        k_length = int(len(token_log_probs) * ratio)
        topk = np.sort(token_log_probs.cpu().float().numpy())[:k_length]
        pred[f"mink_{ratio}"] = np.mean(topk).item()
        
    ## mink++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    for ratio in [0.2]:
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu().float().numpy())[:k_length]
        pred[f"mink++_{ratio}"] = np.mean(topk).item()

    ex["pred"] = pred
    return ex
def prepare_model(peft_model_path):
    print(f"Loading PeftConfig from {peft_model_path}")
    peft_config = PeftConfig.from_pretrained(peft_model_path)


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Usually 'nf4' or 'fp4'
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16  # Can also be torch.float16
    )

    # First try pure GPU auto mapping; if failed, enable offload and max_memory; if still failed, fix to device 0
    offload_dir = os.path.join(os.path.dirname(peft_model_path), ".offload")
    os.makedirs(offload_dir, exist_ok=True)
    # Estimate memory limit
    if torch.cuda.is_available():
        try:
            total_mem_gib = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            gpu_cap = f"{max(1, int(total_mem_gib * 0.9))}GiB"
        except Exception:
            gpu_cap = "20GiB"
        max_memory = {0: gpu_cap, "cpu": "120GiB"}
    else:
        max_memory = {"cpu": "120GiB"}

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    except ValueError as e1:
        print("[prepare_model] Auto mapping failed, retry with offload. Error:", e1)
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                offload_folder=offload_dir,
                max_memory=max_memory,
                trust_remote_code=True
            )
        except Exception as e2:
            print("[prepare_model] Offload auto mapping failed, force device_map {0:0}. Error:", e2)
            forced_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                quantization_config=bnb_config,
                device_map=forced_map,
                low_cpu_mem_usage=True,
                offload_folder=offload_dir,
                max_memory=max_memory,
                trust_remote_code=True
            )

    target_model = PeftModel.from_pretrained(base_model, peft_model_path)

    tokenizer = LlamaTokenizer.from_pretrained(peft_config.base_model_name_or_path, add_eos_token=True,
                                                  add_bos_token=True, use_fast=True)
    return target_model, tokenizer
def generate_prompt(example):
    return f"Generate a passage that is similar to the given text in length, domain, and style.\nGiven text:{example}\nPassage :"

def get_completion(prompt):
    message = [{"role": "user", "content": prompt}]
    responses = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=message,
        max_tokens=1024,
        temperature=1,
    )
    return responses.choices[0]["message"]["content"]

def gpt_synthetic_prefix (original_prefixes):
    # default generate synthetic prefix from non-member data
    synthetic_prefixes = []
    for original_prefix in original_prefixes:
        prompt = generate_prompt(original_prefix)
        response = get_completion(prompt)      
        synthetic_prefixes.append(response)
    return synthetic_prefixes

    
def process_prefix(target_model, prefix, avg_length, pass_window, total_shots):
    if pass_window:
        return prefix
    max_length = model1.config.max_position_embeddings if "mamba" not in target_model else 2048
    token_counts = [len(tokenizer1.encode(shot)) for shot in prefix]
    target_token_count = avg_length
    total_tokens = sum(token_counts) + target_token_count
    if total_tokens <= max_length:
        return prefix
    # Determine the maximum number of shots that can fit within the max_length
    max_shots = 0
    cumulative_tokens = target_token_count
    for count in token_counts:
        if cumulative_tokens + count <= max_length:
            max_shots += 1
            cumulative_tokens += count
        else:
            break
    # Truncate the prefix to include only the maximum number of shots
    truncated_prefix = prefix[-max_shots:]
    total_shots = max_shots
    return truncated_prefix

def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, prefix, accelerator, total_shots, pass_window, synehtic_prefix):
    all_output = []
    if int(total_shots) != 0:   
        avg_length = int(np.mean([len(tokenizer1.encode(ex["input"])) for ex in test_data])) 
        prefix = process_prefix(target_model, prefix, avg_length, pass_window, total_shots) 
        if synehtic_prefix:
            prefix = gpt_synthetic_prefix(prefix)
    for ex in tqdm(test_data):
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, ex["input"], prefix, accelerator, total_shots, ex)
        all_output.append(new_ex)
    return all_output

if __name__ == "__main__":
    accelerator = Accelerator()
    fix_seed(42)
    args = Options()
    args = args.parser.parse_args()

    output_dir = args.output_dir
    dataset = args.dataset
    target_model = args.target_model
    ref_model = args.ref_model
    sub_dataset = args.sub_dataset
    num_shots = args.num_shots
    pass_window = args.pass_window
    synehtic_prefix = args.synehtic_prefix
    api_key_path = args.api_key_path
    print("args.rag:", args.rag)
    print("llm judge: ", args.llm_judge)
    print("hamp: ", args.llm_hamp)
    print("llm defense: ", args.llm_defense, "defense type: ", args.defense_type)

    if synehtic_prefix and api_key_path is not None:
        api_key_setup(api_key_path)
    # load models
    if getattr(args, 'llm_judge', False):
        if getattr(args, 'llm_judge_noise', False):
            from llm_judge_noise import LLMJudgerNoise
            judge_model_path = args.judge_model if args.judge_model is not None else args.ref_model
            model1 = LLMJudgerNoise(target_model_path=args.target_model, base_model_path=args.base_model, judge_model_path=judge_model_path, is_rag=args.rag, device=accelerator.device)
            print("Using LLMJudgerNoise with MIA defense capabilities")
        else:
            from llm_judge import LLMJudger
            judge_model_path = args.judge_model if args.judge_model is not None else args.ref_model
            model1 = LLMJudger(target_model_path=args.target_model, base_model_path=args.base_model, judge_model_path=judge_model_path, is_rag=args.rag, device=accelerator.device)
            print("Using standard LLMJudger")
        #model2 = None  # Not used in LLMJudger mode
        tokenizer1 = model1.target_tokenizer
        #tokenizer2 = None  # Not used in LLMJudger mode
    elif getattr(args, 'llm_hamp', False):
        from llm_hamp import LLMHAMP
        print("Loading LLM-HAMP model...")
        model1 = LLMHAMP(
            target_model_path=args.target_model,
            base_model_path=args.base_model,
            is_rag=args.rag,
            device=accelerator.device,
            entropy_percentile=getattr(args, 'entropy_percentile', 0.95),
            modification_strength=getattr(args, 'modification_strength', 0.5),
            non_member_samples=getattr(args, 'non_member_samples', 3)
        )
        tokenizer1 = model1.target_tokenizer
        print(f"LLM-HAMP loaded successfully with parameters:")
        print(f"  - Entropy percentile: {getattr(args, 'entropy_percentile', 0.95)}")
        print(f"  - Modification strength: {getattr(args, 'modification_strength', 0.5)}")
        print(f"  - Non-member samples: {getattr(args, 'non_member_samples', 10)}")
    elif getattr(args, 'llm_defense', False):
        from llm_defense import LLMDefender
        print("Loading LLM-Defense model...")
        # Build defense parameters dictionary
        defense_kwargs = {}
        if args.defense_type == 'gaussian':
            defense_kwargs['noise_std'] = args.noise_std
        elif args.defense_type == 'laplace':
            defense_kwargs['epsilon'] = args.epsilon
        elif args.defense_type == 'temperature':
            defense_kwargs['temperature'] = args.temperature
        elif args.defense_type == 'confidence':
            defense_kwargs['threshold'] = args.confidence_threshold
        elif args.defense_type == 'ensemble':
            defense_kwargs['n_models'] = args.n_models
            defense_kwargs['dropout_rate'] = args.dropout_rate
        
        model1 = LLMDefender(
            target_model_path=args.target_model,
            base_model_path=args.base_model,
            is_rag=args.rag,
            device=accelerator.device,
            defense_type=args.defense_type,
            **defense_kwargs
        )
        tokenizer1 = model1.target_tokenizer
        print(f"LLM-Defense loaded successfully with parameters:")
        print(f"  - Defense type: {args.defense_type}")
        print(f"  - Defense parameters: {defense_kwargs}")
    else:
        if args.rag:
            # If it's squad dataset, use RAGModelSquad
            if dataset == "squad":
                print("Loading RAGModelSquad for squad dataset...")
                model1 = RAGModelSquad(
                    model_name=args.base_model,
                    cache_dir=args.cache_path,
                    device=accelerator.device,
                    top_k_contexts=getattr(args, 'top_k_contexts', 1)
                )
                tokenizer1 = model1.tokenizer
            else:
                model1 = RAGModel(model_name=args.base_model, cache_dir=args.cache_path)
                tokenizer1 = model1.tokenizer
        else:
            model1, tokenizer1 = prepare_model(target_model)
    model2, tokenizer2 = prepare_model(ref_model)
    # process and prepare the data
    full_data, nonmember_prefix, member_data_prefix = create_dataset(dataset, sub_dataset, output_dir, num_shots, args, tokenizer1)

    
    all_output = evaluate_data(full_data, model1, model2, tokenizer1, tokenizer2, nonmember_prefix, accelerator, num_shots, pass_window, synehtic_prefix)
    # save the results
    all_output_path = os.path.join(output_dir, f"{dataset}", f"{target_model.split('/')[1]}_{ref_model.split('/')[1]}", f"{sub_dataset}", f"{num_shots}_shot_{sub_dataset}.json")
    os.makedirs(os.path.dirname(all_output_path), exist_ok=True)
    dump_jsonl(all_output, all_output_path)
    print(f"Saved results to {all_output_path}")
    
    # evaluate the results
    fig_fpr_tpr(all_output, all_output_path)        
    
    # result visualizations to show 0 to n shot results - make sure you have these results 
    analyze_final_results(os.path.join(output_dir, f"{dataset}", f"{target_model.split('/')[1]}_{ref_model.split('/')[1]}", f"{sub_dataset}"), show_values=True)
            