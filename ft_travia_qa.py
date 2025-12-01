import argparse
import logging
import os

# Set temporary directory to /largedisk before importing any modules to avoid root partition space issues
# Set Python temporary directory
os.environ["TMPDIR"] = "/largedisk/tmp"
os.environ["TMP"] = "/largedisk/tmp"
os.environ["TEMP"] = "/largedisk/tmp"
# Set Python bytecode cache directory
os.environ["PYTHONPYCACHEPREFIX"] = "/largedisk/pycache"
# Ensure temporary directories exist
os.makedirs("/largedisk/tmp", exist_ok=True)
os.makedirs("/largedisk/pycache", exist_ok=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoConfig
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from utils import get_logger
from huggingface_hub import login
r
logger = get_logger("finetune", "info")


if __name__ == "__main__":
    # Use your own token
    token = ""
    
    try:
        login(token=token, add_to_git_credential=False)
        logger.info("Successfully logged in to HuggingFace")
    except Exception as e:
        logger.warning(f"Login attempt failed: {e}, continuing with token parameter")
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_name", type=str, default="google/gemma-2b", help="Model name or path")
    parser.add_argument("--cache_path", type=str, default="/largedisk/ragmia_models/gemma2b", help="Cache directory for model and data")

    parser.add_argument("--block_size", type=int, default=1024, help="Block size for tokenization")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="/largedisk/ragmia_models/gm2b_sft", help="Output directory")
    parser.add_argument("--log_steps", type=int, default=50, help="Steps between logging")
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps between evaluation")
    parser.add_argument("--save_epochs", type=int, default=1, help="Save checkpoints every N epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Learning rate warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Scheduler type")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples for faster training")
    parser.add_argument("--save_limit", type=int, default=2, help="Max number of checkpoints to save")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of training steps (overrides epochs if set)")

    parser.add_argument("--peft", type=str, default="lora", help="Type of PEFT method")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--use_int4", action="store_true", help="Enable INT4 quantization")
    parser.add_argument("--use_int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--disable_peft", action="store_true", help="Disable PEFT")

    args = parser.parse_args()


    cache_dir = os.path.join(args.cache_path, "hub")
    os.makedirs(cache_dir, exist_ok=True)

    os.environ["HF_HOME"] = args.cache_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    logger.info(f"Using cache directory: {cache_dir}")
    logger.info(f"Cache path: {args.cache_path}")

    # Load Model
    accelerator = Accelerator()
    device = accelerator.device

    logger.info(f"Loading model {args.model_name}...")

    config = AutoConfig.from_pretrained(
        args.model_name, 
        token=token,
        cache_dir=cache_dir
    )
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    bnb_config = None
    if args.use_int4 or args.use_int8:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=args.use_int4, load_in_8bit=args.use_int8)
        logger.info(f"Using quantization: INT4={args.use_int4}, INT8={args.use_int8}")
    else:
        logger.info("No quantization applied")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config,
        device_map="auto",
        token=token,
        cache_dir=cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        token=token,
        cache_dir=cache_dir
    )
    # Gemma models need special pad_token handling
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data Loading - TriviaQA

    logger.info("Loading TriviaQA dataset...")
    raw_datasets = load_dataset("trivia_qa", "unfiltered", cache_dir=args.cache_path)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    # Preprocessing for QA

    def preprocess_function(examples):
        inputs = []
        targets = []
        for q, r, a in zip(examples["question"], examples["search_results"], examples["answer"]):
            context = r[0]["snippet"].strip() if isinstance(r, list) and r and "snippet" in r[0] else ""
            target = a["normalized_value"] if isinstance(a, dict) and "normalized_value" in a else ""
            inputs.append(f"Question: {q.strip()}\nContext: {context}")
            targets.append(target)
        return {"text": f"{i}\nAnswer: {t}" for i, t in zip(inputs, targets)}

    train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess_function, remove_columns=eval_dataset.column_names)
    
    # Limit dataset size for faster training
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
        logger.info(f"Limited training dataset to {len(train_dataset)} samples")

    
    # PEFT
    if not args.disable_peft:
        logger.info("Applying LoRA configuration...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)


    use_fp16 = torch.cuda.is_available() and not (args.use_int4 or args.use_int8)
    
    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.log_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "save_steps": args.eval_steps,
        "save_total_limit": args.save_limit,
        "fp16": use_fp16,
        "report_to": "none"
    }

    if args.max_steps is not None:
        training_kwargs["max_steps"] = args.max_steps
        training_kwargs["num_train_epochs"] = None
    else:
        training_kwargs["num_train_epochs"] = args.epochs
    
    training_args = TrainingArguments(**training_kwargs)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    logger.info("Starting training...")
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")
