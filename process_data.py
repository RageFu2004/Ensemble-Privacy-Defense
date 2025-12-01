from datasets import load_dataset
import os
import random
import numpy as np
import json
import datasets
from datasets import Dataset
   
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
def dataset_prepare(args, tokenizer=None, num_of_sequences=1024, chars_per_token=3.6):
    if args.dataset == "trivia_qa":
        args.dataset_config_name = "unfiltered"
    elif args.dataset == "squad":
        args.dataset_config_name = None

    print(f"using {args.dataset} with config {args.dataset_config_name}")

    if args.dataset == "squad":
        train_dataset = datasets.load_dataset(
            args.dataset,
            split="train"
        )
        valid_dataset = datasets.load_dataset(
            args.dataset,
            split="validation"
        )
    else:
        train_dataset = datasets.load_dataset(
            args.dataset,
            args.dataset_config_name,
            split=f"train[:{int((1-args.validation_split_percentage)*100)}%]"
        )
        valid_dataset = datasets.load_dataset(
            args.dataset,
            args.dataset_config_name,
            split=f"train[{int((1-args.validation_split_percentage)*100)}%:]",
        )

    if args.dataset == "trivia_qa":
        def process_trivia_qa(examples):
            inputs = []
            #print("check:", examples["search_results"].get("description"))
            q = examples["question"]
            #for q, r in zip(examples["question"], examples["search_results"]):
            desc_list = []
            for entry in examples["search_results"].get("description")[:5]:
                #print("entries:", entry)
                if isinstance(entry, dict):
                    desc = entry.get("description", "").strip()
                elif isinstance(entry, str):
                    desc = entry.strip()
                else:
                    desc = ""
                #print("desc:", desc)
                desc_list.append(desc)
            context = " ".join(desc_list)
            inputs.append(f"Context: {context}\n Use the above contexts to answer the question: {q.strip()}")
            #print("inputs:", inputs)
            return {"text": inputs, "question": q.strip()}
        def process_trivia_qa_sft(examples):
            q = examples["question"]
            a = examples["answer"]["value"] if isinstance(examples["answer"], dict) else examples["answer"]
            text = f"Question: {q.strip()}\nAnswer: {a.strip()}"
            return {"text": text, "question": q.strip()}


        if args.rag:
            train_dataset = train_dataset.map(
                process_trivia_qa,
                remove_columns=train_dataset.column_names
            )
            valid_dataset = valid_dataset.map(
                process_trivia_qa,
                remove_columns=valid_dataset.column_names
            )
        else:
            train_dataset = train_dataset.map(
                process_trivia_qa_sft,
                remove_columns=train_dataset.column_names
            )
            valid_dataset = valid_dataset.map(
                process_trivia_qa_sft,
                remove_columns=valid_dataset.column_names
            )
        return train_dataset, valid_dataset
    
    elif args.dataset == "squad":
        def process_squad_rag(examples):

            inputs = []
            questions = examples["question"]
            contexts = examples["context"]
            answers = examples["answers"]
            
            for q, context, answer in zip(questions, contexts, answers):
 
                try:
                    if isinstance(answer, dict) and "text" in answer:
                        if isinstance(answer["text"], list) and len(answer["text"]) > 0:
                            answer_text = answer["text"][0]
                        elif isinstance(answer["text"], str):
                            answer_text = answer["text"]
                        else:
                            answer_text = ""
                    else:
                     
                        answer_text = str(answer) if answer else ""
                except Exception as e:
                    
                    answer_text = ""
                
                if answer_text:  
                   
                    rag_text = f"Context: {context}\nQuestion: {q}\nAnswer: {answer_text}"
                    inputs.append(rag_text)
            
            return {"text": inputs, "question": questions}
        
        def process_squad_sft(examples):
            inputs = []
            questions = examples["question"]
            answers = examples["answers"]
            
            for q, answer in zip(questions, answers):
                # Extract answer text - fix answer format handling
                try:
                    if isinstance(answer, dict) and "text" in answer:
                        if isinstance(answer["text"], list) and len(answer["text"]) > 0:
                            answer_text = answer["text"][0]
                        elif isinstance(answer["text"], str):
                            answer_text = answer["text"]
                        else:
                            answer_text = ""
                    else:
                        # If answer is not in dict format, use it directly
                        answer_text = str(answer) if answer else ""
                except Exception as e:
                    print(f"Error processing answer: {e}, answer type: {type(answer)}, answer content: {answer}")
                    answer_text = ""
                print("answer_text:", answer_text)
                
                if answer_text:  # Only process samples with answers
                    #print("yes")
                    # SFT format: question + answer
                    sft_text = f"Question: {q}\nAnswer: {answer_text}"
                    inputs.append(sft_text)
            
            return {"text": inputs, "question": questions}
        
        if args.rag:
            train_dataset = train_dataset.map(
                process_squad_rag,
                remove_columns=train_dataset.column_names
            )
            valid_dataset = valid_dataset.map(
                process_squad_rag,
                remove_columns=valid_dataset.column_names
            )
        else:
            train_dataset = train_dataset.map(
                process_squad_sft,
                remove_columns=train_dataset.column_names
            )
            valid_dataset = valid_dataset.map(
                process_squad_sft,
                remove_columns=valid_dataset.column_names
            )
        return train_dataset, valid_dataset


    global text_column
    column = train_dataset.column_names
    if "text" in column:
        text_column = "text"
    elif "document" in column:
        text_column = "document"
    elif "content" in column:
        text_column = "content"
    else:
        raise ValueError(f"No supported text column found in dataset: {column}")

    train_dataset = train_dataset.select_columns(text_column)
    valid_dataset = valid_dataset.select_columns(text_column)
    if text_column != "text":
        train_dataset = train_dataset.rename_column(text_column, "text")
        valid_dataset = valid_dataset.rename_column(text_column, "text")

    if args.packing:
        global block_size, tokenizer_, max_buff_size
        block_size = args.block_size
        max_buff_size = block_size * chars_per_token * num_of_sequences
        tokenizer_ = tokenizer
        create_folder(f"{args.cache_path}/{args.dataset}/{args.dataset_config_name}")
        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset}/{args.dataset_config_name}/train_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset}/{args.dataset_config_name}/valid_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )

    return train_dataset, valid_dataset
def create_dataset(dataset_name,sub_dataset_name, output_dir, num_shots, cfg, tokenizer):
    
    if dataset_name == "wikimia":
        dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{sub_dataset_name}")
        member_data = []
        nonmember_data = []
        for data in dataset:
            if data["label"] == 1:
                member_data.append(data["input"])
            elif data["label"] == 0:
                nonmember_data.append(data["input"])
        dataset_folder = os.path.join(output_dir, f"{sub_dataset_name}", "data")
        
        # shuffle the datasets
        random.shuffle(member_data)
        random.shuffle(nonmember_data)
        num_shots = int(num_shots)

        nonmember_prefix = nonmember_data[:num_shots]
        nonmember_data = nonmember_data[num_shots:]
        member_data_prefix = member_data[:num_shots]
        member_data = member_data[num_shots:]
        
        # # Create the output directory
        # os.makedirs(dataset_folder, exist_ok=True)

        # data_folder = os.path.join(dataset_folder,  "target_data")  
        # os.makedirs(data_folder, exist_ok=True)

        # prefix_folder_path = os.path.join(dataset_folder,  "prefixes")   
        # os.makedirs(prefix_folder_path, exist_ok=True)

        # # save the original datasets
        # member_data_path = os.path.join(data_folder, "train.jsonl")
        # nonmember_data_path = os.path.join(data_folder, "test.jsonl")\
            
        # save_jsonl(member_data, member_data_path)
        # save_jsonl(nonmember_data, nonmember_data_path)
        # save_jsonl(nonmember_prefix, os.path.join(prefix_folder_path, "nonmember_prefix.jsonl"))
        # save_jsonl(member_data_prefix, os.path.join(prefix_folder_path, "member_prefix.jsonl"))
        
    elif dataset_name == "trivia_qa":
        train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
        train_dataset = Dataset.from_dict(train_dataset[cfg.train_sta_idx:cfg.train_end_idx])
        valid_dataset = Dataset.from_dict(valid_dataset[cfg.eval_sta_idx:cfg.eval_end_idx])
        train_dataset = Dataset.from_dict(train_dataset[random.sample(range(len(train_dataset["text"])), cfg.maximum_samples)])
        valid_dataset = Dataset.from_dict(valid_dataset[random.sample(range(len(valid_dataset["text"])), cfg.maximum_samples)])
        for data in train_dataset:
            print(data['question'])
        member_data = [data["question"] for data in train_dataset]
        nonmember_data = [data["question"] for data in valid_dataset]
        random.shuffle(member_data)
        random.shuffle(nonmember_data)
        num_shots = int(num_shots)
        nonmember_prefix = nonmember_data[:num_shots]
        nonmember_data = nonmember_data[num_shots:]
        member_data_prefix = member_data[:num_shots]
        member_data = member_data[num_shots:]
    elif dataset_name == "squad":
        train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
        train_dataset = Dataset.from_dict(train_dataset[cfg.train_sta_idx:cfg.train_end_idx])
        valid_dataset = Dataset.from_dict(valid_dataset[cfg.eval_sta_idx:cfg.eval_end_idx])
        train_dataset = Dataset.from_dict(train_dataset[random.sample(range(len(train_dataset["text"])), cfg.maximum_samples)])
        valid_dataset = Dataset.from_dict(valid_dataset[random.sample(range(len(valid_dataset["text"])), cfg.maximum_samples)])
        for data in train_dataset:
            print(data['question'])
        print("train_dataset:", len(train_dataset))
        member_data = [data["question"] for data in train_dataset]
        nonmember_data = [data["question"] for data in valid_dataset]
        random.shuffle(member_data)
        random.shuffle(nonmember_data)
        num_shots = int(num_shots)
        nonmember_prefix = nonmember_data[:num_shots]
        nonmember_data = nonmember_data[num_shots:]
        member_data_prefix = member_data[:num_shots]
        member_data = member_data[num_shots:]
    else: 
        raise ValueError(f"Unknown dataset: {dataset_name}. Please modify the code to include the dataset. Make sure the dataset is in the same format.")

    full_data = [] 
    # binary classification, the data need to be balanced. 
    for nm_data, m_data in zip(nonmember_data, member_data):
        full_data.append({"input": nm_data, "label": 0})
        full_data.append({"input": m_data, "label": 1})
    
    return full_data, nonmember_prefix, member_data_prefix
