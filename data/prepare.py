import os
import random
import datasets
import trl
from attack.utils import create_folder

block_size = None
tokenizer_ = None
max_buff_size = None
text_column = None

def packing_texts(examples):
    more_examples = True
    packed_texts = []
    packed_ids = []
    # for key in examples.keys():
    assert list(examples.keys()) == ["text"]
    iterator = iter(examples["text"])
    # for sentence in examples["text"]:
    total_num = 0
    drop_num = 0
    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= max_buff_size:
                break
            try:
                buffer.append(next(iterator))
                buffer_len += len(buffer[-1])
            except StopIteration:
                more_examples = False
                break
        tokenized_inputs = tokenizer_(buffer, truncation=False)["input_ids"]
        inputs = tokenizer_.batch_decode(tokenized_inputs)
        tokenized_inputs = tokenizer_(inputs, truncation=False)["input_ids"]
        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend(tokenized_input)
        for i in range(0, len(all_token_ids), block_size):
            input_ids = all_token_ids[i: i + block_size]
            if len(input_ids) == block_size:
                packed_ids.append(input_ids)
                input_text = tokenizer_.decode(input_ids)
                total_num += 1
                if len(tokenizer_.encode(input_text)) == block_size:
                    packed_texts.append(input_text)
                    drop_num += 1
    # print(f"Total examples: {total_num}, dropped num: {drop_num}, dropped rate: {1 - drop_num/total_num}")
    return {
        "text": packed_texts
    }

def dataset_prepare(args, tokenizer=None, num_of_sequences=1024, chars_per_token=3.6):
    if args.dataset_name == "trivia_qa":
        args.dataset_config_name = "unfiltered"
    elif args.dataset_name == "squad":

        args.dataset_config_name = None

    print(f"using {args.dataset_name} with config {args.dataset_config_name}")


    if args.dataset_name == "squad":

        train_dataset = datasets.load_dataset(
            args.dataset_name,
            split="train"
        )
        valid_dataset = datasets.load_dataset(
            args.dataset_name,
            split="validation"
        )
    else:

        train_dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{int((1-args.validation_split_percentage)*100)}%]"
        )
        valid_dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{int((1-args.validation_split_percentage)*100)}%:]",
        )

    if args.dataset_name == "trivia_qa":
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
    
    elif args.dataset_name == "squad":

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

                    rag_text = f"Context: {context}\n Use the above contexts to answer the question: {q}"
                    inputs.append(rag_text)
            
            return {"text": inputs, "question": questions}
        
        def process_squad_sft(examples):
        
            inputs = []
            questions = examples["question"]
            answers = examples["answers"]
            
            for q, answer in zip(questions, answers):
           
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
                    
                    answer_text = ""
                
                if answer_text:  
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
        create_folder(f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}")
        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}/train_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}/valid_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )

    return train_dataset, valid_dataset

def dataset_prepare1(args, tokenizer=None, num_of_sequences=1024, chars_per_token=3.6):
    # raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config_name)['train']
    # if "validation" in raw_datasets.keys():
    #     train_dataset = raw_datasets["train"]
    #     valid_dataset = raw_datasets["validation"]
    # else:
    if args.dataset_name == "trivia_qa":
        args.dataset_config_name = "unfiltered"
    print(f"using {args.dataset_name} with config {args.dataset_config_name}")
    train_dataset = datasets.load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=f"train[:{int((1-args.validation_split_percentage)*100)}%]"
    )
    valid_dataset = datasets.load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=f"train[{int((1-args.validation_split_percentage)*100)}%:]"
    )

# train_idxs = set(random.sample(range(len(raw_datasets)), int(len(raw_datasets) * (1 - args.validation_split_percentage))))
# valid_idxs = set(range(len(raw_datasets))) - train_idxs
# train_dataset = datasets.Dataset.from_dict(raw_datasets[train_idxs])
# valid_dataset = datasets.Dataset.from_dict(raw_datasets[valid_idxs])


    global text_column
    column = train_dataset.column_names
    if "text" in column:
        text_column = "text"
    elif "document" in column:
        text_column = "document"
    elif "content" in column:
        text_column = "content"

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
        create_folder(f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}")
        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}/train_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}/valid_dataset",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        return train_dataset, valid_dataset