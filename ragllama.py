# model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from rag_retriever import *
from rag_retriever import retrieve_contexts

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class RAGModel:
    def __init__(
        self,
        model_name="meta-llama/Llama-2-7b-hf",
        cache_dir="./cache",
        device="cuda",
        top_k_contexts=5,
        preloaded_model=None,
        preloaded_tokenizer=None,
    ):
        # Support preloaded model and tokenizer to avoid duplicate loading
        if preloaded_model is not None and preloaded_tokenizer is not None:
            print("Using preloaded model and tokenizer")
            self.model = preloaded_model
            self.tokenizer = preloaded_tokenizer
        else:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)

        self.model.eval()
        self.device = device
        self.top_k_contexts = top_k_contexts

        # Load trivia_qa dataset for retrieval
        print("Loading TriviaQA dataset...")
        dataset = load_dataset("trivia_qa", "unfiltered", cache_dir=cache_dir)
        self.trivia_data = dataset["train"]  # Can be changed to ["validation"] if needed
        print("TriviaQA loaded with {} questions.".format(len(self.trivia_data)))
        if self.tokenizer.pad_token is None:
            print("[INFO] tokenizer.pad_token is None, setting to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _find_matching_example(self, clipped_question_text):
   
        for ex in self.trivia_data:
            full_question = ex.get("question", "").strip()
            if full_question.startswith(clipped_question_text):
                return ex
        return False


    def _build_rag_prompt(self, question, search_results):
        """Build RAG template prompt"""
        descriptions = search_results.get("description", [])
        selected_descriptions = descriptions if len(descriptions) <= self.top_k_contexts else descriptions[:self.top_k_contexts]

        context = " ".join([desc.strip() for desc in selected_descriptions if desc.strip()])

        prompt = f"Context: {context}\n Use the above contexts to answer the question: {question.strip()}"
                  
        return prompt

    def generate(self, questions, max_new_tokens=128, num_beams=1, do_sample=True, labels=None):
        results = []
        losses = []
        logits_list = []
        input_ids_list = []
        for idx, question_text in enumerate(questions):
            #print("question_text:", question_text)
            
            example = self._find_matching_example(question_text)
            
            if not example:
                # No matching example found, use RAG retrieval
                topk_contexts = retrieve_contexts(question_text,model=embedding_model, top_k=5)
                context = " ".join([c.strip() for c in topk_contexts if c.strip()])
                rag_prompt = f"Context: {context}\n Use the above contexts to answer the question: {question_text.strip()}"
            else:
                rag_prompt = self._build_rag_prompt(question_text, example["search_results"])

            #print("rag_prompt:", rag_prompt)
            prompt_inputs = self.tokenizer(rag_prompt, return_tensors="pt", padding=True).to(self.device)
            input_ids_list.append(prompt_inputs.input_ids.detach().cpu())
            prompt_length = prompt_inputs.input_ids.shape[1]  # Record prompt token count
            gen_tokens = self.model.generate(
                prompt_inputs.input_ids,
                num_beams=num_beams,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
            )
            # Only decode newly generated tokens
            gen_text = self.tokenizer.decode(gen_tokens[0][prompt_length:], skip_special_tokens=True)
            #print("results:", gen_text)
            results.append(gen_text)
            # If labels is not None, calculate loss
            if labels is not None:
                label = labels[idx] if isinstance(labels, (list, tuple)) else labels
                prompt_len = prompt_inputs.input_ids.shape[1]
                if hasattr(label, 'ndim') and label.ndim == 2:
                    answer_len = label.shape[1]
                    label = label[0]
                else:
                    answer_len = label.shape[0]
                # Pad front with -100, then concatenate real label at the end
                new_labels = torch.full((1, prompt_len), -100, dtype=label.dtype, device=label.device)
                new_labels[0, -answer_len:] = label
                outputs = self.model(input_ids=prompt_inputs.input_ids, labels=new_labels)
                losses.append(outputs.loss.detach().cpu() if hasattr(outputs, "loss") else torch.tensor(float('nan')))
                logits_list.append(outputs.logits.detach().cpu() if hasattr(outputs, "logits") else None)
            else:
                losses.append(torch.tensor(float('nan')))
                logits_list.append(None)
        # If labels is not None, return text and loss (loss is a tensor)
        if labels is not None:
            # Fill invalid losses with nan to ensure all are tensors
            losses = [l if isinstance(l, torch.Tensor) else torch.tensor(float('nan')) for l in losses]
            losses = torch.stack(losses).view(-1)  # Ensure it's a 1D tensor
            return results, losses, logits_list, input_ids_list
        else:
            return results, None, logits_list, input_ids_list

    def __call__(self, input_ids=None, **kwargs):
        if input_ids is None:
            raise ValueError("RAGModel: input_ids must be provided.")
        #print("input_ids shape:\n", input_ids.shape)
        # If input_ids is (64,1), transpose to (1,64)
        if input_ids.ndim == 2 and input_ids.shape[0] != 1 and input_ids.shape[1] == 1:
            input_ids = input_ids.T
        input_ids = input_ids.to(self.device)
        questions = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        allowed_keys = {"max_new_tokens", "num_beams", "do_sample", "labels"}
        generate_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        print("questions------------:", questions)
        if "labels" in generate_kwargs:
            results, losses, logits_list, input_ids_list = self.generate(questions, **generate_kwargs)
        else:
            results, _, logits_list, input_ids_list = self.generate(questions, **generate_kwargs)
            losses = None
        class Output:
            def __init__(self, generated_texts, loss, logits, input_ids_list):
                self.generated_texts = generated_texts
                self.loss = loss
                self.logits = logits
                self.input_ids_list = input_ids_list
            def __getitem__(self, idx):
                #print("logits-----------------:", self.logits.shape)
                # Compatible with outputs[:2] syntax
                res= (self.loss, self.logits, self.input_ids_list)
                return res[idx]
        return Output(results, losses, logits_list[0], input_ids_list[0]) 