import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os



class LLMJudger:
    """
    LLMJudger: A MIA defense wrapper class compatible with RAG and finetuned LLMs, supporting judge functionality.
    Method interfaces are compatible with ragllama.RAGModel, including __init__, generate, __call__, etc.
    """
    def __init__(self, target_model_path, base_model_path, judge_model_path, is_rag=False, device='cuda'):
        """
        Initialize LLMJudger.
        target_model_path: Path to finetuned model or RAG flag
        base_model_path: Path to base model
        judge_model_path: Path to judge LLM
        is_rag: Whether in RAG mode
        device: Device to use
        """
        self.device = device
        self.is_rag = is_rag
        self.target_model_path = target_model_path
        self.base_model_path = base_model_path
        self.judge_model_path = judge_model_path
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load base model and tokenizer (for RAG to reuse directly, avoiding duplicate loading)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map='auto'
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)

        # Load target model
        if is_rag:
            print("RAG mode")
            from ragllama import RAGModel
            # Reuse already loaded base model and tokenizer to avoid duplicate from_pretrained in RAG
            self.target_model = RAGModel(
                model_name=base_model_path,
                device=device,
                preloaded_model=self.base_model,
                preloaded_tokenizer=self.base_tokenizer
            )
            self.target_tokenizer = self.target_model.tokenizer
        else:
            print("Finetune mode")
            if os.path.exists(os.path.join(target_model_path, 'adapter_config.json')):
                peft_config = PeftConfig.from_pretrained(target_model_path)
                # Use target model's backbone (read from adapter_config) instead of the passed base_model
                target_base_model_path = peft_config.base_model_name_or_path
                print(f"Loading target model's backbone: {target_base_model_path}")
                # Load target model's backbone
                target_base_model = AutoModelForCausalLM.from_pretrained(
                    target_base_model_path,
                    quantization_config=bnb_config,
                    device_map='auto'
                )
                # Load PEFT adapter using the correct backbone
                self.target_model = PeftModel.from_pretrained(target_base_model, target_model_path)
                self.target_tokenizer = AutoTokenizer.from_pretrained(target_base_model_path, use_fast=True)
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    target_model_path,
                    quantization_config=bnb_config,
                    device_map='auto'
                )
                self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_path, use_fast=True)
            self.target_model.eval()

        self.base_model.eval()

        # Load judge model
        self.judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_path,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True
        )
        self.judge_tokenizer = AutoTokenizer.from_pretrained(
            judge_model_path, use_fast=True, trust_remote_code=True
        )
        self.judge_model.eval()

    def truncate_text(self, text, tokenizer, max_tokens=50):
        """
        Truncate text to maximum token count to prevent answers from being too long.
        """
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def generate(self, questions, max_new_tokens=256, num_beams=1, do_sample=True, labels=None):
        """
        Compatible with ragllama.RAGModel's generate method.
        Input questions (list), generate answers using target and base models respectively, then combine with judge.
        Returns:
            results: List of judge's final answers
            losses: List of judge losses
            logits_list: List of judge logits
            input_ids_list: List of input token ids
        """
        if isinstance(questions, str):
            questions = [questions]
        results = []
        losses = []
        logits_list = []
        input_ids_list = []
        for question in questions:
            # 1. Generate with target model and calculate loss
            answer_target, loss_target, logits_target, _, answer_loss_target = self._generate_with_token_loss(
                self.target_model, self.target_tokenizer, question, max_new_tokens
            )
            answer_target = self.truncate_text(answer_target, self.target_tokenizer, max_tokens=100)
            
            # 2. Generate with base model and calculate loss
            answer_base, loss_base, logits_base, _, answer_loss_base = self._generate_with_token_loss(
                self.base_model, self.base_tokenizer, question, max_new_tokens
            )
            answer_base = self.truncate_text(answer_base, self.base_tokenizer, max_tokens=100)
            
            # 3. Build enhanced judge prompt
            judge_prompt = self._build_enhanced_judge_prompt(
                question,
                answer_target,
                answer_base,
                loss_target,
                loss_base,
                answer_loss_target,
                answer_loss_base
            )
            
            # 4. Generate with judge model and calculate loss/logits
            answer_judge, loss_judge, logits_judge = self._generate_answer(self.judge_model, self.judge_tokenizer, judge_prompt, self.device, max_new_tokens=128, calc_loss=True)
            
            # 5. Aggregate results
            del answer_target, answer_base
            results.append(answer_judge)
            losses.append(loss_judge)
            logits_list.append(logits_judge)
            input_ids_list.append(self.judge_tokenizer(judge_prompt, return_tensors="pt").input_ids)
        return results, losses, logits_list, input_ids_list

    def _generate_answer(self, model, tokenizer, query, device, max_new_tokens=256, calc_loss=False):
        #print("query:", query)
        # If it's a RAGModel
        if hasattr(model, 'generate') and 'questions' in model.generate.__code__.co_varnames:
            print("RAGModel gen")
            results, losses, logits_list, input_ids_list = model.generate([query], max_new_tokens=max_new_tokens)
            answer = results[0]
            #print("answer:", answer)
            loss = losses[0] if losses is not None else None
            logits = logits_list[0] if logits_list is not None else None
            return answer, loss, logits
        else:
            # HuggingFace model
            if not calc_loss:
                prompt = f"{query}\nAnswer:"
            else:
                prompt = query
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens,eos_token_id=tokenizer.eos_token_id )
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            loss = None
            logits = None
            #print("answer:", answer)
            if calc_loss:
                with torch.no_grad():
                    labels = inputs['input_ids']
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss.item() if hasattr(outputs, 'loss') else None
                    logits = outputs.logits.detach().cpu() if hasattr(outputs, 'logits') else None
            return answer, loss, logits 

    def _generate_with_token_loss(self, model, tokenizer, query, max_new_tokens=256):
        # Handle RAGModel case
        if hasattr(model, 'generate') and 'questions' in model.generate.__code__.co_varnames:
            results, losses, logits_list, input_ids_list = model.generate([query], max_new_tokens=max_new_tokens)
            answer = results[0]
            loss = losses[0] if losses is not None else None
            logits = logits_list[0] if logits_list is not None else None
            token_losses = []
            answer_loss = loss
            return answer, loss, logits, token_losses, answer_loss
        
        # Handle HuggingFace model case
        prompt = f"{query}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(output[0], skip_special_tokens=True)

            labels = inputs['input_ids'].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item() if hasattr(outputs, 'loss') else None
            logits = outputs.logits.detach().cpu() if hasattr(outputs, 'logits') else None
            
            token_losses = self._calculate_token_level_loss(outputs.logits, labels)
  
            generated_tokens = output[0][inputs['input_ids'].size(1):]
            if len(generated_tokens) > 0:
                answer_loss = self._calculate_generated_answer_loss(model, inputs, generated_tokens)
            else:
                answer_loss = loss
        
        return answer, loss, logits, token_losses, answer_loss

    def _calculate_token_level_loss(self, logits, labels):
        log_probs = torch.log_softmax(logits, dim=-1)
        token_losses = []
        
        for i in range(logits.size(1)):
            if i < labels.size(1):
                target_token = labels[0, i]
                token_loss = -log_probs[0, i, target_token].item()
                token_losses.append(token_loss)
        
        return token_losses

    def _calculate_generated_answer_loss(self, model, inputs, generated_tokens):
        try:
            full_input_ids = torch.cat([inputs['input_ids'], generated_tokens.unsqueeze(0)], dim=1)

            with torch.no_grad():
                outputs = model(input_ids=full_input_ids)
                logits = outputs.logits

                generated_logits = logits[0, inputs['input_ids'].size(1)-1:-1]
                generated_labels = generated_tokens
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                answer_loss = loss_fct(generated_logits, generated_labels).item()
                
            return answer_loss
        except Exception as e:
            return None

    def _build_enhanced_judge_prompt(self, query, answer_target, answer_base, loss_target=None, loss_base=None, answer_loss_target=None, answer_loss_base=None):
        prompt = (
            "You are a judge that needs to generate a final answer based on two candidate answers.\n"
            "Consider both PRIVACY (low-loss answers that might leak training data) and ACCURACY (being factually correct).\n\n"
            f"Question: {query}\n"
            f"Answer A (target): {answer_target}\n"
            f"Answer B (base): {answer_base}\n\n"
            "Model confidence (lower loss indicates higher confidence):\n"
            f" - Target answer loss: {answer_loss_target}\n"
            f" - Base   answer loss: {answer_loss_base}\n\n"
            "Your task:\n"
            "1. Weigh privacy and factual accuracy.\n"
            "2. Generate a new final answer that:\n"
            "   - Maintains accuracy while protecting privacy\n"
            "   - Uses general language instead of specific details\n"
            "   - Avoids exact quotes or precise measurements\n"
            "   - Combines the best aspects of both answers\n\n"
            "Generate your final answer:"
        )
        return prompt

    def __call__(self, input_ids=None, **kwargs):
        """
        Compatible with ragllama.RAGModel's __call__ method.
        input_ids: tensor, decode to text then go through generate process.
        """
        if input_ids is None:
            raise ValueError("LLMJudger: input_ids must be provided.")
        if isinstance(input_ids, torch.Tensor):
            questions = self.target_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        else:
            questions = input_ids
        results, losses, logits_list, input_ids_list = self.generate(questions)
        class Output:
            def __init__(self, generated_texts, loss, logits, input_ids_list):
                self.generated_texts = generated_texts
                self.loss = loss
                self.logits = logits
                self.input_ids_list = input_ids_list
            def __getitem__(self, idx):
                res = (self.loss, self.logits, self.input_ids_list)
                return res[idx]
        return Output(results, losses, logits_list, input_ids_list)

# Usage example:
# judger = LLMJudger(target_model_path, base_model_path, judge_model_path, is_rag=False, device='cuda')
# results, losses, logits_list, input_ids_list = judger.generate(["your question"])
# output = judger(torch.tensor([[...]]))


