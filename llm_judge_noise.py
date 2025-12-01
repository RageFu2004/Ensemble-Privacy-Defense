import argparse
import torch
import torch.nn.functional as F
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os
from typing import List, Tuple, Dict, Any
import random

class LLMJudgerNoise:

    def __init__(self, target_model_path, base_model_path, judge_model_path, is_rag=False, device='cuda'):

        self.device = device
        self.is_rag = is_rag
        self.target_model_path = target_model_path
        self.base_model_path = base_model_path
        self.judge_model_path = judge_model_path
        
        self.noise_params = {
            'loss_diff_threshold': 0.1,
            'noise_strength': 0.6,   
            'gaussian_std': 0.2,        
            'token_noise_prob': 0.5,   
            'noise_strategy': 'gaussian',  #  'controlled', 'logits', 'simple', 'gaussian', 'laplace', 'temperature', 'confidence', 'ensemble', 'token_level'

            'temperature': 2.0,          
            'confidence_threshold': 0.8, 
            'ensemble_models': 5,       
            'dropout_rate': 0.1,
            
            # Token-level noise parameters (from paper)
            'sigma_base': 0.1,           # Base noise scale
            'lambda_amp': 1.0,          # Amplification factor
            'alpha': 0.5,                # Confidence decay parameter
            'beta': 1.0,                # Additional scaling factor
        }
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        use_same_base = (target_model_path == base_model_path) or is_rag

        if is_rag:
            print("RAG mode")
            from ragllama import RAGModel

            self.target_model = RAGModel(
                model_name=base_model_path,
                device=device
            )
            self.target_tokenizer = self.target_model.tokenizer

            self.base_model = self.target_model
            self.base_tokenizer = self.target_tokenizer
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map='auto'
            )
            self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
            self.base_model.eval()

            print("Finetune mode")
            if os.path.exists(os.path.join(target_model_path, 'adapter_config.json')):
                peft_config = PeftConfig.from_pretrained(target_model_path)
                self.target_model = PeftModel.from_pretrained(self.base_model, target_model_path)
                self.target_tokenizer = self.base_tokenizer
            else:
                if use_same_base:

                    print("Target and base model are the same, reusing base model")
                    self.target_model = self.base_model
                    self.target_tokenizer = self.base_tokenizer
                else:

                    self.target_model = AutoModelForCausalLM.from_pretrained(
                        target_model_path,
                        quantization_config=bnb_config,
                        device_map='auto'
                    )
                    self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_path, use_fast=True)
            self.target_model.eval()

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

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def generate(self, questions, max_new_tokens=256, num_beams=1, do_sample=True, labels=None):

        if isinstance(questions, str):
            questions = [questions]
        
        results = []
        losses = []
        logits_list = []
        input_ids_list = []
        

        detailed_info = []
        
        for i, question in enumerate(questions):
            #print(f"\n{'='*50}")
            #print(f"{i+1}: {question}")
            #print(f"{'='*50}")
        
            answer_target, loss_target, logits_target, token_losses_target, answer_loss_target = self._generate_with_token_loss(
                self.target_model, self.target_tokenizer, question, max_new_tokens
            )
            answer_target = self.truncate_text(answer_target, self.target_tokenizer, max_tokens=100)
            
            answer_base, loss_base, logits_base, token_losses_base, answer_loss_base = self._generate_with_token_loss(
                self.base_model, self.base_tokenizer, question, max_new_tokens
            )
            answer_base = self.truncate_text(answer_base, self.base_tokenizer, max_tokens=100)
            
            #print(f"\n【Base Model】:")
            #print(f"Prompt Loss: {loss_base:.4f}")
            #print(f"Answer Loss: {answer_loss_base:.4f}" if answer_loss_base is not None else "Answer Loss: N/A")
            #print(f"Token Losses: {token_losses_base[:10]}...") 
            
            similarity = self._calculate_answer_similarity(answer_target, answer_base)
            #print(f"\n: {similarity:.4f}")
            

            judge_prompt = self._build_enhanced_judge_prompt(
                question,
                answer_target,
                answer_base,
                loss_target,
                loss_base,
                answer_loss_target,
                answer_loss_base
            )
            

            judge_response, loss_judge, logits_judge = self._generate_answer(
                self.judge_model, self.judge_tokenizer, judge_prompt, max_new_tokens=128, calc_loss=True
            )
            

            final_answer = self._extract_final_answer_from_judge(judge_response)
            

            if answer_loss_target is not None and answer_loss_base is not None:
                loss_diff = abs(answer_loss_target - answer_loss_base)
            else:
                loss_diff = abs(loss_target - loss_base) if loss_target is not None and loss_base is not None else 0

            final_answer = judge_response.strip()

            final_answer_before_noise = final_answer
            final_answer = self._inject_noise_based_on_loss_diff(final_answer, loss_diff)
            

            final_answer = self._inject_noise_based_on_token_overlap(answer_target, final_answer)
            

            del answer_target, answer_base, judge_response
            
            results.append(final_answer)
            losses.append(loss_judge)
            logits_list.append(logits_judge)
            input_ids_list.append(self.judge_tokenizer(judge_prompt, return_tensors="pt").input_ids)
        

        
        return results, losses, logits_list, input_ids_list

    def _generate_with_token_loss(self, model, tokenizer, query, max_new_tokens=256):
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

    def _calculate_answer_similarity(self, answer1, answer2):

        tokens1 = set(answer1.lower().split())
        tokens2 = set(answer2.lower().split())
        
        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0

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

    def _parse_judge_scores(self, judge_response):

        scores = {}
        try:
            patterns = {
                'accuracy_a': r'Accuracy_A:\s*([0-9.]+)',
                'accuracy_b': r'Accuracy_B:\s*([0-9.]+)',
                'preference': r'Preference:\s*([AB])',
                'confidence': r'Confidence:\s*([0-9.]+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, judge_response)
                if match:
                    if key == 'preference':
                        scores[key] = match.group(1)
                    else:
                        scores[key] = float(match.group(1))
                else:
                    scores[key] = 0.5 if key != 'preference' else 'A'
                    
        except Exception as e:
            print(f"Error parsing judge scores: {e}")

            scores = {
                'accuracy_a': 0.5, 'accuracy_b': 0.5,
                'preference': 'A', 'confidence': 0.5
            }
        
        return scores

    def _extract_final_answer_from_judge(self, judge_response):

        lines = judge_response.strip().split('\n')

        answer_lines = []
        for line in lines:
            line = line.strip()

            if (line and 
                not line.startswith('Accuracy') and 
                not line.startswith('Privacy') and 
                not line.startswith('Preference') and 
                not line.startswith('Confidence') and
                not line.startswith('Score') and
                not line.startswith('Analysis') and
                not line.startswith('Based on') and
                not line.startswith('I will') and
                not line.startswith('Let me') and
                not line.startswith('The answer') and
                len(line) > 10): 
                answer_lines.append(line)
        
        if answer_lines:
            return answer_lines[-1]
        else:
            return lines[-1] if lines else judge_response.strip()

    def _inject_noise_based_on_loss_diff(self, answer, loss_diff):

        base_threshold = self.noise_params['loss_diff_threshold']
        
        if loss_diff > base_threshold:
            if loss_diff > 0.5:
                noise_strength = self.noise_params['noise_strength'] * 1.5
            elif loss_diff > 0.2:
                noise_strength = self.noise_params['noise_strength'] * 1.2
            else:
                noise_strength = self.noise_params['noise_strength'] * 0.8
            
            print(f"Loss difference ({loss_diff:.3f}) exceeds threshold ({base_threshold:.3f}), injecting noise with strength {noise_strength:.3f}...")
            return self._apply_noise_strategy(answer, noise_strength)
        else:
            print(f"Loss difference ({loss_diff:.3f}) below threshold ({base_threshold:.3f}), no noise injection.")
        return answer

    def _calculate_token_confidence(self, logits):
        """
        Calculate confidence for each token position using logit margin between top-2 predictions.
        confidence_i = logit_margin = top1_logit - top2_logit
        """
        # logits shape: [batch_size, seq_len, vocab_size]
        if len(logits.shape) == 3:
            logits = logits[0]  # Take first batch: [seq_len, vocab_size]
        
        # Get top-2 logits for each position
        top2_logits, top2_indices = torch.topk(logits, k=2, dim=-1)  # [seq_len, 2]
        
        # Calculate margin: top1 - top2
        confidence = top2_logits[:, 0] - top2_logits[:, 1]  # [seq_len]
        
        return confidence
    
    def _inject_token_level_noise(self, target_answer, final_answer):
        """
        Implement token-level noise injection as described in the paper.
        Inject calibrated Gaussian noise into logits for tokens that appear in both 
        target answer and judge-generated answer.
        
        Formula: z_hat_i = z_i + N(0, sigma_i^2 * I)
        where sigma_i = sigma_base * lambda_amp * beta * exp(-alpha * confidence_i)
        """
        # Encode both answers to get token IDs for comparison
        target_input_ids = self.target_tokenizer.encode(target_answer, add_special_tokens=False, return_tensors="pt").to(self.device)
        final_input_ids = self.target_tokenizer.encode(final_answer, add_special_tokens=False, return_tensors="pt").to(self.device)
        
        # Find overlapping token IDs (more accurate than string matching for subword tokenization)
        target_token_set = set(target_input_ids[0].cpu().tolist())
        final_token_list = final_input_ids[0].cpu().tolist()
        
        # Create mask for overlapping token positions in final answer
        overlap_mask = torch.tensor([token_id in target_token_set for token_id in final_token_list], 
                                   dtype=torch.bool, device=self.device)
        
        if not overlap_mask.any():
            print("No overlapping tokens found, skipping token-level noise injection.")
            return final_answer
        
        num_overlap = overlap_mask.sum().item()
        print(f"Token-level noise: Found {num_overlap} overlapping token positions out of {len(final_token_list)}")
        
        with torch.no_grad():
            # Get original logits
            outputs = self.target_model(final_input_ids)
            original_logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Calculate confidence for each token position
            confidence = self._calculate_token_confidence(original_logits)  # [seq_len]
            
            # Ensure confidence and overlap_mask have the same length as logits sequence length
            seq_len = original_logits.shape[1]
            if len(confidence) != seq_len:
                # Adjust confidence length to match (should be rare)
                if len(confidence) > seq_len:
                    confidence = confidence[:seq_len]
                else:
                    # Pad with mean confidence value
                    mean_conf = confidence.mean() if len(confidence) > 0 else 0.0
                    padding = torch.full((seq_len - len(confidence),), mean_conf, device=confidence.device)
                    confidence = torch.cat([padding, confidence])
            
            if len(overlap_mask) != seq_len:
                # Adjust overlap_mask length to match
                if len(overlap_mask) > seq_len:
                    overlap_mask = overlap_mask[:seq_len]
                else:
                    # Pad with False
                    padding = torch.zeros(seq_len - len(overlap_mask), dtype=torch.bool, device=overlap_mask.device)
                    overlap_mask = torch.cat([padding, overlap_mask])
            
            # Calculate adaptive noise scale for each position
            # sigma_i = sigma_base * lambda_amp * beta * exp(-alpha * confidence_i)
            sigma_base = self.noise_params['sigma_base']
            lambda_amp = self.noise_params['lambda_amp']
            beta = self.noise_params['beta']
            alpha = self.noise_params['alpha']
            
            # Expand confidence to match logits shape for broadcasting
            confidence_expanded = confidence.unsqueeze(-1)  # [seq_len, 1]
            
            # Calculate adaptive sigma for each position
            adaptive_sigma = sigma_base * lambda_amp * beta * torch.exp(-alpha * confidence_expanded)  # [seq_len, 1]
            
            # Generate Gaussian noise: N(0, sigma_i^2 * I)
            noise = torch.randn_like(original_logits[0]) * adaptive_sigma  # [seq_len, vocab_size]
            
            # Apply noise only to overlapping token positions
            noisy_logits = original_logits[0].clone()  # [seq_len, vocab_size]
            noisy_logits[overlap_mask] = noisy_logits[overlap_mask] + noise[overlap_mask]
            
            # Sample new tokens from noisy logits
            probs = torch.softmax(noisy_logits, dim=-1)
            new_input_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)
            
            # Decode the noisy answer
            noisy_answer = self.target_tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
            
            print(f"Token-level noise injected at {overlap_mask.sum().item()} positions")
            return noisy_answer
    
    def _inject_noise_based_on_token_overlap(self, target_answer, final_answer):
        """
        Inject noise based on token overlap between target and final answer.
   
        """
        # Check if token-level noise strategy is enabled
        if self.noise_params.get('noise_strategy') == 'token_level':
            return self._inject_token_level_noise(target_answer, final_answer)
        
        # Original text-level method
        target_tokens = set(target_answer.lower().split())
        final_tokens = set(final_answer.lower().split())
        overlap_tokens = target_tokens & final_tokens

        overlap_ratio = len(overlap_tokens) / max(len(target_tokens), len(final_tokens), 1)
        
        if len(overlap_tokens) > 5 or overlap_ratio > 0.1:  
            print(f"Detected {len(overlap_tokens)} overlapping tokens (ratio: {overlap_ratio:.3f}): {overlap_tokens}")

            if overlap_ratio > 0.3:
                noise_strength = self.noise_params['noise_strength'] * 1.3
            elif overlap_ratio > 0.15:
                noise_strength = self.noise_params['noise_strength'] * 1.1
            else:
                noise_strength = self.noise_params['noise_strength'] * 0.9
            
            return self._apply_noise_strategy(final_answer, noise_strength)
        else:
            print(f"Token overlap ({len(overlap_tokens)} tokens, ratio: {overlap_ratio:.3f}) below threshold, no noise injection.")
        
        return final_answer

    def _add_gaussian_noise_to_tokens(self, answer, noise_strength):

        input_ids = self.target_tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt")

        noise = torch.randn_like(input_ids.float()) * noise_strength
        
        noisy_input_ids = input_ids.float() + noise

        noisy_input_ids = torch.round(noisy_input_ids).long()

        vocab_size = self.target_tokenizer.vocab_size
        noisy_input_ids = torch.clamp(noisy_input_ids, 0, vocab_size - 1)

        noisy_answer = self.target_tokenizer.decode(noisy_input_ids[0], skip_special_tokens=True)
        
        return noisy_answer

    def _add_controlled_gaussian_noise(self, answer, noise_strength):

        input_ids = self.target_tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt")
        
        noise = torch.randn_like(input_ids.float()) * self.noise_params['gaussian_std']
        

        mask = torch.rand_like(input_ids.float()) < self.noise_params['token_noise_prob']
        noise = noise * mask.float()

        noisy_input_ids = input_ids.float() + noise

        noisy_input_ids = torch.round(noisy_input_ids).long()

        vocab_size = self.target_tokenizer.vocab_size
        noisy_input_ids = torch.clamp(noisy_input_ids, 0, vocab_size - 1)

        noisy_answer = self.target_tokenizer.decode(noisy_input_ids[0], skip_special_tokens=True)
        
        return noisy_answer

    def _add_logits_level_gaussian_noise(self, answer, noise_strength):

        input_ids = self.target_tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt").to(self.device)
        
        with torch.no_grad():

            outputs = self.target_model(input_ids)
            logits = outputs.logits
            

            noise = torch.randn_like(logits) * self.noise_params['gaussian_std']
            

            mask = torch.rand_like(logits) < self.noise_params['token_noise_prob']
            noisy_logits = logits + noise * mask.float()
            

            probs = torch.softmax(noisy_logits, dim=-1)
            new_input_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)
            

            noisy_answer = self.target_tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
            
            return noisy_answer


    
    def gaussian_noise_defense(self, logits, noise_std=None):

        if noise_std is None:
            noise_std = self.noise_params['gaussian_std']
        noise = torch.randn_like(logits) * noise_std
        return logits + noise

    def laplace_noise_defense(self, logits, epsilon=None):

        if epsilon is None:
            epsilon = self.noise_params['laplace_epsilon']
        sensitivity = 2.0  
        scale = sensitivity / epsilon
        noise = torch.distributions.Laplace(0, scale).sample(logits.shape).to(logits.device)
        return logits + noise

    def temperature_scaling_defense(self, logits, temperature=None):

        if temperature is None:
            temperature = self.noise_params['temperature']
        return logits / temperature

    def confidence_threshold_defense(self, logits, threshold=None):

        if threshold is None:
            threshold = self.noise_params['confidence_threshold']
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)

        vocab_size = logits.shape[-1]
        uniform_logits = torch.zeros_like(logits)
        mask = (max_probs < threshold).unsqueeze(-1)
        
        return torch.where(mask, uniform_logits, logits)

    def ensemble_dropout_defense(self, model, input_ids, n_models=None, dropout_rate=None):

        if n_models is None:
            n_models = self.noise_params['ensemble_models']
        if dropout_rate is None:
            dropout_rate = self.noise_params['dropout_rate']
            
        predictions = []
        model.train() 
        
        for _ in range(n_models):
            with torch.no_grad():
                outputs = model(input_ids)
                predictions.append(outputs.logits)
        
        model.eval()
        return torch.mean(torch.stack(predictions), dim=0)

    def apply_enhanced_defense(self, logits, model=None, input_ids=None, noise_strength=1.0):
        strategy = self.noise_params.get('noise_strategy', 'logits')
        
        if strategy == 'gaussian':
            noise_std = self.noise_params['gaussian_std'] * noise_strength

            return self.gaussian_noise_defense(logits, noise_std)
        
        elif strategy == 'laplace':
            epsilon = self.noise_params['laplace_epsilon'] / noise_strength

            return self.laplace_noise_defense(logits, epsilon)
        
        elif strategy == 'temperature':
            temperature = self.noise_params['temperature'] * noise_strength

            return self.temperature_scaling_defense(logits, temperature)
        
        elif strategy == 'confidence':
            threshold = self.noise_params['confidence_threshold']

            return self.confidence_threshold_defense(logits, threshold)
        
        elif strategy == 'ensemble':
            if model is not None and input_ids is not None:
                n_models = self.noise_params['ensemble_models']
                dropout_rate = self.noise_params['dropout_rate']
                return self.ensemble_dropout_defense(model, input_ids, n_models, dropout_rate)
            else:

                return self.gaussian_noise_defense(logits, self.noise_params['gaussian_std'] * noise_strength)
        
        else:
            return logits

    def _apply_noise_strategy(self, answer, noise_strength):
        strategy = self.noise_params.get('noise_strategy', 'controlled')
        
        if strategy in ['gaussian', 'laplace', 'temperature', 'confidence', 'ensemble']:
            return self._apply_enhanced_logits_defense(answer, noise_strength)
        elif strategy == 'logits':
            return self._add_logits_level_gaussian_noise(answer, noise_strength)
        elif strategy == 'simple':
            return self._add_gaussian_noise_to_tokens(answer, noise_strength)
        else:  # default: controlled
            return self._add_controlled_gaussian_noise(answer, noise_strength)

    def _apply_enhanced_logits_defense(self, answer, noise_strength):

        input_ids = self.target_tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt").to(self.device)
        
        with torch.no_grad():

            outputs = self.target_model(input_ids)
            original_logits = outputs.logits
            

            defended_logits = self.apply_enhanced_defense(
                original_logits, 
                model=self.target_model, 
                input_ids=input_ids, 
                noise_strength=noise_strength
            )
            
            probs = torch.softmax(defended_logits, dim=-1)
            new_input_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)
            
            defended_answer = self.target_tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
            
            return defended_answer


    def _generate_answer(self, model, tokenizer, query, max_new_tokens=256, calc_loss=False):
        if hasattr(model, 'generate') and 'questions' in model.generate.__code__.co_varnames:
            results, losses, logits_list, input_ids_list = model.generate([query], max_new_tokens=max_new_tokens)
            answer = results[0]
            loss = losses[0] if losses is not None else None
            logits = logits_list[0] if logits_list is not None else None
            return answer, loss, logits
        else:
            if not calc_loss:
                prompt = f"{query}\nAnswer:"
            else:
                prompt = query
            
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(output[0], skip_special_tokens=True)
                
                loss = None
                logits = None
                
                if calc_loss:
                    labels = inputs['input_ids']
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss.item() if hasattr(outputs, 'loss') else None
                    logits = outputs.logits.detach().cpu() if hasattr(outputs, 'logits') else None
                
                return answer, loss, logits

    def __call__(self, input_ids=None, **kwargs):
        if input_ids is None:
            raise ValueError("LLMJudgerNoise: input_ids must be provided.")
        
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

    def set_defense_parameters(self, noise_strategy=None, **kwargs):
        old_strategy = self.noise_params['noise_strategy']
        if noise_strategy is not None:
            self.noise_params['noise_strategy'] = noise_strategy
            print(f"Noise strategy changed: {old_strategy} -> {noise_strategy}")
        
        if kwargs:
            print(f"Updating defense parameters: {kwargs}")
        self.noise_params.update(kwargs)

    def get_defense_info(self):
        return {
            'noise_strategy': self.noise_params['noise_strategy'],
            'noise_parameters': self.noise_params.copy()
        }

    def reset_defense_to_default(self):
        self.noise_params = {
            'loss_diff_threshold': 0.1,
            'noise_strength': 0.6,
            'gaussian_std': 0.2,
            'token_noise_prob': 0.5,
            'noise_strategy': 'logits',
            'laplace_epsilon': 1.0,
            'temperature': 2.0,
            'confidence_threshold': 0.8,
            'ensemble_models': 5,
            'dropout_rate': 0.1,
            # Token-level noise parameters
            'sigma_base': 0.1,
            'lambda_amp': 1.0,
            'alpha': 0.5,
            'beta': 1.0,
        }

