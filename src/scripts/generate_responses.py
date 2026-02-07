# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from datasets import load_dataset, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


@dataclass
class GenerationScriptArguments:
    """
    Script arguments for response generation.
    """
    model_name_or_path: str = field(
        metadata={"help": "Model checkpoint path"}
    )
    dataset_name: str = field(
        metadata={"help": "Dataset path"}
    )
    output_file: str = field(
        default="generated_responses.json",
        metadata={"help": "Output JSON file path"}
    )
    step_output_dir: str = field(
        default="./steps",
        metadata={"help": "Directory to save step outputs"}
    )
    num_generations: int = field(
        default=5,
        metadata={"help": "Number of responses to generate per prompt"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU"}
    )
    max_new_tokens: int = field(
        default=768,
        metadata={"help": "Maximum new tokens to generate"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature for sampling"}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p for sampling"}
    )
    max_pixels: int = field(
        default=401408,
        metadata={"help": "Maximum number of pixels for the image"}
    )
    min_pixels: int = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"}
    )
    use_accelerate: bool = field(
        default=True,
        metadata={"help": "Use accelerate for multi-GPU"}
    )


def accuracy_reward(completions, solution, **kwargs):
    """Calculate accuracy reward based on answer extraction and comparison."""
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)

    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    
    question_type = kwargs['problem_type']
    
    rewards = []
    for content in completions:
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(solution)
            
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                else:
                    rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                    rel_diff = min(1.0, max(0.0, rel_diff))
                    reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards


def format_reward(completions):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    rewards = []
    for content in completions:
        match = re.fullmatch(pattern, content, re.DOTALL)
        rewards.append(1.0 if match else 0.0)
    return rewards


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return batch


def process_batch(model, processor, batch, args, accelerator):
    """Process a batch of examples and generate responses"""
    
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )
    
    TYPE_TEMPLATE = {
        "multiple choice": " Please provide the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }
    
    batch_results = []
    
    for example in batch:
        # Prepare question text
        if example.get('problem_type') == 'multiple choice' and 'options' in example:
            question = example['problem'] + "\nOptions:\n"
            for op in example['options']:
                question += op + "\n"
        else:
            question = example['problem']
        
        # Prepare the full prompt text
        prompt_text = QUESTION_TEMPLATE.format(Question=question)
        if example.get('problem_type'):
            prompt_text += TYPE_TEMPLATE.get(example['problem_type'], "")
        
        # Prepare messages for the processor
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": example.get('data_type', 'image'),
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]
        
        try:
            # Process the input
            if 'path' in example:
                media_path = example['path']
                messages[0]['content'][0][example.get('data_type', 'image')] = media_path
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                if example.get('data_type') == 'video':
                    image_inputs, video_inputs = process_vision_info(messages)
                else:
                    image_inputs = process_vision_info(messages)[0]
                    video_inputs = None
                
                inputs = processor(
                    text=[text],
                    images=image_inputs if example.get('data_type') == 'image' else None,
                    videos=video_inputs if example.get('data_type') == 'video' else None,
                    padding=True,
                    return_tensors="pt",
                ).to(accelerator.device)
            else:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                ).to(accelerator.device)
            
            # Generate multiple responses
            generated_responses = []
            
            for gen_idx in range(args.num_generations):
                with torch.no_grad():
                    # Use .module to access the underlying model if wrapped by DDP
                    model_to_use = model.module if hasattr(model, 'module') else model
                    generated_ids = model_to_use.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True,
                    )
                
                # Decode the generated text
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                generated_responses.append(output_text)
            
            # Calculate rewards
            format_rewards = format_reward(generated_responses)
            
            if 'solution' in example:
                acc_rewards = accuracy_reward(
                    generated_responses, 
                    example['solution'], 
                    problem_type=example.get('problem_type', 'free-form')
                )
            else:
                acc_rewards = [0.0] * len(generated_responses)
            
            # Store results
            result = {
                'problem_id': example['problem_id'],
                'problem': example['problem'],
                'problem_type': example.get('problem_type', 'unknown'),
                'solution': example.get('solution', ''),
                'generated_responses': generated_responses,
                'format_rewards': format_rewards,
                'accuracy_rewards': acc_rewards,
                'data_type': example.get('data_type', 'unknown'),
                'path': example.get('path', '')
            }
            
            if 'options' in example:
                result['options'] = example['options']
            
            batch_results.append(result)
            
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
    
    return batch_results


def main(args):
    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator()
    
    # Load model and processor
    if accelerator.is_main_process:
        print(f"Loading model from {args.model_name_or_path}")
        os.makedirs(args.step_output_dir, exist_ok=True)
    
    # Auto-detect model type
    if "Qwen2.5-VL" in args.model_name_or_path or "Qwen2_5_VL" in args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": accelerator.device}
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": accelerator.device}
        )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    
    # Set image processor parameters
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = args.max_pixels
        processor.image_processor.min_pixels = args.min_pixels
    
    # Wrap model with accelerator
    model = accelerator.prepare(model)
    
    # Load dataset
    if accelerator.is_main_process:
        print(f"Loading dataset from {args.dataset_name}")
    
    if args.dataset_name.endswith('.json') or args.dataset_name.endswith('.jsonl'):
        dataset = Dataset.from_json(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name)
        if isinstance(dataset, DatasetDict):
            dataset = dataset['train']
    
    # Create DataLoader with DistributedSampler for multi-GPU
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues with vision data
    )
    
    # Prepare dataloader with accelerator
    dataloader = accelerator.prepare(dataloader)
    
    results = []
    
    # Process batches
    progress_bar = None
    if accelerator.is_main_process:
        progress_bar = tqdm(total=len(dataloader), desc="Generating responses")
    
    for batch_idx, batch in enumerate(dataloader):
        batch_results = process_batch(model, processor, batch, args, accelerator)
        
        # Gather results from all processes
        gathered_results = accelerator.gather_for_metrics(batch_results)
        
        if accelerator.is_main_process:
            results.extend(gathered_results)
            progress_bar.update(1)
            
            # Save intermediate results periodically
            base_name = os.path.basename(args.output_file).replace('.json', '')
            step_output_file = os.path.join(args.step_output_dir, f'{base_name}_step_{batch_idx + 1}.json')
            
            with open(step_output_file, 'w', encoding='utf-8') as f:
                json.dump(gathered_results, f, ensure_ascii=False, indent=2)
    
    if progress_bar:
        progress_bar.close()
    
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    
    # Save final results (only on main process)
    if accelerator.is_main_process:
        print(f"\nSaving final results to {args.output_file}")
        
        # Add indices to results
        for idx, result in enumerate(results):
            result['index'] = idx
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Print summary statistics
        total_samples = len(results)
        total_responses = sum(len(r['generated_responses']) for r in results)
        avg_format_reward = sum(sum(r['format_rewards']) for r in results) / total_responses if total_responses > 0 else 0
        avg_acc_reward = sum(sum(r['accuracy_rewards']) for r in results) / total_responses if total_responses > 0 else 0
        
        print("\n" + "="*50)
        print("GENERATION SUMMARY")
        print("="*50)
        print(f"Total samples processed: {total_samples}")
        print(f"Total responses generated: {total_responses}")
        print(f"Average format reward: {avg_format_reward:.4f}")
        print(f"Average accuracy reward: {avg_acc_reward:.4f}")
        print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate responses and calculate rewards with multi-GPU support")
    parser.add_argument("--model_name_or_path", type=str, required=True, 
                       help="Path to the model")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Path to the dataset")
    parser.add_argument("--output_file", type=str, default="generated_responses.json",
                       help="Output JSON file path")
    parser.add_argument("--step_output_dir", type=str, default="./steps",
                       help="Output JSON file path")
    parser.add_argument("--num_generations", type=int, default=5,
                       help="Number of responses to generate per prompt")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per GPU")
    parser.add_argument("--max_new_tokens", type=int, default=768,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p for sampling")
    parser.add_argument("--max_pixels", type=int, default=401408,
                       help="Maximum number of pixels")
    parser.add_argument("--min_pixels", type=int, default=3136,
                       help="Minimum number of pixels")
    parser.add_argument("--use_accelerate", type=bool, default=True,
                       help="Use accelerate for multi-GPU")
    
    args = parser.parse_args()
    main(args)