import re
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import wandb
import os

def format_reward(completions, **kwargs):
    """Reward for proper thinking syntax <think>...</think><answer>...</answer>"""
    rewards = []
    for completion in completions:
        # Check if output follows format
        if re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", completion, re.DOTALL):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward(completions, gt_answers, **kwargs):
    """Reward for extracting the correct final answer."""
    rewards = []
    for completion, gt in zip(completions, gt_answers):
        # Extract the content inside <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match:
            prediction = match.group(1).strip()
            # Basic validation
            if prediction == gt:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def format_gsm8k(example):
    """Format GSM8K dataset into conversational prompt for GRPO."""
    # Extract ground truth from the solution
    ground_truth = example["answer"].split("####")[-1].strip()
    return {
        "prompt": [{"role": "user", "content": example["question"]}],
        "gt_answers": ground_truth
    }

def main():
    wandb.init(project="rlvr-qwen", name="phase1-grpo-training")

    max_seq_length = 1024
    
    # Load Qwen2.5-1.5B-Instruct using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=32,
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_gsm8k)
    
    training_args = GRPOConfig(
        output_dir="outputs/phase1",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        logging_steps=10,
        max_steps=200,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=1024,
        save_steps=50,
        report_to="wandb",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[accuracy_reward, format_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting GRPO phase 1 training...")
    # NOTE: "Aha moment" expected around step 70 where reward jumps!
    trainer.train()
    
    FastLanguageModel.save_lora(model, "outputs/phase1/final_lora")
    print("Training finished.")

if __name__ == "__main__":
    main()
