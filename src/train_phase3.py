import os, re, sys, io, torch, wandb, glob
from unsloth import FastLanguageModel
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# ── Safe distributed setup ──────────────────────────────────
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_main = (local_rank == 0)

if world_size > 1:
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
else:
    torch.cuda.set_device(0)

if is_main:
    print(f"Running with {world_size} GPU(s)")

# ── Resume or start fresh ───────────────────────────────────
checkpoints = sorted(glob.glob("/kaggle/working/phase3_step*"))
if checkpoints:
    MODEL_PATH = checkpoints[-1]
    if is_main: print(f"Resuming from {MODEL_PATH}")
else:
    MODEL_PATH = "/kaggle/input/notebooks/pasha1701/phase1-rl-reasioning-training/grpo_gsm8k/checkpoint-500"
    if is_main: print(f"Starting fresh from {MODEL_PATH}")

# ── Model ───────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto",
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)
if is_main: print("✅ Model loaded!")

# ── System prompt ───────────────────────────────────────────
SYSTEM_PROMPT = """You are an advanced math reasoning assistant.
Think step-by-step inside <think> tags.
For complex calculations, write Python inside ```python``` blocks and use print().
The system will execute your code and return the output.
Once you have the answer, output it inside <answer> tags and stop."""

# ── Tool ────────────────────────────────────────────────────
def safe_execute(code: str) -> str:
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue().strip()
        sys.stdout = old_stdout
        return output if output else "No output printed."
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {e}"

# ── Reward ──────────────────────────────────────────────────
def score_trajectory(model_turns: list, ground_truth: str) -> float:
    reward = 0.0
    reward -= (len(model_turns) - 1) * 0.2
    first_exec_rewarded = False

    for i, turn in enumerate(model_turns):
        is_final = (i == len(model_turns) - 1)

        if i == 0 and "<think>" in turn and "</think>" in turn:
            reward += 0.3

        code_match = re.search(r'```python(.*?)```', turn, re.DOTALL)
        if code_match and not first_exec_rewarded:
            result = safe_execute(code_match.group(1).strip())
            if ground_truth in result:
                reward += 1.5
                first_exec_rewarded = True
            elif "Error" in result:
                reward -= 0.3

        if is_final and "<answer>" in turn:
            reward += 0.5
            ans_match = re.search(r'<answer>(.*?)</answer>', turn, re.DOTALL)
            if ans_match:
                nums = re.findall(r'\d+\.?\d*', ans_match.group(1))
                if nums and nums[-1].strip() == ground_truth:
                    reward += 2.0

    return reward

# ── Trajectory collection ───────────────────────────────────
def generate_turn(messages: list) -> str:
    FastLanguageModel.for_inference(model)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text += "<think>\n"
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return "<think>\n" + tokenizer.decode(new_tokens, skip_special_tokens=True)


def collect_trajectory(problem: str, ground_truth: str, max_turns=2):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem}
    ]
    model_turns = []

    for _ in range(max_turns):
        output = generate_turn(messages)
        model_turns.append(output)
        messages.append({"role": "assistant", "content": output})

        code_match = re.search(r'```python(.*?)```', output, re.DOTALL)
        if code_match:
            result = safe_execute(code_match.group(1).strip())
            messages.append({
                "role": "user",
                "content": f"Terminal Output: {result}\nNow give your final answer in <answer> tags."
            })

        if "<answer>" in output:
            break

    reward = score_trajectory(model_turns, ground_truth)
    return messages, reward


def collect_group(problem: str, ground_truth: str, group_size=4):
    all_messages, rewards = [], []

    for _ in range(group_size):
        msgs, reward = collect_trajectory(problem, ground_truth)
        all_messages.append(msgs)
        rewards.append(reward)

    mean_r = sum(rewards) / len(rewards)
    std_r  = (sum((r - mean_r)**2 for r in rewards) / len(rewards))**0.5 + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]

    return all_messages, advantages, rewards

# ── GRPO loss ───────────────────────────────────────────────
def compute_grpo_loss(messages_group: list, advantages: list):
    FastLanguageModel.for_training(model)
    total_loss = torch.tensor(0.0, requires_grad=True).to("cuda")
    valid = 0

    for messages, adv in zip(messages_group, advantages):
        if abs(adv) < 1e-6:
            continue
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=2048
        ).to("cuda")
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["input_ids"]
        )
        log_prob = -outputs.loss
        total_loss = total_loss + (-adv * log_prob)
        valid += 1

    return total_loss / valid if valid > 0 else total_loss

# ── Training loop ───────────────────────────────────────────
def train():
    if is_main:
        wandb.login()
        wandb.init(project="my-rlvr-agent", name="phase3-multiturn-grpo")

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.shuffle(seed=42)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    NUM_STEPS  = 80
    GROUP_SIZE = 4

    if is_main:
        print(f"\n🚀 Phase 3 | Steps: {NUM_STEPS} | Group: {GROUP_SIZE}\n")

    running_rewards = []

    for step, example in enumerate(dataset):
        if step >= NUM_STEPS:
            break

        problem = example["question"]
        gt      = example["answer"].split("####")[-1].strip().replace(",", "")

        try:
            messages_group, advantages, rewards = collect_group(
                problem, gt, GROUP_SIZE
            )
        except Exception as e:
            if is_main: print(f"⚠️  Step {step} failed: {e}")
            continue

        mean_reward = sum(rewards) / len(rewards)
        running_rewards.append(mean_reward)

        optimizer.zero_grad()
        loss = compute_grpo_loss(messages_group, advantages)
        if loss.requires_grad:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Save every 10 steps
        if is_main and (step + 1) % 10 == 0:
            avg = sum(running_rewards[-10:]) / min(len(running_rewards), 10)
            print(f"Step {step+1:3d}/{NUM_STEPS} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg Reward: {avg:.3f}")
            wandb.log({
                "step": step + 1,
                "loss": loss.item(),
                "mean_reward": mean_reward,
                "avg_reward": avg,
            })
            # Save checkpoint
            path = f"/kaggle/working/phase3_step{step+1}"
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            print(f"💾 Saved: {path}")

    if is_main:
        model.save_pretrained("/kaggle/working/phase3_final")
        tokenizer.save_pretrained("/kaggle/working/phase3_final")
        print("\n✅ Done!")
        wandb.finish()

train()