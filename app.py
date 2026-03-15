import gradio as gr
import sys, io, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. LOAD MODEL
model_name = "mpasha1701/RLVR-Qwen2.5-1.5B-Agent"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)
if device == "cpu":
    model = model.to("cpu")

print(f"✅ Model loaded on {device}")

# 2. TOOL
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

# 3. AGENT
def agent_loop(problem, history):
    SYSTEM_PROMPT = """You are an advanced math reasoning assistant.
Think step-by-step inside <think> tags.
For complex calculations, write Python inside ```python``` blocks and use print().
The system will execute your code and return the output.
Once you have the answer, output it inside <answer> tags and stop."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add history - Gradio passes list of [user, assistant] pairs
    for turn in history:
        if turn[0]:
            messages.append({"role": "user", "content": turn[0]})
        if turn[1]:
            messages.append({"role": "assistant", "content": turn[1]})

    messages.append({"role": "user", "content": problem})

    full_response = ""

    for turn_num in range(3):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text += "<think>\n"
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.2,
                do_sample=True,           # ← required with temperature
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        turn_response = "<think>\n" + tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )

        # Format nicely for display
        # Convert <think> blocks to markdown
        formatted = re.sub(
            r'<think>(.*?)</think>',
            r'💭 **Thinking...**\n> \1',
            turn_response, flags=re.DOTALL
        )
        # Highlight final answer
        formatted = re.sub(
            r'<answer>(.*?)</answer>',
            r'✅ **Answer: \1**',
            formatted, flags=re.DOTALL
        )

        full_response += f"\n\n---\n**Turn {turn_num + 1}:**\n{formatted}"
        messages.append({"role": "assistant", "content": turn_response})

        if "<answer>" in turn_response:
            break

        code_match = re.search(r'```python(.*?)```', turn_response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            result = safe_execute(code)
            full_response += f"\n\n🖥️ **Terminal Output:**\n```\n{result}\n```"
            messages.append({
                "role": "user",
                "content": f"Terminal Output: {result}\nNow provide your final answer in <answer> tags."
            })

    return full_response.strip()

# 4. UI
with gr.Blocks(title="RLVR Reasoning Agent") as demo:
    gr.Markdown("""
    # 🧠 RLVR Autonomous Reasoning Agent
    **Trained with GRPO (Group Relative Policy Optimization)** — reproducing DeepSeek-R1 style emergent reasoning.
    
    This agent thinks step-by-step, writes Python code to verify calculations, and self-corrects.
    Built on Qwen2.5-1.5B fine-tuned with Reinforcement Learning on free Kaggle GPUs.
    
    📄 [GitHub](https://github.com/mpasha1701/Reinforcement-Learning-for-LLM-Agentic-Reasoning) | 
    🤗 [Model](https://huggingface.co/mpasha1701/RLVR-Qwen2.5-1.5B-Agent)
    """)

    gr.ChatInterface(
        fn=agent_loop,
            chatbot=gr.Chatbot(height=500, render_markdown=True),
        textbox=gr.Textbox(
            placeholder="Try: 'A store buys 100 items at $5 each and sells 80% at $9 and rest at $3. What is the profit?'",
            container=False,
            scale=7
        ),
        examples=[
            ["What is 15% of 840?"],
            ["A store buys 200 shirts for $8 each. They sell 75% at $15 and the rest at $5. What is the profit?"],
            ["Tom has 3 times as many marbles as Jerry. Together they have 48. How many does Tom have?"],
            ["A train travels 120km at 60km/h then 180km at 90km/h. What is the average speed?"],
        ],
        title="",
    )

demo.launch(theme=gr.themes.Soft())