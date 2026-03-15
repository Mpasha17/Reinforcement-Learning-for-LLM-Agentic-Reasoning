"""
Phase 2: LangGraph ReAct Agent with Python Tool
"""
import re, sys, io, torch
from unsloth import FastLanguageModel
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

def load_model(model_path: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

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

def build_agent(model, tokenizer):
    SYSTEM_PROMPT = """You are an advanced math reasoning assistant.
Think step-by-step inside <think> tags.
For complex calculations, write Python inside ```python``` blocks and use print().
The system will execute your code and return the output.
Once you have the answer, output it inside <answer> tags and stop."""

    def llm_node(state: AgentState):
        chat_history = []
        for msg in state["messages"]:
            if hasattr(msg, 'content'):
                role = "assistant" if msg.type == "ai" else ("system" if msg.type == "system" else "user")
                chat_history.append({"role": role, "content": msg.content})
            else:
                chat_history.append(msg)

        text = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        if not text.endswith("<think>\n"):
            text += "<think>\n"

        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512,
                temperature=0.1, use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = "<think>\n" + tokenizer.decode(new_tokens, skip_special_tokens=True)

        if "```python" in response:
            parts = response.split("```python")
            if "```" in parts[1]:
                code_part = parts[1].split("```")[0]
                response = parts[0] + "```python\n" + code_part.strip() + "\n```"

        return {"messages": [{"role": "assistant", "content": response}]}

    def python_execution_node(state: AgentState):
        last = state["messages"][-1]
        content = last.content if hasattr(last, 'content') else last["content"]
        code_match = re.search(r'```python(.*?)```', content, re.DOTALL)
        result = safe_execute(code_match.group(1).strip()) if code_match else "No code found."
        feedback = f"Terminal Output: {result}\nNow give your final answer in <answer> tags."
        return {"messages": [{"role": "user", "content": feedback}]}

    def route_step(state: AgentState):
        last = state["messages"][-1]
        content = last.content if hasattr(last, 'content') else last["content"]
        if "<answer>" in content:
            return END
        messages = state["messages"]
        if len(messages) >= 2:
            prev = messages[-2]
            prev_content = prev.content if hasattr(prev, 'content') else prev.get("content", "")
            if "Terminal Output:" in str(prev_content):
                return END
        if "```python" in content:
            return "execute"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", llm_node)
    workflow.add_node("execute", python_execution_node)
    workflow.add_edge(START, "llm")
    workflow.add_conditional_edges("llm", route_step, {"execute": "execute", END: END})
    workflow.add_edge("execute", "llm")

    return workflow.compile(), SYSTEM_PROMPT


def run_agent(problem: str, model_path: str):
    model, tokenizer = load_model(model_path)
    agent_app, SYSTEM_PROMPT = build_agent(model, tokenizer)

    initial_state = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]
    }

    final_state = agent_app.invoke(initial_state, {"recursion_limit": 10})

    for msg in reversed(final_state["messages"]):
        content = msg.content if hasattr(msg, 'content') else msg.get("content", "")
        matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if matches:
            nums = re.findall(r'-?\d+\.?\d*', matches[-1])
            if nums:
                return nums[-1]
        match = re.search(r'Terminal Output:\s*(-?\d+\.?\d*)', content)
        if match:
            return match.group(1)

    return "No answer found"


if __name__ == "__main__":
    problem = "A store buys 200 shirts for $8 each. Sells 75% at $15, rest at $5. What is profit?"
    answer = run_agent(problem, "mpasha1701/RLVR-Qwen2.5-1.5B-Agent")
    print(f"Answer: {answer}")