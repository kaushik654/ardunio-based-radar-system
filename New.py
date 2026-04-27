import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Load BASE Gemma 4 E2B-IT (no fine-tuning)
# ============================================================
BASE_MODEL = "google/gemma-4-E2B-it"

print(f"Loading base model: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model.eval()


def generate_response(messages, max_new_tokens=512):
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


from fake_executor import execute_tool


def parse_tool_calls(text):
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            print(f"[!] Failed to parse tool call: {m[:100]}")
    return calls


def run_conversation(user_message, max_rounds=5):
    messages = [{"role": "user", "content": user_message}]

    for round_num in range(max_rounds):
        print(f"\n{'='*60}\nROUND {round_num + 1}\n{'='*60}")

        response = generate_response(messages)
        print(f"\n[MODEL]:\n{response}\n")

        tool_calls = parse_tool_calls(response)

        if not tool_calls:
            print("[+] Conversation complete (no tool calls)")
            messages.append({"role": "assistant", "content": response})
            break

        messages.append({"role": "assistant", "content": response})

        responses = []
        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("arguments", {})
            print(f"[TOOL CALL] {name}({args})")
            result = execute_tool(name, args)
            print(f"[TOOL RESULT] {result[:200]}")
            responses.append(f"<tool_response>{result}</tool_response>")

        messages.append({"role": "user", "content": "\n".join(responses)})

    return messages


# Run
final_messages = run_conversation("how to optimize battery?")

print("\n" + "="*60)
print("FINAL CONVERSATION:")
print("="*60)
for m in final_messages:
    print(f"\n[{m['role'].upper()}]: {m['content'][:300]}")
