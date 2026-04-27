import json
import re

def generate_response(messages, max_new_tokens=512):
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, 
                       max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,    # Gemma team's recommended values
            top_p=0.95,
            top_k=64,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True,
    )
    return response


# Fake tool executor — replace with your real Samsung tool dispatcher
def execute_tool(name, arguments):
    """In production, this calls your actual Android tool."""
    if name == "battery":
        return {"level": 67, "health": "good", "temperature": 32.1, 
                "top_drains": ["bluetooth", "screen"]}
    elif name == "device_control":
        return {"success": True, **arguments}
    elif name == "diagnostics":
        return {"memory_usage": "62%", "storage": "45GB free", "battery": 67}
    # ... etc for all your tools
    return {"error": f"unknown tool {name}"}


def parse_tool_calls(text):
    """Extract <tool_call>{...}</tool_call> blocks from model output."""
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            print(f"[!] Failed to parse tool call: {m}")
    return calls


def run_conversation(user_message, max_rounds=5):
    """Full multi-turn loop: generate → execute tools → feed back → repeat."""
    messages = [{"role": "user", "content": user_message}]
    
    for round_num in range(max_rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num + 1}")
        print('='*60)
        
        # 1. Generate model output
        response = generate_response(messages)
        print(f"\n[MODEL]:\n{response}\n")
        
        # 2. Check if model called any tools
        tool_calls = parse_tool_calls(response)
        
        if not tool_calls:
            # No more tools — model has given final answer
            print("[+] Conversation complete (no more tool calls)")
            messages.append({"role": "assistant", "content": response})
            break
        
        # 3. Add the model's response to messages
        messages.append({"role": "assistant", "content": response})
        
        # 4. Execute each tool and build a tool_response
        responses = []
        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("arguments", {})
            print(f"[TOOL CALL] {name}({args})")
            result = execute_tool(name, args)
            print(f"[TOOL RESULT] {result}")
            responses.append(f"<tool_response>{json.dumps(result)}</tool_response>")
        
        # 5. Add tool responses as a user turn
        messages.append({"role": "user", "content": "\n".join(responses)})
    
    return messages


# Usage
final_messages = run_conversation("how to optimize battery?")

print("\n" + "="*60)
print("FINAL CONVERSATION:")
print("="*60)
for m in final_messages:
    print(f"\n[{m['role'].upper()}]: {m['content'][:300]}")
