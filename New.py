#!/usr/bin/env python3
"""
generator_core.py
-----------------
Shared core for grouped dataset generators. Group-specific scripts import from
here so the 700 lines of generation logic isn't duplicated.

Each group script needs to define:
    GROUP_NAME: str                        # e.g. "system_device_info"
    GROUP_TOOL_ALLOWLIST: set[str]         # which tools from the full catalog belong to this group
    GROUP_SEEDS: list[dict]                # scenarios for this group only
    DEFAULT_OUTPUT: str                    # default output filename

then call:
    from generator_core import run_group
    run_group(GROUP_NAME, GROUP_TOOL_ALLOWLIST, GROUP_SEEDS, DEFAULT_OUTPUT)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a synthetic dataset generator. Your job is to produce ONE high-quality TRULY MULTI-TURN conversation between a user and an Android assistant that controls the device via tool calls.

⚠️ CRITICAL: You MUST strictly follow the Intent given below. Do NOT drift to an unrelated scenario. If the Intent is about calendar, write a calendar scenario. If it's about memory, write a memory scenario. Do NOT substitute with a "save battery" scenario or any other topic.

⚠️ MULTI-TURN REQUIREMENT (most important):
The conversation MUST contain AT LEAST 3-4 separate user turns — not just one.
This means the user speaks, the assistant responds (maybe with tools), then the user speaks AGAIN with a follow-up, the assistant responds again, and so on.
A sample with only ONE user turn is WRONG. Generate realistic back-and-forth dialogue.

Examples of good user follow-ups to weave in:
- Asking for clarification ("what do you mean by that?")
- Refining the request ("actually make it 50% instead of 30%")
- Asking a related follow-up ("and what about my location settings?")
- Reacting to the assistant's result ("hmm that's lower than I expected, can you check again?")
- Changing their mind ("wait, undo that and try something else")
- Asking why ("why did you turn that off?")
- Confirming before action ("yes go ahead")
- Introducing a new but related need ("now can you also check my storage?")

The output MUST be valid JSON matching this EXACT format:

{
  "scenario": "<one-sentence description of the situation>",
  "expected_outcome": "<one-sentence description of what the assistant should accomplish>",
  "chatbot_role": "<short role description>",
  "turns": [
    {"role": "user", "content": "<realistic user request matching the Intent>"},
    {
      "role": "assistant",
      "content": "<brief reasoning>",
      "tools_called": [
        {
          "name": "<exact tool name from AVAILABLE TOOLS>",
          "description": "<brief paraphrase of what the tool does>",
          "input_parameters": { <arguments as JSON OBJECT, NOT a string> },
          "output": "<realistic result as a JSON-encoded string>"
        }
      ]
    },
    {"role": "assistant", "content": "<response to user with tool results>"},
    {"role": "user", "content": "<FOLLOW-UP question, refinement, or reaction>"},
    {"role": "assistant", "content": "<reasoning for next action>",
     "tools_called": [ ... ]},
    {"role": "assistant", "content": "<response to second user turn>"},
    {"role": "user", "content": "<THIRD user turn — another follow-up or new related ask>"},
    {"role": "assistant", "content": "<final summary, NO tools_called>"}
  ]
}

HARD REQUIREMENTS:
1. `role` is ONLY "user" or "assistant". NEVER "system", NEVER "tool".
2. First turn MUST be "user". Last turn MUST be "assistant" with NO "tools_called".
3. **AT LEAST 3 user turns required. 4 user turns is even better.** Not just one.
4. Every tool name MUST appear in AVAILABLE TOOLS.
5. `input_parameters` is a JSON OBJECT (dict), NOT a string.
6. `output` is a JSON-encoded STRING simulating what the tool would return.
7. Include 2-6 tool calls total across the conversation (spread across turns).
8. Required parameters for each tool MUST be present in `input_parameters`.
9. Do NOT include argument keys not in the tool's schema.
10. The user's FIRST message must match the scenario Intent.
11. Later user turns should be related to the same scenario — a continuing dialogue, not a topic switch.

VARIETY:
- Vary the user's opening phrasing. Do NOT always start with "My", "Can you", "Please".
- Vary the assistant's phrasing. Do NOT always start with "Let me check".
- Later user turns should sound natural — short reactions, "yes do it", "hmm", "what about X", "actually make it Y" — not another full formal request each time.

Output ONLY the JSON object. No prose, no markdown fences.
"""


def build_user_prompt(tools_block: str, seed: dict[str, Any]) -> str:
    target = seed.get("target_tools")
    target_line = (f"- **Target tools (MUST use at least 60% of these)**: {', '.join(target)}"
                   if target else
                   "- Target tools: choose 2-5 from AVAILABLE TOOLS.")
    expected = seed.get("expected_outcome", "Assistant resolves the user's request.")

    # Pick a dialogue shape — different conversational arcs for variety
    dialogue_shapes = [
        ("The user asks a first question, the assistant answers with tools, the user "
         "asks a CLARIFYING follow-up, the assistant answers again (possibly more tools), "
         "the user REACTS to the result, the assistant gives a final summary."),
        ("The user makes an initial request, the assistant takes action, the user "
         "changes their mind or refines the request, the assistant adjusts, the user "
         "confirms, the assistant wraps up."),
        ("The user asks for information, the assistant provides it, the user asks a "
         "RELATED follow-up question, the assistant handles that too, the user asks "
         "ONE MORE related thing, the assistant summarizes everything."),
        ("The user states a problem, the assistant investigates, the user provides "
         "EXTRA context when asked, the assistant acts on it, the user approves or "
         "pushes back, the assistant finalizes."),
    ]
    dialogue_shape = random.choice(dialogue_shapes)

    variety_hints = [
        "User is casual and uses short, colloquial phrasing.",
        "User is formal and professional.",
        "User is impatient — short, direct replies.",
        "User is curious and asks WHY about things.",
        "User is analytical — wants details and numbers.",
    ]
    variety = random.choice(variety_hints)

    return f"""AVAILABLE TOOLS:
{tools_block}

SCENARIO (match this exactly — do not drift to a different topic):
- **Intent**: {seed['intent']}
{target_line}
- Strategy hint: {seed['hint']}
- User persona: {seed['persona']}
- Expected outcome: {expected}
- Variety nudge: {variety}

DIALOGUE SHAPE (follow this structure — produces a natural back-and-forth):
{dialogue_shape}

⚠️ The conversation MUST have AT LEAST 3 user turns (4 is better). The user speaks, the assistant responds, the user follows up, the assistant responds again, and so on. Do NOT make a 1-user-turn conversation.

The user's FIRST message must be about: {seed['intent']}
Subsequent user turns should continue the same scenario (follow-ups, reactions, refinements) — not switch topics.

Generate the sample now. Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Tool catalog filtering — per-group
# ---------------------------------------------------------------------------

def load_tools_filtered(path: Path, allowlist: set[str]):
    """Load full catalog, keep only tools in allowlist. Return (list, schema_index)."""
    with path.open() as f:
        raw = json.load(f)
    tool_list, index = [], {}
    for entry in raw:
        fn = entry["function"]
        if fn["name"] not in allowlist:
            continue
        tool_list.append(entry)
        index[fn["name"]] = fn["parameters"]
    return tool_list, index


def compact_tools_for_prompt(tools):
    return json.dumps([{
        "name": t["function"]["name"],
        "description": t["function"].get("description", ""),
        "parameters": t["function"]["parameters"],
    } for t in tools], indent=2)


# ---------------------------------------------------------------------------
# Seed prep
# ---------------------------------------------------------------------------

def prepare_seeds(handcrafted, schema_index, tools, verbose=True):
    """Drop seeds referencing tools outside the group allowlist; auto-cover uncovered."""
    available = set(schema_index.keys())
    kept, dropped = [], []
    for s in handcrafted:
        unknown = [t for t in (s.get("target_tools") or []) if t not in available]
        if unknown:
            dropped.append((s["intent"][:50], unknown))
            continue
        kept.append(s)

    if verbose and dropped:
        print(f"[!] Dropped {len(dropped)} seeds referencing out-of-group tools:",
              flush=True)
        for intent, unk in dropped[:3]:
            print(f"    - {intent!r} → unknown {unk}", flush=True)

    covered = set()
    for s in kept:
        for t in s.get("target_tools") or []:
            covered.add(t)
    uncovered = sorted(available - covered)
    desc_by_name = {t["function"]["name"]: t["function"].get("description", "") for t in tools}
    for name in uncovered:
        kept.append({
            "intent": f"User request requiring the '{name}' tool. Purpose: {desc_by_name.get(name,'')[:140]}",
            "target_tools": [name],
            "hint": f"Use '{name}' with valid arguments per schema.",
            "persona": "casual",
            "expected_outcome": f"Request resolved using {name}.",
        })

    if verbose:
        print(f"[+] Seeds in group: {len(kept)} ({len(uncovered)} auto-coverage)",
              flush=True)
    return kept


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------

JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate_sample(sample, schema_index, target_tools=None, strict_targets=True,
                    group_allowlist=None, min_user_turns=3):
    if not isinstance(sample, dict) or "turns" not in sample:
        return False, "missing 'turns'"
    turns = sample["turns"]
    if not isinstance(turns, list) or len(turns) < 2:
        return False, "too few turns"
    if turns[0].get("role") != "user":
        return False, "first turn must be 'user'"
    if turns[-1].get("role") != "assistant":
        return False, "last turn must be 'assistant'"
    if turns[-1].get("tools_called"):
        return False, "last assistant turn must not have tools_called"

    # NEW: count user turns — we want a genuinely multi-turn conversation
    n_user_turns = sum(1 for t in turns if t.get("role") == "user")
    if n_user_turns < min_user_turns:
        return False, f"only {n_user_turns} user turn(s), need at least {min_user_turns}"

    n_tool_calls = 0
    tools_called = []
    for i, t in enumerate(turns):
        role = t.get("role")
        if role not in ("user", "assistant"):
            return False, f"turn {i}: invalid role '{role}'"
        if not isinstance(t.get("content"), str):
            return False, f"turn {i}: content not string"
        if role == "user" and t.get("tools_called"):
            return False, f"turn {i}: user turn has tools_called"
        if role == "assistant":
            for j, tc in enumerate(t.get("tools_called") or []):
                n_tool_calls += 1
                name = tc.get("name")
                if name not in schema_index:
                    return False, f"turn {i} tc{j}: unknown tool '{name}' (out of group)"
                tools_called.append(name)
                args = tc.get("input_parameters")
                if not isinstance(args, dict):
                    return False, f"turn {i} tc{j}: input_parameters not dict"
                if not isinstance(tc.get("output"), str):
                    return False, f"turn {i} tc{j}: output not string"
                params = schema_index[name]
                missing = [k for k in params.get("required", []) if k not in args]
                if missing:
                    return False, f"turn {i} tc{j} '{name}': missing required {missing}"
                allowed = set((params.get("properties") or {}).keys())
                unknown = [k for k in args if k not in allowed]
                if unknown:
                    return False, f"turn {i} tc{j} '{name}': unknown args {unknown}"
    if n_tool_calls == 0:
        return False, "no tool_calls"
    if n_tool_calls > 10:
        return False, f"too many tool_calls ({n_tool_calls})"

    if strict_targets and target_tools:
        target_set = set(target_tools)
        called_set = set(tools_called)
        hits = called_set & target_set
        if not hits:
            return False, f"no target tools called (wanted {sorted(target_set)}, got {sorted(called_set)})"
        if len(target_set) >= 2 and len(hits) / len(target_set) < 0.6:
            return False, f"only {len(hits)}/{len(target_set)} target tools used"
        if len(called_set - target_set) > 2:
            return False, f"too many off-target tools: {sorted(called_set - target_set)}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Model loading (HuggingFace, no vLLM)
# ---------------------------------------------------------------------------

def load_model_hf(model_name, device, dtype, load_in_4bit, load_in_8bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = {"trust_remote_code": True}
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    else:
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}.get(dtype, torch.bfloat16)
        kwargs["torch_dtype"] = torch_dtype
        if device == "auto":
            kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not (load_in_4bit or load_in_8bit) and device != "auto":
        model = model.to(device)
    model.eval()
    return model, tok


def build_chat_prompt(tokenizer, system, user, enable_thinking=False):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)


def generate_batch(model, tokenizer, prompts, max_new_tokens, temperature, top_p, device):
    import torch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=4096)
    if device == "auto":
        emb_device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(emb_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# The main generation loop — used by every group script
# ---------------------------------------------------------------------------

def _normalize(text):
    t = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def run_group_generation(
    group_name: str,
    group_allowlist: set[str],
    group_seeds: list[dict],
    tools_file: Path,
    output_file: Path,
    num_samples: int,
    model,
    tokenizer,
    batch_size: int = 4,
    temperature: float = 0.85,
    top_p: float = 0.95,
    max_new_tokens: int = 2048,
    enable_thinking: bool = False,
    device: str = "auto",
    debug_dir: Path | None = None,
    strict_targets: bool = True,
    min_user_turns: int = 3,
) -> int:
    """Run generation for one tool group. Returns count of valid samples written."""
    tools, schema_index = load_tools_filtered(tools_file, group_allowlist)
    if not tools:
        print(f"[!] Group '{group_name}': no tools from allowlist found in {tools_file}",
              flush=True)
        return 0
    print(f"[+] Group '{group_name}': {len(tools)} tools "
          f"({list(schema_index.keys())})", flush=True)

    tools_block = compact_tools_for_prompt(tools)
    active_seeds = prepare_seeds(group_seeds, schema_index, tools)
    if not active_seeds:
        print(f"[!] Group '{group_name}': no usable seeds after filtering", flush=True)
        return 0

    # Resume
    completed_ids = set()
    accepted_sigs = set()
    seed_counts = Counter()
    if output_file.exists():
        with output_file.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    completed_ids.add(obj.get("id", ""))
                    turns = obj.get("turns", [])
                    if turns and turns[0].get("role") == "user":
                        accepted_sigs.add(_normalize(turns[0]["content"])[:200])
                    seed_counts[obj.get("scenario_intent", "")] += 1
                except json.JSONDecodeError:
                    pass
        if completed_ids:
            print(f"[+] Resuming '{group_name}': {len(completed_ids)} samples existing",
                  flush=True)
    if len(completed_ids) >= num_samples:
        print(f"[+] Group '{group_name}' already has {len(completed_ids)} samples. Skipping.",
              flush=True)
        return len(completed_ids)

    # Per-seed cap: 1.5x ideal share, floor 3
    ideal = max(1, num_samples / len(active_seeds))
    per_seed_cap = max(3, int(ideal * 2.0))
    print(f"[+] Balance: per_seed_cap={per_seed_cap} ({len(active_seeds)} seeds)",
          flush=True)

    # Shuffled seed queue
    seed_q = list(active_seeds)
    random.shuffle(seed_q)
    cursor = [0]  # mutable from inner fn

    def next_seed():
        tries = 0
        while tries < len(seed_q) * 2:
            if cursor[0] >= len(seed_q):
                random.shuffle(seed_q)
                cursor[0] = 0
            s = seed_q[cursor[0]]
            cursor[0] += 1
            tries += 1
            if seed_counts[s["intent"]] >= per_seed_cap:
                continue
            return s
        return seed_q[cursor[0] % len(seed_q)]

    n_valid = len(completed_ids)
    n_attempts = 0
    max_attempts = num_samples * 6

    if debug_dir:
        debug_dir = Path(debug_dir) / group_name
        debug_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=num_samples, initial=n_valid, desc=f"[{group_name}]")
    out_fh = output_file.open("a")
    try:
        while n_valid < num_samples and n_attempts < max_attempts:
            this_batch = min(batch_size, num_samples - n_valid)
            prompts, seeds_used = [], []
            for _ in range(this_batch):
                s = next_seed()
                seeds_used.append(s)
                up = build_user_prompt(tools_block, s)
                prompts.append(build_chat_prompt(tokenizer, SYSTEM_PROMPT, up,
                                                  enable_thinking=enable_thinking))
            t0 = time.time()
            texts = generate_batch(model, tokenizer, prompts, max_new_tokens,
                                    temperature, top_p, device)
            dt = time.time() - t0

            batch_valid = 0
            batch_rej = Counter()
            for text, s in zip(texts, seeds_used):
                n_attempts += 1
                text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
                text = re.sub(r"^[\s\S]*?</think>\s*", "", text).strip()
                obj = extract_json(text)
                if obj is None:
                    batch_rej["no_json"] += 1
                    if debug_dir and n_attempts <= 3:
                        (debug_dir / f"reject_{n_attempts}_no_json.txt").write_text(
                            f"=== SEED ===\n{s['intent']}\n\n=== RAW ===\n{text[:4000]}")
                    continue
                ok, reason = validate_sample(
                    obj, schema_index,
                    target_tools=s.get("target_tools"),
                    strict_targets=strict_targets,
                    group_allowlist=group_allowlist,
                    min_user_turns=min_user_turns,
                )
                if not ok:
                    batch_rej[reason] += 1
                    if debug_dir and n_attempts <= 5:
                        safe = reason[:30].replace(" ", "_").replace("/", "_")
                        (debug_dir / f"reject_{n_attempts}_{safe}.json").write_text(
                            json.dumps(obj, indent=2)[:4000])
                    continue
                # Dup check
                first_user = next((t["content"] for t in obj["turns"]
                                   if t.get("role") == "user"), "")
                sig = _normalize(first_user)[:200]
                if sig and sig in accepted_sigs:
                    batch_rej["duplicate"] += 1
                    continue
                if seed_counts[s["intent"]] >= per_seed_cap:
                    batch_rej["seed_cap"] += 1
                    continue
                accepted_sigs.add(sig)
                seed_counts[s["intent"]] += 1

                record = {
                    "id": f"sample_{group_name}_{uuid.uuid4().hex[:8]}",
                    "group": group_name,
                    "scenario": obj.get("scenario", s["intent"]),
                    "expected_outcome": obj.get("expected_outcome",
                                                s.get("expected_outcome", "")),
                    "chatbot_role": obj.get("chatbot_role",
                        f"Android on-device assistant specializing in {group_name} tools."),
                    "scenario_intent": s["intent"],
                    "scenario_persona": s["persona"],
                    "target_tools": s.get("target_tools", []),
                    "turns": obj["turns"],
                }
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fh.flush()
                n_valid += 1
                batch_valid += 1
                pbar.update(1)
                if n_valid >= num_samples:
                    break

            top_rej = dict(batch_rej.most_common(3))
            print(f"[{group_name}][batch] attempts={n_attempts} "
                  f"valid={n_valid}/{num_samples} batch={batch_valid}/{this_batch} "
                  f"sec={dt:.1f} rej={top_rej}", flush=True)
    finally:
        out_fh.close()
        pbar.close()

    print(f"[+] Group '{group_name}' done: {n_valid} samples written to {output_file}",
          flush=True)
    return n_valid


# ---------------------------------------------------------------------------
# CLI entry — each group script calls this
# ---------------------------------------------------------------------------

def run_group(group_name: str, allowlist: set[str], seeds: list[dict],
              default_output: str):
    ap = argparse.ArgumentParser(description=f"Generate {group_name} samples")
    ap.add_argument("--tools-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path(default_output))
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=3072,
                    help="Max output tokens per sample. Bumped from 2048 to 3072 "
                         "to fit 3+ user turns.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cuda:0", "cuda:1", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--debug-dir", type=str, default="debug_rejects")
    ap.add_argument("--no-strict-targets", action="store_true")
    ap.add_argument("--min-user-turns", type=int, default=3,
                    help="Minimum user turns per conversation (default: 3). "
                         "Lower to 2 if rejection rate is too high, raise to 4 for "
                         "longer dialogues.")
    args = ap.parse_args()

    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    print(f"[+] Loading model {args.model}...", flush=True)
    model, tokenizer = load_model_hf(
        args.model, args.device, args.dtype, args.load_in_4bit, args.load_in_8bit,
    )
    print(f"[+] Model ready.", flush=True)

    run_group_generation(
        group_name=group_name,
        group_allowlist=allowlist,
        group_seeds=seeds,
        tools_file=args.tools_file,
        output_file=args.output,
        num_samples=args.num_samples,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        device=args.device,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        strict_targets=not args.no_strict_targets,
        min_user_turns=args.min_user_turns,
    )
