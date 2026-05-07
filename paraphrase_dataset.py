#!/usr/bin/env python3
"""
paraphrase_dataset.py
=====================
Multiplies a validated tool-call dataset by paraphrasing the natural-language
content of every turn while keeping tool calls VERBATIM.

What gets paraphrased:
  • Every turn's `content` field (user requests, assistant clarifications,
    assistant reasoning, assistant summaries).

What stays IDENTICAL:
  • turn count, roles, ordering
  • every `tools_called` entry (name, description, input_parameters, output)
  • scenario / expected_outcome / target_tools / group / chatbot_role

Usage:
    python paraphrase_dataset.py \\
        --input  out/master_dataset.jsonl \\
        --output out/master_dataset_paraphrased.jsonl \\
        --num-paraphrases 2 \\
        --model Qwen/Qwen3-8B \\
        --load-in-4bit

The output JSONL contains the originals PLUS the paraphrases, each with a
fresh `id` and a `source_id` field linking back to the original sample.
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

# Reuse exactly the same model/generation/parsing helpers as the generator
from generator_core import (
    load_model_hf,
    build_chat_prompt,
    generate_batch,
    extract_json,
    _normalize,
)


PARAPHRASE_SYSTEM = """You are a paraphrasing assistant for a tool-calling dataset.

You will receive an ordered list of conversation turn contents. Paraphrase each one — keep the same meaning, intent, and persona, but use different wording.

HARD RULES:
1. Preserve VERBATIM: all numbers, times, dates, durations, percentages, file paths, app names, contact names, identifiers, tool-related technical terms (e.g. "battery level", "DND", "WiFi", "calendar event"). Do NOT change "30 minutes" to "half an hour" or "Mom" to "Mother".
2. Preserve the persona and level of formality (casual stays casual; professional stays professional).
3. Preserve the role's intent — a clarifying question stays a question, a summary stays a summary.
4. Do NOT add new information, do NOT drop information.
5. Output EXACTLY the same number of paraphrases as the input, in the same order.
6. Each paraphrase MUST differ noticeably from the original (different sentence shape or vocabulary), not a trivial rewording.

Output ONLY this JSON object — no prose, no markdown:
{"paraphrased": ["...", "...", ...]}
"""


def build_paraphrase_user_prompt(turns: list[dict]) -> str:
    """Build the user-side prompt listing all turns to paraphrase."""
    lines = [f"Paraphrase each of the following {len(turns)} turn contents:\n"]
    for i, t in enumerate(turns):
        role = t.get("role", "?")
        content = (t.get("content") or "").replace("\n", " ").strip()
        lines.append(f"[{i}] ({role}) {json.dumps(content, ensure_ascii=False)}")
    lines.append(
        f"\nReturn JSON: {{\"paraphrased\": [<{len(turns)} strings in order>]}}"
    )
    return "\n".join(lines)


def splice_paraphrase(original: dict, new_contents: list[str]) -> dict | None:
    """Return a new sample with each turn's content replaced. tools_called preserved."""
    turns = original.get("turns", [])
    if len(new_contents) != len(turns):
        return None
    new_turns = []
    for orig_turn, new_text in zip(turns, new_contents):
        if not isinstance(new_text, str) or not new_text.strip():
            return None
        # shallow copy so tools_called list is shared (we don't mutate it)
        nt = dict(orig_turn)
        nt["content"] = new_text.strip()
        new_turns.append(nt)

    out = dict(original)
    out["turns"] = new_turns
    out["id"] = f"para_{uuid.uuid4().hex[:8]}"
    out["source_id"] = original.get("id", "")
    out["paraphrased"] = True
    return out


def structurally_valid(orig: dict, paraphrased: dict) -> tuple[bool, str]:
    """Cheap check — paraphrase only edits content, so structure must match exactly."""
    o, p = orig.get("turns", []), paraphrased.get("turns", [])
    if len(o) != len(p):
        return False, "turn-count mismatch"
    for i, (a, b) in enumerate(zip(o, p)):
        if a.get("role") != b.get("role"):
            return False, f"turn {i}: role changed"
        if (a.get("tools_called") or []) != (b.get("tools_called") or []):
            return False, f"turn {i}: tools_called changed"
        if not isinstance(b.get("content"), str) or not b["content"].strip():
            return False, f"turn {i}: empty content"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description="Paraphrase a tool-call dataset")
    ap.add_argument("--input", type=Path, required=True, help="Input JSONL")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL")
    ap.add_argument("--num-paraphrases", type=int, default=2,
                    help="Paraphrased copies per source sample")
    ap.add_argument("--include-originals", action="store_true",
                    help="Also copy originals into output (default: paraphrases only)")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cuda:0", "cuda:1", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--max-attempts-per-sample", type=int, default=3,
                    help="Max regen attempts if a paraphrase fails validation/dedup")
    args = ap.parse_args()

    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    # ---- load source samples ----
    sources: list[dict] = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sources.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"[+] Loaded {len(sources)} source samples from {args.input}",
          flush=True)
    if not sources:
        sys.exit("No source samples — exiting.")

    # ---- resume / dedup ----
    accepted_sigs: set[str] = set()
    already_done: Counter = Counter()  # source_id -> count of paraphrases already produced
    if args.output.exists():
        with args.output.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turns = obj.get("turns", [])
                if turns:
                    first_user = next(
                        (t["content"] for t in turns if t.get("role") == "user"),
                        "",
                    )
                    sig = _normalize(first_user)[:200]
                    if sig:
                        accepted_sigs.add(sig)
                src = obj.get("source_id")
                if src:
                    already_done[src] += 1
        print(f"[+] Resume: {sum(already_done.values())} paraphrases already on disk",
              flush=True)

    # also seed accepted_sigs with originals so paraphrases don't collide with them
    for s in sources:
        turns = s.get("turns", [])
        first_user = next(
            (t["content"] for t in turns if t.get("role") == "user"), "",
        )
        sig = _normalize(first_user)[:200]
        if sig:
            accepted_sigs.add(sig)

    # ---- load model ----
    print(f"[+] Loading model {args.model}...", flush=True)
    model, tokenizer = load_model_hf(
        args.model, args.device, args.dtype,
        args.load_in_4bit, args.load_in_8bit,
    )
    print(f"[+] Model ready.", flush=True)

    # ---- main loop ----
    out_fh = args.output.open("a")
    n_written = sum(already_done.values())
    n_attempts = 0
    n_rejects = Counter()

    target_total = len(sources) * args.num_paraphrases

    # If user asked, copy originals through first (only on a fresh file)
    if args.include_originals and not args.output.exists():
        for s in sources:
            out_fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        out_fh.flush()
        print(f"[+] Copied {len(sources)} originals into output", flush=True)

    pbar = tqdm(total=target_total, initial=n_written, desc="paraphrases")

    try:
        # Build a flat task list: (source, paraphrase_index) for all needed work
        tasks: list[tuple[dict, int]] = []
        for s in sources:
            sid = s.get("id", "")
            done = already_done.get(sid, 0)
            for k in range(done, args.num_paraphrases):
                tasks.append((s, k))

        # Batch processing
        i = 0
        while i < len(tasks):
            batch = tasks[i : i + args.batch_size]
            i += args.batch_size

            prompts = []
            srcs = []
            for src_sample, _k in batch:
                up = build_paraphrase_user_prompt(src_sample.get("turns", []))
                prompts.append(build_chat_prompt(
                    tokenizer, PARAPHRASE_SYSTEM, up,
                    enable_thinking=args.enable_thinking,
                ))
                srcs.append(src_sample)

            t0 = time.time()
            texts = generate_batch(
                model, tokenizer, prompts,
                args.max_new_tokens, args.temperature, args.top_p, args.device,
            )
            dt = time.time() - t0

            batch_ok = 0
            for raw, src_sample in zip(texts, srcs):
                n_attempts += 1
                # strip <think>...</think> if present
                raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
                raw = re.sub(r"^[\s\S]*?</think>\s*", "", raw).strip()
                obj = extract_json(raw)
                if obj is None or "paraphrased" not in obj:
                    n_rejects["no_json"] += 1
                    continue
                contents = obj["paraphrased"]
                if not isinstance(contents, list):
                    n_rejects["paraphrased_not_list"] += 1
                    continue

                new_sample = splice_paraphrase(src_sample, contents)
                if new_sample is None:
                    n_rejects["splice_failed"] += 1
                    continue

                ok, reason = structurally_valid(src_sample, new_sample)
                if not ok:
                    n_rejects[reason] += 1
                    continue

                # dedup
                first_user = next(
                    (t["content"] for t in new_sample["turns"]
                     if t.get("role") == "user"),
                    "",
                )
                sig = _normalize(first_user)[:200]
                if not sig or sig in accepted_sigs:
                    n_rejects["duplicate"] += 1
                    continue
                accepted_sigs.add(sig)

                out_fh.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
                out_fh.flush()
                n_written += 1
                batch_ok += 1
                pbar.update(1)

            top_rej = dict(n_rejects.most_common(3))
            print(f"[paraphrase] attempts={n_attempts} written={n_written}/"
                  f"{target_total} batch={batch_ok}/{len(batch)} "
                  f"sec={dt:.1f} rej={top_rej}", flush=True)
    finally:
        out_fh.close()
        pbar.close()

    print(f"\n[+] Done. {n_written} paraphrases written to {args.output}",
          flush=True)
    print(f"[+] Total rejections: {sum(n_rejects.values())} "
          f"(top: {dict(n_rejects.most_common(5))})", flush=True)


if __name__ == "__main__":
    main()
