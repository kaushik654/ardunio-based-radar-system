#!/usr/bin/env python3
"""
convert_all_formats.py
----------------------
Converts THREE different dataset formats into a single unified SFT format
for fine-tuning Gemma 4 / Gemma 3n.

Common output format (Hermes-style, one example per line in JSONL):
{
  "messages": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "Okay, checking...\n<tool_call>{...}</tool_call>"},
    {"role": "user",      "content": "<tool_response>{...}</tool_response>"},
    {"role": "assistant", "content": "Final natural-language reply"}
  ],
  "source_format": "deepeval" | "conversation_list" | "blueprint_turns",
  "id": "..."
}

The three input formats:

A) DEEPEVAL format (your generated dataset):
   {"turns": [{"role": "user/assistant", "content": "...", "tools_called": [...]}, ...]}

B) CONVERSATION_LIST format (image 1 — your additional samples):
   {"user_intent_id": ..., "conversation_list": [
     {"role": "user/assistant/tool", "content": "..."}, ...
   ]}
   - assistant tool calls are stored as a Python-repr STRING in content
   - tool responses are stored as JSON strings in content

C) BLUEPRINT_TURNS format (image 2 — your other additional samples):
   {"blueprint_id": ..., "turns": [
     {"role": "user/assistant/tool", "content": "...", "tool_call": {...}|null}
   ], "final_response": "..."}
   - tool calls are in a separate `tool_call` field
   - content can be null, str, or dict

Usage:
    python convert_all_formats.py \\
        --deepeval dataset_all_groups.jsonl \\
        --conv-list dataset_format_a.jsonl \\
        --blueprint dataset_format_b.jsonl \\
        --output combined_sft.jsonl \\
        --train-split 0.9 \\
        --shuffle
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers — parse messy input fields
# ---------------------------------------------------------------------------

def _parse_maybe_json(value: Any) -> Any:
    """Try to parse a value as JSON or Python literal. Return original if both fail."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return s
    # Try JSON first (most common)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Try Python literal (handles single quotes from repr())
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    return s


def _to_json_string(value: Any) -> str:
    """Coerce anything (dict, list, str, None) into a JSON-encoded string."""
    if value is None:
        return "{}"
    if isinstance(value, str):
        # Already a string — assume it's already serialized; wrap if not valid JSON
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            return json.dumps(value)
    return json.dumps(value, ensure_ascii=False)


def _build_tool_call_tag(name: str, arguments: dict) -> str:
    """Build a <tool_call>{...}</tool_call> block."""
    payload = {"name": name, "arguments": arguments or {}}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def _build_tool_response_tag(output: Any) -> str:
    """Build a <tool_response>{...}</tool_response> block from any output shape."""
    return f"<tool_response>{_to_json_string(output)}</tool_response>"


# ---------------------------------------------------------------------------
# Converter A: DeepEval format → unified messages
# ---------------------------------------------------------------------------

def convert_deepeval(sample: dict) -> dict | None:
    turns = sample.get("turns") or []
    if not turns:
        return None

    messages = []
    for t in turns:
        role = t.get("role")
        content = t.get("content", "") or ""
        tools_called = t.get("tools_called") or []

        if role == "user":
            messages.append({"role": "user", "content": content})

        elif role == "assistant":
            if tools_called:
                # Combine reasoning text + tool_call tags into one assistant message
                tool_call_tags = [
                    _build_tool_call_tag(tc.get("name"), tc.get("input_parameters", {}))
                    for tc in tools_called
                ]
                assistant_text = content
                if assistant_text:
                    assistant_text += "\n"
                assistant_text += "\n".join(tool_call_tags)
                messages.append({"role": "assistant", "content": assistant_text})

                # Then a synthetic user turn carrying all tool_response blocks
                response_tags = [
                    _build_tool_response_tag(tc.get("output"))
                    for tc in tools_called
                ]
                messages.append({"role": "user", "content": "\n".join(response_tags)})
            else:
                messages.append({"role": "assistant", "content": content})
        # role == "tool" not expected in deepeval format, but handle defensively
        elif role == "tool":
            messages.append({
                "role": "user",
                "content": _build_tool_response_tag(content),
            })

    return _finalize_messages(messages, source="deepeval", sample_id=sample.get("id"))


# ---------------------------------------------------------------------------
# Converter B: Conversation_list format → unified messages
# ---------------------------------------------------------------------------

def convert_conversation_list(sample: dict) -> dict | None:
    conv = sample.get("conversation_list") or []
    if not conv:
        return None

    messages = []
    for t in conv:
        role = t.get("role")
        content = t.get("content", "")

        if role == "user":
            messages.append({"role": "user", "content": str(content) if content else ""})

        elif role == "assistant":
            # In format A, tool calls live as a Python-repr string in content,
            # like: "[{'name': 'notifications_status', 'arguments': {'action': 'dnd_status'}}]"
            # We need to detect this and convert.
            parsed = _parse_maybe_json(content)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) \
                    and "name" in parsed[0]:
                # It's a list of tool calls
                tool_call_tags = [
                    _build_tool_call_tag(tc.get("name"), tc.get("arguments", {}))
                    for tc in parsed
                ]
                messages.append({"role": "assistant", "content": "\n".join(tool_call_tags)})
            else:
                # Plain assistant text
                messages.append({"role": "assistant", "content": str(content) if content else ""})

        elif role == "tool":
            # Tool responses: content is a JSON string (or already a dict)
            parsed = _parse_maybe_json(content)
            messages.append({
                "role": "user",
                "content": _build_tool_response_tag(parsed),
            })

    return _finalize_messages(
        messages,
        source="conversation_list",
        sample_id=sample.get("user_intent_id"),
    )


# ---------------------------------------------------------------------------
# Converter C: Blueprint_turns format → unified messages
# ---------------------------------------------------------------------------

def convert_blueprint_turns(sample: dict) -> dict | None:
    turns = sample.get("turns") or []
    if not turns:
        return None

    messages = []
    for t in turns:
        role = t.get("role")
        content = t.get("content")
        tool_call = t.get("tool_call")

        if role == "user":
            messages.append({"role": "user", "content": str(content) if content else ""})

        elif role == "assistant":
            if tool_call:
                # Combine optional text + the tool call tag
                text_part = str(content) if content else ""
                tag = _build_tool_call_tag(
                    tool_call.get("name"),
                    tool_call.get("arguments", {}),
                )
                combined = (text_part + "\n" + tag).strip() if text_part else tag
                messages.append({"role": "assistant", "content": combined})
            else:
                # Plain assistant text — but content might be None (skip empties)
                if content:
                    messages.append({"role": "assistant", "content": str(content)})

        elif role == "tool":
            # content is often a dict here, e.g. {"ok": true, "result": {...}}
            messages.append({
                "role": "user",
                "content": _build_tool_response_tag(content),
            })

    # Format C has a separate `final_response` field — append it as the last
    # assistant turn IF the last message isn't already a plain assistant turn
    final = sample.get("final_response")
    if final and (not messages or messages[-1]["role"] != "assistant"
                   or "<tool_call>" in messages[-1]["content"]):
        messages.append({"role": "assistant", "content": str(final)})

    return _finalize_messages(
        messages,
        source="blueprint_turns",
        sample_id=sample.get("blueprint_id"),
    )


# ---------------------------------------------------------------------------
# Shared finalization: merge adjacent same-role turns, validate shape
# ---------------------------------------------------------------------------

def _finalize_messages(
    messages: list[dict],
    source: str,
    sample_id: str | None = None,
) -> dict | None:
    if not messages:
        return None

    # Collapse adjacent same-role turns (avoids "user, user" or "assistant, assistant"
    # which Gemma's chat template doesn't expect)
    collapsed = []
    for m in messages:
        if collapsed and collapsed[-1]["role"] == m["role"]:
            collapsed[-1]["content"] = (
                collapsed[-1]["content"].rstrip() + "\n" + m["content"].lstrip()
            ).strip()
        else:
            collapsed.append(m)

    # Must start with user
    if collapsed[0]["role"] != "user":
        return None

    # Must end with assistant. If the last turn is a synthetic tool_response
    # (user turn injected by us), drop it — conversation has no final summary.
    while collapsed and collapsed[-1]["role"] == "user":
        collapsed.pop()
    if not collapsed or collapsed[-1]["role"] != "assistant":
        return None

    # Drop conversations with no meaningful content (e.g. empty assistant)
    if not any(m.get("content", "").strip() for m in collapsed):
        return None

    return {
        "messages": collapsed,
        "source_format": source,
        "id": sample_id,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[!] {path.name}: skipping malformed line: {e}")
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main: convert each input file with its specific converter, then merge.
# ---------------------------------------------------------------------------

CONVERTERS = {
    "deepeval": convert_deepeval,
    "conv-list": convert_conversation_list,
    "blueprint": convert_blueprint_turns,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepeval", type=Path, default=None,
                    help="Path to DeepEval-format JSONL")
    ap.add_argument("--conv-list", type=Path, default=None,
                    help="Path to conversation_list-format JSONL")
    ap.add_argument("--blueprint", type=Path, default=None,
                    help="Path to blueprint_turns-format JSONL")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output combined train JSONL")
    ap.add_argument("--train-split", type=float, default=0.9)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inputs = [
        ("deepeval", args.deepeval),
        ("conv-list", args.conv_list),
        ("blueprint", args.blueprint),
    ]
    inputs = [(name, p) for name, p in inputs if p is not None]
    if not inputs:
        print("[!] At least one input file is required.")
        return

    all_converted = []
    stats = {}

    for fmt_name, path in inputs:
        if not path.exists():
            print(f"[!] Skipping {path}: file not found")
            continue
        raw = load_jsonl(path)
        converter = CONVERTERS[fmt_name]
        converted = []
        for sample in raw:
            out = converter(sample)
            if out is not None:
                converted.append(out)
        n_skipped = len(raw) - len(converted)
        stats[fmt_name] = {"loaded": len(raw), "converted": len(converted),
                           "skipped": n_skipped}
        all_converted.extend(converted)
        print(f"[+] {fmt_name:12s} {path}: {len(converted)} converted, {n_skipped} skipped")

    print(f"\n[+] Total combined: {len(all_converted)} samples")
    for fmt, s in stats.items():
        pct = 100 * s["converted"] / max(len(all_converted), 1)
        print(f"    {fmt:12s} contributes {s['converted']} samples ({pct:.1f}%)")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(all_converted)

    n_train = int(len(all_converted) * args.train_split)
    train = all_converted[:n_train]
    val = all_converted[n_train:]

    train_path = args.output
    val_path = args.output.with_name(args.output.stem + "_val.jsonl")

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    print(f"\n[+] Wrote train: {len(train)} → {train_path}")
    print(f"[+] Wrote val:   {len(val)} → {val_path}")

    # Show one example per source so you can eyeball the conversion
    seen_sources = set()
    print("\n[+] Example converted samples (one per source format):")
    for s in all_converted:
        src = s["source_format"]
        if src in seen_sources:
            continue
        seen_sources.add(src)
        print(f"\n--- source: {src}, id: {s.get('id')} ---")
        for i, m in enumerate(s["messages"][:6]):
            preview = m["content"][:100].replace("\n", " ")
            print(f"  [{i}] {m['role']:9s} | {preview}")
        if len(s["messages"]) > 6:
            print(f"  ... and {len(s['messages']) - 6} more turns")
        if len(seen_sources) == 3:
            break


if __name__ == "__main__":
    main()
