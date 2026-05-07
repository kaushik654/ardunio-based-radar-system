"""
generator_core.py — UPDATES for clarification turns, multi-tool, inter-group
=============================================================================
This file shows the NEW and MODIFIED pieces only. Integrate into your existing
generator_core.py. Sections are clearly marked REPLACE / ADD / MODIFY.

Summary of what's new:
  • Every sample now requires a clarification turn (assistant asks user back
    BEFORE any tool call). Configurable per-seed via `requires_clarification`.
  • Every sample requires ≥2 tool calls (configurable via `min_tool_calls`).
  • Every sample requires ≥6 turns (configurable via `min_turns`).
  • New `run_intergroup()` entry point for cross-group seeds.
  • Stricter validator with all 7 new rules.
  • Backward compatible: existing seeds without new fields get safe defaults.
"""

from __future__ import annotations
from typing import Any, Iterable
import json
import re


# ============================================================================
# 1. SYSTEM_PROMPT  ——  REPLACE  ——  the existing constant in generator_core.py
# ============================================================================
SYSTEM_PROMPT = """\
You generate multi-turn tool-calling training samples for an Android assistant.

OUTPUT FORMAT
Return ONE JSON object with keys: "scenario", "expected_outcome",
"chatbot_role", "turns". No markdown fences, no prose outside the JSON.

CONVERSATION RULES
1.  Roles are "user" or "assistant" ONLY. Never "system". Never "tool".
2.  First turn = user. Last turn = assistant with NO tools_called
    (the final natural-language summary).
3.  At least 6 turns total.
4.  The user speaks at least TWICE: an initial request, plus at least one
    follow-up that answers the assistant's clarification.
5.  CLARIFICATION (mandatory unless the seed disables it): the FIRST assistant
    turn must NOT contain tool calls — it must ask the user a clarifying
    question to pin down missing info (which item, what time, whose number,
    where to, etc.). The next user turn answers that clarification.
6.  MULTI-TOOL: at least 2 tool calls total across the conversation. They may
    be parallel (multiple entries in one turn's tools_called) or sequential
    (split across multiple assistant turns).
7.  After every assistant turn that contains tool calls, the next turn is a
    user turn (continuing the conversation) — never a synthetic "tool" role.
8.  Keep dialogue natural for the given persona. Don't restate tool JSON in
    user messages.

TOOL CALL FORMAT
9.  `tools_called` ONLY on assistant turns that actually call tools.
    Each entry: {"name": <str>, "input_parameters": <dict>, "output": <str>}.
10. `input_parameters` is a JSON OBJECT, not a stringified one.
11. `output` is a JSON-ENCODED STRING simulating the tool's response.
12. Every `name` MUST exist in AVAILABLE TOOLS.
13. Include all required parameters; omit keys not in the schema.
14. Every tool listed in the seed's target_tools must be called somewhere.
15. For inter-group scenarios: tool calls MUST cover all listed groups.
"""


# ============================================================================
# 2. build_user_prompt()  ——  REPLACE
# ============================================================================
def build_user_prompt(tools_block: str, seed: dict[str, Any]) -> str:
    """Build the user-side prompt for a single generation."""
    target = seed.get("target_tools") or []
    if target:
        target_line = (
            f"- Target tools (CALL ALL of these, in a sensible order): "
            f"{', '.join(target)}"
        )
    else:
        target_line = (
            "- Target tools: CHOOSE 2-5 from AVAILABLE TOOLS that fit the intent."
        )

    requires_clar = seed.get("requires_clarification", True)
    min_tc = seed.get("min_tool_calls", 2)
    min_turns = seed.get("min_turns", 6)
    groups_req = seed.get("groups_required") or []
    expected = seed.get("expected_outcome", "Assistant resolves the request.")

    clar_line = (
        "- Clarification: REQUIRED. The first assistant turn must ask a "
        "clarifying question and contain NO tool calls. The next user turn "
        "supplies the answer."
        if requires_clar else
        "- Clarification: optional for this seed (request is fully specified)."
    )

    multi_group_line = ""
    if groups_req:
        multi_group_line = (
            f"- Inter-group: tool calls MUST span these groups: "
            f"{', '.join(groups_req)}. Use at least one tool from EACH group."
        )

    clar_hint = seed.get("clarification_hint")
    clar_hint_line = f"- Suggested clarification: {clar_hint}" if clar_hint else ""

    return f"""AVAILABLE TOOLS:
{tools_block}

SCENARIO:
- Intent: {seed['intent']}
{target_line}
- Strategy hint: {seed['hint']}
- User persona / style: {seed['persona']}
- Expected outcome: {expected}
{clar_line}
{clar_hint_line}
- Minimum tool calls: {min_tc}
- Minimum turns: {min_turns}
{multi_group_line}

Generate the sample now. Output ONLY the JSON object."""


# ============================================================================
# 3. validate_sample()  ——  REPLACE
# ============================================================================
def validate_sample(
    obj: dict[str, Any],
    seed: dict[str, Any],
    schema_index: dict[str, dict[str, Any]],
    tool_to_group: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """
    Returns (ok, reason). reason is "" when ok is True; otherwise a short
    human-readable rejection cause for logs.

    `tool_to_group` maps tool_name -> group_name. Required for inter-group
    coverage validation (pass None for intra-group runs).
    """
    # --- shape ---------------------------------------------------------------
    if not isinstance(obj, dict):
        return False, "top-level not a dict"
    turns = obj.get("turns")
    if not isinstance(turns, list) or len(turns) == 0:
        return False, "missing/empty turns"

    # --- per-seed thresholds (defaults are strict) --------------------------
    requires_clar = seed.get("requires_clarification", True)
    min_tc = int(seed.get("min_tool_calls", 2))
    min_turns = int(seed.get("min_turns", 6))
    groups_req = set(seed.get("groups_required") or [])

    # --- min turns -----------------------------------------------------------
    if len(turns) < min_turns:
        return False, f"only {len(turns)} turns (need ≥{min_turns})"

    # --- role sequence + first/last ------------------------------------------
    if turns[0].get("role") != "user":
        return False, "first turn must be user"
    last = turns[-1]
    if last.get("role") != "assistant":
        return False, "last turn must be assistant"
    if last.get("tools_called"):
        return False, "last turn must have no tool calls (it's the summary)"

    user_count = 0
    for t in turns:
        role = t.get("role")
        if role not in ("user", "assistant"):
            return False, f"bad role {role!r}"
        if not isinstance(t.get("content", ""), str):
            return False, "non-string content"
        if role == "user":
            user_count += 1
        # tools_called only on assistant turns
        if role == "user" and t.get("tools_called"):
            return False, "user turn has tools_called"

    # --- ≥2 user turns -------------------------------------------------------
    if user_count < 2:
        return False, f"only {user_count} user turn(s) (need ≥2)"

    # --- gather all tool calls in order --------------------------------------
    all_calls: list[tuple[int, dict]] = []  # (turn_idx, call_dict)
    first_call_turn = None
    for i, t in enumerate(turns):
        if t.get("role") != "assistant":
            continue
        tc = t.get("tools_called") or []
        if not isinstance(tc, list):
            return False, f"turn {i} tools_called not a list"
        for c in tc:
            if not isinstance(c, dict):
                return False, f"turn {i} tool call not a dict"
            all_calls.append((i, c))
        if tc and first_call_turn is None:
            first_call_turn = i

    # --- multi-tool requirement ---------------------------------------------
    if len(all_calls) < min_tc:
        return False, f"only {len(all_calls)} tool call(s) (need ≥{min_tc})"

    # --- clarification turn before first tool call --------------------------
    if requires_clar:
        if first_call_turn is None:
            return False, "no tool calls present"
        # find FIRST assistant turn; it must have no tools_called
        first_assistant_idx = next(
            (i for i, t in enumerate(turns) if t.get("role") == "assistant"),
            None,
        )
        if first_assistant_idx is None:
            return False, "no assistant turn"
        if first_assistant_idx >= first_call_turn:
            return False, "first assistant turn already has tool calls (no clarification)"
        first_assistant = turns[first_assistant_idx]
        if first_assistant.get("tools_called"):
            return False, "clarification turn has tool calls"
        # heuristic: clarification turn should look like a question
        content = (first_assistant.get("content") or "").strip()
        if "?" not in content and len(content) < 20:
            return False, "clarification turn doesn't look like a question"

    # --- per-call schema check ----------------------------------------------
    called_names: list[str] = []
    for turn_idx, call in all_calls:
        name = call.get("name")
        if name not in schema_index:
            return False, f"unknown tool {name!r}"
        params = call.get("input_parameters")
        if not isinstance(params, dict):
            return False, f"{name}: input_parameters not a dict"
        out = call.get("output")
        if not isinstance(out, str):
            return False, f"{name}: output must be a JSON string"
        # required params present
        spec = schema_index[name]
        required = set(spec.get("parameters", {}).get("required", []) or [])
        missing = required - set(params.keys())
        if missing:
            return False, f"{name}: missing required {sorted(missing)}"
        # no extraneous keys
        allowed = set((spec.get("parameters", {}).get("properties") or {}).keys())
        if allowed:
            extra = set(params.keys()) - allowed
            if extra:
                return False, f"{name}: extraneous keys {sorted(extra)}"
        called_names.append(name)

    # --- target tools all called --------------------------------------------
    target_tools = seed.get("target_tools") or []
    missing_targets = [t for t in target_tools if t not in called_names]
    if missing_targets:
        return False, f"target_tools not called: {missing_targets}"

    # --- inter-group: tool calls span ≥2 distinct groups --------------------
    if groups_req:
        if tool_to_group is None:
            return False, "tool_to_group required for inter-group validation"
        groups_seen = {tool_to_group.get(n) for n in called_names}
        groups_seen.discard(None)
        missing_groups = groups_req - groups_seen
        if missing_groups:
            return False, f"groups not covered: {sorted(missing_groups)}"
        if len(groups_seen) < 2:
            return False, f"only 1 group hit (need ≥2 for inter-group)"

    return True, ""


# ============================================================================
# 4. run_intergroup()  ——  ADD  (new public entry point alongside run_group)
# ============================================================================
def run_intergroup(
    name: str,
    allowlists_by_group: dict[str, set[str]],
    seeds: list[dict[str, Any]],
    output_path: str,
) -> None:
    """
    Inter-group generation entry point.

    Parameters
    ----------
    name : str
        Identifier for logs and the `group` field on output records
        (e.g., "inter_comms_system").
    allowlists_by_group : dict[str, set[str]]
        Maps group_name -> set of tool names from that group that may be used.
        The combined allowlist is the union. Example:
            {"comms":  {"sms", "email", "phone"},
             "system": {"battery", "device_info"}}
    seeds : list[dict]
        Seeds in the new format. Every seed MUST set `groups_required` to a
        list/set of ≥2 group names from `allowlists_by_group`.
    output_path : str
        JSONL output path.

    The shared engine (model loading, prompt building, batched generation,
    JSON extraction, validation, dedup, resume, write) is reused — this
    wrapper just builds the combined allowlist and the tool→group map, then
    delegates to the same loop used by run_group().
    """
    # 1. Build combined allowlist (union)
    combined_allowlist: set[str] = set().union(*allowlists_by_group.values())

    # 2. Build tool→group reverse map for validator
    tool_to_group: dict[str, str] = {}
    for grp, tools in allowlists_by_group.items():
        for t in tools:
            tool_to_group[t] = grp

    # 3. Sanity-check seeds reference valid groups & tools
    valid_groups = set(allowlists_by_group.keys())
    cleaned_seeds = []
    for s in seeds:
        gr = set(s.get("groups_required") or [])
        if len(gr) < 2:
            print(f"[!] inter-group seed without ≥2 groups_required, skipping: "
                  f"{s.get('intent', '')[:60]!r}")
            continue
        unknown_groups = gr - valid_groups
        if unknown_groups:
            print(f"[!] seed references unknown groups {unknown_groups}, "
                  f"skipping: {s.get('intent','')[:60]!r}")
            continue
        unknown_tools = [t for t in (s.get("target_tools") or [])
                         if t not in combined_allowlist]
        if unknown_tools:
            print(f"[!] seed targets tools not in combined allowlist "
                  f"{unknown_tools}, skipping: {s.get('intent','')[:60]!r}")
            continue
        cleaned_seeds.append(s)

    # 4. Delegate to the same generation loop run_group() uses.
    #    The loop calls validate_sample(obj, seed, schema_index, tool_to_group=tool_to_group)
    #    so cross-group coverage is enforced automatically.
    #
    #    Pseudocode of what your existing run_group_generation() needs to do
    #    differently for inter-group:
    #      - filter schema by combined_allowlist (not single allowlist)
    #      - pass tool_to_group through to validate_sample
    #
    # Easiest integration: refactor your existing run_group() to call a private
    # _run_generation(name, allowlist, seeds, output_path, tool_to_group=None),
    # and have run_intergroup() call the same with tool_to_group populated.
    #
    # Example:
    _run_generation(
        name=name,
        allowlist=combined_allowlist,
        seeds=cleaned_seeds,
        output_path=output_path,
        tool_to_group=tool_to_group,   # <-- only addition over intra-group
    )


# ============================================================================
# 5. Refactor note: split run_group() so both entry points share the loop
# ============================================================================
# In your existing generator_core.py, run_group() currently does CLI parsing,
# model loading, and the main generation loop together. Refactor like this:
#
#   def run_group(group_name, allowlist, seeds, output_path):
#       # ... CLI parsing, model loading ...
#       _run_generation(group_name, allowlist, seeds, output_path,
#                       tool_to_group=None)
#
#   def run_intergroup(name, allowlists_by_group, seeds, output_path):
#       # (this file, above)
#       _run_generation(name, combined_allowlist, cleaned_seeds, output_path,
#                       tool_to_group=tool_to_group)
#
#   def _run_generation(name, allowlist, seeds, output_path, tool_to_group=None):
#       # main loop: pick seed → build prompt → generate → extract JSON
#       # → validate_sample(obj, seed, schema_index, tool_to_group)
#       # → dedup → write
#
# Only validate_sample() needs the new tool_to_group param. Everything else
# works as-is.
