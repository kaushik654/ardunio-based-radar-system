#!/usr/bin/env python3
"""
gen_08_personal_info.py — UPDATED seed format
==============================================
Same structure as before. Changes:
  • Every seed now has `requires_clarification`, `min_tool_calls`, `min_turns`.
  • Added `clarification_hint` to guide the LLM on what to ask.
  • Several seeds upgraded to multi-tool scenarios (alarms+calendar, etc).

Apply this same pattern to gen_01..gen_07, gen_09..gen_11.
"""
from generator_core import run_group


GROUP_NAME = "personal_info"
ALLOWLIST = {"calendar", "alarms", "accounts"}


SEEDS = [
    # ---- single-tool with clarification (most common pattern) ----
    {
        "intent": "Wake me up early tomorrow.",
        "target_tools": ["alarms"],
        "hint": "Use alarms with a set action; pin time after clarification.",
        "persona": "casual",
        "expected_outcome": "Alarm set for the chosen time.",
        "requires_clarification": True,
        "clarification_hint": "Ask exact time and whether to repeat on weekdays.",
        "min_tool_calls": 2,   # set + verify, OR snooze config + set
        "min_turns": 6,
    },
    {
        "intent": "What's on my calendar today?",
        "target_tools": ["calendar"],
        "hint": "Calendar list for today, then maybe details for a specific event.",
        "persona": "casual",
        "expected_outcome": "Today's events listed; one expanded on request.",
        "requires_clarification": True,
        "clarification_hint": "Ask whether to include all-day events or just timed.",
        "min_tool_calls": 2,
        "min_turns": 6,
    },

    # ---- multi-tool same-group (chain alarms + calendar) ----
    {
        "intent": "I have a flight at 6am on Friday — set it up properly.",
        "target_tools": ["calendar", "alarms"],
        "hint": "Create calendar event for the flight + alarm 2h before.",
        "persona": "professional",
        "expected_outcome": "Calendar entry and pre-flight alarm both set.",
        "requires_clarification": True,
        "clarification_hint": "Ask flight number/destination and alarm lead time.",
        "min_tool_calls": 2,
        "min_turns": 7,
    },
    {
        "intent": "Schedule team standups for the next 3 weekdays and remind me.",
        "target_tools": ["calendar", "alarms"],
        "hint": "Recurring calendar entry + recurring alarm.",
        "persona": "professional",
        "expected_outcome": "Recurring standup calendar + reminder alarm created.",
        "requires_clarification": True,
        "clarification_hint": "Ask time and whether to skip Fridays.",
        "min_tool_calls": 2,
        "min_turns": 6,
    },

    # ---- accounts coverage with multi-step ----
    {
        "intent": "Who am I logged in as?",
        "target_tools": ["accounts"],
        "hint": "Accounts list, then details on one if asked.",
        "persona": "neutral",
        "expected_outcome": "Active accounts shown with primary highlighted.",
        "requires_clarification": True,
        "clarification_hint": "Ask whether to include work and personal accounts.",
        "min_tool_calls": 2,
        "min_turns": 6,
    },

    # ---- a fully-specified seed where clarification is optional ----
    {
        "intent": "Set an alarm for 7am tomorrow, no snooze, label 'Gym'.",
        "target_tools": ["alarms"],
        "hint": "Direct set; everything specified.",
        "persona": "direct",
        "expected_outcome": "Alarm created exactly as described.",
        "requires_clarification": False,   # rare — use sparingly for variety
        "min_tool_calls": 2,               # still multi-call: set + verify
        "min_turns": 6,
    },
]


if __name__ == "__main__":
    run_group(
        GROUP_NAME,
        ALLOWLIST,
        SEEDS,
        "group_08_personal_info.jsonl",
    )
