# Groupwise Multi-Turn Tool-Calling Dataset Generation

### A Walkthrough of the Pipeline Design

---

## TL;DR

We use **Qwen3-8B** to synthesize a multi-turn conversation dataset where users ask an Android assistant to do things and the assistant responds with structured tool calls. Instead of generating freely (which causes mode collapse and uneven tool coverage), we **split the 39-tool Samsung catalog into 11 thematic groups** and run a dedicated generator per group. Each generator is a **thin recipe** (~80 lines defining seeds and a tool allowlist) that delegates all heavy lifting to a **shared core engine** (`generator_core.py`). A shell orchestrator (`run_all_groups.sh`) runs the 11 generators sequentially. The result is ~200+ diverse, validated, deduplicated samples spread evenly across all tool categories.

---

## 1. The Problem

We need training data for a small on-device tool-calling model. The data must be:

- **Multi-turn** — real conversations, not single Q&A pairs.
- **Tool-grounded** — every assistant response must include valid, well-formed tool calls.
- **Diverse** — covering all 39 tools, not just the 5 most obvious ones.
- **Realistic** — phrased the way actual users speak (casual, professional, technical).

Three things make naive "ask an LLM to generate examples" approaches fail badly:

| Problem | What goes wrong |
|---|---|
| **Mode collapse** | Free generation drifts to the same 3-4 themes ("save battery", "check weather"). You get 80 samples that all say the same thing. |
| **Tool coverage** | Some tools get 50 samples, others get zero. The fine-tuned model never learns the rare ones. |
| **Quality** | Malformed JSON, hallucinated tool names, missing arguments, scenarios that don't actually use the targeted tools. |

The pipeline is built around solving all three.

---

## 2. The Core Idea

> **Don't generate freely. Generate within constrained slices, then merge.**

We partition the tools into 11 themed groups. For each group, the LLM is given **only the tools in that group** plus **pre-written scenario seeds** that pin the topic. It cannot drift, because the off-topic tools aren't in its prompt at all.

This single decision solves mode collapse (constrained tool space), guarantees coverage (every tool belongs to a group, every group has dedicated quota), and lets us validate against a much smaller schema per call (faster, fewer false positives).

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  run_all_groups.sh                           │
│                                                              │
│   Orchestrator. Runs gen_01 → gen_11 sequentially.           │
│   Each script writes its own JSONL. Merge at the end.        │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              │   for each of the 11 groups:
                              ▼
       ┌──────────────────────────────────────────────┐
       │   gen_NN_<group>.py     (thin recipe)        │
       │                                              │
       │   GROUP_NAME   = "personal_info"             │
       │   ALLOWLIST    = {calendar, alarms, ...}     │
       │   SEEDS        = [scenario1, scenario2, ...] │
       │                                              │
       │   run_group(NAME, ALLOWLIST, SEEDS, output)  │
       └─────────────────────┬────────────────────────┘
                             │
                             ▼
       ┌──────────────────────────────────────────────┐
       │   generator_core.py     (shared engine)      │
       │                                              │
       │   • Load Qwen3-8B once                       │
       │   • Filter tools by allowlist                │
       │   • Loop: pick seed → prompt → generate →    │
       │           validate → dedup → write           │
       └──────┬──────────────────┬─────────────────┬──┘
              │                  │                 │
              ▼                  ▼                 ▼
      ┌────────────┐    ┌────────────────┐  ┌──────────────────┐
      │ Qwen3-8B   │    │ Tool catalog   │  │ group_NN_*.jsonl │
      │ (HF model) │    │ (39 tools JSON)│  │   (output)       │
      └────────────┘    └────────────────┘  └──────────────────┘
```

After all 11 finish, we concatenate the per-group files into one master dataset.

---

## 4. The Two-Tier Design

The codebase has exactly two kinds of files:

**Tier 1 — Recipes (thin, data-only).**
Each `gen_NN_<group>.py` defines three things and nothing else: a name, an allowlist, and a list of seeds. There is **zero generation logic** in these files.

```python
from generator_core import run_group

GROUP_NAME = "personal_info"
ALLOWLIST  = {"calendar", "alarms", "accounts"}

SEEDS = [
    {
        "intent":           "What's on my calendar today?",
        "target_tools":     ["calendar"],
        "hint":             "Use calendar with today action.",
        "persona":          "casual",
        "expected_outcome": "Today's events listed.",
    },
    # ... ~15 more seeds
]

if __name__ == "__main__":
    run_group(GROUP_NAME, ALLOWLIST, SEEDS, "group_08_personal_info.jsonl")
```

**Tier 2 — The engine (shared, ~500 LOC).**
`generator_core.py` contains every piece of logic: model loading, prompt building, batched generation, JSON extraction, validation, deduplication, file I/O, resume support. The 11 group scripts all call into the same engine.

**Why this matters:**

- Adding a new group is ~80 lines. Just write a new recipe.
- Fixing a validation bug is one edit. All 11 groups benefit immediately.
- Swapping the LLM backend (HF Transformers → vLLM → llama.cpp) touches two functions in the core. Recipes don't change.
- The split keeps reasoning about "what to generate" (recipes) separate from "how to generate" (engine).

---

## 5. End-to-End Flow — Walking Through One Group

Here's what happens when you run `python gen_08_personal_info.py --num-samples 20`:

```
                ┌──────────────────────────────┐
                │  run_group()  — CLI entry    │
                └──────────────┬───────────────┘
                               │
            parse args, seed RNGs, load Qwen3-8B once (~3 min)
                               │
                               ▼
                ┌──────────────────────────────┐
                │  run_group_generation()      │
                └──────────────┬───────────────┘
                               │
                ┌──────────────┴──────────────┐
                │   Setup phase (one-time)    │
                ├─────────────────────────────┤
                │ 1. Load tool catalog        │
                │    Filter to allowlist:     │
                │    39 tools → 3 tools       │
                │                             │
                │ 2. Prepare seeds            │
                │    Validate against schema  │
                │    Add coverage seeds for   │
                │    any tools missing from   │
                │    SEEDS                    │
                │                             │
                │ 3. Resume support           │
                │    Read existing JSONL,     │
                │    rebuild signature set    │
                └──────────────┬──────────────┘
                               │
                               ▼
        ╔══════════════════════════════════════════╗
        ║          MAIN GENERATION LOOP            ║
        ║   (until valid_count >= 20  OR           ║
        ║    attempts >= num_samples × 6)          ║
        ╚══════════════════════╤═══════════════════╝
                               │
       ┌───────────────────────┴───────────────────────┐
       │                                               │
       │  next_seed()   round-robin pick               │
       │     ↓                                         │
       │  build_user_prompt()                          │
       │     intent + persona + hint + tool defs       │
       │     ↓                                         │
       │  build_chat_prompt()                          │
       │     Qwen3 chat template wrapping              │
       │     ↓                                         │
       │  ───── batch B=4 prompts ─────                │
       │     ↓                                         │
       │  generate_batch()  ← Qwen3 inference (~30s)   │
       │     ↓                                         │
       │  for each generated string:                   │
       │     • strip <think>...</think>                │
       │     • extract_json()                          │
       │     • validate_sample()                       │
       │     • dedup check (signature hash)            │
       │     • per-seed cap check                      │
       │     • write to JSONL                          │
       │     ↓                                         │
       │  print progress: valid=X/N attempts=Y         │
       │                                               │
       └───────────────────────┬───────────────────────┘
                               │
                       loop done → close file
                       print final stats per seed
```

This same loop runs once per group. The only thing that changes between groups is the recipe inputs.

---

## 6. The Stages, Explained

### Stage 1 — Loading the model (once)

Qwen3-8B takes 2–5 minutes to load and ~16 GB of VRAM (8 GB with 4-bit quantization). We load it **once** at the start of the script and reuse it for every sample. Loading per-sample would take days.

### Stage 2 — Filtering the tool catalog

The full Samsung tool JSON has 39 tools. The group's `ALLOWLIST` says "we only care about these 3". We drop the other 36 before they ever reach the prompt.

> **Why this is the most important step.** The LLM literally cannot suggest tools it has not been shown. Mode collapse is impossible because the off-topic tools don't exist in its world.

### Stage 3 — Preparing seeds

We start with the hand-written SEEDS list. Two things happen:

1. **Schema check.** Drop any seed whose `target_tools` reference tools not in the schema (typos, stale lists).
2. **Coverage augmentation.** For every tool in the allowlist that isn't mentioned in any seed, auto-generate a minimal seed: `"Use the X tool"`. This guarantees every tool gets at least one prompt.

### Stage 4 — Picking the next seed

A round-robin picker with a per-seed cap. The cap is `max(3, 2 × ideal_share)` where `ideal_share = num_samples / num_seeds`. Without this, one "easy" seed would produce 60% of the dataset because the LLM finds it most natural.

### Stage 5 — Building the prompt

The prompt sent to Qwen3-8B contains:

- A strict system prompt: *"You MUST follow the Intent. Do NOT drift."*
- The tool definitions (filtered to allowlist).
- The seed: intent, hint, persona.
- Format instructions: produce a JSON object with `turns` and `tools_called`.

Then this gets wrapped in Qwen3's chat template (`<|im_start|>...<|im_end|>`).

### Stage 6 — Batched generation

Four prompts are processed together in a single forward pass. Batching is roughly 3× faster than serial generation on an A6000. Output: four raw strings.

### Stage 7 — JSON extraction

Qwen3 emits reasoning inside `<think>...</think>` blocks before the actual answer. We strip those, find the JSON object in the remainder, and parse it. Returns `None` if the output is malformed (rare with a strict prompt, but happens).

### Stage 8 — Validation

A sample passes validation if and only if:

1. Top-level shape is correct (`turns` array, `tools_called` array).
2. Every turn has a valid role (user/assistant) and content.
3. Every tool call references a tool that exists in the schema.
4. Every tool call's arguments match the tool's parameter schema.
5. The `target_tools` from the seed all actually appear in `tools_called` (unless `--no-strict-targets` is passed).

A failure on any check rejects the sample and the loop tries again.

### Stage 9 — Deduplication

We compute a signature for each candidate sample:

```
signature = normalize(first_user_turn[:150]) + "||" + sorted(tool_names_called)
```

If we've seen that signature before, reject. The signature includes tool names so that "check clipboard with `clipboard_get`" and "check clipboard with `clipboard_get` + `clipboard_set`" count as different scenarios.

### Stage 10 — Writing

Successful samples are appended to `group_NN_<name>.jsonl` and `flush()`ed immediately. If the script is killed, no data is lost.

---

## 7. Anti-Mode-Collapse — How We Actually Get Diversity

Three layers of defense:

| Layer | Mechanism |
|---|---|
| **Tool space** | Allowlist filtering — LLM only sees this group's tools. |
| **Topic space** | Pre-written seeds — each scenario is pinned to a specific intent. |
| **Sample space** | Per-seed cap + signature dedup — no single seed dominates, no two samples are near-duplicates. |

If you removed any one of these, the dataset would degrade. The combination is what makes the output usable for fine-tuning.

---

## 8. Resume and Robustness

The pipeline is designed to survive crashes, OOMs, and Ctrl-C without losing work.

- **Append-only writes** with explicit flush after every sample.
- **Signature reload** on startup. If `group_08_personal_info.jsonl` already has 12 samples, the script reads them, rebuilds `accepted_sigs`, and continues from sample 13.
- **Hard cap on attempts.** `max_attempts = num_samples × 6`. If the LLM keeps producing duplicates or invalid output, the script gives up cleanly rather than spinning forever.

You can interrupt any group, restart it days later, and pick up where it left off.

---

## 9. The 11 Groups

| # | Script | Group | Tools (allowlist) | Seeds |
|---|--------|-------|-------------------|-------|
| 01 | `gen_01_system_device.py` | system_device | device_info, sensors, diagnostics | 20 |
| 02 | `gen_02_display_ui.py` | display_ui | brightness, theme, rotation | 19 |
| 03 | `gen_03_audio.py` | audio | volume, ringer, vibration | 12 |
| 04 | `gen_04_app_mgmt.py` | app_mgmt | apps_info, app_launch | 17 |
| 05 | `gen_05_connectivity.py` | connectivity | wifi, bluetooth, hotspot | 16 |
| 06 | `gen_06_battery_power.py` | battery_power | battery, power_saver | 16 |
| 07 | `gen_07_memory_files.py` | memory_files | memory, files, storage | 18 |
| 08 | `gen_08_personal_info.py` | personal_info | calendar, alarms, accounts | 15 |
| 09 | `gen_09_communication.py` | communication | sms, contacts, share | 14 |
| 10 | `gen_10_clipboard.py` | clipboard | clipboard_get, clipboard_set | 11 |
| 11 | `gen_11_misc_util.py` | misc_util | watcher_manage, exec | 17 |

Total: ~175 distinct seeds across 11 groups. With per-seed sampling caps, the realistic max is ~210 samples; targeting 20 per group gives a clean, balanced 220-sample dataset.

---

## 10. The Output

Each line of `group_NN_<name>.jsonl` is one self-contained training example:

```json
{
  "id": "uuid",
  "group": "personal_info",
  "seed_intent": "What's on my calendar today?",
  "persona": "casual",
  "tools": [ ...filtered tool schemas... ],
  "turns": [
    {"role": "user",
     "content": "yo what's on my calendar today"},
    {"role": "assistant",
     "tools_called": [
       {"name": "calendar", "arguments": {"action": "today"}}
     ],
     "content": "You have 3 events today: ..."}
  ],
  "tools_called": [...],
  "expected_outcome": "Today's events listed."
}
```

After all groups finish:

```bash
cat group_*.jsonl > dataset_all_groups.jsonl
```

This master file is what gets fed into the fine-tuning pipeline.

---

## 11. Why This Design Holds Up

**The two-tier split** keeps the surface area for change small. Recipes are data; engine is logic. You only ever edit one or the other, not both.

**Group partitioning** turns one impossible generation problem ("cover 39 tools without drift") into 11 easy ones ("cover 3 tools without drift").

**Seeds + allowlist** give us deterministic control over topic distribution, while leaving the LLM free to produce natural language variation in user phrasing, persona, and assistant response style.

**Layered validation** (schema check → target-tool check → signature dedup → per-seed cap) catches every common failure mode before the data hits disk.

**Append + resume** means we never have to re-run successful work. Long generation runs are safe to interrupt.

The whole pipeline took about 800 lines of code (500 in the engine, ~80 × 11 in recipes, ~30 in the orchestrator). It produces dataset quality that free-form generation cannot match at any scale.

---

## How to Run It

**One group at a time:**

```bash
python gen_08_personal_info.py \
    --tools-file tool_suwon_ocfk_v2.json \
    --num-samples 20 \
    --model Qwen/Qwen3-8B \
    --batch-size 4 \
    --temperature 0.85 \
    --load-in-4bit
```

**All 11 groups in sequence:**

```bash
bash run_all_groups.sh tool_suwon_ocfk_v2.json 20 Qwen/Qwen3-8B
```

**Check progress:**

```bash
for f in group_*.jsonl; do
    echo "$f: $(wc -l < "$f") samples"
done
```

**Resume after interruption:** just re-run the same command. The script will pick up where it stopped.
