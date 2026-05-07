#!/usr/bin/env bash
# =============================================================================
# run_all_groups.sh
# =============================================================================
# Sequentially runs all 11 intra-group generators + the new inter-group
# generator(s). Each script writes its own JSONL into $OUT_DIR.
#
# Usage:
#   ./run_all_groups.sh <tools_file> <samples_per_group> [model] [out_dir] \
#                       [inter_samples]
#
# Examples:
#   ./run_all_groups.sh tool_suwon_ocfk_v2.json 30
#   ./run_all_groups.sh tool_suwon_ocfk_v2.json 30 Qwen/Qwen3-8B ./out 60
# =============================================================================

set -euo pipefail

TOOLS_FILE="${1:?usage: $0 <tools_file> <samples_per_group> [model] [out_dir] [inter_samples]}"
N_PER_GROUP="${2:?samples per group required}"
MODEL="${3:-Qwen/Qwen3-8B}"
OUT_DIR="${4:-./out}"
N_INTER="${5:-$(( N_PER_GROUP * 2 ))}"   # default: 2x per-group quota

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

COMMON_ARGS=(
    --tools-file  "../$TOOLS_FILE"
    --model       "$MODEL"
    --load-in-4bit
)

INTRA_GROUP_SCRIPTS=(
    gen_01_system_device.py
    gen_02_display_ui.py
    gen_03_audio.py
    gen_04_connectivity.py
    gen_05_comms.py
    gen_06_media.py
    gen_07_location.py
    gen_08_personal_info.py
    gen_09_notifications.py
    gen_10_apps.py
    gen_11_misc.py
)

INTER_GROUP_SCRIPTS=(
    gen_12_inter_group.py
)

echo "================================================================"
echo "Stage 1/2 — INTRA-GROUP generation (11 groups, $N_PER_GROUP each)"
echo "================================================================"
for script in "${INTRA_GROUP_SCRIPTS[@]}"; do
    echo
    echo "▶ Running $script"
    python "../$script" --num-samples "$N_PER_GROUP" "${COMMON_ARGS[@]}"
done

echo
echo "================================================================"
echo "Stage 2/2 — INTER-GROUP generation ($N_INTER samples)"
echo "================================================================"
for script in "${INTER_GROUP_SCRIPTS[@]}"; do
    echo
    echo "▶ Running $script"
    python "../$script" --num-samples "$N_INTER" "${COMMON_ARGS[@]}"
done

echo
echo "================================================================"
echo "Merging all JSONL files → master_dataset.jsonl"
echo "================================================================"
cat group_*.jsonl > master_dataset.jsonl
echo "Total samples: $(wc -l < master_dataset.jsonl)"
echo "Per-group breakdown:"
for f in group_*.jsonl; do
    printf "  %-40s %s\n" "$f" "$(wc -l < "$f")"
done
