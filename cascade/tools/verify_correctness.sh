#!/bin/bash
# verify_correctness.sh — Token-for-token correctness check for EAGLE3 speculation.
#
# At temp=0 (greedy decoding), speculative output MUST exactly match baseline
# (non-speculative) output, token for token. Any divergence is a bug.
#
# Usage:
#   ./cascade/tools/verify_correctness.sh [--prompts-dir DIR] [--bin PATH]
#
# Required env or args:
#   MODEL      base model path     (or --model PATH)
#   EAGLE      eagle draft model   (or --eagle PATH)
#
# Example:
#   MODEL=~/models/Meta-Llama-3.1-8B-Instruct-bf16.gguf \
#   EAGLE=~/models/llama3.1-8b_eagle.bf16.gguf \
#   ./cascade/tools/verify_correctness.sh

set -euo pipefail

BIN="${BIN:-./build/bin/llama-speculative-simple}"
PROMPTS_DIR="${PROMPTS_DIR:-/tmp/bench_prompts}"
N_PREDICT="${N_PREDICT:-128}"
COMMON="-n $N_PREDICT -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on"
EAGLE_OPTS="--spec-type eagle3 --eagle-max-depth 7 --eagle-max-proposals 16 --eagle-beam-width 8 --eagle-per-beam-topk-candidates 128"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2"; shift 2 ;;
        --eagle)    EAGLE="$2"; shift 2 ;;
        --bin)      BIN="$2"; shift 2 ;;
        --prompts-dir) PROMPTS_DIR="$2"; shift 2 ;;
        --serial)   EAGLE_OPTS="--spec-type eagle3 --eagle-serial --eagle-max-depth 7"; shift ;;
        --n-predict) N_PREDICT="$2"; COMMON="-n $N_PREDICT -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL not set. Use MODEL=... or --model PATH"
    exit 1
fi
if [[ -z "${EAGLE:-}" ]]; then
    echo "ERROR: EAGLE not set. Use EAGLE=... or --eagle PATH"
    exit 1
fi
if [[ ! -d "$PROMPTS_DIR" ]]; then
    echo "ERROR: prompts directory not found: $PROMPTS_DIR"
    exit 1
fi

TMPDIR=$(mktemp -d /tmp/verify_correctness.XXXXXX)
trap "rm -rf $TMPDIR" EXIT

pass=0
fail=0
total=0

echo "=== EAGLE3 Token-for-Token Correctness Check ==="
echo "Model: $MODEL"
echo "Eagle: $EAGLE"
echo "Eagle opts: $EAGLE_OPTS"
echo "Prompts: $PROMPTS_DIR"
echo "Tokens: $N_PREDICT"
echo ""

for f in "$PROMPTS_DIR"/*.txt; do
    name=$(basename "$f" .txt)
    total=$((total + 1))

    prompt=$(cat "$f")

    # Run baseline (no speculation) — capture only generated text via --log-verbosity 0
    $BIN -m "$MODEL" $COMMON --spec-type none \
        --log-verbosity 0 -p "$prompt" \
        > "$TMPDIR/baseline_${name}.txt" 2>/dev/null

    # Run speculative — same params + eagle model
    $BIN -m "$MODEL" $COMMON $EAGLE_OPTS \
        -md "$EAGLE" --log-verbosity 0 -p "$prompt" \
        > "$TMPDIR/spec_${name}.txt" 2>/dev/null

    if diff -q "$TMPDIR/baseline_${name}.txt" "$TMPDIR/spec_${name}.txt" > /dev/null 2>&1; then
        printf "  %-8s PASS\n" "$name"
        pass=$((pass + 1))
    else
        printf "  %-8s FAIL\n" "$name"
        fail=$((fail + 1))
        # Show first divergence
        echo "    --- baseline ---"
        head -c 200 "$TMPDIR/baseline_${name}.txt"
        echo ""
        echo "    --- speculative ---"
        head -c 200 "$TMPDIR/spec_${name}.txt"
        echo ""
        echo "    --- diff ---"
        diff "$TMPDIR/baseline_${name}.txt" "$TMPDIR/spec_${name}.txt" | head -20
        echo ""
    fi
done

echo ""
echo "Results: $pass/$total passed, $fail failed"

if [[ $fail -gt 0 ]]; then
    echo "FAIL: Speculative output does not match baseline!"
    exit 1
else
    echo "ALL PASS: Speculative output matches baseline token-for-token."
    exit 0
fi
