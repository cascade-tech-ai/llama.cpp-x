---
name: benchmarking
description: Run EAGLE3 benchmarks and compare performance/correctness against known-good baselines
user-invocable: true
---

# Apples-to-Apples EAGLE3 Benchmark

Run standardized benchmarks comparing EAGLE3 speculative decoding performance AND correctness. All runs use the same prompts, parameters, and hardware.

This document has two sections:
1. **Performance benchmarks** — throughput (t/s), acceptance length, speedup
2. **Correctness verification** — token-for-token comparison against baseline, greedy divergence diagnostics

## CRITICAL: Exact command line fidelity

**Always use the EXACT command lines documented here. Do NOT add, remove, or change any flags.**

The most important rule: **chat templating MUST be applied before benchmarking.** The canonical path is now the dataset wrapper:

```
cascade/scripts/benchmark_llama_speculative.py
```

This wrapper renders prompts with the HF tokenizer chat template using the same prompt-construction semantics as `kestrel train.py eval`, then invokes `llama-speculative-simple` with `--no-conversation` on the already-rendered prompt text via `-p`. It also disables GGUF-side BOS insertion by default (`--override-kv tokenizer.ggml.add_bos_token=bool:false`) so llama.cpp does not prepend a second BOS to a prompt that already starts with one. For future benchmarking, use the wrapper rather than hand-building prompt files or calling the binary directly.

Do not use `-f` for parity benchmarks with rendered prompts. In llama.cpp's common arg parser, `-f/--file` strips one trailing newline from the loaded prompt. That corrupts prompts that intentionally end with `\n\n`, which changes the final prompt token for Llama 3 chat headers and breaks tensor-level parity.

If you bypass the wrapper and call `llama-speculative-simple` directly, you must still ensure the prompt is chat-templated first. Raw user text is not a valid parity benchmark setup and can collapse acceptance length from ~2.0 to ~0.6.

If you need to deviate from the documented commands for any reason, explain the deviation to the user BEFORE running. Do not silently add or remove flags.

## Canonical benchmark driver

For future dataset benchmarks in this repo, the canonical driver is:

```
python3 cascade/scripts/benchmark_llama_speculative.py
```

Its defaults are split into two groups:

- Shared semantic defaults intentionally match `kestrel train.py eval`: `--spec-type eagle3`, `--eval-mode beam_prefix`, `--max-proposals 8`, `--max-depth 7`, `--max-new-tokens 128`, `--seed 42`, `--temp 0`, `--top-k 0`, and `--enable-thinking` unset by default.
- llama.cpp-only runtime defaults are explicit local defaults: `--ctx-size 4096`, `--batch-size 4096`, `--ubatch-size 4096`, `--gpu-layers 999`, `--gpu-layers-draft 999`, `--flash-attn on`, `--eagle-adaptive-depth 0.05`, `--eagle-beam-width 8`, and `--eagle-per-beam-topk-candidates 128`.

When running parity benchmarks against Kestrel serial eval, override the shared defaults as needed, especially:

- `--eval-mode serial`
- `--enable-thinking false`
- `--eagle-adaptive-depth 0`

The raw `llama-speculative-simple` commands below remain useful as implementation history and for one-off diagnostics, but the wrapper is the preferred benchmark entry point going forward.

## Test set

Use 10 prompts extracted from `perfect_1k.jsonl` (NOT the full dataset). If `/tmp/bench_prompts/` doesn't exist, create it:

```python
import json, os
os.makedirs('/tmp/bench_prompts', exist_ok=True)
with open('/home/alvion/projects/kestrel/datasets/perfect_1k.jsonl') as f:
    lines = f.readlines()
candidates = []
for i, line in enumerate(lines):
    d = json.loads(line)
    text = ' '.join(m['content'] for m in d['messages'] if m['role'] == 'user')
    if 2000 <= len(text) <= 8000:
        candidates.append((len(text), text))
candidates.sort()
step = len(candidates) // 10
for j in range(10):
    with open(f'/tmp/bench_prompts/{j:02d}.txt', 'w') as f:
        f.write(candidates[j * step][1])
```

Each prompt is 2-8k characters from user turns only. Run each prompt file with `-p "$(cat /tmp/bench_prompts/NN.txt)"`.

## Available models

| Model | Base GGUF | EAGLE Head |
|-------|-----------|------------|
| Llama 3.2-1B bf16 | ~/models/unsloth-Llama-3.2-1B-Instruct-bf16.gguf | ~/projects/kestrel/models/llama3-1b_eagle.bf16.gguf |
| Llama 3.1-8B bf16 | ~/models/unsloth-Meta-Llama-3.1-8B-Instruct-bf16.gguf | ~/projects/kestrel/models/llama3.1-8b_eagle.bf16.gguf |
| Llama 3.1-8B bf16 (RedHatAI) | ~/models/unsloth-Meta-Llama-3.1-8B-Instruct-bf16.gguf | ~/models/RedHatAI-Llama-3.1-8B-Instruct-speculator.eagle3-bf16.gguf |
| Llama 3.1-8B q8_0 | ~/models/unsloth-Meta-Llama-3.1-8B-Instruct-Q8_0.gguf | ~/projects/kestrel/models/llama3.1-8b_eagle.bf16.gguf |
| Qwen3.5-4B bf16 | ~/models/Qwen3.5-4B-bf16.gguf | ~/models/qwen3.5-4b_eagle.bf16.gguf |

Note: The RedHatAI eagle head uses `norm_before_residual=true` and significantly outperforms the kestrel and yuhuili heads for Llama 8B (~2.95 vs ~2.0 acc_len).

## Binary

```
build/bin/llama-speculative-simple
```

---

# Part 1: Performance Benchmarks

## Standard parameters

Common flags for ALL runs (baseline and speculative):
```
-n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on
```

EAGLE3 tree mode additional flags:
```
--spec-type eagle3 --eagle-max-depth 7 --eagle-max-proposals 16
--eagle-beam-width 8 --eagle-per-beam-topk-candidates 128 -md <EAGLE>
```

EAGLE3 serial mode additional flags:
```
--spec-type eagle3 --eagle-max-depth 7 --eagle-serial -md <EAGLE>
```

Note: In serial mode, `--eagle-max-proposals` and `--eagle-beam-width` are automatically overridden to match depth. You can pass them but they will be ignored.

Baseline runs use `--spec-type none` (same binary, no draft model, no `-md`).

Adaptive depth (`--eagle-adaptive-depth 0.05`) is ON by default. It cuts off low-confidence drafts early, improving throughput with minimal impact on acceptance length. To disable, pass `--eagle-adaptive-depth 0`. To use a different threshold, pass e.g. `--eagle-adaptive-depth 0.10`.

## How to compute metrics

From the output:
- **t/s**: from `decoded N tokens in X seconds, speed: Y t/s`
- **Acceptance length**: `acc_len` in the output. This is `n_accept / (n_predict - n_accept)` = average accepted draft tokens per speculative cycle.
- **Average depth**: `avg_depth` in the output. This is the average number of draft tokens proposed per cycle. With adaptive depth enabled, this will be less than `eagle_max_depth`.
- **Speedup**: `eagle_tps / baseline_tps`

When reporting acceptance metrics:
- Always report **acceptance length**, because that is the metric that matters for performance comparisons.
- Always report **average depth** — it shows how aggressively adaptive depth is pruning.
- Do not substitute draft acceptance rate (`n_accept / n_drafted`) for acceptance length. That ratio is low-signal noise.

## Performance history

### Run 3: 2026-04-05, eagle3-reorg branch, CUDA 13.1, Driver 590.48.01, RTX 5090

Exact commands:

Baseline:
```
./build/bin/llama-speculative-simple -m <MODEL> -n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on --spec-type none -p "$(cat /tmp/bench_prompts/NN.txt)"
```

Tree:
```
./build/bin/llama-speculative-simple -m <MODEL> -n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on --spec-type eagle3 --eagle-max-depth 7 --eagle-max-proposals 16 --eagle-beam-width 8 --eagle-per-beam-topk-candidates 128 -md <EAGLE> -p "$(cat /tmp/bench_prompts/NN.txt)"
```

Serial:
```
./build/bin/llama-speculative-simple -m <MODEL> -n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on --spec-type eagle3 --eagle-max-depth 7 --eagle-serial -md <EAGLE> -p "$(cat /tmp/bench_prompts/NN.txt)"
```

Note: Run 3 was performed before adaptive depth became the default. The `avg_depth` column was not recorded. Future runs should always report `avg_depth`.

#### Llama 3.1-8B bf16 — kestrel eagle (10-prompt average)

| Mode | t/s | Acc length | Avg depth | Speedup |
|------|-----|------------|-----------|---------|
| Baseline | 97.2 | - | - | - |
| Tree (no adaptive) | 136.8 | 2.03 | 7.0 | 1.41x |

#### Llama 3.1-8B bf16 — RedHatAI eagle (10-prompt average)

| Mode | t/s | Acc length | Avg depth | Speedup |
|------|-----|------------|-----------|---------|
| Baseline | 97.2 | - | - | - |
| Tree (no adaptive) | 173.4 | 2.95 | 7.0 | 1.78x |
| Tree (default, adaptive 0.05) | 193.5 | 2.89 | n/a | 1.99x |
| Tree + adaptive 0.10 | 191.1 | 2.74 | n/a | 1.97x |
| Serial (no adaptive) | 190.0 | 2.37 | 7.0 | 1.95x |
| Serial + adaptive 0.10 | 191.5 | 2.25 | n/a | 1.97x |

Note: RedHatAI eagle head (`norm_before_residual=true`) dramatically outperforms kestrel. Tree with default adaptive 0.05 is the best config at 2x speedup.

#### Qwen3.5-4B bf16 — serial mode sweep (10-prompt average)

| Mode | t/s | Acc length | Avg depth | Speedup |
|------|-----|------------|-----------|---------|
| Baseline | 147.1 | - | - | - |
| Tree (no adaptive) | ~132 | 2.19 | 7.0 | 0.90x |
| Tree + adaptive 0.02 | ~146 | 2.13 | n/a | 0.99x |
| Serial (no adaptive) | 158.6 | 1.57 | 7.0 | 1.08x |
| Serial (default, adaptive 0.05) | 167.5 | 1.41 | n/a | 1.14x |
| Serial + adaptive 0.10 | 169.2 | 1.35 | n/a | 1.15x |

Note: Tree mode is slower than baseline for Qwen3.5-4B due to per-token conv unrolling overhead. Serial + adaptive 0.10 is the best config at 1.15x speedup.

### Run 2: 2026-04-04, commit 253a5aba7 (verified and reproduced in Run 3)

Run 2 numbers for kestrel eagle models have been independently verified. The kestrel 8B tree acc_len=2.00 was reproduced as 2.03 in Run 3.

### Run 1: 2026-03-24, commit 327b6f2fa (STALE — NOT REPRODUCIBLE)

The acceptance lengths from Run 1 (3.1, 3.2, 2.3) could not be reproduced. The original numbers were likely computed from a different metric or test setup.

---

# Part 2: Correctness Verification

Tree-based speculative decoding can produce output that diverges from baseline (non-speculative) at temp=0. This is a numerical precision issue: tree batch evaluation uses different GPU kernel code paths than sequential single-token evaluation, producing slightly different logits. These small differences can flip close greedy argmax decisions.

This is NOT a logic bug — it affects all quantization types (bf16, Q8_0) and both tree implementations (flat tree and coupled sequences produce identical divergence patterns).

## Tool 1: Output text comparison

Compare the text output of baseline vs speculative runs. This is the simplest correctness check.

**Method:** Run baseline and speculative on the same prompt, capture stdout (logs go to stderr), diff.

```bash
# Baseline
./build/bin/llama-speculative-simple -m <MODEL> -n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on --spec-type none -p "$(cat /tmp/bench_prompts/00.txt)" 2>/dev/null > /tmp/baseline.txt

# Speculative (tree)
./build/bin/llama-speculative-simple -m <MODEL> -n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on --spec-type eagle3 --eagle-max-depth 7 --eagle-max-proposals 16 --eagle-beam-width 8 --eagle-per-beam-topk-candidates 128 -md <EAGLE> -p "$(cat /tmp/bench_prompts/00.txt)" 2>/dev/null > /tmp/speculative.txt

diff /tmp/baseline.txt /tmp/speculative.txt && echo "PASS" || echo "FAIL"
```

**Important:** stdout/stderr separation is critical. Text goes to stdout, logs to stderr. Use `2>/dev/null` to suppress logs. Do NOT use `--no-conversation` — it changes the prompt and invalidates the comparison.

### Correctness results: Run 3 (2026-04-05, 10 prompts each)

Llama 3.1-8B bf16 + kestrel eagle, tree mode:

| Prompt | Match? |
|--------|--------|
| 00 | PASS |
| 01 | FAIL |
| 02 | FAIL |
| 03 | PASS |
| 04 | FAIL |
| 05 | PASS |
| 06 | PASS |
| 07 | FAIL |
| 08 | FAIL |
| 09 | FAIL |

**6/10 prompts diverge** from baseline in tree mode. Serial mode for Qwen3.5-4B passes all prompts (serial verification uses standard sequential batch, not tree batch).

## Tool 2: Greedy divergence diagnostics (`--eagle-verify-greedy`)

The `--eagle-verify-greedy` flag re-evaluates each accepted token one-at-a-time after tree acceptance, comparing the sequential greedy argmax against the tree-accepted token. It reports:
- How many tokens diverged
- The probability gap between the sequential-greedy token and the tree-accepted token (min/max/mean)

This flag also CORRECTS the accepted tokens to match sequential greedy, so the final output will match baseline. The correction is a necessary side effect of accurate measurement — each position must be evaluated in the correct context to measure subsequent positions.

**Warning:** This flag makes speculative decoding significantly slower (each accepted token requires a separate decode call). It is a diagnostic tool, not for production use.

```bash
./build/bin/llama-speculative-simple -m <MODEL> -n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on --spec-type eagle3 --eagle-max-depth 7 --eagle-max-proposals 16 --eagle-beam-width 8 --eagle-per-beam-topk-candidates 128 --eagle-verify-greedy -md <EAGLE> -p "$(cat /tmp/bench_prompts/NN.txt)"
```

Output includes lines like:
```
verify-greedy: 5/129 tokens differ (3.9%)
verify-greedy: prob_diff min=0.000784 max=0.791968 mean=0.309793
```

The `prob_diff` is `p(sequential_greedy) - p(tree_accepted)` under sequential softmax. A small prob_diff means the model was nearly indifferent between the two tokens. A large prob_diff means the tree batch produced a meaningfully wrong logit ranking.

### Greedy divergence results: Run 3 (2026-04-05)

Llama 3.1-8B bf16 + kestrel eagle, tree mode, 10 prompts:

| Prompt | Tokens differ | % | prob_diff (mean) |
|--------|--------------|---|-----------------|
| 00 | 0/129 | 0.0% | - |
| 01 | 5/129 | 3.9% | 0.310 |
| 02 | 7/129 | 5.4% | 0.273 |
| 03 | 0/129 | 0.0% | - |
| 04 | 1/129 | 0.8% | 0.018 |
| 05 | 0/129 | 0.0% | - |
| 06 | 0/129 | 0.0% | - |
| 07 | 1/129 | 0.8% | 0.008 |
| 08 | 1/129 | 0.8% | 0.009 |
| 09 | 1/129 | 0.8% | 0.007 |

**Total: 15/1290 tokens differ (1.2%)**. Most mismatches have small probability gaps (< 2%), meaning the model was nearly indifferent. A few have large gaps (up to 100%), indicating the tree batch logits were meaningfully wrong at those positions.

Key findings:
- Both flat tree and coupled sequences produce identical divergence patterns — the issue is in batched attention, not tree structure.
- All quantization types are affected (bf16 and Q8_0 show similar ~1% divergence rates).
- Serial mode does not have this issue (verification passes 10/10).
- The root cause is under investigation — likely related to flash attention kernel differences between batched and single-token evaluation.

## What to report after a rebase or code change

Run both tools and compare against the baselines above:

1. **Performance:** Run the 10-prompt benchmark for at least Llama 8B tree mode. Report t/s, acc_len, avg_depth, speedup. Compare against Run 3 numbers.

2. **Correctness:** Run text comparison on all 10 prompts for Llama 8B tree mode. Report how many pass. Then run `--eagle-verify-greedy` on all 10 prompts and report total tokens differing and mean prob_diff. Compare against the 15/1290 (1.2%) baseline.

3. A regression in performance (acc_len drop > 10%) or correctness (divergence rate increase > 2x) warrants investigation before proceeding.
