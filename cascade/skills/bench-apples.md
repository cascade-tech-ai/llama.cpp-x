---
name: bench-apples
description: Run apples-to-apples EAGLE3 benchmarks and compare against known-good baselines
user-invocable: true
---

# Apples-to-Apples EAGLE3 Benchmark

Run standardized benchmarks comparing EAGLE3 speculative decoding performance. All runs use the same prompts, parameters, and hardware.

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

## Standard parameters

```
-n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on
```

EAGLE3 additional flags:
```
--spec-type eagle3 --eagle-max-depth 7 --eagle-max-proposals 16
--eagle-beam-width 8 --eagle-per-beam-topk-candidates 128
```

Baseline runs use `--spec-type none` (same binary, no draft model).

## Available models

| Model | Base GGUF | EAGLE Head |
|-------|-----------|------------|
| Llama 3.2-1B bf16 | ~/models/unsloth-Llama-3.2-1B-Instruct-bf16.gguf | ~/projects/kestrel/models/llama3-1b_eagle.bf16.gguf |
| Llama 3.1-8B bf16 | ~/models/unsloth-Meta-Llama-3.1-8B-Instruct-bf16.gguf | ~/projects/kestrel/models/llama3.1-8b_eagle.bf16.gguf |
| Llama 3.1-8B q8_0 | ~/models/unsloth-Meta-Llama-3.1-8B-Instruct-Q8_0.gguf | (same eagle head) |
| Llama 3.1-8B q4_k_m | ~/models/unsloth-Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | (same eagle head) |
| Qwen3.5-4B bf16 | ~/models/Qwen3.5-4B-bf16.gguf | ~/models/qwen3.5-4b_eagle.bf16.gguf |

## Binary

```
build/bin/llama-speculative-simple
```

## How to compute metrics

From the output:
- **t/s**: from `decoded N tokens in X seconds, speed: Y t/s`
- **Acceptance length**: accepted draft tokens per speculative cycle. When using traces, compute `sum(cycle.accepted_count) / len(cycles)`.
- **Draft acceptance rate**: `n_accept / n_drafted` from the example binary. This is only a low-signal debugging ratio and should not be used as the main performance metric.
- **Speedup**: `eagle_tps / baseline_tps`

## Baselines

### Run 2: 2026-04-04, commit 253a5aba7, CUDA 13.1, Driver 590.48.01

Exact command (baseline):
```
./build/bin/llama-speculative-simple -m <MODEL> -n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on --spec-type none -p "$(cat /tmp/bench_prompts/NN.txt)"
```

Exact command (EAGLE3 tree, depth=7, proposals=16, beam=8):
```
./build/bin/llama-speculative-simple -m <MODEL> -n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on --spec-type eagle3 --eagle-max-depth 7 --eagle-max-proposals 16 --eagle-beam-width 8 --eagle-per-beam-topk-candidates 128 -md <EAGLE> -p "$(cat /tmp/bench_prompts/NN.txt)"
```

Exact command (EAGLE3 serial, depth=7, proposals=16, beam=8):
```
./build/bin/llama-speculative-simple -m <MODEL> -n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123 -fa on --spec-type eagle3 --eagle-max-depth 7 --eagle-max-proposals 16 --eagle-beam-width 8 --eagle-per-beam-topk-candidates 128 --eagle-serial -md <EAGLE> -p "$(cat /tmp/bench_prompts/NN.txt)"
```

#### Llama 3.2-1B bf16 (10-prompt average, tree mode, depth=7, proposals=16, beam=8)

| Mode | t/s | Acc length |
|------|-----|------------|
| Baseline (no spec) | 462.6 | - |
| EAGLE3 | 376.1 | 2.12 |
| Speedup | 0.81x | - |

Note: 1B is too fast on RTX 5090 for spec decoding to help. Draft head overhead dominates.

#### Llama 3.1-8B bf16 (10-prompt average, tree mode, depth=7, proposals=16, beam=8)

| Mode | t/s | Acc length |
|------|-----|------------|
| Baseline (no spec) | 97.2 | - |
| EAGLE3 | 136.8 | 2.00 |
| Speedup | 1.41x (+41%) | - |

#### Qwen3.5-4B bf16 serial (10-prompt average, serial mode, depth=7, proposals=16, beam=8)

| Mode | t/s | Acc length |
|------|-----|------------|
| Baseline (no spec) | 140.5 | - |
| EAGLE3 serial | 150.7 | 1.58 |
| Speedup | 1.07x (+7%) | - |

#### Qwen3.5-4B bf16 tree (10-prompt average, tree mode, depth=7, proposals=16, beam=8)

| Mode | t/s | Acc length |
|------|-----|------------|
| Baseline (no spec) | 140.4 | - |
| EAGLE3 tree | 121.9 | 2.05 |
| Speedup | 0.87x | - |

Note: Tree mode is slower than serial for Qwen3.5-4B due to per-token conv unrolling overhead.

#### Kestrel reference comparison (same prompts, same params, PyTorch bf16)

Kestrel command (tree): `cd ~/projects/kestrel && .venv/bin/python -u train.py eval --dataset /tmp/bench_kestrel.jsonl --base-model <HF_ID> --head-model models/<EAGLE_DIR> --eval-mode beam_prefix --max-depth 7 --max-proposals 16 --beam-width 8 --max-new-tokens 128 --temp 0 --top-k 0 --seed 123 --enable-thinking false --max-items 10`

Kestrel command (serial): same but `--eval-mode serial`

| Model | Kestrel Tree | llama.cpp Tree | Ratio | Kestrel Serial | llama.cpp Serial | Ratio |
|-------|-------------|---------------|-------|---------------|-----------------|-------|
| Llama 3.2-1B | 2.103 | 2.123 | 101.0% | 1.736 | 1.626 | 93.7% |
| Llama 3.1-8B | 2.378 | 2.000 | 84.1% | 1.931 | 1.882 | 97.5% |
| Qwen 3.5-4B | 1.482 | 2.050 | 138.3% | 1.046 | 1.580 | 151.1% |

Notes:
- Llama 1B tree matches kestrel exactly. Serial has a small gap from greedy chain differences.
- Llama 8B tree is 84% of kestrel — beam search may have room for improvement.
- Qwen 3.5 llama.cpp outperforms kestrel — likely because the ggml delta-net kernel handles the hybrid model differently than PyTorch.

### Run 1: 2026-03-24, commit 327b6f2fa (STALE — numbers were not independently verified)

The acceptance lengths below (3.1, 3.2, 2.3) could not be reproduced. Re-running the
exact same prompts and parameters on commit 327b6f2fa produces acc_len=1.99 for Llama 8B,
matching Run 2. The original numbers were likely computed from a different metric or test.
The throughput difference (72 vs 97 t/s baseline for 8B) is attributed to a CUDA
driver update (590.48.01 vs whatever was installed on 2026-03-24).

| Model | Baseline t/s | EAGLE3 t/s | Acc length | Notes |
|-------|-------------|------------|------------|-------|
| Llama 3.2-1B bf16 | ~455 | ~350 | 3.1 | NOT REPRODUCIBLE |
| Llama 3.1-8B bf16 | ~72 | ~103 | 3.2 | NOT REPRODUCIBLE |
| Qwen3.5-4B bf16 | ~134 | ~88 | 2.3 | NOT REPRODUCIBLE |

## What to report

For each model config, report a table with: Model, Baseline t/s, EAGLE3 t/s, Speedup %, Acceptance Length. Compare against the baselines above to detect regressions.

When reporting acceptance metrics:
- Always report acceptance length, because that is the acceptance metric that matters for performance comparisons.
- Do not substitute draft acceptance rate for acceptance length.
