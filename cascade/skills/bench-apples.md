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
-n 128 -c 4096 -b 4096 -ngl 999 --temp 0 --seed 123
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
- **Acceptance length**: `n_predict / (n_predict - n_accept)`
- **Speedup**: `eagle_tps / baseline_tps`

## Last known good baselines (RTX 5090, 2026-03-24, commit 327b6f2fa)

### Llama 3.2-1B bf16

| Mode | t/s | Acc length |
|------|-----|------------|
| Baseline (no spec) | ~466 | - |
| EAGLE3 | ~436 | 3.8 |
| Speedup | 0.94x | - |

Note: 1B is too fast on RTX 5090 for spec decoding to help. Draft head overhead dominates. This model is useful for correctness validation, not speedup testing.

### Llama 3.1-8B bf16 (10-prompt average)

| Mode | t/s | Acc length |
|------|-----|------------|
| Baseline (no spec) | ~72 | - |
| EAGLE3 | ~103 | 3.2 |
| Speedup | 1.44x (+44%) | - |

### Qwen3.5-4B bf16

| Mode | t/s | Acc length |
|------|-----|------------|
| Baseline (no spec) | ~134 | - |
| EAGLE3 | ~88 | 2.3 |
| Speedup | 0.66x | - |

Note: 4B is also too fast on RTX 5090. Acceptance length 2.3 matches kestrel exactly, confirming correctness. Eagle head is still training (step 900k).

## What to report

For each model config, report a table with: Model, Baseline t/s, EAGLE3 t/s, Speedup %, Acceptance Length. Compare against the baselines above to detect regressions.
