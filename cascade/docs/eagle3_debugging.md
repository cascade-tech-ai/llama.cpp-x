# EAGLE3 Debugging Notes (llama.cpp-x fork)

AI / upstream note:
This fork contains AI-assisted code and docs for experimentation. Do not upstream as-is. See `CONTRIBUTING.md`.

## What Was Wrong

We had a large mismatch between the llama.cpp EAGLE3 head forward pass and the PyTorch reference (Kestrel).
The earliest divergence showed up in the attention path at **RoPE on Q/K**:

- The PyTorch head uses HuggingFace `LlamaRotaryEmbedding` + `apply_rotary_pos_emb`, which corresponds to **NeoX-style RoPE pairing**:
  - pairs are `(i, i + head_dim/2)` (first half with second half).
- The llama.cpp head was effectively applying **"normal" RoPE pairing**:
  - pairs are `(0,1) (2,3) ...` (adjacent pairs).

This makes step 0 look OK (position 0 => RoPE is identity), then Q/K diverge massively starting at position 1, cascading into incorrect logits.

Also: the Kestrel CLI output that showed `"7" 97%` was the **base model** top-k, not the EAGLE head top-k (Kestrel prints base on the left, head on the right).

## What We Fixed

In `src/llama-eagle3.cpp`, the head runtime now forces:

- `rt.rope_type = LLAMA_ROPE_TYPE_NEOX`

Rationale: the exported EAGLE head matches HF's RoPE convention, and the base model's internal rope type in llama.cpp is not necessarily the same due to model-specific layout/permutations.

Additionally, for Llama 3-style RoPE scaling, the head needs the per-dimension RoPE factors tensor. We added a safe path to copy the base model's `rope_factors` tensor into a small CPU ggml context owned by the EAGLE3 runtime, so `ggml_rope_ext(..., rope_factors, ...)` behaves the same even when the base model is GPU-offloaded.

After this change, Q/K parity vs PyTorch becomes tight (sub-millifloat diffs), and the head logits match PyTorch within ~1e-3.

## Debugging Tooling Added

### 1) Forward-pass dumps from llama.cpp

Environment variable:

- `CASCADE_EAGLE_DUMP_DIR=/path/to/dir`

This writes a "state-dict style" dump as a directory of named tensors (`.npy`) plus `meta.json`, including:

- Teacher hidden states:
  - `teacher_hidden_layer_{layer}.npy` (per selected layer)
  - `teacher_hidden_concat_by_step.npy`
- Head forward intermediates (by step):
  - `head_embd_by_step.npy`
  - `head_embd_norm_by_step.npy`
  - `head_hidden_proj_by_step.npy`
  - `head_hidden_norm_by_step.npy`
  - `head_cat_by_step.npy`
  - `head_q_by_step.npy`
  - `head_k_by_step.npy`
  - `head_v_by_step.npy`
  - `head_hidden_after_step.npy`
- Final head logits (draft vocab):
  - `head_root_logits_draft.npy`

Code lives in:
- `common/speculative.cpp` (dump writer + orchestration)
- `src/llama-eagle3.{h,cpp}` (optional debug capture in `llama_eagle3_step()`)

### 2) PyTorch parity script

Script:
- `tools/eagle3/eagle3_parity.py`

It loads the llama.cpp dump and runs the Kestrel head in PyTorch, comparing tensors stage-by-stage and printing max-abs / RMS diffs and top-k logits.

Important implementation detail:
It does not need to run a full base model in PyTorch. It patches a minimal embedding table using `head_embd_by_step.npy` so the PyTorch head sees the exact embeddings llama.cpp used.

## Repro Commands (Known Prompt)

Generate a dump with llama.cpp:

```bash
rm -rf /tmp/eagle_llama && mkdir -p /tmp/eagle_llama
CASCADE_EAGLE_DUMP_DIR=/tmp/eagle_llama CASCADE_EAGLE_VERBOSE=1 \
  ./build/bin/llama-speculative-simple \
    -m /home/alvion/models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --model-draft /home/alvion/models/llama3-1b_eagle_001.gguf \
    --spec-type eagle3 \
    --eagle-max-depth 1 --eagle-max-proposals 8 --eagle-beam-width 8 --eagle-per-beam-topk-candidates 0 \
    -p "0, 1, 2, 3, 4, 5, 6, " \
    -n 1 -ngl -1 --temp 0 --top-k 1
```

Compare against PyTorch/Kestrel:

```bash
python3 tools/eagle3/eagle3_parity.py \
  --llama-dump /tmp/eagle_llama \
  --head-model /home/alvion/models/llama3-1b_eagle_001 \
  --topk 20
```

If parity is good, the first mismatching stage should be "none" (tiny numeric noise only).

## Notes

If the head does not place `"7"` as top-1 for this prompt, that can be expected: the EAGLE head is not the base model.
When comparing "should be 7", ensure you're comparing the same distribution (base vs head, and same vocab mapping).
