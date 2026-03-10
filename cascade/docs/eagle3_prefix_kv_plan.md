# EAGLE3 Prompt-Prefilled KV Plan

AI / upstream note:
- This fork contains AI-assisted code and docs for experimentation. Do not upstream as-is. See `AGENTS.md` and `CONTRIBUTING.md`.

## Problem

Current llama.cpp Eagle3 rollout keeps only a short draft-depth KV cache per beam.

That differs from:
- reference EAGLE3 in `~/lib/EAGLE`
- Kestrel in `~/projects/kestrel`

Those implementations keep a prompt-length Eagle KV cache and draft over `S + depth`, where:
- `S` = accepted prefix length in the Eagle head
- `depth` = current speculative rollout depth

This mismatch is the most likely reason current llama.cpp acceptance length is materially worse than Kestrel even when first-step parity is good.

## Goal

Move llama.cpp Eagle3 to a Kestrel-style KV model:
- maintain a canonical accepted-prefix Eagle KV cache
- incrementally prefill only newly accepted prefix tokens
- keep a batched per-beam KV cache for rollout
- attend over the full Eagle prefix plus drafted suffix
- avoid per-depth graph rebuilds within a target round

## Non-Goal

The accepted prefix is not the target model KV cache.

We must build and store the Eagle head's own KV cache:
- same hidden inputs from the target capture
- same Eagle token inputs
- Eagle attention projections produce Eagle K/V

## Proposed Data Model

Persistent canonical accepted-prefix state:
- `prefix_hidden`
- `prefix_k`
- `prefix_v`
- `prefix_len`

Batched rollout state for one target round:
- `beam_hidden`: current hidden per beam
- `beam_k`: `[head_dim, round_capacity, n_kv_heads, beam_width]`
- `beam_v`: `[head_dim, round_capacity, n_kv_heads, beam_width]`
- `beam_mask`: mask over the same logical sequence length
- `beam_prefix_len`: accepted prefix length copied into the batched buffers
- `beam_suffix_parent`: temporary parent-remap buffer for the drafted suffix

Temporary remap buffers:
- drafted suffix K/V for active beams only
- parent index buffer for the next depth
- token / position inputs per beam

## Mask Semantics

If we use the current ggml attention semantics:
- valid positions should contribute `0`
- masked positions should contribute `-INF`

Do not use raw `1/0` unless the attention op is changed to interpret it that way.

## Round Lifecycle

At the start of a speculative round:

1. Let:
   - `S_old` = previous accepted Eagle prefix length
   - `S_new` = current accepted Eagle prefix length after the latest target verification
   - `delta = S_new - S_old`

2. Incrementally prefill the canonical Eagle cache for only the new accepted segment:
   - consume the newly accepted target tokens
   - consume the corresponding captured target hidden states
   - append Eagle K/V rows for positions `[S_old, S_new)`

3. Produce the root Eagle state for the current frontier as part of that same incremental update.
   - The easiest v1 is to end the prefill/root path with a single canonical current hidden state for the last accepted position.

4. Fan the canonical prefix delta out to the batched beam KV:
   - copy only `[S_old, S_new)` into every beam slice
   - do not recopy `[0, S_old)` if the batched beam buffers are persistent across rounds

5. Reset any stale drafted suffix beyond `S_new`.

6. Initialize the round mask:
   - positions `< S_new` valid for all beams
   - positions `>= S_new` masked

## Root Step

The root step remains special.

Recommended v1:
- maintain the canonical accepted-prefix Eagle cache without a beam dimension
- after incremental prefill, copy the canonical hidden/KV state into all beam slots
- then start batched rollout from that shared root state

This keeps the implementation simpler than trying to make the canonical and beam caches the same tensor immediately.

## Depth Rollout Lifecycle

For each rollout depth `d`:

1. Run one batched Eagle step for all active beams.
   - query length is 1 per beam
   - KV length is `S_new + d`

2. Write new K/V rows into position `S_new + d` in each active beam slice.

3. Update the mask for those newly written positions.

4. Score proposals and choose the next parent beams.

5. Remap only the drafted suffix, not the full accepted prefix.
   - prefix `[0, S_new)` is identical for all beams
   - only suffix `[S_new, S_new + d]` depends on beam ancestry

This suffix-only remap is the key optimization over the current "copy the whole short KV state around" design.

## Tensor Sizing

Recommended first implementation:
- per target round, build batched rollout tensors with exact capacity `S_new + max_depth`

Benefits:
- one graph shape for the whole round
- easy correctness model
- no per-depth graph rebuild

Later optimization:
- over-allocate and grow only occasionally
- preserve graphs until capacity changes

## Attention Backend

Once the KV layout is fixed, the rollout path should use the same optimized attention op already wired into Eagle:
- `ggml_flash_attn_ext` when supported
- generic attention fallback otherwise

The main structural change is the KV layout and mask, not the attention math.

## Initial Implementation Plan

Phase 1: Canonical prefix cache
- add persistent canonical Eagle prefix KV state
- incrementally append only newly accepted tokens
- keep existing rollout logic otherwise

Phase 2: Batched beam KV
- replace per-beam independent short KV caches with one batched KV tensor
- initialize beam prefix from the canonical cache
- use exact round capacity `S + max_depth`

Phase 3: Suffix remap
- after beam-parent selection, remap only drafted suffix rows
- stop copying or rebuilding full per-beam prefix state

Phase 4: Persistent capacities
- preserve batched beam buffers across rounds
- append only prefix delta to all beam slices
- avoid full round reallocation unless capacity changes

## Correctness Checks

After each phase, verify:
- first-step parity against Kestrel using `tools/eagle3/eagle3_parity.py`
- full first-10 prompt acceptance-length comparison against Kestrel
- deterministic greedy output with `-fa off` and `-fa on`
- no regression in the thresholded per-beam top-k path

Primary metric to watch:
- acceptance length on the first 10 `perfect_10k` prompts

## Expected Outcome

If this matches Kestrel/reference behavior more closely, acceptance length should move materially upward from the current llama.cpp range toward the Kestrel range, while keeping the Flash Attention speedups already recovered.
