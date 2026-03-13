# EAGLE batched attention plan

## Goal

Stop running rollout attention as effectively separate per-beam attention calls with per-beam KV materialization.

The target end state is:

1. one persistent full-length global KV store for rollout beam state
2. one persistent full-length global mask store aligned with that KV
3. one small depth-bounded workspace for parent/child remap copies
4. one slice of length `current_length + DEPTH` taken once per target-model pass
5. reuse that same slice across every rollout depth in the draft loop
6. do not rebuild the draft attention graph per rollout depth

This note is only about the rollout attention path in EAGLE.

## Current state

Current rollout batching is only partial.

### What is already batched

In [`src/llama-eagle3.cpp`](../../src/llama-eagle3.cpp), `build_step_batch_graph()` batches:

- token embedding
- hidden projection
- RMS norms
- Q/K/V projections
- output projection
- FFN

That starts at:

- [`src/llama-eagle3.cpp:1555`](../../src/llama-eagle3.cpp:1555)

### What is still not truly batched

The attention/KV path is still beam-local inside that graph.

For each beam, the current code:

1. slices `q_i`, `k_i`, `v_i`
2. allocates separate per-beam past inputs:
   - `t_k_past_input[ib]`
   - `t_v_past_input[ib]`
3. concatenates `k_past + k_i`
4. concatenates `v_past + v_i`
5. runs attention for that one beam
6. concatenates the resulting beam outputs back together

So the graph is not doing one real batched attention over all beams. It is doing one graph execution that contains N separate beam-local attention subgraphs.

### Current rollout storage

There is already rollout-batch storage:

- `t_hidden`
- `t_k`
- `t_v`
- `t_mask`

allocated in:

- [`src/llama-eagle3.cpp:447`](../../src/llama-eagle3.cpp:447)

and slots are bound in:

- [`src/llama-eagle3.cpp:495`](../../src/llama-eagle3.cpp:495)

However, the hot `step_batch` path does not use that storage directly for attention.

### Current hot-path copies

In `llama_eagle3_step_batch_from_parents(...)`:

- each parent beam state is copied into per-beam step-graph inputs
- each parent KV prefix is copied into per-beam `t_k_past_input[ib]` / `t_v_past_input[ib]`
- after the graph runs, the resulting hidden/KV are copied back into output beam states

That path is here:

- [`src/llama-eagle3.cpp:2892`](../../src/llama-eagle3.cpp:2892)
- KV copy-in starts around [`src/llama-eagle3.cpp:2972`](../../src/llama-eagle3.cpp:2972)

So the current state is:

1. rollout beams live in a global backing store
2. but attention does not read that store directly
3. each rollout step still copies full effective KV into per-step graph inputs
4. attention is still built beam-by-beam

## Intended design

The intended design is:

### 1. Persistent full-length KV storage

One global persistent tensor for rollout KV state, sized to the full active sequence region.

Conceptually:

- `K_persistent[head_dim, n_kv_heads, seq_capacity, n_beams]`
- `V_persistent[head_dim, n_kv_heads, seq_capacity, n_beams]`

This is the canonical rollout state.

### 2. Persistent full-length mask storage

One global persistent mask tensor aligned with the same sequence region.

Conceptually:

- `M_persistent[seq_capacity, 1, 1, n_beams]`

This is kept fully updated for the entire visible sequence state.

Important detail:

- we should keep the full persistent mask state correct
- but we do **not** want to hand the entire full-capacity tensor to the attention graph every rollout depth

### 3. Small depth-bounded workspace

A separate temporary workspace only for parent/child copy safety during remap.

Conceptually:

- `K_work[head_dim, n_kv_heads, DEPTH, n_beams]`
- `V_work[head_dim, n_kv_heads, DEPTH, n_beams]`
- `M_work[DEPTH, 1, 1, n_beams]`

Purpose:

- when multiple children read from parent suffix rows
- and parent/child slots alias the same persistent storage
- we need temporary non-aliasing space so we do not overwrite a parent suffix before all children have copied from it

This workspace only needs to cover the speculative suffix, not the full prefix.

## Slice-based graph reuse plan

This is the important detail for graph stability.

We should not try to pass the entire full persistent KV/mask tensors into the draft attention graph every time.

Instead:

1. after each target-model pass, compute the current visible attention extent once:
   - `slice_len = current_length + DEPTH`
2. create views/slices of persistent KV and mask up to that length once
3. use that same slice across the whole draft rollout for that pass
4. only rebuild the graph when the slice bucket changes

The user-requested immediate target is even simpler:

- do **not** implement lazy slice-bucket graph reuse yet
- just slice once at `current_length + DEPTH` per target pass
- then reuse that same slice for every rollout depth
- this alone should avoid per-depth graph rebuilds

So the draft graph should be built against:

- one K slice view
- one V slice view
- one mask slice view

for the whole rollout round.

## Why the small workspace is still necessary

Even with persistent full-length storage, we still need temporary copy space during the parent->child remap phase.

Reason:

1. multiple children can share a parent
2. selected children may be assigned into slots that alias currently-live parent slots
3. if we copy suffix rows directly in-place, we can destroy a parent suffix before another child finishes reading it

So the correct model is:

- persistent full-length storage is the canonical source of truth
- depth-bounded workspace is only for collision-free speculative suffix remap

## Validation / commit consequences

After rollout, the speculative suffix region in the persistent beam slots is not necessarily canonical.

So after target validation there must be a commit/compaction stage that:

1. identifies the accepted path
2. copies the accepted suffix rows into the canonical persistent sequence state
3. updates persistent mask state to match
4. treats rejected rollout slots as garbage / reusable scratch

Important nuance:

- we do **not** need to restore every beam slot for the next round
- we only need to restore the committed state that survives into the next pass

This is why the persistent + workspace design pairs naturally with:

- GPU greedy verify
- GPU KV/mask commit and compaction

## What needs to change in the current code

### A. Rollout storage model

The rollout batch object should explicitly separate:

1. persistent full-length storage
2. temporary depth-bounded workspace
3. current slice views used by the draft graph

The current single `t_k/t_v/t_mask` rollout-batch storage is not enough to express the intended lifetime split cleanly.

### B. Step-batch graph inputs

`build_step_batch_graph()` should stop taking per-beam:

- `t_k_past_input[ib]`
- `t_v_past_input[ib]`

and should instead consume:

1. one batched K slice
2. one batched V slice
3. one batched mask slice
4. current per-beam token ids / positions
5. per-beam slot or valid-length metadata if needed

### C. Attention itself

The per-beam attention loop inside `build_step_batch_graph()` must be removed.

Instead, the graph should build one batched attention call over the full beam dimension.

That means the attention op should see something conceptually like:

- `Q[head_dim, n_heads, 1, n_beams]`
- `K_slice[head_dim, n_kv_heads, slice_len, n_beams]`
- `V_slice[head_dim, n_kv_heads, slice_len, n_beams]`
- `M_slice[slice_len, 1, 1, n_beams]`

where `slice_len` is fixed for the whole rollout round.

### D. Parent/child remap

The parent->child remap phase should:

1. copy only the speculative suffix rows that need movement
2. use the depth-bounded workspace to avoid alias clobbering
3. leave the full prefix rows in persistent storage untouched
4. update persistent masks accordingly

## Minimal staged implementation plan

### Stage 1. Add the correct storage model

Introduce explicit rollout storage for:

1. persistent full-length K/V/mask
2. depth-bounded K/V/mask workspace
3. current K/V/mask slices for the active round

No behavior change yet.

### Stage 2. Slice once per target pass

When the target pass finishes and the draft round starts:

1. compute `slice_len = current_length + DEPTH`
2. create/update K/V/mask slice views once
3. reuse them for the whole rollout loop

No lazy bucketed graph reuse yet.

### Stage 3. Replace per-beam past-KV inputs

Change `step_batch_from_parents(...)` and `build_step_batch_graph()` so the batched graph reads from the sliced persistent tensors instead of per-beam copied `t_k_past_input/t_v_past_input` tensors.

### Stage 4. Remove the per-beam attention loop

Replace the beam-local attention construction in `build_step_batch_graph()` with one batched attention call over the beam dimension.

### Stage 5. Use workspace only for remap collisions

Move parent->child suffix copying to the explicit depth-bounded workspace path.

## Exact target behavior

The target behavior after these changes should be:

1. target model runs once
2. draft runtime computes `slice_len = current_length + DEPTH`
3. one K/V/mask slice is created for this round
4. every rollout depth reuses the same sliced graph inputs
5. attention is processed in parallel across beams
6. only speculative suffix rows are copied during remap
7. after validation, accepted suffix rows are committed back to canonical persistent state

That is the intended design this branch should move toward.
