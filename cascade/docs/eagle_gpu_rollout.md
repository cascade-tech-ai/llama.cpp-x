# EAGLE GPU rollout plan

## Goal

Remove all CPU work that currently blocks the EAGLE rollout loop, so the hot path becomes a single contiguous device-side pipeline per draft round.

This note is only about the **rollout loop** in [`common/speculative.cpp`](../../common/speculative.cpp), not `prefill_to()`.

## Current rollout loop structure

The relevant code is the EAGLE `draft()` rollout in:

- [`common/speculative.cpp:984`](../../common/speculative.cpp:984) root alignment step
- [`common/speculative.cpp:1081`](../../common/speculative.cpp:1081) depth loop
- [`common/speculative.cpp:1121`](../../common/speculative.cpp:1121) select stage
- [`common/speculative.cpp:1256`](../../common/speculative.cpp:1256) batched step stage
- [`common/speculative.cpp:1287`](../../common/speculative.cpp:1287) post-step bookkeeping
- [`common/speculative.cpp:1363`](../../common/speculative.cpp:1363) final tree rebuild

The current per-depth shape is:

1. GPU: `llama_eagle3_select_state_slots(...)`
2. CPU: consume selected rows, filter/expand candidates, build next frontier
3. GPU: `llama_eagle3_step_batch_from_parents(...)`
4. CPU: append `all_nodes`, update `prefix_states`, update debug trace

Important detail: the `select` stage is already more than just logits/top-k. On the device path it runs a fused select graph and then reads back:

- `selected_linear`
- `selected_draft_idx`
- `selected_logprob`

That readback happens in:

- [`src/llama-eagle3.cpp:2761`](../../src/llama-eagle3.cpp:2761)
- [`src/llama-eagle3.cpp:2779`](../../src/llama-eagle3.cpp:2779)

That readback is the main GPU->CPU blocking boundary in the loop.

## Exact CPU work in the hot loop today

### Blocking CPU block A: after `select_state_slots`, before `step_batch_from_parents`

This is the code between:

- [`common/speculative.cpp:1121`](../../common/speculative.cpp:1121)
- [`common/speculative.cpp:1256`](../../common/speculative.cpp:1256)

It currently does all of the following on CPU:

1. Iterate `selected_linear`, `selected_draft_idx`, `selected_logprob`
2. Decode `parent_slot = linear / k`
3. Skip inactive parents
4. Convert `draft_idx -> base_id` via `model->d2t`
5. Recompute per-parent probability from logprobs
6. Apply `prob_threshold`
7. Build `expansions`
8. Truncate expansions to `beam_width`
9. Build next-depth beam metadata:
   - `active_nxt`
   - `beam_logprob_nxt`
   - `child.logprob`
   - `child.tokens = parent.tokens + token`
10. Build the next step inputs:
   - `parent_states`
   - `out_states`
   - `candidate_input_ids`
11. Clear inactive next slots

This is the only CPU block that directly determines the next GPU step.

### Non-blocking CPU block B: after `step_batch_from_parents`

This is the code starting at:

- [`common/speculative.cpp:1287`](../../common/speculative.cpp:1287)

It currently does:

1. `common_speculative_trace_insert_path(...)`
2. `prefix_states[cand.tokens] = cand.state`
3. `all_nodes.push_back(...)`

This block does **not** determine the next depth. It only preserves information later used to rebuild:

- `last_tree`
- `last_tree_states`

in:

- [`common/speculative.cpp:1363`](../../common/speculative.cpp:1363)

Because it is not part of next-depth control, it does not have to stay in the hot loop.

## What should move into the new GPU op

The new op should absorb **all of block A**.

A good name would be something like:

- `GGML_OP_EAGLE_FRONTIER_UPDATE`

or, if we want to commit to a more fused design:

- `GGML_OP_EAGLE_ROLLOUT`

For the incremental plan in this document, assume a frontier update op.

### Inputs to the new op

Per depth, the op needs device-visible inputs equivalent to:

1. `selected_linear`
2. `selected_draft_idx`
3. `selected_logprob`
4. current frontier active mask
5. current frontier beam logprobs
6. current frontier parent/node indices
7. `model->d2t` mapping or an equivalent draft->base token table on device
8. constants:
   - `beam_width`
   - `k`
   - `prob_threshold`
   - inactive logprob sentinel

### Outputs from the new op

The op should produce everything needed for the next depth without CPU intervention:

1. `next_active_mask`
2. `next_beam_logprob`
3. `next_parent_slot` or equivalent row mapping
4. `next_input_ids`
5. `next_out_slot` or child state slot mapping
6. persistent node metadata for final tree reconstruction:
   - `node_token`
   - `node_parent_node`
   - `node_depth`
   - `node_logprob`
   - `node_state_slot`
7. `n_next` / live child count

### Responsibilities moved from CPU into the new op

The op should do all of this internally:

1. Decode selected rows into parent beam indices
2. Convert draft vocab ids to base vocab ids
3. Apply inactive-mask filtering
4. Apply probability threshold filtering
5. Truncate/sort to beam budget as needed
6. Allocate child slots for the next frontier
7. Write next frontier tensors for the next `step_batch`
8. Append persistent node records for later tree reconstruction

## What can move after rollout completes

Everything in block B can leave the hot loop.

Specifically, we should remove these hot-loop host containers:

- `std::vector<proposal_path> all_nodes`
- `std::map<llama_tokens, llama_eagle3_state> prefix_states`

and replace them with persistent device-side rollout buffers.

After the rollout is finished, CPU can do one readback of compact node/state metadata and then build:

1. `last_tree.tokens`
2. `last_tree.parents`
3. `last_tree.depths`
4. `last_tree_states`
5. `last_tree.row_indices` / metadata via `common_speculative_tree_build_metadata(...)`

That deferred reconstruction replaces the current code in:

- [`common/speculative.cpp:1363`](../../common/speculative.cpp:1363)
- [`common/speculative.cpp:1416`](../../common/speculative.cpp:1416)

## Required persistent rollout buffers

To defer final tree construction safely, the rollout must preserve enough information during the loop.

At minimum, keep these device-side arrays for the whole draft round:

1. `node_token[max_nodes]`
2. `node_parent[max_nodes]`
3. `node_depth[max_nodes]`
4. `node_logprob[max_nodes]`
5. `node_state_slot[max_nodes]`
6. `node_count`

And for the live frontier:

1. `frontier_node_idx[beam_width]`
2. `frontier_active[beam_width]`
3. `frontier_beam_logprob[beam_width]`
4. `frontier_input_id[beam_width]`
5. `frontier_parent_state_slot[beam_width]`
6. `frontier_child_state_slot[beam_width]`

This is enough to:

- run the next step without CPU work
- rebuild `last_tree` later
- recover the correct `llama_eagle3_state` for each selected node later

## What does not need to stay in the rollout loop

These pieces can move out of the hot path entirely:

1. `common_speculative_trace_insert_path(...)`
2. building `all_nodes`
3. building `prefix_states`
4. rebuilding `last_tree`
5. rebuilding `last_tree_states`
6. marking selected nodes in the final proposal tree

They can all happen after the rollout device work completes.

## Concrete refactor plan

### Stage 1: remove the blocking GPU->CPU readback boundary

1. Keep current root step and current `step_batch_from_parents(...)`.
2. Keep current device select graph.
3. Replace the CPU code between select and step with one new device op.
4. Make that op emit the next frontier tensors directly.
5. Stop reading back `selected_linear`, `selected_draft_idx`, `selected_logprob` to CPU inside the depth loop.

Result:

- the rollout loop no longer blocks on CPU between `select` and `step_batch`
- the main GPU hot path becomes contiguous

### Stage 2: remove non-blocking hot-loop bookkeeping

1. Remove `all_nodes` from the per-depth loop.
2. Remove `prefix_states` from the per-depth loop.
3. Make the new device op append persistent node metadata every depth.
4. After the rollout ends, do one readback of compact node metadata and state-slot info.
5. Rebuild `last_tree` and `last_tree_states` once on CPU.

Result:

- no CPU work remains inside the rollout loop except outer loop control
- tree reconstruction becomes a post-pass

### Stage 3: fuse the rollout loop further if desired

Once stage 1 and 2 are done, the remaining per-depth device sequence is conceptually:

1. select graph
2. frontier update op
3. step batch graph

At that point there are two good follow-ups:

1. fold frontier update into the select graph
2. replace `select + frontier_update + step_batch` with one bigger `eagle_rollout` op

Stage 3 is optional. Stages 1 and 2 are enough to remove the current CPU blocking path.

## Exact answer to the planning question

All CPU code that currently blocks the rollout loop should be split like this:

### Move into the new GPU op

Move all of the CPU code between:

- [`common/speculative.cpp:1121`](../../common/speculative.cpp:1121)
- [`common/speculative.cpp:1256`](../../common/speculative.cpp:1256)

That entire block is rollout-critical frontier construction and should become device code.

### Move after rollout completion

Move the bookkeeping currently done after:

- [`common/speculative.cpp:1287`](../../common/speculative.cpp:1287)

and the final tree rebuild at:

- [`common/speculative.cpp:1363`](../../common/speculative.cpp:1363)

into one CPU post-pass after the rollout device work is finished.

## Net result

After this refactor, the rollout loop should no longer contain any CPU work that blocks the next depth.

The hot path becomes:

1. root step
2. for each depth:
   - select graph
   - frontier update device op
   - step batch
3. after rollout ends:
   - one CPU reconstruction pass for `last_tree` / `last_tree_states`

That is the cleanest path from the current code to a single contiguous GPU-side rollout pipeline.
