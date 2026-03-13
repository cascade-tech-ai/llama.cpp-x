# EAGLE 3 in `llama.cpp-x`: implementation walkthrough

This is a code-oriented walkthrough of how EAGLE 3 is implemented in this fork of `llama.cpp`.

It assumes you already understand the EAGLE 3 algorithm at a model level. The goal here is narrower:

- show where the EAGLE 3 path lives in the codebase
- explain the runtime pipeline in the order it executes
- point out the expensive operations
- identify the main optimization targets

## File map

The core implementation is split across four places:

- `common/speculative.cpp`: speculative driver, prefix-state management, rollout, proposal-tree construction
- `src/llama-eagle3.cpp` and `src/llama-eagle3.h`: EAGLE head model loader, runtime, graphs, per-step and batched step execution
- `src/models/llama.cpp`: hook that captures selected hidden states from the base model during target decode
- `src/llama-context.cpp`: stores the captured hidden tensors for the current decode pass and exposes them to the EAGLE driver

## High-level overview

At a high level, the implementation does this:

1. Load a separate EAGLE 3 GGUF head.
2. Read from that GGUF which base-model layers must be captured.
3. Configure the target `llama_context` to capture those layer inputs during normal decode.
4. Maintain an EAGLE prefix state (`prefix_state`) that mirrors the accepted target prefix.
5. When speculation starts, advance one EAGLE step from the last teacher hidden state plus `id_last` to produce the root draft state.
6. Run beam rollout from that root:
   - score active beams with the EAGLE LM head
   - select the best beam/token expansions
   - advance the selected children with a batched EAGLE step
7. Convert the resulting paths into a proposal tree for the verifier.
8. After verification, update `prefix_state` to the deepest accepted node so the next round resumes from the accepted EAGLE KV/cache frontier.

The most important structural point is this:

- the base model is still the source of truth
- EAGLE is attached as a sidecar head
- the bridge between them is the captured hidden-state stream from selected base-model layers

## 1. Loading the EAGLE head and enabling hidden capture

The speculative state for EAGLE lives in `common_speculative_state_eagle3` in [`common/speculative.cpp`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:673).

Its constructor:

- loads the EAGLE head GGUF
- reads `hidden_layer_ids`
- checks that `target_hidden_size` matches the base model embedding size
- enables hidden capture on the target context
- builds an EAGLE runtime

Relevant code:

```cpp
llama_eagle3_model * raw = llama_eagle3_load(params.mparams_dft.path, err);
model.reset(raw);

layer_ids = model->hidden_layer_ids;
hidden_size = model->hparams.hidden_size;
target_hidden_size = model->hparams.target_hidden_size;
hidden_in_dim = model->hparams.hidden_concat * model->hparams.target_hidden_size;

if (!llama_eagle3_set_layers(ctx_tgt, layer_ids.data(), layer_ids.size())) {
    LOG_ERR("%s: failed to enable eagle3 hidden capture\n", __func__);
    return;
}

rt = llama_eagle3_make_runtime(ctx_tgt, model.get(), llama_n_threads(ctx_tgt));
```

Source:

- [`common/speculative.cpp:820`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:820)
- [`common/speculative.cpp:842`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:842)
- [`common/speculative.cpp:847`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:847)

The GGUF loader is in [`src/llama-eagle3.cpp:1864`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:1864). It validates:

- `general.architecture == "eagle3"`
- hidden size / FFN size / head counts
- `hidden_state_layer_ids`
- `d2t`
- all required EAGLE tensors

This is where the implementation decides the base-to-draft interface shape:

- `hidden_in_dim = hidden_concat * target_hidden_size`
- if `hidden_in_dim != hidden_size`, the EAGLE head uses `fc_w` to project the concatenated teacher hidden stream into the head hidden size

## 2. Where the base-model hidden states come from

The base model graph captures the input to selected transformer layers in [`src/models/llama.cpp:36`](/home/alvion/projects/llama.cpp-x/src/models/llama.cpp:36):

```cpp
if (capture_eagle3_layer(il)) {
    ggml_tensor * captured = ggml_cast(ctx0, inpL, GGML_TYPE_F32);
    captured = ggml_cont(ctx0, captured);
    cb(captured, "eagle3_hidden", il);
    res->t_eagle3_hidden[il] = captured;
    ggml_build_forward_expand(gf, captured);
}
```

That means the captured tensor is:

- the layer input, not the output
- cast to `f32`
- made contiguous
- attached to the graph result map by layer id

This is one of the first important cost centers:

- every captured layer adds a cast and contiguous materialization
- every decode ubatch now carries extra output tensors

The `llama_context` side stores those per-layer captures in backend-resident tensors in [`src/llama-context.cpp:876`](/home/alvion/projects/llama.cpp-x/src/llama-context.cpp:876). For each ubatch, `eagle3_capture_append()` copies the graph result into a per-layer capture buffer:

```cpp
if (!layer.tensor || layer.backend != backend_src || layer.capacity < (int32_t) capacity) {
    ggml_tensor * t_capture = ggml_new_tensor_2d(ctx_capture.get(), GGML_TYPE_F32, n_embd, capacity);
    ...
    layer.tensor = t_capture;
}

const size_t dst_offset = (size_t) eagle3_capture_n_tokens * (size_t) layer.tensor->nb[1];
if (!tensor_copy_bytes_async(backend_src, src, 0, layer.tensor, dst_offset, n_bytes)) {
    return false;
}
```

Sources:

- [`src/llama-context.cpp:903`](/home/alvion/projects/llama.cpp-x/src/llama-context.cpp:903)
- [`src/llama-context.cpp:933`](/home/alvion/projects/llama.cpp-x/src/llama-context.cpp:933)

The capture lifecycle is attached to target decode in [`src/llama-context.cpp:1910`](/home/alvion/projects/llama.cpp-x/src/llama-context.cpp:1910) and [`src/llama-context.cpp:1968`](/home/alvion/projects/llama.cpp-x/src/llama-context.cpp:1968).

## 3. Runtime setup: why EAGLE has its own backend/runtime

`llama_eagle3_make_runtime()` in [`src/llama-eagle3.cpp:2044`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:2044) builds a separate runtime for the head.

Important details:

- it reuses the base model token embedding tensor
- it copies RoPE parameters from the target model
- it forces `LLAMA_ROPE_TYPE_NEOX` for the EAGLE head
- it prefers a GPU backend for EAGLE execution
- it copies the EAGLE head weights into that backend if possible

The weight copy is in [`src/llama-eagle3.cpp:749`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:749).

This design gives the implementation two important properties:

- the head can run independently of the main target graph
- hidden capture can stay on-device and be copied directly into the EAGLE step graph

Performance implication:

- startup pays a one-time weight duplication cost
- steady-state benefits by avoiding repeated host-side head execution

## 4. Prefix-state model: how accepted target tokens become EAGLE state

The key mental model is:

- `prefix_state` is the canonical EAGLE state for the currently accepted target prefix
- `prefix_prompt_len` tells you how much of the target prefix that state covers
- `prefix_tail_hidden_concat` caches the teacher hidden concat for the prefix tail token, because the next EAGLE step needs it

These fields are defined in [`common/speculative.cpp:684`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:684).

`begin()` resets that cache and immediately calls `prefill_to(prompt, seq_id)` in [`common/speculative.cpp:853`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:853).

### What `prefill_to()` does

`prefill_to()` is in [`common/speculative.cpp:1789`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1789). Its job is to bring the EAGLE state up to the accepted target prompt.

The consecutive logic is:

1. Fetch the captured hidden tensors for all requested layers.
2. If this is an incremental extension, use the cached `prefix_tail_hidden_concat` for the first new token.
3. For the remaining prompt positions, call `llama_eagle3_step_from_hidden_capture()` token by token.
4. Cache the hidden concat for the new tail token into `prefix_tail_hidden_concat`.
5. Update `prefix_prompt_len`.

The core loop looks like this:

```cpp
for (size_t i = start; i + 1 < prompt_tgt.size(); ++i) {
    if (!llama_eagle3_step_from_hidden_capture(
                *model,
                rt,
                prefix_state,
                layer_tensors,
                i - local_offset,
                prompt_tgt[i + 1],
                nullptr,
                dump_steps ? &dbg : nullptr)) {
        return false;
    }
}
```

Source: [`common/speculative.cpp:1936`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1936)

Two important observations:

- this is a real EAGLE forward over the accepted prefix, not just cached bookkeeping
- prefill cost scales with the amount of new accepted target context not yet reflected in `prefix_state`

Major cost centers here:

- extracting per-token hidden slices from the capture tensors
- running one EAGLE step per newly accepted token
- copying and growing EAGLE KV/state

## 5. Root alignment: the extra step with `id_last`

The speculative driver keeps `id_last` separate from `prompt_tgt`, so EAGLE needs one extra alignment step before rollout.

That is the comment at [`common/speculative.cpp:941`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:941):

> advance the head one more step using the teacher hidden at the last prompt token and input_id = id_last

Implementation:

```cpp
llama_eagle3_state root_state = prefix_state;

if (!llama_eagle3_step_from_hidden_capture(
            *model,
            rt,
            root_state,
            layer_tensors,
            prefix_tail_capture_idx,
            id_last,
            nullptr,
            dump_root ? &dbg_root : nullptr)) {
    return;
}
```

Source: [`common/speculative.cpp:950`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:950)

This produces the draft root state used for beam rollout.

## 6. What a single EAGLE step actually does

The fast path is `llama_eagle3_step_from_hidden_capture()` in [`src/llama-eagle3.cpp:3096`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:3096).

On the optimized path, it:

1. Reuses a cached step graph keyed by `(past_len, hidden_in_dim, with_logits)`.
2. Copies the selected hidden slices from the target capture tensors directly into the EAGLE graph input.
3. Uploads `input_id` and `past_len`.
4. Copies past EAGLE KV into the graph inputs.
5. Runs the step graph.
6. Copies `hidden_out`, `k_total`, and `v_total` into the next state storage.

The direct target-backend to EAGLE-backend copy is here:

```cpp
for (int32_t il = 0; il < hp.hidden_concat; ++il) {
    ...
    if (!tensor_copy_bytes_async(
                rt.target_backend ? rt.target_backend : rt.backend_compute.get(),
                rt.backend_compute.get(),
                src, src_offset,
                graph.t_hidden_in, dst_offset,
                n_bytes)) {
        return false;
    }
}
```

Source: [`src/llama-eagle3.cpp:3127`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:3127)

The graph itself is built in [`src/llama-eagle3.cpp:1376`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:1376). The computation is:

1. optional `fc_w` projection from concatenated teacher hidden to head hidden size
2. token embedding lookup from the base model embedding table
3. RMSNorm on teacher hidden and token embedding
4. concat of normalized embedding and normalized teacher hidden
5. Q/K/V projections
6. RoPE on Q and K
7. attention against past + current KV
8. output projection + residual
9. post-attention RMSNorm
10. SwiGLU FFN
11. residual add to get next hidden
12. optional final norm + LM head for logits

The expensive part is exactly what you would expect:

- Q/K/V projections
- attention
- output projection
- FFN
- copying total KV back into state storage

## 7. Beam rollout

The rollout loop starts at [`common/speculative.cpp:1081`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1081).

Each depth does two heavy things:

1. score/select expansions from the current active beams
2. advance the selected children with a batched EAGLE step

### 7.1 Selection

Driver side:

```cpp
const bool ok = llama_eagle3_select_state_slots(
        *model,
        rt,
        beam_states,
        active_cur,
        beam_logprob_cur,
        k,
        prob_threshold,
        selected_linear,
        selected_draft_idx,
        selected_logprob);
```

Source: [`common/speculative.cpp:1121`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1121)

The device graph for this is built by `build_select_batch_graph()` in [`src/llama-eagle3.cpp:1038`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:1038).

What it computes:

- batched norm + LM head for all beams
- softmax over draft vocab
- `top_k_threshold` per beam
- add per-beam accumulated logprob
- flatten beam/rank into one vector
- global top-k over the flattened beam-token candidates

This is the heart of draft scoring, and it is a major hotspot because it effectively does:

- `LM head * active_beams`
- softmax over `draft_vocab_size * active_beams`
- selection over `k * active_beams`

The host-side follow-up in the driver maps draft ids back to base ids with `d2t` and applies the probability filter:

- [`common/speculative.cpp:1168`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1168)
- [`common/speculative.cpp:1170`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1170)

### 7.2 Batched child advance

Once the best expansions are chosen, the driver calls:

```cpp
const bool ok = llama_eagle3_step_batch_from_parents(
        *model,
        rt,
        parent_states,
        hidden_size,
        candidate_input_ids,
        out_states,
        reserve_kv);
```

Source: [`common/speculative.cpp:1256`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1256)

The batched implementation is in [`src/llama-eagle3.cpp:2892`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:2892), with the graph builder in [`src/llama-eagle3.cpp:1555`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:1555).

Important implementation detail:

- the expensive matmuls are batched across beams
- attention is still handled per beam inside the graph

The comment in the builder says why:

- duplicating the full block per beam caused a GEMV-heavy CUDA path
- batching the matmuls is better for the small-beam regime EAGLE usually runs in

That is one of the key performance choices in this implementation.

## 8. Building the proposal tree

After rollout, the best paths are turned into `common_speculative_tree` in [`common/speculative.cpp:1388`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1388).

The implementation deduplicates shared prefixes, records parent/depth arrays, and stores the EAGLE state for each tree node in `last_tree_states`.

That state association is what makes incremental acceptance cheap in the next phase.

## 9. How accepted verifier tokens update EAGLE state

After verification, `accept_tokens()` in [`common/speculative.cpp:1450`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1450) walks the accepted token sequence through the last proposal tree.

If no proposal token was accepted, it falls back to the root state:

```cpp
if (accepted_nodes.empty()) {
    set_prefix_frontier(last_root_state, 0);
}
```

If some path was accepted, it promotes the deepest accepted node state:

```cpp
const int32_t deepest = accepted_nodes.back();
...
set_prefix_frontier(last_tree_states[(size_t) deepest], row_idx);
```

Source:

- [`common/speculative.cpp:1514`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1514)
- [`common/speculative.cpp:1525`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1525)

This is why the next `prefill_to()` call can usually do incremental work instead of rebuilding the whole prefix state.

## 10. Major expensive operations

If you are profiling this code, these are the big-ticket items.

### 10.1 Base-model hidden capture

Cost comes from:

- `ggml_cast(..., F32)` and `ggml_cont(...)` per captured layer
- copying each ubatch’s capture into `llama_context` storage

Code:

- [`src/models/llama.cpp:36`](/home/alvion/projects/llama.cpp-x/src/models/llama.cpp:36)
- [`src/llama-context.cpp:876`](/home/alvion/projects/llama.cpp-x/src/llama-context.cpp:876)

### 10.2 Prefix prefill

Cost comes from:

- one EAGLE step per newly accepted token
- hidden-slice extraction
- KV/state copies during each step

Code:

- [`common/speculative.cpp:1789`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1789)

### 10.3 Root step and rollout step compute

Cost comes from:

- EAGLE attention block
- FFN
- KV concat and copy-out

Code:

- [`src/llama-eagle3.cpp:1376`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:1376)
- [`src/llama-eagle3.cpp:3252`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:3252)
- [`src/llama-eagle3.cpp:2892`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:2892)

### 10.4 Beam scoring / selection

Cost comes from:

- LM head over all active beams
- softmax over the draft vocab
- top-k selection and beam-logprob combination

Code:

- [`src/llama-eagle3.cpp:1038`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:1038)
- [`common/speculative.cpp:1113`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1113)

### 10.5 KV copying

This implementation copies KV a lot:

- from parent state into step graph inputs
- from graph outputs back into child state storage
- from root state into rollout slots

Code:

- [`src/llama-eagle3.cpp:554`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:554)
- [`src/llama-eagle3.cpp:2972`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:2972)
- [`src/llama-eagle3.cpp:3045`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:3045)

This is probably the most obvious memory-traffic bottleneck in the current design.

## 11. Major optimization targets

If you want to make this faster, these are the first places worth attacking.

### 11.1 Reduce or eliminate KV copying

Current behavior is copy-heavy and safe, but expensive.

Best target:

- keep rollout states as views into preallocated beam storage for the whole round
- avoid materializing full `k_total`/`v_total` copies when the child can append in place

Related code:

- [`src/llama-eagle3.cpp:447`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:447)
- [`src/llama-eagle3.cpp:495`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:495)

### 11.2 Cut hidden-capture overhead

Current capture path always casts to `f32` and materializes contiguous tensors.

Best target:

- minimize `hidden_concat`
- capture only the exact layers needed
- if possible, avoid redundant format/layout conversions

Related code:

- [`src/models/llama.cpp:36`](/home/alvion/projects/llama.cpp-x/src/models/llama.cpp:36)

### 11.3 Keep more of selection on-device

Selection is already mostly device-side, which is good. But the driver still reads selected results back to host and does path assembly there.

Best target:

- keep candidate compaction and maybe even rollout bookkeeping on-device longer
- reduce the host round-trip per depth

Related code:

- [`src/llama-eagle3.cpp:2544`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:2544)
- [`common/speculative.cpp:1151`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1151)

### 11.4 Reuse graphs aggressively

The implementation already caches graphs by shape keys:

- step graph keyed by `(past_len, hidden_in_dim, with_logits)`
- step-batch graph keyed by `(past_len, hidden_in_dim, n_beams, with_logits)`

Code:

- [`src/llama-eagle3.cpp:342`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:342)
- [`src/llama-eagle3.cpp:348`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:348)

This is good, but `past_len` still changes as rollout advances, so graph reuse is not perfect across all depths.

### 11.5 Watch fallback paths

When device-to-device async copy is unavailable, `tensor_copy_bytes_async()` can fall back to synchronize + host staging:

```cpp
ggml_backend_synchronize(backend_src);
ggml_backend_synchronize(backend_dst);

std::vector<uint8_t> tmp(size);
ggml_backend_tensor_get(src, tmp.data(), src_offset, size);
ggml_backend_tensor_set(dst, tmp.data(), dst_offset, size);
```

Source: [`src/llama-eagle3.cpp:710`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:710)

If you ever hit this path in steady state, performance will collapse.

## 12. Practical mental model

The simplest way to think about this implementation is:

- `llama.cpp` runs the target model normally
- selected base-model layer inputs are siphoned off during decode
- the EAGLE head consumes those hidden slices plus token ids to maintain its own KV/state
- rollout is a repeated `select -> step_batch -> build tree`
- acceptance just moves the EAGLE frontier to the accepted node so the next round starts from the right state

If you keep that model in your head, the code organization makes sense:

- base-model code owns teacher hidden generation
- `llama_context` owns capture storage
- `llama-eagle3.cpp` owns head execution
- `common/speculative.cpp` owns the speculative policy and round-to-round state transitions

## 13. Suggested reading order in code

If you want to continue from here, read in this order:

1. [`common/speculative.cpp:800`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:800) constructor
2. [`src/models/llama.cpp:36`](/home/alvion/projects/llama.cpp-x/src/models/llama.cpp:36) capture hook
3. [`src/llama-context.cpp:876`](/home/alvion/projects/llama.cpp-x/src/llama-context.cpp:876) capture append
4. [`common/speculative.cpp:1789`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1789) `prefill_to`
5. [`common/speculative.cpp:885`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:885) `draft`
6. [`src/llama-eagle3.cpp:3096`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:3096) `step_from_hidden_capture`
7. [`src/llama-eagle3.cpp:1376`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:1376) single-step graph builder
8. [`src/llama-eagle3.cpp:1555`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:1555) batched-step graph builder
9. [`src/llama-eagle3.cpp:1038`](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp:1038) selection graph builder
10. [`common/speculative.cpp:1450`](/home/alvion/projects/llama.cpp-x/common/speculative.cpp:1450) `accept_tokens`

That sequence matches the actual runtime pipeline.
