# EAGLE3 Compact-Tree Pivot Plan

AI / upstream note:
- This fork contains AI-assisted code and docs for experimentation. Do not upstream as-is. See `CONTRIBUTING.md` and `AGENTS.md`.

## Branch / Base

Current working branch:
- `pivot/eagle3-sglang-study`

Implementation base for this plan:
- `61744981` `eagle3: make compact verifier graph reusable`

Saved side branches:
- `backup/eagle3-graph-reuse-544976b5`
- `failed/eagle3-step-static-exactness`

## Why We Are Pivoting

The current llama.cpp EAGLE3 implementation has three structural problems:

1. Draft state is modeled as standalone hidden/KV blobs per beam.
2. Tree build and verification are still largely host-orchestrated.
3. Correct compact-tree verification is blocked by ambiguous KV ownership/writes.

SGLang is fast because it does the opposite:

1. Shared GPU-resident speculative KV slots are first-class.
2. Tree build/verify/compaction are GPU kernels.
3. Draft and verify paths are fixed-capacity CUDA-graph runners.

The goal of this pivot is to move our implementation toward that structure without giving up the exact greedy-decoding guarantee.

## End Goal

Build a compact-tree EAGLE3 pipeline with:

1. One root row plus `max_proposals` node rows.
2. Explicit speculative KV slot control.
3. GPU-side tree metadata and greedy verification.
4. GPU-side KV compaction / accepted-path commit.
5. Static-shape draft + verify runners that are graph-reusable.
6. Exact greedy output equivalence with non-speculative decoding.
7. Preserved first-token parity against the Kestrel reference dump.

## Required Verification At Every Stopping Point

After each implementation step below, run all of these checks before moving on.

### A. 1B Greedy Exactness

Model pair:
- Base: `/home/alvion/models/unsloth-Llama-3.2-1B-Instruct-bf16.gguf`
- Draft: `/home/alvion/projects/kestrel/models/llama3-1b_eagle.f16.gguf`

Prompt source:
- Use the same chat-templated prompt path that matches the Kestrel dump setup.

Required checks:
1. CPU baseline vs CPU EAGLE greedy output tokens are identical.
2. CUDA baseline vs CUDA EAGLE greedy output tokens are identical.
3. CPU and CUDA EAGLE output tokens are identical to each other.

### B. First-Token Kestrel Parity

Reference:
- `/home/alvion/projects/kestrel/reports/llama32_1b_eagle_first_token_dump.pt`

Required checks:
1. Prompt token ids exact.
2. Head input ids exact.
3. Target hidden-state capture remains close to Kestrel.
4. First-token top-k logits remain roughly aligned.
5. Top-100 KL remains within the previously established small-error regime.

Acceptance criteria:
- We do not require bitwise parity.
- We do require that changes do not materially worsen the previous parity envelope.

### C. Logging / Evidence

For each step, save:
- exact command lines
- relevant logs in `/tmp` or `cascade/reports` if worth preserving
- one short note in this doc under the step describing pass/fail
- two summary numbers in this doc under the step:
  - first-token KL divergence vs Kestrel
  - CUDA EAGLE decode throughput in tok/s from the standardized 1B perf check

Standardized 1B perf check:
- Base: `/home/alvion/models/unsloth-Llama-3.2-1B-Instruct-bf16.gguf`
- Draft: `/home/alvion/projects/kestrel/models/llama3-1b_eagle.bf16.gguf`
- Prompt: `/tmp/perfect1k_prompts/00.txt`
- EAGLE command:
  - `/home/alvion/projects/llama.cpp-x/build-cuda/bin/llama-speculative-simple -f /tmp/perfect1k_prompts/00.txt -n 64 -c 4096 -ngl 999 --temp 0 --seed 123 -m /home/alvion/models/unsloth-Llama-3.2-1B-Instruct-bf16.gguf -md /home/alvion/projects/kestrel/models/llama3-1b_eagle.bf16.gguf --spec-type eagle3 --eagle-max-depth 7 --eagle-max-proposals 16 --eagle-beam-width 8 -ngld 999`
- Baseline command:
  - `/home/alvion/projects/llama.cpp-x/build-cuda/bin/llama-speculative-simple -f /tmp/perfect1k_prompts/00.txt -n 64 -c 4096 -ngl 999 --temp 0 --seed 123 -m /home/alvion/models/unsloth-Llama-3.2-1B-Instruct-bf16.gguf --spec-type none -ngld 999`

Do not proceed to the next step with broken greedy exactness unless the step is explicitly checkpointed as failed work on another branch.

## Implementation Plan

### Step 1. Add explicit speculative KV slot control

Goal:
- Make target verification batches able to specify where speculative tokens write K/V.

Deliverable:
- Batch / ubatch / KV plumbing for explicit speculative slot indices.

Design:
- Add optional per-token KV slot indices to the target decode path.
- If unset, current allocator behavior remains unchanged.
- If set, speculative verify can place siblings at the same logical `pos` into distinct physical KV cells.

Why:
- This is the foundation for compact-tree correctness.
- Without explicit slot control, tree-mask-only verification cannot disambiguate writes.

Verification gate:
- All required checks A/B/C.

Status:
- in progress

Verification notes:
- 2026-03-08:
  - Added `kv_slot` plumbing through `llama_batch`, `llama_ubatch`, and `llama_kv_cache::find_slot()`.
  - `kv_slot == nullptr` still means allocator-managed placement.
  - If `kv_slot` is present, all entries must be explicit and non-negative.
  - `llama_kv_cache` now accepts explicit cell indices and updates `head` from the max written slot, not row order.
  - Build passed on both `build` and `build-cuda`.
  - Found and fixed an unrelated baseline crash in `examples/speculative-simple/speculative-simple.cpp`:
    - `--spec-type none` was still calling `common_speculative_init()`, which dereferenced a null draft context.
    - Baseline path now correctly skips speculative initialization.
- Greedy exactness check:
  - Prompt: `What is the tallest mountain on earth?`
  - CPU baseline vs CPU EAGLE:
    - identical generated text for `-n 16`
    - output: `The tallest mountain on Earth is Mount Everest, located in the Himalayas on the border`
  - CUDA baseline vs CUDA EAGLE:
    - identical generated text for `-n 16`
    - same output as CPU
  - Logs:
    - `/tmp/eagle1b_cpu_none.log`
    - `/tmp/eagle1b_cpu_eagle.log`
    - `/tmp/eagle1b_cuda_none.log`
    - `/tmp/eagle1b_cuda_eagle.log`
- First-token Kestrel parity check:
  - Used exact rendered prompt from `/home/alvion/projects/kestrel/reports/llama32_1b_eagle_first_token_dump.json`
  - Fresh traces:
    - `/tmp/llama_base_trace_cpu_kestrelprompt`
    - `/tmp/llama_base_trace_cuda_kestrelprompt`
  - CPU:
    - prompt ids exact
    - hidden cosine mean `0.9998229`, min `0.9993985`
    - top-10 differs by a near-tie swap at ranks 7/8 (`44` vs `10640`)
    - top-100 KL `0.0001017`
  - CUDA:
    - prompt ids exact
    - hidden cosine mean `0.9997464`, min `0.9990575`
    - top-10 exact
    - top-100 KL `0.0003182`
  - CPU head-dump parity still looks healthy:
    - `/tmp/eagle_dump_cpu_step1.parity`

Step result:
- Verification gate A/B/C passed well enough to continue.
- The `kv_slot` path is not used by the verifier yet; this step only establishes the plumbing and preserves current behavior.
- Summary numbers for this step:
  - first-token KL divergence vs Kestrel:
    - CPU `0.0001017`
    - CUDA `0.0003182`
  - CUDA EAGLE decode throughput:
    - standardized 1B perf check: `216.607 tok/s`
    - matching baseline: `451.740 tok/s`
  - Perf logs:
    - `/tmp/eagle1b_step1_perf.log`
    - `/tmp/eagle1b_step1_perf_base.log`

### Step 2. Switch target verification back to a compact fixed-shape tree

Goal:
- Replace duplicated-path verification with a compact tree layout.

Deliverable:
- One verifier batch with:
  - row 0 = root (`id_last`)
  - rows `1..max_proposals` = speculative node slots

Design:
- Fixed physical shape every pass.
- Unused node rows are padded/inactive.
- Tree metadata remains explicit:
  - parent
  - depth
  - token
  - active/inactive

Why:
- `max_proposals` is the tree size budget.
- The selected set is prefix-closed under cumulative logprob scoring.
- This should bound verifier rows to `max_proposals + 1`.

Verification gate:
- All required checks A/B/C.

Status:
- completed

Verification notes:
- 2026-03-08:
  - No net code change was required here because `61744981` already has the compact verifier shape we want for this step:
    - row `0` = root (`id_last`)
    - rows `1..max_proposals` = speculative node slots
    - unused node rows padded
  - The compact verifier remained the active exact path after Step 1.
  - I tested a follow-on experiment that bound explicit `kv_slot` values directly to the compact verifier rows.
  - That experiment failed exactness immediately on both CPU and CUDA.
  - Root cause:
    - accepted seq `0` tokens are still carried forward in sparse speculative cells
    - naive fixed slot reuse collides with those sparse accepted cells on later passes
    - so explicit slot binding for the compact verifier must wait until accepted-path compaction / commit exists
  - The failed experiment was reverted before proceeding.
- Greedy exactness check:
  - inherited exact compact path remains the same as Step 1
  - CPU baseline vs CPU EAGLE: identical
  - CUDA baseline vs CUDA EAGLE: identical
  - CPU/CUDA EAGLE outputs identical to each other
- First-token Kestrel parity check:
  - unchanged from Step 1 because this step made no behavior change in the accepted implementation

Step result:
- Step 2 is satisfied by the inherited compact verifier design on this branch.
- Explicit `kv_slot` use in the compact verifier is deferred until Step 5, where accepted-path compaction/commit can make slot reuse safe.
- Summary numbers for this step:
  - first-token KL divergence vs Kestrel:
    - CPU `0.0001017`
    - CUDA `0.0003182`
  - CUDA EAGLE decode throughput:
    - standardized 1B perf check: `216.607 tok/s`
    - matching baseline: `451.740 tok/s`

### Step 3. Keep compact tree metadata GPU-native

Goal:
- Stop using host-only tree metadata as the source of truth in the hot path.

Deliverable:
- GPU buffers for tree-node metadata used by verify and commit.

Design:
- Keep fixed-capacity arrays for:
  - parent indices
  - child traversal info
  - active mask
  - positions
  - slot indices

Why:
- This is the bridge to GPU-side verify and compaction.

Verification gate:
- All required checks A/B/C.

Status:
- not started

### Step 4. Move greedy tree verification onto GPU

Goal:
- Stop doing greedy tree walk on the host.

Deliverable:
- GPU kernel or equivalent backend path that:
  - walks the compact tree
  - matches target tokens against child nodes
  - emits accepted node indices / accepted length

Scope:
- Greedy only first.

Why:
- This removes a major host bottleneck and avoids repeated GPU->CPU syncs in the verify loop.

Verification gate:
- All required checks A/B/C.

Status:
- not started

### Step 5. Add GPU KV compaction / commit for accepted paths

Goal:
- Commit accepted nodes to canonical sequence state without host-side manual reconstruction.

Deliverable:
- GPU-side compaction/commit path that:
  - preserves accepted slots
  - removes rejected slots
  - makes accepted path the new committed sequence state

Why:
- Correctness across passes depends on exact accepted-path carry-forward.
- This is the piece SGLang effectively handles with explicit slot movement/compaction.

Verification gate:
- All required checks A/B/C.

Status:
- not started

### Step 6. Remove per-beam standalone draft KV state from the hot path

Goal:
- Stop treating each draft beam as a standalone self-contained state object with copied KV tensors.

Deliverable:
- Shared speculative-slot based draft state management for the hot path.

Why:
- Per-state KV copying is expensive and pushes us into dynamic graph shapes and high memory traffic.

Verification gate:
- All required checks A/B/C.

Status:
- not started

### Step 7. Make draft + verify runners statically shaped and graph-reusable

Goal:
- Build fixed-capacity reusable runners once the semantics are correct.

Deliverable:
- Stable-shape draft rollout runner.
- Stable-shape target verify runner.
- Reuse counters showing expected reuse after first build.

Why:
- CUDA graph capture only pays off once graph shape and state semantics are stable.

Verification gate:
- All required checks A/B/C.

Status:
- not started

### Step 8. Final validation and report

Goal:
- Confirm that the pivot landed correctly and did not regress exactness/parity.

Deliverable:
- 1B BF16 CPU/CUDA exactness pass
- Kestrel first-token parity pass
- updated performance notes

Minimum success condition:
- All prior steps pass the verification gates.

Stretch goal:
- Performance is at least not worse than the current pre-pivot base while preserving correctness.

Status:
- not started

## Comparison Notes From SGLang

Concrete architecture differences worth copying:

1. Explicit speculative KV placement and movement.
2. Compact tree build on GPU.
3. Greedy verify on GPU.
4. Accepted-path KV compaction on GPU.
5. Fixed-capacity CUDA-graph runners for draft and draft-extend.
6. Shared cache ownership model instead of per-beam copied state.

Relevant SGLang files:
- `/home/alvion/lib/sglang/python/sglang/srt/speculative/eagle_worker.py`
- `/home/alvion/lib/sglang/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`
- `/home/alvion/lib/sglang/python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`
- `/home/alvion/lib/sglang/python/sglang/srt/speculative/eagle_utils.py`
- `/home/alvion/lib/sglang/sgl-kernel/csrc/speculative/eagle_utils.cu`
- `/home/alvion/lib/sglang/sgl-kernel/csrc/speculative/speculative_sampling.cu`

## Operating Rule While Implementing

After each step:
1. update this document
2. run the required verification
3. only proceed if exactness is preserved
4. if exactness breaks and cannot be fixed quickly, checkpoint the work on a side branch and return to the last good commit
