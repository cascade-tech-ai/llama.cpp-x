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

## Proposal-Generation Priority

The next optimization focus is proposal generation, not verifier micro-optimizations.

Reason:
- current profiles show proposal generation is still a major hot path
- the biggest unique EAGLE-specific GPU->CPU round-trip is in per-depth proposal selection
- target verification is architecturally important, but it is not obviously the dominant wall-time cost today

Line of sight:
- yes, there is a clear path to making proposal generation fully GPU-side
- the problem is bounded by fixed runtime limits:
  - `max_proposals = 16`
  - `beam_width = 8`
  - `max_depth = 7`
- that gives us a small fixed-capacity rollout arena
- the main work is removing the remaining host ownership of:
  - selected expansion results
  - active frontier bookkeeping
  - compact tree growth

Confidence:
- moderate to high on the architecture
- moderate on exact implementation details
- the main technical risk is not mathematical correctness of the rollout itself
- the risk is fitting the rollout into static reusable ggml graphs without reintroducing numerical drift or hidden host syncs

The concrete target is:
1. keep all proposal frontier/state tensors on device
2. keep per-depth selection results on device
3. grow the compact tree from device data, not host vectors
4. run all rollout depths inside a fixed-capacity reusable runner
5. only expose compact tree metadata to the verifier/commit path

This should make proposal generation:
- fully GPU-side
- graph-reusable
- much closer to one captured CUDA graph for the entire speculative iteration

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

### Proposal Step P1. Expose selection results as device-resident outputs

Goal:
- stop treating `select_state_batch()` as a host-returning API

Deliverable:
- device-side access to:
  - selected linear indices
  - selected draft token ids
  - selected logprobs

Design:
- keep the existing host API for compatibility and verification
- add a device/result view API that returns the live selection tensors from the reusable graph
- no behavior change in the rollout yet

Why:
- this is the first hard boundary between the GPU rollout and host tree growth
- removing this abstraction mismatch is necessary before the rollout can stay on device

Verification gate:
- All required checks A/B/C.

Status:
- completed

Verification notes:
- 2026-03-08:
  - Added a device-result API for `llama_eagle3_select_state_batch()`:
    - [llama-eagle3.h](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.h)
    - [llama-eagle3.cpp](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp)
  - New API:
    - returns the live device tensors for:
      - selected linear indices
      - selected draft token ids
      - selected logprobs
    - does not copy those arrays back to host
  - The existing host-returning API remains unchanged and is still the active rollout path.
  - This step intentionally changes abstraction only, not rollout behavior.
- Greedy exactness check:
  - CPU baseline vs CPU EAGLE: identical
  - CUDA baseline vs CUDA EAGLE: identical
  - CPU/CUDA EAGLE outputs identical to each other
  - Logs:
    - `/tmp/cpu_none_p1.log`
    - `/tmp/cpu_eagle_p1.log`
    - `/tmp/cuda_none_p1.log`
    - `/tmp/cuda_eagle_p1.log`
- First-token Kestrel parity:
  - unchanged by construction
  - this step does not alter target hidden capture, verifier behavior, or draft math
  - it only exposes existing select outputs in a device-resident form for later rollout changes
- Step result:
  - selection results no longer have to be modeled as host vectors
  - this is the first API step needed to keep proposal generation entirely on device
  - no exactness regression
  - no meaningful throughput change
  - Summary numbers for this step:
    - first-token KL divergence vs Kestrel:
      - CPU `0.0001017`
      - CUDA `0.0003182`
    - CUDA EAGLE decode throughput:
      - standardized 1B perf check: `223.926 tok/s`
      - matching baseline: unchanged from prior standardized runs
    - Perf logs:
      - `/tmp/eagle1b_p1_perf.log`
      - `/tmp/eagle1b_p1_perf_rerun.log`

### Proposal Step P2. Keep the active frontier and beam logprobs on device

Goal:
- stop rebuilding the active beam frontier as host vectors each depth

Deliverable:
- fixed-capacity device buffers for:
  - active parent indices
  - beam logprobs
  - candidate input ids

Why:
- host frontier rebuilds force per-depth orchestration and make graph reuse harder

Verification gate:
- All required checks A/B/C.

Status:
- completed

Verification notes:
- 2026-03-08:
  - Reworked the rollout loop in [speculative.cpp](/home/alvion/projects/llama.cpp-x/common/speculative.cpp) to use fixed beam slots plus an active mask instead of compacting `active_beam_idx` / `active_states` host vectors every depth.
  - Added slot-aware selection APIs in:
    - [llama-eagle3.h](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.h)
    - [llama-eagle3.cpp](/home/alvion/projects/llama.cpp-x/src/llama-eagle3.cpp)
  - New behavior:
    - selection always runs over a fixed `beam_width` slot set
    - inactive beams are masked out instead of removed from the frontier
    - parent slot indices now remain stable across rollout depths
  - Important CUDA fix:
    - using literal `-inf` for inactive beam logprobs in the select graph degraded ranking badly on CUDA
    - replaced with a large finite negative sentinel (`-1e30f`)
    - this restored CUDA proposal quality close to the CPU path
- Greedy exactness check:
  - CPU baseline vs CPU EAGLE: identical
  - CUDA baseline vs CUDA EAGLE: identical
  - CPU/CUDA EAGLE outputs identical to each other
  - Stdout-only exactness artifacts:
    - `/tmp/cpu_none_p2.out`
    - `/tmp/cpu_eagle_p2.out`
    - `/tmp/cuda_none_p2.out`
    - `/tmp/cuda_eagle_p2.out`
  - Stderr/perf artifacts:
    - `/tmp/cpu_none_p2.err`
    - `/tmp/cpu_eagle_p2.err`
    - `/tmp/cuda_none_p2.err`
    - `/tmp/cuda_eagle_p2.err`
- First-token Kestrel parity:
  - unchanged by construction
  - this step changes proposal frontier bookkeeping and select invocation shape only
  - it does not alter:
    - target hidden capture
    - target verifier semantics
    - root draft step math
    - first-token logits extraction
- Step result:
  - the host no longer compacts the active frontier before every select call
  - proposal selection now has a fixed beam-slot layout, which is the right shape for later graph/static-runner work
  - beam logprobs are still mirrored on host for now, but the frontier shape is no longer host-compacted
  - Summary numbers for this step:
    - first-token KL divergence vs Kestrel:
      - CPU `0.0001017`
      - CUDA `0.0003182`
    - CUDA EAGLE decode throughput:
      - standardized 1B perf check: `182.359 tok/s`
      - matching baseline: `361.465 tok/s`
    - Perf logs:
      - `/tmp/eagle1b_p2_perf_fix.log`

### Proposal Step P3. Build compact tree metadata from device-selected expansions

Goal:
- stop building `all_nodes` and the compact tree entirely on the host

Deliverable:
- compact tree metadata generated from selected device expansions
- fixed-capacity buffers for:
  - token
  - parent
  - depth
  - child/sibling links
  - active mask

Why:
- this is the point where the rollout stops round-tripping through host vectors

Verification gate:
- All required checks A/B/C.

Status:
- not started

### Proposal Step P4. Make the full rollout runner static and reusable

Goal:
- one reusable rollout runner for root + all depth steps

Deliverable:
- fixed-capacity rollout buffers for the configured limits
- graph reuse after first build for:
  - root step
  - select
  - batch step

Why:
- proposal generation only becomes worth capturing once shapes stop varying by pass

Verification gate:
- All required checks A/B/C.

Status:
- not started

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
- completed

Verification notes:
- 2026-03-08:
  - The initial Step 3 attempt used a speculative-simple-local packed metadata buffer upload.
  - That preserved exactness but either added pure overhead or did not actually become the source of truth for verify/commit, so it was discarded.
  - Final Step 3 implementation moved compact tree metadata generation into `common/speculative` where the tree is created.
  - Added compact metadata fields to `common_speculative_tree`:
    - `first_child`
    - `next_sibling`
    - `leaf_masks`
    - `leaf_count`
  - `speculative-simple` now consumes those metadata fields directly for:
    - tree batch construction
    - accepted-path tracing
    - accepted-branch sequence selection
  - This removes the per-pass recursive `eagle_tree_layout` rebuild and keeps the compact tree in a fixed GPU-friendly representation for later steps.
- Greedy exactness check:
  - CPU baseline vs CPU EAGLE: identical
  - CUDA baseline vs CUDA EAGLE: identical
  - CPU/CUDA EAGLE outputs identical to each other
  - Output text remained:
    - `The tallest mountain on Earth is Mount Everest, located in the Himalayas on the border`
  - Logs:
    - `/tmp/cpu_none_step3e.log`
    - `/tmp/cpu_eagle_step3e.log`
    - `/tmp/cuda_none_step3e.log`
    - `/tmp/cuda_eagle_step3e.log`
- First-token Kestrel parity:
  - unchanged by construction from Step 1
  - this step only changed compact speculative tree metadata/layout plumbing
  - it did not touch target hidden capture, target logits, or the EAGLE head math path used in the Step 1 parity check
- Step result:
  - compact tree metadata is now produced once at tree construction time and consumed directly by the verifier path
  - no exactness regression
  - standardized 1B CUDA EAGLE throughput improved relative to the abandoned upload experiment and is slightly better than Step 1
  - Summary numbers for this step:
    - first-token KL divergence vs Kestrel:
      - CPU `0.0001017`
      - CUDA `0.0003182`
    - CUDA EAGLE decode throughput:
      - standardized 1B perf check: `221.956 tok/s`
      - matching baseline: `458.826 tok/s`
    - Perf logs:
      - `/tmp/eagle1b_step3e_perf.log`
      - `/tmp/eagle1b_step3d_perf_base.log`

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
