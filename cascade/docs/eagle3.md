# EAGLE3 speculative decoding support plan (llama.cpp)

AI-Generated Disclaimer
- This document was generated with AI assistance and must not be submitted upstream as-is.
- If any code or docs derived from this plan are to be submitted as part of a PR, they must be rewritten or exhaustively reviewed by a human contributor who can explain every line.

Status: Draft plan only. Implementation in progress in this fork.

Notes
- This repo does not accept PRs that are fully or predominantly AI-generated. Please review CONTRIBUTING.md and be ready to author the core implementation yourself.
- This plan references local Kestrel code for semantics and naming. The implementation decisions below should be confirmed against your desired training/export format.

AI usage in forks vs PRs
- A fork is your private workspace; you can generate or modify code however you want for internal use.
- The upstream policy governs what maintainers will accept in PRs. If you intend to submit upstream, you must follow CONTRIBUTING.md requirements (disclosure, manual review, and line‑by‑line understanding).
- This plan (and any AI-generated code that follows from it) is meant for a forked reference implementation only. Do not submit it upstream unless it is rewritten or exhaustively reviewed by a human and properly disclosed.

Goals
- Add EAGLE3 speculative decoding with a single-layer draft head.
- Hidden-state layer selection is stored in the EAGLE3 GGUF (no CLI layer-selection flag).
- Do not implement vLLM norm-before-residual variants.
- Prefer a separate GGUF for the EAGLE3 head.
- Support standard GGUF quant formats for the head weights.

Non-goals (for initial implementation)
- Support other speculative head variants (EAGLE2, Medusa, etc).
- Support draft heads that change norm placement.
- Rework general speculative decoding APIs beyond what is needed for EAGLE3.

Current codebase touchpoints
- Speculative decoding entry points:
  - common/speculative.h
  - common/speculative.cpp (EAGLE3 stub present)
  - common/common.h (common_params_speculative)
  - common/arg.cpp (CLI params for speculative decoding)
  - tools/cli/README.md, tools/completion/README.md (document CLI flags)
  - tools/server/server-context.cpp, tools/server/server-task.cpp (server params and JSON)
- Model graph and hidden states:
  - src/models/llama.cpp (layer loop for LLaMA arch)
  - src/llama-graph.h / src/llama-graph.cpp (graph outputs plumbing)
  - src/llama-context.h / src/llama-context.cpp (output buffers and copies)
- GGUF loading and metadata:
  - src/llama-arch.h / src/llama-arch.cpp (LLM_KV names, arch list)
  - src/llama-model-loader.h / src/llama-model-loader.cpp (GGUF read)
  - gguf-py/gguf/constants.py, gguf-py/gguf/metadata.py (GGUF key mirror)

Reference (Kestrel)
- Head architecture and config:
  - ~/projects/kestrel/kestrel/head.py
  - ~/projects/kestrel/kestrel/runtime.py (hidden layer selection)
  - ~/projects/kestrel/kestrel/draft_loader.py (config formats)
- Important semantics from Kestrel:
  - Hidden states are selected from HF outputs where hidden_states[0] is embeddings and hidden_states[i] is input to block i.
  - Head consumes (prev_hidden, cur_token) alignment: tokens are shifted by 1 relative to teacher hidden stream.
  - Head has a single transformer layer with: RMSNorm -> concat(input_emb, hidden) -> attention -> MLP -> LM head.
  - Optional draft vocab mapping via d2t/t2d.

Open design decisions (need confirmation)
1) GGUF packaging
   - Option A: Separate EAGLE3 GGUF with its own architecture key (e.g. "eagle3") and EAGLE-specific KV metadata.
   - Option B: Store EAGLE3 tensors in the base model GGUF under a new tensor namespace.
   - Strong preference is Option A per user request, but Option B avoids a second file and reduces loader changes.

2) Token embedding source
   - Decision: Reuse base model token embedding tensor at runtime. Do not store embeddings in the EAGLE3 GGUF.

3) Draft vocab mapping
   - Decision: Must support subset draft vocabularies. d2t mapping is required.
   - Kestrel uses d2t/t2d; for inference only d2t is required, but we may keep t2d for validation.

4) Hidden state capture semantics
   - Decision: Match Kestrel. Use "input to block i" (pre-norm), i.e. inpL in src/models/llama.cpp.

5) Target architectures
   - MVP: LLaMA-like architectures (LLM_ARCH_LLAMA and close relatives) only.
   - Future: generalize hidden-state capture hooks across all architectures.

6) CLI shape
   - Decision: Fit llama.cpp’s existing speculative decoding flags. Reuse --model-draft + --spec-type eagle3.
   - Layer selection is read from the EAGLE3 GGUF (no CLI flag).
   - Add rollout control flags matching speculators eval mode: --eagle-max-depth, --eagle-max-proposals, --eagle-prob-threshold.

Proposed GGUF schema for EAGLE3 head (draft)
- general.architecture = "eagle3" (new arch string)
- eagle3.hidden_size (int)
- eagle3.intermediate_size (int)
- eagle3.num_attention_heads (int)
- eagle3.num_key_value_heads (int)
- eagle3.head_dim (int, optional)
- eagle3.rms_norm_eps (float)
- eagle3.rope_theta (float)
- eagle3.rope_scaling.type (string, optional)
- eagle3.rope_scaling.factor (float, optional)
- eagle3.hidden_concat (int)  # number of hidden layers concatenated
- eagle3.target_hidden_size (int, optional) # base model hidden size if different
- eagle3.draft_vocab_size (int)
- eagle3.vocab_size (int) # base model vocab size
- eagle3.hidden_state_layer_ids (int array) # resolved indices
- eagle3.norm_before_residual (bool) # expect false; error if true
- eagle3.d2t (int array) # optional; length = draft_vocab_size

Note: names and placement should be aligned with LLM_KV_NAMES and gguf-py constants.

Implementation plan (phased)

Phase 0: Align on format and semantics
- Confirm which Kestrel export format will be used (Kestrel vs speculators).
- Decide on GGUF schema and whether to store embeddings in the head GGUF.
- Decide draft vocab requirements (full vocab only vs subset w/ d2t).
- Confirm rollout CLI parameters and names (use --eagle-max-depth, --eagle-max-proposals, --eagle-prob-threshold).

Phase 1: Hidden state capture in llama.cpp
- Add a config path for "capture hidden states for layers X" that is controlled by EAGLE3 GGUF metadata.
- Extend graph outputs to include selected per-layer hidden state tensors.
- Capture the pre-layer input (inpL) at each transformer block for LLaMA architecture.

Implementation sketch
- Add fields to llama_context / llama_cparams for:
  - bool eagle3_enabled
  - vector<int> eagle3_hidden_layer_ids (resolved)
- In src/models/llama.cpp, add a callback hook at the start of each layer loop:
  - when il in hidden_layer_ids, register inpL as an output tensor (per-token, per-batch).
- Extend llm_graph_result to store extra output tensors and set them as outputs.
- In llama_context, after graph execution, copy selected layer outputs into a ring buffer aligned with tokens.
  - Maintain a per-sequence buffer of hidden states for the selected layers.
  - Append new hidden states for newly decoded tokens.

Files to modify
- src/models/llama.cpp
- src/llama-graph.h
- src/llama-graph.cpp
- src/llama-context.h
- src/llama-context.cpp
- include/llama.h (if a public API is needed; otherwise keep internal)

Phase 2: EAGLE3 head model loader and runtime
- Add a lightweight head model struct to hold EAGLE3 tensors + config.
- Load from GGUF (separate file) using llama_model_loader or a new minimal loader.
- Build a GGML graph for the head that mirrors Kestrel's EagleHead forward.

Implementation sketch
- Create new files:
  - src/llama-eagle3.h
  - src/llama-eagle3.cpp
- Define structs:
  - llama_eagle3_hparams (config)
  - llama_eagle3_model (tensors)
  - llama_eagle3_context (KV cache + graph inputs/outputs)
- Head forward passes:
  - Inputs: teacher_hidden (concatenated), input_ids
  - Use base model embedding weight for input_ids (Option A), or head gguf embeddings (Option B).
  - Build attention, MLP, final norm, LM head.
  - Provide logits over draft vocab.
- Manage a small KV cache for the head (single layer).

Files to modify/create
- New: src/llama-eagle3.h
- New: src/llama-eagle3.cpp
- src/llama-model-loader.h / src/llama-model-loader.cpp (if reusing loader)
- src/llama-arch.h / src/llama-arch.cpp (new arch + KV names)
- gguf-py/gguf/constants.py (new keys)
- gguf-py/gguf/metadata.py (if metadata override support is desired)

Phase 3: Speculative decoding integration
- Implement common_speculative_state_eagle3 in common/speculative.cpp.
- Add new CLI params for EAGLE3 rollout control (match speculators eval: max-depth, max-proposals, prob-threshold).
- Ensure speculative decoding uses head to propose tokens and maps to base tokens.

Implementation sketch
- Extend common_params_speculative with:
  - mparams_eagle (path/hf repo) OR reuse mparams_dft + new type enum
  - eagle_max_depth, eagle_max_proposals, eagle_prob_threshold (match speculators eval; CLI: --eagle-max-depth/--eagle-max-proposals/--eagle-prob-threshold)
- In common_speculative_init:
  - detect eagle3 head file, build eagle3 state
- In common_speculative_state_eagle3::draft:
  - Build head prefill from cached teacher hidden states (prompt)
  - Unroll head to produce up to n_max draft tokens
  - Map draft ids to base ids if needed (d2t)
  - Return draft token list (base vocab ids)

Files to modify
- common/common.h
- common/speculative.h
- common/speculative.cpp
- common/arg.cpp
- tools/cli/README.md
- tools/completion/README.md
- tools/server/server-context.cpp
- tools/server/server-task.cpp

Phase 4: GGUF conversion pipeline
- Add a conversion script to export Kestrel head to GGUF.
- Because this is not intended for upstream yet, place scripts under ./cascade.

New files (cascade)
- cascade/scripts/convert_eagle3_to_gguf.py (converter for Kestrel/speculators head checkpoints)
  - Example: `python cascade/scripts/convert_eagle3_to_gguf.py /path/to/head --out /path/to/eagle3.gguf`
  - If the config lacks resolved layer ids, pass `--base-n-layers` to resolve `eagle_aux_hidden_state_layers`.
- cascade/docs/eagle3.md (this file)

Phase 5: Tests and validation
- Add a minimal unit test that loads a small EAGLE3 head GGUF and runs a 1-2 token draft on a tiny base model.
- Add a basic server test to verify deterministic output with and without eagle3 (draft should not alter final tokens).

Potential test locations
- tests/ (new unit tests)
- tools/server/tests/ (speculative integration)

Risks and mitigations
- Hidden state capture adds memory and copy overhead.
  - Mitigate by only enabling when eagle3 is active and only for selected layers.
- Head graph may require extra allocations and per-step rebuilds.
  - Mitigate with a persistent graph and small KV cache (single layer).
- Draft vocab mapping errors can corrupt acceptance.
  - Add strict validation of d2t size and bounds on load.

Outstanding questions for you
- Do you want to require full vocab draft heads (no d2t), at least for MVP?
- Should the head GGUF include token embeddings, or always reuse the base model embeddings?
- Confirm the exact names/defaults for the three rollout CLI params (use --eagle-max-depth, --eagle-max-proposals, --eagle-prob-threshold).
- Do you want a new CLI flag (e.g. --model-eagle3) or reuse --model-draft with --spec-type eagle3?
