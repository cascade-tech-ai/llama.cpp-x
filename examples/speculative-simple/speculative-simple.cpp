#include "arg.h"
#include "chat.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "ggml-cuda.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace {
using json = nlohmann::ordered_json;

struct scoped_prof {
    bool enabled = false;
    const char * name = nullptr;
    int64_t t_start_us = 0;

    scoped_prof(bool enabled, const char * name) : enabled(enabled), name(name) {
        if (!enabled) {
            return;
        }
        t_start_us = ggml_time_us();
#if defined(GGML_USE_CUDA)
        ggml_backend_cuda_nvtx_push(name);
#endif
    }

    ~scoped_prof() {
        if (!enabled) {
            return;
        }
#if defined(GGML_USE_CUDA)
        ggml_backend_cuda_nvtx_pop();
#endif
    }

    double elapsed_ms() const {
        return enabled ? (ggml_time_us() - t_start_us) / 1000.0 : 0.0;
    }
};

static int32_t find_tree_child_token(const common_speculative_tree & tree, int32_t parent, llama_token tok) {
    const int32_t first = parent < 0
        ? (tree.parents.empty() ? -1 : 0)
        : tree.first_child[(size_t) parent];

    if (parent < 0) {
        for (size_t i = 0; i < tree.tokens.size(); ++i) {
            if (tree.parents[i] < 0 && tree.tokens[i] == tok) {
                return (int32_t) i;
            }
        }
        return -1;
    }

    for (int32_t node = first; node >= 0; node = tree.next_sibling[(size_t) node]) {
        if (tree.tokens[(size_t) node] == tok) {
            return node;
        }
    }
    return -1;
}

static uint32_t tree_leaf_count(const common_speculative_tree & tree) {
    return tree.leaf_count;
}

static void append_seq_ids_from_mask(uint32_t mask, std::vector<llama_seq_id> & out) {
    out.clear();
    while (mask != 0) {
        const uint32_t bit = __builtin_ctz(mask);
        out.push_back((llama_seq_id) (1 + bit));
        mask &= mask - 1;
    }
}

static llama_seq_id first_seq_id_from_mask(uint32_t mask) {
    GGML_ASSERT(mask != 0);
    return (llama_seq_id) (1 + __builtin_ctz(mask));
}

static std::vector<int32_t> trace_accepted_tree_nodes(const common_speculative_tree & tree, const llama_tokens & ids) {
    std::vector<int32_t> nodes;
    if (ids.empty()) {
        return nodes;
    }

    int32_t node = find_tree_child_token(tree, -1, ids[0]);
    while (node >= 0) {
        nodes.push_back(node);
        if (nodes.size() >= ids.size() - 1) {
            break;
        }
        node = find_tree_child_token(tree, node, ids[nodes.size()]);
    }

    return nodes;
}

static void build_eagle_tree_batch(
        llama_batch & batch_tgt,
        const llama_token id_last,
        const llama_pos base_pos,
        const int32_t max_nodes,
        common_speculative_tree & tree) {
    std::vector<llama_seq_id> shared_seq_ids;
    shared_seq_ids.reserve(1 + (size_t) max_nodes);
    shared_seq_ids.push_back(0);
    for (int32_t i = 0; i < max_nodes; ++i) {
        shared_seq_ids.push_back((llama_seq_id) (1 + i));
    }
    common_batch_add(batch_tgt, id_last, base_pos, shared_seq_ids, true);

    tree.row_indices.assign(tree.tokens.size(), std::numeric_limits<uint32_t>::max());
    std::vector<int32_t> nodes(tree.tokens.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        nodes[i] = (int32_t) i;
    }

    std::stable_sort(nodes.begin(), nodes.end(), [&](int32_t a, int32_t b) {
        const int32_t depth_a = tree.depths[(size_t) a];
        const int32_t depth_b = tree.depths[(size_t) b];
        if (depth_a != depth_b) {
            return depth_a < depth_b;
        }
        return a < b;
    });

    for (int32_t node : nodes) {
        std::vector<llama_seq_id> seq_ids;
        append_seq_ids_from_mask(tree.leaf_masks[(size_t) node], seq_ids);
        GGML_ASSERT(!seq_ids.empty());

        const llama_pos pos = base_pos + 1 + tree.depths[(size_t) node];
        const uint32_t row = batch_tgt.n_tokens;
        common_batch_add(batch_tgt, tree.tokens[(size_t) node], pos, seq_ids, true);
        tree.row_indices[(size_t) node] = row;
    }

    for (size_t i = 0; i < tree.row_indices.size(); ++i) {
        GGML_ASSERT(tree.row_indices[i] != std::numeric_limits<uint32_t>::max());
    }

    GGML_ASSERT(max_nodes >= 0);
    GGML_ASSERT((int32_t) tree.tokens.size() <= max_nodes);
}

// Build a flat tree batch: all tree tokens on seq 1 with unique cache positions
// and depth-based RoPE positions. Returns the rope_pos array and tree parent array
// for the tree mask.
static void build_flat_tree_batch(
        llama_batch & batch_tgt,
        const llama_token id_last,
        const llama_pos cache_base,
        const llama_pos rope_base,
        common_speculative_tree & tree,
        std::vector<llama_pos> & rope_pos_out,
        std::vector<int32_t> & tree_parents_out) {
    // Root token: on both seq 0 and seq 1
    common_batch_add(batch_tgt, id_last, cache_base, { 0, 1 }, true);
    rope_pos_out.push_back(rope_base);

    // Sort tree nodes by depth (same ordering as coupled version)
    tree.row_indices.assign(tree.tokens.size(), std::numeric_limits<uint32_t>::max());
    std::vector<int32_t> nodes(tree.tokens.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        nodes[i] = (int32_t) i;
    }
    std::stable_sort(nodes.begin(), nodes.end(), [&](int32_t a, int32_t b) {
        const int32_t depth_a = tree.depths[(size_t) a];
        const int32_t depth_b = tree.depths[(size_t) b];
        if (depth_a != depth_b) {
            return depth_a < depth_b;
        }
        return a < b;
    });

    // Map from original tree node index → sorted position
    std::vector<int32_t> node_to_sorted(tree.tokens.size(), -1);
    for (size_t si = 0; si < nodes.size(); ++si) {
        node_to_sorted[(size_t) nodes[si]] = (int32_t) si;
    }

    tree_parents_out.clear();
    for (size_t si = 0; si < nodes.size(); ++si) {
        const int32_t node = nodes[si];
        const uint32_t row = batch_tgt.n_tokens;
        const llama_pos cache_pos = cache_base + 1 + (llama_pos) si;
        const llama_pos rope_pos  = rope_base  + 1 + tree.depths[(size_t) node];

        common_batch_add(batch_tgt, tree.tokens[(size_t) node], cache_pos, { 1 }, true);
        rope_pos_out.push_back(rope_pos);
        tree.row_indices[(size_t) node] = row;

        // Build parent array for tree mask (indices within the tree node array)
        const int32_t orig_parent = tree.parents[(size_t) node];
        if (orig_parent < 0) {
            tree_parents_out.push_back(-1);  // parent is root (outside tree)
        } else {
            tree_parents_out.push_back(node_to_sorted[(size_t) orig_parent]);
        }
    }

    for (size_t i = 0; i < tree.row_indices.size(); ++i) {
        GGML_ASSERT(tree.row_indices[i] != std::numeric_limits<uint32_t>::max());
    }
}

static std::string trace_token_text(llama_context * ctx, llama_token tok) {
    return common_token_to_piece(ctx, tok);
}

static json trace_top_candidates_json(
        llama_context * ctx,
        const std::vector<common_sampler_trace_candidate> & candidates) {
    json out = json::array();
    for (const auto & cand : candidates) {
        out.push_back({
            {"token", (int) cand.token},
            {"text_escaped", trace_token_text(ctx, cand.token)},
            {"prob", cand.p},
        });
    }
    return out;
}

static float trace_find_draft_prob(
        const common_speculative_trace & trace,
        size_t depth,
        llama_token tok) {
    std::vector<const common_speculative_trace_node *> level;
    for (const auto & root : trace.proposal_tree) {
        level.push_back(&root);
    }

    for (size_t d = 0; d < depth; ++d) {
        std::vector<const common_speculative_trace_node *> next;
        for (const auto * node : level) {
            for (const auto & child : node->children) {
                next.push_back(&child);
            }
        }
        level.swap(next);
        if (level.empty()) {
            return 0.0f;
        }
    }

    float best = 0.0f;
    for (const auto * node : level) {
        if (node->token == tok && node->prob > best) {
            best = node->prob;
        }
    }
    return best;
}

static json trace_proposal_tree_json(
        llama_context * ctx,
        const common_speculative_trace & trace) {
    std::function<json(const common_speculative_trace_node &)> encode =
            [&](const common_speculative_trace_node & node) -> json {
        json out = {
            {"token", (int) node.token},
            {"text_escaped", trace_token_text(ctx, node.token)},
            {"prob", node.prob},
            {"cum_prob", node.cum_prob},
            {"selected", node.selected},
            {"accepted", node.accepted},
            {"children", json::array()},
        };
        for (const auto & child : node.children) {
            out["children"].push_back(encode(child));
        }
        return out;
    };

    json tree = json::array();
    for (const auto & root : trace.proposal_tree) {
        tree.push_back(encode(root));
    }
    return tree;
}
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.type != COMMON_SPECULATIVE_TYPE_NONE &&
        params.speculative.mparams_dft.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    if (params.speculative.type == COMMON_SPECULATIVE_TYPE_EAGLE3) {
        params.kv_unified = true;
        params.n_parallel = std::max(params.n_parallel, params.speculative.eagle_max_proposals + 1);
    }

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    auto chat_templates = common_chat_templates_init(model_tgt, params.chat_template);
    const bool profile_spec = std::getenv("CASCADE_SPEC_PROFILE") != nullptr;

    // load the draft model (non-eagle3)
    llama_model_ptr model_dft;
    if (params.speculative.type != COMMON_SPECULATIVE_TYPE_EAGLE3 &&
        params.speculative.type != COMMON_SPECULATIVE_TYPE_NONE) {
        // TODO: simplify this logic
        {
            const auto & params_spec = params.speculative;

            auto params_dft = params;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx;
            params_dft.n_batch      = llama_n_ctx_seq(ctx_tgt);
            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams_dft;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params.speculative.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params.speculative.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return 1;
            }

            params.speculative.model_dft = model_dft.get();
            params.speculative.cparams_dft = common_context_params_to_llama(params_dft);
        }
    }

    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        params.conversation_mode = has_chat_template
            ? COMMON_CONVERSATION_MODE_ENABLED
            : COMMON_CONVERSATION_MODE_DISABLED;
    }

    if (params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }

    std::string prompt = params.prompt;
    if (params.conversation_mode && params.enable_chat_template) {
        if (!params.prompt.empty() && params.system_prompt.empty()) {
            LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
        }

        std::vector<common_chat_msg> chat_msgs;
        if (!params.system_prompt.empty()) {
            common_chat_msg system_msg;
            system_msg.role = "system";
            system_msg.content = params.system_prompt;
            chat_msgs.push_back(std::move(system_msg));
        }
        if (!params.prompt.empty()) {
            common_chat_msg user_msg;
            user_msg.role = "user";
            user_msg.content = params.prompt;
            chat_msgs.push_back(std::move(user_msg));
        }

        if (!chat_msgs.empty()) {
            common_chat_templates_inputs inputs;
            inputs.use_jinja = params.use_jinja;
            inputs.messages = std::move(chat_msgs);
            inputs.add_generation_prompt = !params.prompt.empty();

            prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
        }
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // init the speculator (before prompt eval so EAGLE3 can capture hidden states)
    const auto & params_spec = params.speculative;
    const bool trace_enabled = !params_spec.eagle_trace_yaml.empty();
    const float trace_prob_threshold = params_spec.eagle_per_beam_topk_candidates > 0
        ? 1.0f / (float) params_spec.eagle_per_beam_topk_candidates
        : 0.0f;
    json trace_tokens = json::array();
    json trace_cycles = json::array();
    int trace_emitted_idx = 0;

    struct common_speculative * spec = params.speculative.type != COMMON_SPECULATIVE_TYPE_NONE
        ? common_speculative_init(params.speculative, ctx_tgt)
        : nullptr;

    if (params.speculative.type == COMMON_SPECULATIVE_TYPE_EAGLE3 && llama_model_is_hybrid(model_tgt)) {
        LOG_WRN("%s: hybrid model detected — tree verification disabled, using linear verification\n", __func__);
    }

    // eval the prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, prompt_tgt, 0);

    const bool is_hybrid = llama_model_is_hybrid(model_tgt);
    const bool use_flat_tree = !is_hybrid && (std::getenv("COUPLED_TREE") == nullptr);
    bool flat_tree_diverged = false; // once true, cache positions diverge from rope positions
    // For hybrid models during linear verification, we use a temporary seq to
    // save/restore the recurrent state (which can't be partially rolled back).
    // Must be < n_seq_max (= n_parallel) so the recurrent memory accepts it.
    // Tree verification is disabled for hybrid models, so branch seq_ids are unused.
    const llama_seq_id hybrid_save_seq = (llama_seq_id)(llama_n_seq_max(ctx_tgt) - 1);

    const int max_tree_seq_ids = params.speculative.type == COMMON_SPECULATIVE_TYPE_EAGLE3
        ? std::max(1, params.speculative.eagle_max_proposals + 1)
        : 1;
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, max_tree_seq_ids);

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();
    int64_t t_eagle_total_us = 0;
    int64_t t_target_total_us = 0;
    int64_t t_target_fwd_us = 0;
    int64_t t_target_sampling_us = 0;
    int32_t n_target_passes = 0;

    while (true) {
        const int pass_idx = n_target_passes;
        const int n_past_before = n_past;
        // optionally, generate draft tokens that can be appended to the target batch
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        double pass_eagle_ms = 0.0;
        llama_tokens draft;
        {
            scoped_prof zone_eagle(profile_spec, "spec/target_pass/eagle_total");
            draft = spec
                ? common_speculative_draft(spec, params_spec, prompt_tgt, id_last, 0)
                : llama_tokens{};
            pass_eagle_ms = zone_eagle.elapsed_ms();
            t_eagle_total_us += (int64_t) (pass_eagle_ms * 1000.0);
        }
        common_speculative_tree tree;
        common_speculative_trace spec_trace;
        const bool has_tree = spec ? common_speculative_get_tree(spec, tree) : false;
        bool has_trace = spec ? common_speculative_get_trace(spec, spec_trace) : false;
        // Tree verification creates coupled sequences (root token shared across all branches),
        // which the hybrid memory's batch splitting cannot handle for recurrent layers.
        bool use_tree = params.speculative.type == COMMON_SPECULATIVE_TYPE_EAGLE3 && has_tree && !tree.tokens.empty()
            && !llama_model_is_hybrid(model_tgt);

        //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

        // always have a token to evaluate from before - id_last
        scoped_prof zone_target(profile_spec, "spec/target_pass/total");
        common_batch_clear(batch_tgt);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        double pass_target_fwd_ms = 0.0;
        llama_pos cache_base = 0; // cache position of root token (for flat tree cleanup)
        // tree mask data must outlive llama_decode (set_kq_mask_tree stores a raw pointer)
        std::vector<int32_t> tree_parents;
        llama_kq_mask_tree tree_mask = { 0, nullptr, 0 };
        {
            // do not waste time on small drafts
            if (draft.size() < (size_t) params_spec.n_min) {
                draft.clear();
                tree.clear();
                use_tree = false;
            }

            if (use_tree && use_flat_tree) {
                // Flat tree: all tree tokens on seq 1 with unique cache positions,
                // depth-based RoPE positions, and a tree attention mask.
                auto * mem = llama_get_memory(ctx_tgt);
                cache_base = llama_memory_seq_pos_max(mem, 0) + 1;

                llama_memory_seq_rm(mem, 1, -1, -1);
                llama_memory_seq_cp(mem, 0, 1, -1, -1);

                std::vector<llama_pos> rope_pos;
                build_flat_tree_batch(batch_tgt, id_last, cache_base, n_past, tree, rope_pos, tree_parents);

                tree_mask = { tree_parents.size(), tree_parents.data(), 1 };
                llama_set_kq_mask_tree(ctx_tgt, &tree_mask);
                llama_set_rope_pos_override(ctx_tgt, (uint32_t) rope_pos.size(), rope_pos.data());

                flat_tree_diverged = true;
                n_past++;

                common_speculative_set_tree(spec, tree);
            } else if (use_tree) {
                // Coupled tree: one sequence per branch, shared root token
                if ((int) tree_leaf_count(tree) + 1 > (int) llama_n_seq_max(ctx_tgt)) {
                    LOG_ERR("%s: insufficient n_seq_max=%u for %zu EAGLE branches\n",
                            __func__, llama_n_seq_max(ctx_tgt), (size_t) tree_leaf_count(tree));
                    return 1;
                }

                auto * mem = llama_get_memory(ctx_tgt);
                for (int32_t i = 0; i < params_spec.eagle_max_proposals; ++i) {
                    const llama_seq_id seq_id = (llama_seq_id) (1 + i);
                    llama_memory_seq_rm(mem, seq_id, -1, -1);
                    llama_memory_seq_cp(mem, 0, seq_id, -1, -1);
                }
                const llama_pos tree_base_pos = n_past++;
                build_eagle_tree_batch(batch_tgt, id_last, tree_base_pos, params_spec.eagle_max_proposals, tree);

                common_speculative_set_tree(spec, tree);
            } else {
                // Linear decode (no tree)
                auto * mem = llama_get_memory(ctx_tgt);
                if (flat_tree_diverged) {
                    cache_base = llama_memory_seq_pos_max(mem, 0) + 1;

                    std::vector<llama_pos> rope_pos;
                    common_batch_add(batch_tgt, id_last, cache_base, { 0 }, true);
                    rope_pos.push_back(n_past);
                    n_past++;
                    for (size_t i = 0; i < draft.size(); ++i) {
                        common_batch_add(batch_tgt, draft[i], cache_base + 1 + (llama_pos)i, { 0 }, true);
                        rope_pos.push_back(n_past + (llama_pos)i);
                    }
                    llama_set_rope_pos_override(ctx_tgt, (uint32_t) rope_pos.size(), rope_pos.data());
                } else {
                    // For hybrid models, save the recurrent state before processing drafts.
                    // The recurrent state can't be partially rolled back, so we need to
                    // restore it if any drafts are rejected.
                    if (is_hybrid && !draft.empty()) {
                        llama_memory_seq_rm(mem, hybrid_save_seq, -1, -1);
                        llama_memory_seq_cp(mem, 0, hybrid_save_seq, -1, -1);
                    }
                    common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);
                    for (size_t i = 0; i < draft.size(); ++i) {
                        common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
                    }
                }
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            scoped_prof zone_target_fwd(profile_spec, "spec/target_pass/forward");
            const int ret = llama_decode(ctx_tgt, batch_tgt);
            if (ret != 0) {
                LOG_ERR("%s: llama_decode failed with %d\n", __func__, ret);
                return 1;
            }
            pass_target_fwd_ms = zone_target_fwd.elapsed_ms();
            t_target_fwd_us += (int64_t) (zone_target_fwd.elapsed_ms() * 1000.0);
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        scoped_prof zone_target_sampling(profile_spec, "spec/target_pass/sampling");
        const int64_t t_sampling_start = ggml_time_us();
        std::vector<common_sampler_tree_trace_pass> trace_passes;
        const auto ids = use_tree
            ? (trace_enabled
                ? common_sampler_sample_and_accept_tree_trace(smpl, ctx_tgt, 0, tree, trace_passes)
                : common_sampler_sample_and_accept_tree(smpl, ctx_tgt, 0, tree))
            : common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
        llama_tokens ids_limited = ids;

        if (params.n_predict >= 0) {
            const int n_max_emit = params.n_predict + 1;
            const int n_remaining = n_max_emit - n_predict;
            if (n_remaining > 0 && (int) ids_limited.size() > n_remaining) {
                ids_limited.resize((size_t) n_remaining);
            }
        }
        const double pass_target_sampling_ms = zone_target_sampling.elapsed_ms();
        t_target_sampling_us += ggml_time_us() - t_sampling_start;

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids_limited.size() > 0); // there will always be at least one accepted token

        n_past    += ids_limited.size() - 1;
        n_drafted += draft.size(); // note: we ignore the discarded small drafts
        n_accept  += ids_limited.size() - 1;
        n_predict += ids_limited.size();
        t_target_total_us += (int64_t) (zone_target.elapsed_ms() * 1000.0);
        ++n_target_passes;

        if (profile_spec) {
            LOG_INF("spec profile: pass=%d target_total=%.3fms target_forward=%.3fms target_sampling=%.3fms eagle_total=%.3fms draft_tokens=%zu accepted=%zu use_tree=%d\n",
                    pass_idx,
                    zone_target.elapsed_ms(),
                    pass_target_fwd_ms,
                    pass_target_sampling_ms,
                    pass_eagle_ms,
                    draft.size(),
                    ids_limited.size() - 1,
                    use_tree ? 1 : 0);
        }

        if (use_tree) {
            common_speculative_accept_tokens(spec, ids_limited, 0);
        }
        common_speculative_accept(spec, ids_limited.size() - 1);

        if (use_tree && trace_enabled) {
            has_trace = spec ? common_speculative_get_trace(spec, spec_trace) : false;
        }

        std::vector<int32_t> accepted_nodes;
        if (use_tree && use_flat_tree) {
            // Flat tree cleanup: copy accepted cells from seq 1 to seq 0, delete seq 1
            accepted_nodes = trace_accepted_tree_nodes(tree, ids_limited);
            auto * mem = llama_get_memory(ctx_tgt);
            for (const auto & node : accepted_nodes) {
                const llama_pos p = cache_base + (llama_pos) tree.row_indices[(size_t) node];
                llama_memory_seq_cp(mem, 1, 0, p, p + 1);
            }
            llama_memory_seq_rm(mem, 1, -1, -1);
            llama_clear_kq_mask_tree(ctx_tgt);
            llama_clear_rope_pos_override(ctx_tgt);
        } else if (use_tree) {
            // Coupled tree cleanup: copy winning branch to seq 0, delete all branches
            accepted_nodes = trace_accepted_tree_nodes(tree, ids_limited);
            auto * mem = llama_get_memory(ctx_tgt);
            if (!accepted_nodes.empty()) {
                const int32_t deepest = accepted_nodes.back();
                const llama_seq_id seq_id = first_seq_id_from_mask(tree.leaf_masks[(size_t) deepest]);
                const llama_pos p0 = n_past_before + 1;
                const llama_pos p1 = p0 + (llama_pos) accepted_nodes.size();
                llama_memory_seq_cp(mem, seq_id, 0, p0, p1);
            }
            for (int32_t i = 0; i < params_spec.eagle_max_proposals; ++i) {
                const llama_seq_id seq_id = (llama_seq_id) (1 + i);
                llama_memory_seq_rm(mem, seq_id, -1, -1);
            }
        }

        if (trace_enabled) {
            json cycle = {
                {"cycle", pass_idx + 1},
                {"context_len", (int) prompt_tgt.size() + 1},
                {"proposal_count", use_tree && has_trace ? (int) spec_trace.proposal_count : 0},
                {"accepted_count", (int) ids_limited.size() - 1},
                {"proposal_tree", use_tree && has_trace ? trace_proposal_tree_json(ctx_tgt, spec_trace) : json::array()},
                {"passes", json::array()},
                {"appended_tokens", json::array()},
            };

            for (const auto & pass : trace_passes) {
                cycle["passes"].push_back({
                    {"pass", pass.pass},
                    {"sampled_token", (int) pass.sampled_token},
                    {"sampled_text_escaped", trace_token_text(ctx_tgt, pass.sampled_token)},
                    {"sampled_target_prob", pass.sampled_prob},
                    {"sampled_draft_prob", use_tree && has_trace ? trace_find_draft_prob(spec_trace, (size_t) pass.pass, pass.sampled_token) : 0.0f},
                    {"accepted", pass.accepted},
                    {"reason", pass.reason},
                    {"target_top10", trace_top_candidates_json(ctx_tgt, pass.target_top_candidates)},
                });
            }

            const int accepted_count = std::max(0, (int) ids_limited.size() - 1);
            for (size_t i = 0; i < ids_limited.size(); ++i) {
                cycle["appended_tokens"].push_back((int) ids_limited[i]);

                const char * kind = (int) i < accepted_count ? "accepted" : (use_tree ? "rejected" : "fallback");
                trace_tokens.push_back({
                    {"index", trace_emitted_idx++},
                    {"cycle", pass_idx + 1},
                    {"token", (int) ids_limited[i]},
                    {"text_escaped", trace_token_text(ctx_tgt, ids_limited[i])},
                    {"kind", kind},
                });
            }

            trace_cycles.push_back(std::move(cycle));
        }

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids_limited.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids_limited[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids_limited.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids_limited.size() - 1, (int) draft.size(), id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            if (!use_tree) {
                if (flat_tree_diverged) {
                    // After flat tree, positions diverged: remove rejected draft tokens by cache position
                    const llama_pos remove_from = cache_base + (llama_pos) ids_limited.size();
                    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, remove_from, -1);
                    llama_clear_rope_pos_override(ctx_tgt);
                } else if (is_hybrid && !draft.empty()) {
                    // For hybrid models, the recurrent state can't be partially rolled back.
                    // Restore the saved state (at position n_past_before - 1) and replay
                    // only the accepted tokens to rebuild both KV cache and recurrent state.
                    auto * mem = llama_get_memory(ctx_tgt);

                    // Clear seq 0 entirely (both attention KV and recurrent state)
                    llama_memory_seq_rm(mem, 0, -1, -1);
                    // Restore the saved state from before draft processing
                    llama_memory_seq_cp(mem, hybrid_save_seq, 0, -1, -1);
                    // Clean up the temporary save sequence
                    llama_memory_seq_rm(mem, hybrid_save_seq, -1, -1);

                    // Replay accepted tokens through the full target model to rebuild
                    // both the attention KV cache and the recurrent state.
                    // prompt_tgt[n_past_before .. n_past-1] contains the original id_last
                    // followed by the accepted draft tokens (pushed during the acceptance loop).
                    const int n_replay = n_past - n_past_before;
                    GGML_ASSERT(n_replay > 0 && n_replay == (int) ids_limited.size());
                    llama_batch replay_batch = llama_batch_init(n_replay, 0, 1);
                    common_batch_clear(replay_batch);
                    for (int i = 0; i < n_replay; i++) {
                        common_batch_add(replay_batch, prompt_tgt[(size_t)(n_past_before + i)], n_past_before + i, { 0 }, false);
                    }
                    llama_decode(ctx_tgt, replay_batch);
                    llama_batch_free(replay_batch);
                } else {
                    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
                }
            }
            if (params.speculative.type == COMMON_SPECULATIVE_TYPE_EAGLE3 && !use_tree) {
                llama_eagle3_trim_seq(ctx_tgt, 0, n_past);
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? (100.0f * n_accept / n_drafted) : 0.0f);

    if (profile_spec) {
        LOG_INF("spec profile total: target_passes=%d target_total=%.3fms target_forward=%.3fms target_sampling=%.3fms eagle_total=%.3fms\n",
                n_target_passes,
                t_target_total_us / 1000.0,
                t_target_fwd_us / 1000.0,
                t_target_sampling_us / 1000.0,
                t_eagle_total_us / 1000.0);
    }

    if (trace_enabled) {
        json payload = {
            {"base_model", params.model.path},
            {"head_model", params_spec.mparams_dft.path},
            {"prompt", prompt},
            {"chat", params.conversation_mode != COMMON_CONVERSATION_MODE_DISABLED},
            {"input_source", {{"type", "prompt"}}},
            {"reference_response", nullptr},
            {"seed", (int64_t) common_sampler_get_seed(smpl)},
            {"max_depth", params_spec.eagle_max_depth},
            {"max_proposals", params_spec.eagle_max_proposals},
            {"prob_threshold", trace_prob_threshold},
            {"temp", params.sampling.temp},
            {"top_k", params.sampling.top_k},
            {"prompt_tokens", json::array()},
            {"generated_count", n_predict},
            {"tokens", std::move(trace_tokens)},
            {"cycles", std::move(trace_cycles)},
        };

        for (llama_token tok : inp) {
            payload["prompt_tokens"].push_back((int) tok);
        }

        std::filesystem::path out_path(params_spec.eagle_trace_yaml);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }
        std::ofstream out(out_path);
        if (!out) {
            LOG_ERR("%s: failed to open eagle trace path '%s'\n", __func__, params_spec.eagle_trace_yaml.c_str());
            return 1;
        }
        out << payload.dump(2) << "\n";
        LOG_INF("wrote eagle trace: %s\n", params_spec.eagle_trace_yaml.c_str());
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    llama_batch_free(batch_tgt);

    common_sampler_free(smpl);
    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
