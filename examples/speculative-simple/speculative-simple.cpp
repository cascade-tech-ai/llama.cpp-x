// AI-GENERATED: This file was modified with AI assistance for an experimental fork.
// DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
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

static std::string tokens_to_debug_string(llama_context * ctx, const llama_tokens & toks) {
    std::string out;
    for (llama_token tok : toks) {
        out += common_token_to_piece(ctx, tok);
    }
    return out;
}

static int32_t find_tree_token(const common_speculative_tree & tree, const std::vector<int32_t> & nodes, llama_token tok) {
    for (int32_t node : nodes) {
        if (tree.tokens[(size_t) node] == tok) {
            return node;
        }
    }
    return -1;
}

static std::vector<int32_t> trace_accepted_tree_nodes(const common_speculative_tree & tree, const llama_tokens & ids) {
    std::vector<int32_t> nodes;
    if (ids.empty() || tree.tokens.empty()) {
        return nodes;
    }

    const size_t n_nodes = tree.tokens.size();
    std::vector<int32_t> roots;
    std::vector<std::vector<int32_t>> children(n_nodes);

    for (size_t i = 0; i < n_nodes; ++i) {
        const int32_t parent = tree.parents[i];
        if (parent < 0) {
            roots.push_back((int32_t) i);
        } else if ((size_t) parent < n_nodes) {
            children[(size_t) parent].push_back((int32_t) i);
        }
    }

    int32_t node = find_tree_token(tree, roots, ids[0]);
    while (node >= 0) {
        nodes.push_back(node);
        if (nodes.size() >= ids.size() - 1) {
            break;
        }
        node = find_tree_token(tree, children[(size_t) node], ids[nodes.size()]);
    }

    return nodes;
}
}

int main(int argc, char ** argv) {
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
    const bool trace_spec   = std::getenv("CASCADE_SPEC_TRACE") != nullptr;
    const bool rebuild_tree_accept = std::getenv("CASCADE_TREE_REBUILD_ACCEPTED") != nullptr;

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

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    // eval the prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, prompt_tgt, 0);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    batch_tgt.kv_idx = (uint32_t *) malloc(sizeof(uint32_t) * llama_n_batch(ctx_tgt));
    for (uint32_t i = 0; i < llama_n_batch(ctx_tgt); ++i) {
        batch_tgt.kv_idx[i] = UINT32_MAX;
    }

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
        const bool has_tree = spec ? common_speculative_get_tree(spec, tree) : false;
        bool use_tree = params.speculative.type == COMMON_SPECULATIVE_TYPE_EAGLE3 && has_tree && !tree.tokens.empty();
        std::vector<uint32_t> tree_kv_slots;

        //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

        // always have a token to evaluate from before - id_last
        scoped_prof zone_target(profile_spec, "spec/target_pass/total");
        common_batch_clear(batch_tgt);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        double pass_target_fwd_ms = 0.0;
        {
            // do not waste time on small drafts
            if (draft.size() < (size_t) params_spec.n_min) {
                draft.clear();
                tree.clear();
                use_tree = false;
            }

            if (use_tree) {
                common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);
                batch_tgt.kv_idx[batch_tgt.n_tokens - 1] = (uint32_t) n_past_before;
                tree.row_indices.resize(tree.tokens.size());
                std::vector<uint32_t> node_order(tree.tokens.size());
                for (uint32_t i = 0; i < node_order.size(); ++i) {
                    node_order[i] = i;
                }
                std::stable_sort(node_order.begin(), node_order.end(), [&](uint32_t a, uint32_t b) {
                    const int32_t da = tree.depths[a];
                    const int32_t db = tree.depths[b];
                    if (da != db) {
                        return da < db;
                    }
                    return a < b;
                });
                for (uint32_t node : node_order) {
                    const llama_pos pos = n_past + tree.depths[node];
                    tree.row_indices[node] = batch_tgt.n_tokens;
                    common_batch_add(batch_tgt, tree.tokens[node], pos, { 0 }, true);
                    batch_tgt.kv_idx[batch_tgt.n_tokens - 1] = (uint32_t) (n_past_before + 1 + node);
                }

                const llama_kq_mask_tree mask = {
                    /* .n_nodes     = */ tree.tokens.size(),
                    /* .parent      = */ tree.parents.data(),
                    /* .batch_start = */ tree.batch_start,
                    /* .row_indices = */ tree.row_indices.data(),
                };
                llama_set_kq_mask_tree(ctx_tgt, &mask);
            } else {
                common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);
                for (size_t i = 0; i < draft.size(); ++i) {
                    common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
                }
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            scoped_prof zone_target_fwd(profile_spec, "spec/target_pass/forward");
            llama_decode(ctx_tgt, batch_tgt);
            pass_target_fwd_ms = zone_target_fwd.elapsed_ms();
            t_target_fwd_us += (int64_t) (zone_target_fwd.elapsed_ms() * 1000.0);

            if (use_tree) {
                size_t n_slots = 0;
                const uint32_t * slots = llama_get_kv_slot_indices(ctx_tgt, &n_slots);
                if (!slots || n_slots != (size_t) batch_tgt.n_tokens) {
                    LOG_ERR("%s: failed to retrieve KV slots for EAGLE tree batch (got %zu, expected %d)\n",
                            __func__, n_slots, batch_tgt.n_tokens);
                    return 1;
                }
                tree_kv_slots.assign(slots, slots + n_slots);
                llama_clear_kq_mask_tree(ctx_tgt);
            }
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
        const auto ids = use_tree
            ? common_sampler_sample_and_accept_tree(smpl, ctx_tgt, 0, tree)
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
        if (trace_spec) {
            std::fprintf(stderr,
                    "spec trace: pass=%d n_past_before=%d use_tree=%d id_last=%d('%s') draft=%zu accepted=%zu out=\"%s\"\n",
                    pass_idx,
                    n_past_before,
                    use_tree ? 1 : 0,
                    (int) id_last,
                    common_token_to_piece(ctx_tgt, id_last).c_str(),
                    draft.size(),
                    ids_limited.size(),
                    tokens_to_debug_string(ctx_tgt, ids_limited).c_str());
        }

        if (use_tree) {
            common_speculative_accept_tokens(spec, ids_limited, 0, &tree);
        }
        common_speculative_accept(spec, ids_limited.size() - 1);

        if (use_tree) {
            const std::vector<int32_t> accepted_nodes = trace_accepted_tree_nodes(tree, ids_limited);
            auto * mem = llama_get_memory(ctx_tgt);

            std::vector<char> keep(tree.tokens.size(), 0);
            for (int32_t node : accepted_nodes) {
                if (node >= 0 && (size_t) node < keep.size()) {
                    keep[(size_t) node] = 1;
                }
            }

            std::vector<uint32_t> reject_slots;
            reject_slots.reserve(tree.tokens.size());
            for (size_t i = 0; i < tree.tokens.size(); ++i) {
                if (!keep[i]) {
                    const uint32_t row = tree.row_indices[i];
                    GGML_ASSERT(row < tree_kv_slots.size());
                    reject_slots.push_back(tree_kv_slots[row]);
                }
            }

            if (!reject_slots.empty() && !llama_memory_kv_idx_rm(mem, reject_slots.data(), reject_slots.size())) {
                LOG_ERR("%s: failed to remove rejected EAGLE KV slots\n", __func__);
                return 1;
            }
        }

        if (use_tree && rebuild_tree_accept) {
            auto * mem = llama_get_memory(ctx_tgt);
            llama_memory_seq_rm(mem, 0, n_past_before, -1);

            llama_tokens rebuild;
            rebuild.reserve(ids_limited.size());
            rebuild.push_back(id_last);
            if (ids_limited.size() > 1) {
                rebuild.insert(rebuild.end(), ids_limited.begin(), ids_limited.end() - 1);
            }

            if (!rebuild.empty()) {
                llama_batch batch_rebuild = llama_batch_get_one(rebuild.data(), rebuild.size());
                if (llama_decode(ctx_tgt, batch_rebuild) != 0) {
                    LOG_ERR("%s: failed to rebuild accepted compact-tree tokens\n", __func__);
                    return 1;
                }
            }
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

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
            if (params.speculative.type == COMMON_SPECULATIVE_TYPE_EAGLE3) {
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
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    if (profile_spec) {
        LOG_INF("spec profile total: target_passes=%d target_total=%.3fms target_forward=%.3fms target_sampling=%.3fms eagle_total=%.3fms\n",
                n_target_passes,
                t_target_total_us / 1000.0,
                t_target_fwd_us / 1000.0,
                t_target_sampling_us / 1000.0,
                t_eagle_total_us / 1000.0);
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
