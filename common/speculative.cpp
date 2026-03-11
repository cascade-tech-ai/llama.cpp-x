#include "speculative.h"
// AI-GENERATED: This file was modified with AI assistance for an experimental fork.
// DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.

#include "common.h"
#include "ggml.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#include "llama-eagle3.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static void common_speculative_tree_build_metadata(common_speculative_tree & tree) {
    const size_t n_nodes = tree.tokens.size();

    tree.first_child.assign(n_nodes, -1);
    tree.next_sibling.assign(n_nodes, -1);
    tree.leaf_masks.assign(n_nodes, 0);
    tree.leaf_count = 0;

    if (n_nodes == 0) {
        return;
    }

    std::vector<int32_t> last_child(n_nodes, -1);
    for (size_t i = 0; i < n_nodes; ++i) {
        const int32_t parent = tree.parents[i];
        if (parent < 0) {
            continue;
        }
        GGML_ASSERT((size_t) parent < n_nodes);
        if (tree.first_child[(size_t) parent] < 0) {
            tree.first_child[(size_t) parent] = (int32_t) i;
        } else {
            GGML_ASSERT(last_child[(size_t) parent] >= 0);
            tree.next_sibling[(size_t) last_child[(size_t) parent]] = (int32_t) i;
        }
        last_child[(size_t) parent] = (int32_t) i;
    }

    for (size_t rev = n_nodes; rev-- > 0;) {
        if (tree.first_child[rev] < 0) {
            GGML_ASSERT(tree.leaf_count < 32);
            tree.leaf_masks[rev] = 1u << tree.leaf_count++;
            continue;
        }

        uint32_t mask = 0;
        for (int32_t child = tree.first_child[rev]; child >= 0; child = tree.next_sibling[(size_t) child]) {
            mask |= tree.leaf_masks[(size_t) child];
        }
        tree.leaf_masks[rev] = mask;
    }
}

static common_speculative_trace_node * common_speculative_trace_find_child(
        std::vector<common_speculative_trace_node> & nodes,
        llama_token token) {
    for (auto & node : nodes) {
        if (node.token == token) {
            return &node;
        }
    }
    return nullptr;
}

static common_speculative_trace_node & common_speculative_trace_insert_path(
        std::vector<common_speculative_trace_node> & roots,
        const llama_tokens & path,
        float prob,
        float cum_prob) {
    GGML_ASSERT(!path.empty());

    std::vector<common_speculative_trace_node> * level = &roots;
    common_speculative_trace_node * node = nullptr;
    for (size_t i = 0; i < path.size(); ++i) {
        node = common_speculative_trace_find_child(*level, path[i]);
        if (!node) {
            level->push_back({ path[i], 0.0f, 0.0f, false, {} });
            node = &level->back();
        }
        if (i + 1 == path.size()) {
            node->prob = prob;
            node->cum_prob = cum_prob;
        }
        level = &node->children;
    }

    return *node;
}

static void common_speculative_trace_mark_accepted(
        std::vector<common_speculative_trace_node> & roots,
        const llama_tokens & accepted_tokens) {
    if (accepted_tokens.empty()) {
        return;
    }

    std::vector<common_speculative_trace_node> * level = &roots;
    for (llama_token token : accepted_tokens) {
        common_speculative_trace_node * node = common_speculative_trace_find_child(*level, token);
        if (!node) {
            return;
        }
        node->accepted = true;
        level = &node->children;
    }
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    // TODO: rename to n_call_draft, n_gen_drafts, n_acc_drafts, n_gen_tokens, n_acc_tokens
    // TODO: add n_call_begin, n_call_accept
    size_t drafts_call_count       = 0; // number of times this implementation was called.
    size_t drafts_generated_count  = 0; // number of times a draft or part was generated by this implementation.
    size_t drafts_accepted_count   = 0; // number of times a draft or part was accepted by the target model.
    size_t drafts_generated_tokens = 0; // number of tokens generated by this implementation.
    size_t drafts_accepted_tokens  = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    // TODO: rename to t_draft_us
    // TODO: add t_begin_us, t_accept_us
    int64_t gen_duration_us = 0; // total time spent in this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt, llama_seq_id seq_id) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_seq_id seq_id,
            llama_tokens & result) = 0;

    virtual void accept(uint16_t n_accepted) = 0;

    virtual void accept_tokens(const llama_tokens & ids, llama_seq_id seq_id) {
        GGML_UNUSED(ids);
        GGML_UNUSED(seq_id);
    }

    virtual bool get_tree(common_speculative_tree & out) const {
        out.clear();
        return false;
    }

    virtual bool get_trace(common_speculative_trace & out) const {
        out.clear();
        return false;
    }

    virtual void set_tree(const common_speculative_tree & tree) {
        GGML_UNUSED(tree);
    }
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt, llama_seq_id seq_id) override {
        GGML_UNUSED(prompt);
        GGML_UNUSED(seq_id);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_seq_id seq_id,
            llama_tokens & result) override {
        auto * spec = this;
        GGML_UNUSED(seq_id);

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < params.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < params.p_min) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

namespace {

static std::string npy_shape_string(const std::vector<size_t> & shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (shape.size() == 1) {
            oss << ",";
        } else if (i + 1 < shape.size()) {
            oss << ", ";
        }
    }
    oss << ")";
    return oss.str();
}

static bool write_npy(
        const std::filesystem::path & path,
        const void * data,
        size_t n_bytes,
        const char * descr,
        const std::vector<size_t> & shape) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        LOG_ERR("%s: failed to open '%s'\n", __func__, path.string().c_str());
        return false;
    }

    // NumPy .npy v1.0
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const unsigned char ver[2] = {1, 0};
    out.write(reinterpret_cast<const char *>(ver), 2);

    std::string header = "{'descr': '";
    header += descr;
    header += "', 'fortran_order': False, 'shape': ";
    header += npy_shape_string(shape);
    header += ", }";

    // Pad header to 16-byte alignment including the trailing newline.
    const size_t base_len = header.size() + 1;
    const size_t pad = (16 - ((10 + base_len) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');

    if (header.size() > std::numeric_limits<uint16_t>::max()) {
        LOG_ERR("%s: header too large for v1.0\n", __func__);
        return false;
    }

    const uint16_t hlen = (uint16_t) header.size();
    out.write(reinterpret_cast<const char *>(&hlen), sizeof(hlen));
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char *>(data), (std::streamsize) n_bytes);
    return (bool) out;
}

static bool write_npy_f32(
        const std::filesystem::path & path,
        const float * data,
        const std::vector<size_t> & shape) {
    size_t count = 1;
    for (size_t d : shape) {
        count *= d;
    }
    return write_npy(path, data, count * sizeof(float), "<f4", shape);
}

static bool write_npy_i32(
        const std::filesystem::path & path,
        const int32_t * data,
        const std::vector<size_t> & shape) {
    size_t count = 1;
    for (size_t d : shape) {
        count *= d;
    }
    return write_npy(path, data, count * sizeof(int32_t), "<i4", shape);
}

} // namespace

struct common_speculative_state_eagle3 : public common_speculative_state {
    struct rollout_beam_slot {
        float logprob = 0.0f;
        std::vector<llama_token> tokens;
        llama_eagle3_state state;
    };

    llama_context * ctx_tgt = nullptr;
    std::unique_ptr<llama_eagle3_model, void (*)(llama_eagle3_model *)> model;
    llama_eagle3_runtime rt{};

    // Canonical accepted-prefix Eagle state. This holds prompt-length Eagle KV and is
    // incrementally extended as target tokens are accepted.
    llama_eagle3_state prefix_state{};
    size_t prefix_prompt_len = 0;
    llama_seq_id active_seq_id = -1;

    std::vector<int32_t> layer_ids;
    int32_t hidden_in_dim = 0;
    int32_t target_hidden_size = 0;
    int32_t hidden_size = 0;
    const llama_vocab * vocab_tgt = nullptr;
    const bool verbose;
    const bool profile;
    const bool profile_gpu;
    const std::string dump_dir;
    std::vector<float> hidden_concat_buf;
    bool enabled = false;
    common_speculative_tree last_tree;
    common_speculative_trace last_trace;
    std::vector<llama_eagle3_state> last_tree_states;
    llama_eagle3_state last_root_state;
    bool has_last_root_state = false;
    std::vector<float> prefix_tail_hidden_concat;
    size_t prefix_tail_capture_idx = 0;

    // Dump buffers (CASCADE_EAGLE_DUMP_DIR)
    bool dump_pending = false;
    llama_tokens dump_prompt_tgt;
    llama_token dump_id_last = -1;
    std::vector<int32_t> dump_head_input_ids; // [n_steps]
    std::vector<float> dump_teacher_hidden_by_step; // [n_steps, hidden_in_dim]
    std::vector<float> dump_head_embd_by_step; // [n_steps, hidden_size]
    std::vector<float> dump_head_embd_norm_by_step; // [n_steps, hidden_size]
    std::vector<float> dump_head_hidden_proj_by_step; // [n_steps, hidden_size]
    std::vector<float> dump_head_hidden_norm_by_step; // [n_steps, hidden_size]
    std::vector<float> dump_head_cat_by_step; // [n_steps, hidden_size*2]
    std::vector<float> dump_head_q_by_step; // [n_steps, head_dim * n_heads]
    std::vector<float> dump_head_k_by_step; // [n_steps, head_dim * n_kv_heads]
    std::vector<float> dump_head_v_by_step; // [n_steps, head_dim * n_kv_heads]
    std::vector<float> dump_head_hidden_after_step; // [n_steps, hidden_size]
    std::vector<float> dump_root_logits_draft; // [draft_vocab_size]

    std::vector<rollout_beam_slot> rollout_beams_a;
    std::vector<rollout_beam_slot> rollout_beams_b;
    std::vector<uint8_t> rollout_active_a;
    std::vector<uint8_t> rollout_active_b;
    std::vector<float> rollout_beam_logprob_a;
    std::vector<float> rollout_beam_logprob_b;
    std::shared_ptr<llama_eagle3_rollout_batch> rollout_batch_a;
    std::shared_ptr<llama_eagle3_rollout_batch> rollout_batch_b;

    static std::string escape_token_piece(const std::string & input) {
        std::string out;
        out.reserve(input.size());
        for (char ch : input) {
            switch (ch) {
                case '\\': out += "\\\\"; break;
                case '\"': out += "\\\""; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out.push_back(ch); break;
            }
        }
        return out;
    }

    void reset_prefix_cache() {
        prefix_prompt_len = 0;
        prefix_state = {};
        prefix_tail_hidden_concat.clear();
        prefix_tail_capture_idx = 0;
    }

    void set_prefix_frontier(const llama_eagle3_state & state, size_t tail_capture_idx) {
        prefix_state = state;
        prefix_tail_capture_idx = tail_capture_idx;
    }

    void ensure_rollout_storage(int32_t beam_width) {
        const size_t n = (size_t) std::max(1, beam_width);
        rollout_beams_a.resize(n);
        rollout_beams_b.resize(n);
        rollout_active_a.resize(n, 0);
        rollout_active_b.resize(n, 0);
        rollout_beam_logprob_a.resize(n, 0.0f);
        rollout_beam_logprob_b.resize(n, 0.0f);
    }

    void clear_rollout_slots(
            std::vector<rollout_beam_slot> & beams,
            std::vector<uint8_t> & active,
            std::vector<float> & beam_logprob,
            float inactive_beam_logprob) {
        for (size_t i = 0; i < beams.size(); ++i) {
            beams[i].logprob = 0.0f;
            beams[i].tokens.clear();
            active[i] = 0;
            beam_logprob[i] = inactive_beam_logprob;
        }
    }

    bool ensure_rollout_batches(int32_t beam_width, int32_t kv_capacity) {
        if (!rt.backend_compute || !rt.buft_compute) {
            return false;
        }
        if (!rollout_batch_a) {
            rollout_batch_a = std::make_shared<llama_eagle3_rollout_batch>();
        }
        if (!rollout_batch_b) {
            rollout_batch_b = std::make_shared<llama_eagle3_rollout_batch>();
        }
        return llama_eagle3_rollout_batch_ensure(*model, rt, beam_width, kv_capacity, *rollout_batch_a) &&
               llama_eagle3_rollout_batch_ensure(*model, rt, beam_width, kv_capacity, *rollout_batch_b);
    }

    common_speculative_state_eagle3(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            const common_params_speculative & params)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , model(nullptr, llama_eagle3_free)
        , verbose(std::getenv("CASCADE_EAGLE_VERBOSE") != nullptr)
        , profile(std::getenv("CASCADE_EAGLE_PROFILE") != nullptr)
        , profile_gpu(std::getenv("CASCADE_EAGLE_PROFILE_GPU") != nullptr)
        , dump_dir(std::getenv("CASCADE_EAGLE_DUMP_DIR") ? std::getenv("CASCADE_EAGLE_DUMP_DIR") : "") {
        if (!ctx_tgt) {
            LOG_ERR("%s: null target context\n", __func__);
            return;
        }
        if (params.mparams_dft.path.empty()) {
            LOG_ERR("%s: eagle3 requires --model-draft pointing to an eagle3 GGUF\n", __func__);
            return;
        }

        std::string err;
        llama_eagle3_model * raw = llama_eagle3_load(params.mparams_dft.path, err);
        if (!raw) {
            LOG_ERR("%s: failed to load eagle3 head: %s\n", __func__, err.c_str());
            return;
        }
        model.reset(raw);

        layer_ids = model->hidden_layer_ids;
        hidden_size = model->hparams.hidden_size;
        target_hidden_size = model->hparams.target_hidden_size;
        hidden_in_dim = model->hparams.hidden_concat * model->hparams.target_hidden_size;

        const llama_model * base_model = llama_get_model(ctx_tgt);
        vocab_tgt = llama_model_get_vocab(base_model);
        const int32_t base_hidden = llama_model_n_embd(base_model);
        if (base_hidden != target_hidden_size) {
            LOG_ERR("%s: eagle3 target_hidden_size=%d does not match base model n_embd=%d\n",
                    __func__, target_hidden_size, base_hidden);
            return;
        }

        if (!llama_eagle3_set_layers(ctx_tgt, layer_ids.data(), layer_ids.size())) {
            LOG_ERR("%s: failed to enable eagle3 hidden capture\n", __func__);
            return;
        }

        const int32_t n_threads = llama_n_threads(ctx_tgt);
        rt = llama_eagle3_make_runtime(ctx_tgt, model.get(), n_threads);

        enabled = true;
    }

    void begin(const llama_tokens & prompt, llama_seq_id seq_id) override {
        if (seq_id != active_seq_id) {
            active_seq_id = seq_id;
        }

        reset_prefix_cache();

        if (!enabled) {
            return;
        }

        if (!dump_dir.empty()) {
            dump_pending = true;
            dump_prompt_tgt = prompt;
            dump_id_last = -1;
            dump_head_input_ids.clear();
            dump_teacher_hidden_by_step.clear();
            dump_head_embd_by_step.clear();
            dump_head_embd_norm_by_step.clear();
            dump_head_hidden_proj_by_step.clear();
            dump_head_hidden_norm_by_step.clear();
            dump_head_cat_by_step.clear();
            dump_head_q_by_step.clear();
            dump_head_k_by_step.clear();
            dump_head_v_by_step.clear();
            dump_head_hidden_after_step.clear();
            dump_root_logits_draft.clear();
        }

        prefill_to(prompt, seq_id);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_seq_id seq_id,
            llama_tokens & draft_tokens) override {
        draft_tokens.clear();
        last_tree.clear();
        last_trace.clear();
        last_tree_states.clear();
        has_last_root_state = false;

        if (!enabled) {
            return;
        }

        if (seq_id != active_seq_id) {
            reset_prefix_cache();
            active_seq_id = seq_id;
        }

        if (params.eagle_max_depth < 1 || params.eagle_max_proposals < 1) {
            return;
        }

        if (prompt_tgt.empty()) {
            return;
        }

        const int64_t t_draft_start = ggml_time_us();
        int64_t t_prefill_us = 0;
        int64_t t_scoring_us = 0;
        int64_t t_logits_us = 0;
        int64_t t_step_us = 0;
        int64_t t_root_step_us = 0;
        int32_t n_logits_calls = 0;
        int32_t n_step_calls = 0;
        std::vector<double> depth_select_ms;
        std::vector<double> depth_score_ms;
        std::vector<double> depth_step_ms;
        std::vector<int32_t> depth_active_beams;
        std::vector<int32_t> depth_expansions;
#if defined(GGML_USE_CUDA)
        double t_logits_gpu_ms = 0.0;
        double t_step_gpu_ms   = 0.0;
        double t_root_step_gpu_ms = 0.0;
        std::vector<double> depth_select_gpu_ms;
        std::vector<double> depth_step_gpu_ms;
#endif

        const int64_t t_prefill_start = ggml_time_us();
        if (!prefill_to(prompt_tgt, seq_id)) {
            return;
        }
        t_prefill_us += ggml_time_us() - t_prefill_start;

        // The speculative-simple example keeps the last token separate (id_last) and only decodes
        // prompt_tgt up to the token before it. For EAGLE alignment, we must advance the head one
        // more step using the teacher hidden at the last prompt token and input_id = id_last.
        std::vector<const ggml_tensor *> layer_tensors;
        size_t n_tokens = 0;
        if (!fetch_hidden_tensors(layer_tensors, n_tokens)) {
            return;
        }

        llama_eagle3_state root_state = prefix_state;
        const bool dump_root = dump_pending && !dump_dir.empty();
        llama_eagle3_step_debug dbg_root;
        if (dump_root) {
            if (prefix_tail_capture_idx >= n_tokens ||
                !build_hidden_concat_host(layer_tensors, prefix_tail_capture_idx, hidden_concat_buf)) {
                return;
            }
            dbg_root.embd        = &dump_head_embd_by_step;
            dbg_root.embd_norm   = &dump_head_embd_norm_by_step;
            dbg_root.hidden_proj = &dump_head_hidden_proj_by_step;
            dbg_root.hidden_norm = &dump_head_hidden_norm_by_step;
            dbg_root.cat         = &dump_head_cat_by_step;
            dbg_root.q           = &dump_head_q_by_step;
            dbg_root.k           = &dump_head_k_by_step;
            dbg_root.v           = &dump_head_v_by_step;

            dump_id_last = id_last;
            dump_head_input_ids.push_back((int32_t) id_last);
            dump_teacher_hidden_by_step.insert(
                dump_teacher_hidden_by_step.end(),
                hidden_concat_buf.begin(),
                hidden_concat_buf.end());
        }

        {
            const int64_t t_step_start = ggml_time_us();
            ++n_step_calls;
#if defined(GGML_USE_CUDA)
            ggml_backend_cuda_profiler_zone zone = {};
            if (profile_gpu) {
                ggml_backend_cuda_profiler_zone_begin(rt.backend_compute.get(), &zone, "eagle3/root_step");
            }
#endif
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
#if defined(GGML_USE_CUDA)
            if (profile_gpu) {
                t_root_step_gpu_ms += ggml_backend_cuda_profiler_zone_end(rt.backend_compute.get(), &zone, "eagle3/root_step");
            }
#endif
            t_step_us += ggml_time_us() - t_step_start;
            t_root_step_us += ggml_time_us() - t_step_start;
        }
        if (!llama_eagle3_state_has_hidden(root_state)) {
            return;
        }
        last_root_state = root_state;
        has_last_root_state = true;

        if (dump_root) {
            std::vector<float> root_hidden;
            if (!llama_eagle3_state_get_hidden(*model, rt, root_state, root_hidden)) {
                return;
            }
            dump_head_hidden_after_step.insert(
                dump_head_hidden_after_step.end(),
                root_hidden.begin(),
                root_hidden.end());

            std::vector<float> root_logits;
            if (!llama_eagle3_logits(*model, rt, root_hidden.data(), root_logits)) {
                return;
            }
            dump_root_logits_draft = std::move(root_logits);

            if (!write_dump(seq_id)) {
                return;
            }
            dump_pending = false;
        }

        const int max_depth = params.eagle_max_depth;
        const int max_proposals = params.eagle_max_proposals;
        const int beam_width = params.eagle_beam_width > 0
            ? std::min(params.eagle_beam_width, max_proposals)
            : max_proposals;
        const float prob_threshold = params.eagle_per_beam_topk_candidates > 0
            ? 1.0f / (float) params.eagle_per_beam_topk_candidates
            : 0.0f;
        const float inactive_beam_logprob = -1e30f;

        struct beam_expansion {
            float logprob = 0.0f;
            size_t beam_idx = 0;
            llama_token token = LLAMA_TOKEN_NULL;
        };

        ensure_rollout_storage(beam_width);
        const int32_t round_kv_capacity = root_state.past_len + std::max(0, max_depth);
        if (rt.backend_compute && rt.buft_compute) {
            if (!ensure_rollout_batches(beam_width, round_kv_capacity)) {
                return;
            }
            for (int32_t i = 0; i < beam_width; ++i) {
                if (!llama_eagle3_rollout_batch_bind_slot(*model, rollout_batch_a, i, rollout_beams_a[(size_t) i].state, root_state.past_len)) {
                    return;
                }
                if (!llama_eagle3_rollout_batch_bind_slot(*model, rollout_batch_b, i, rollout_beams_b[(size_t) i].state, root_state.past_len)) {
                    return;
                }
            }
        }
        clear_rollout_slots(rollout_beams_a, rollout_active_a, rollout_beam_logprob_a, inactive_beam_logprob);
        clear_rollout_slots(rollout_beams_b, rollout_active_b, rollout_beam_logprob_b, inactive_beam_logprob);
        if (rt.backend_compute && rt.buft_compute) {
            if (!llama_eagle3_rollout_batch_copy_state_to_slot(*model, rt, root_state, rollout_batch_a, 0, rollout_beams_a[0].state)) {
                return;
            }
        } else {
            rollout_beams_a[0].state = root_state;
        }
        rollout_active_a[0] = 1;
        rollout_beam_logprob_a[0] = 0.0f;

        struct proposal_path {
            float logprob = 0.0f;
            llama_tokens tokens;
        };

        std::vector<proposal_path> all_nodes;
        std::map<llama_tokens, llama_eagle3_state> prefix_states;
        for (int depth = 0; depth < max_depth; ++depth) {
            auto & beams_cur = (depth & 1) ? rollout_beams_b : rollout_beams_a;
            auto & beams_nxt = (depth & 1) ? rollout_beams_a : rollout_beams_b;
            auto & active_cur = (depth & 1) ? rollout_active_b : rollout_active_a;
            auto & active_nxt = (depth & 1) ? rollout_active_a : rollout_active_b;
            auto & beam_logprob_cur = (depth & 1) ? rollout_beam_logprob_b : rollout_beam_logprob_a;
            auto & beam_logprob_nxt = (depth & 1) ? rollout_beam_logprob_a : rollout_beam_logprob_b;

            std::vector<beam_expansion> expansions;
            expansions.reserve((size_t) beam_width * (size_t) beam_width);

            int32_t n_active = 0;
            for (uint8_t active : active_cur) {
                n_active += active ? 1 : 0;
            }
            depth_active_beams.push_back(n_active);

            const int k = std::min<int>(beam_width, model->hparams.draft_vocab_size);
            if (k <= 0 || n_active == 0) {
                break;
            }

            std::vector<const llama_eagle3_state *> beam_states((size_t) beam_width);
            for (int32_t beam_idx = 0; beam_idx < beam_width; ++beam_idx) {
                beam_states[(size_t) beam_idx] = &beams_cur[(size_t) beam_idx].state;
            }

            std::vector<int32_t> selected_linear;
            std::vector<int32_t> selected_draft_idx;
            std::vector<float> selected_logprob;
            double depth_select_gpu = 0.0;
            {
                const int64_t t_logits_start = ggml_time_us();
                ++n_logits_calls;
#if defined(GGML_USE_CUDA)
                ggml_backend_cuda_profiler_zone zone = {};
                if (profile_gpu) {
                    ggml_backend_cuda_profiler_zone_begin(rt.backend_compute.get(), &zone, "eagle3/select_state_batch");
                }
#endif
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
#if defined(GGML_USE_CUDA)
                if (profile_gpu) {
                    depth_select_gpu = ggml_backend_cuda_profiler_zone_end(rt.backend_compute.get(), &zone, "eagle3/select_state_batch");
                    t_logits_gpu_ms += depth_select_gpu;
                }
#endif
                if (!ok) {
                    break;
                }
                const double depth_select = (ggml_time_us() - t_logits_start) / 1000.0;
                t_logits_us += (int64_t) (depth_select * 1000.0);
                depth_select_ms.push_back(depth_select);
#if defined(GGML_USE_CUDA)
                depth_select_gpu_ms.push_back(depth_select_gpu);
#else
                GGML_UNUSED(depth_select_gpu);
#endif
            }

            const int64_t t_scoring_start = ggml_time_us();
            for (size_t rank = 0; rank < selected_linear.size(); ++rank) {
                const int32_t linear = selected_linear[rank];
                const int32_t draft_idx = selected_draft_idx[rank];
                if (linear < 0 || draft_idx < 0 || draft_idx >= model->hparams.draft_vocab_size) {
                    continue;
                }

                const int32_t parent_slot = linear / k;
                if (parent_slot < 0 || parent_slot >= beam_width) {
                    continue;
                }

                const size_t beam_idx = (size_t) parent_slot;
                if (active_cur[beam_idx] == 0) {
                    continue;
                }
                const float total_logprob = selected_logprob[rank];
                const float prob = std::exp(total_logprob - beam_logprob_cur[beam_idx]);
                const int32_t base_id = draft_idx + model->d2t[draft_idx];
                const bool filtered = prob_threshold > 0.0f && prob < prob_threshold;

                if (verbose) {
                    const bool base_ok = base_id >= 0 && base_id < model->hparams.vocab_size;
                    const std::string piece = base_ok && vocab_tgt
                        ? common_token_to_piece(vocab_tgt, base_id)
                        : std::string("<invalid>");
                    const std::string escaped = escape_token_piece(piece);
                    LOG_INF("eagle3 depth=%d beam=%zu rank=%zu token=%d piece=\"%s\" p=%.6f%s\n",
                            depth,
                            beam_idx,
                            rank,
                            base_id,
                            escaped.c_str(),
                            prob,
                            filtered ? " (filtered)" : "");
                }

                if (filtered) {
                    continue;
                }
                if (base_id < 0 || base_id >= model->hparams.vocab_size) {
                    continue;
                }

                expansions.push_back({
                    /* logprob  = */ total_logprob,
                    /* beam_idx = */ beam_idx,
                    /* token    = */ base_id,
                });
                if ((int) expansions.size() >= beam_width) {
                    break;
                }
            }
            const double depth_score = (ggml_time_us() - t_scoring_start) / 1000.0;
            t_scoring_us += (int64_t) (depth_score * 1000.0);
            depth_score_ms.push_back(depth_score);
            depth_expansions.push_back((int32_t) expansions.size());

            if (expansions.empty()) {
                break;
            }

            std::vector<const llama_eagle3_state *> parent_states;
            std::vector<llama_eagle3_state *> out_states;
            std::vector<llama_token> candidate_input_ids;
            parent_states.reserve(expansions.size());
            out_states.reserve(expansions.size());
            candidate_input_ids.reserve(expansions.size());

            const int32_t n_next = (int32_t) expansions.size();
            for (int32_t i = 0; i < n_next; ++i) {
                const auto & expansion = expansions[(size_t) i];
                const auto & parent = beams_cur[expansion.beam_idx];

                rollout_beam_slot & child = beams_nxt[(size_t) i];
                child.logprob = expansion.logprob;
                child.tokens = parent.tokens;
                child.tokens.push_back(expansion.token);
                active_nxt[(size_t) i] = 1;
                beam_logprob_nxt[(size_t) i] = expansion.logprob;

                parent_states.push_back(&parent.state);
                out_states.push_back(&child.state);
                candidate_input_ids.push_back(expansion.token);
            }

            for (int32_t i = n_next; i < beam_width; ++i) {
                beams_nxt[(size_t) i].logprob = 0.0f;
                beams_nxt[(size_t) i].tokens.clear();
                active_nxt[(size_t) i] = 0;
                beam_logprob_nxt[(size_t) i] = inactive_beam_logprob;
            }

            if (!candidate_input_ids.empty()) {
                const int64_t t_step_start = ggml_time_us();
                ++n_step_calls;
                double depth_step_gpu = 0.0;
#if defined(GGML_USE_CUDA)
                ggml_backend_cuda_profiler_zone zone = {};
                if (profile_gpu) {
                    ggml_backend_cuda_profiler_zone_begin(rt.backend_compute.get(), &zone, "eagle3/step_batch");
                }
#endif
                const int32_t reserve_kv = std::max(0, max_depth - depth - 1);
                const bool ok = llama_eagle3_step_batch_from_parents(
                        *model,
                        rt,
                        parent_states,
                        hidden_size,
                        candidate_input_ids,
                        out_states,
                        reserve_kv);
#if defined(GGML_USE_CUDA)
                if (profile_gpu) {
                    depth_step_gpu = ggml_backend_cuda_profiler_zone_end(rt.backend_compute.get(), &zone, "eagle3/step_batch");
                    t_step_gpu_ms += depth_step_gpu;
                }
#endif
                if (!ok) {
                    break;
                }
                const double depth_step = (ggml_time_us() - t_step_start) / 1000.0;
                t_step_us += (int64_t) (depth_step * 1000.0);
                depth_step_ms.push_back(depth_step);
#if defined(GGML_USE_CUDA)
                depth_step_gpu_ms.push_back(depth_step_gpu);
#else
                GGML_UNUSED(depth_step_gpu);
#endif
            }

            if (candidate_input_ids.empty()) {
                break;
            }

            for (int32_t i = 0; i < n_next; ++i) {
                const auto & cand = beams_nxt[(size_t) i];
                common_speculative_trace_insert_path(
                        last_trace.proposal_tree,
                        cand.tokens,
                        std::exp(cand.logprob - beam_logprob_cur[expansions[(size_t) i].beam_idx]),
                        std::exp(cand.logprob));
                prefix_states[cand.tokens] = cand.state;
                all_nodes.push_back({cand.logprob, cand.tokens});
            }
        }

        if (all_nodes.empty()) {
            return;
        }

        std::sort(all_nodes.begin(), all_nodes.end(),
            [](const auto & a, const auto & b) { return a.logprob > b.logprob; });

        if ((int) all_nodes.size() > max_proposals) {
            all_nodes.resize(max_proposals);
        }

        last_trace.proposal_count = (int32_t) all_nodes.size();

        if (profile) {
            const int64_t t_total_us = ggml_time_us() - t_draft_start;
#if defined(GGML_USE_CUDA)
            if (profile_gpu) {
                LOG_INF("eagle3 profile: prompt=%zu depth=%d beam=%d props=%d out=%zu total=%.3fms prefill=%.3fms logits=%.3fms score=%.3fms step=%.3fms calls(logits=%d,step=%d) gpu(logits=%.3fms,step=%.3fms)\n",
                        prompt_tgt.size(),
                        max_depth,
                        beam_width,
                        max_proposals,
                        all_nodes.size(),
                        t_total_us / 1000.0,
                        t_prefill_us / 1000.0,
                        t_logits_us / 1000.0,
                        t_scoring_us / 1000.0,
                        t_step_us / 1000.0,
                        n_logits_calls,
                        n_step_calls,
                        t_logits_gpu_ms,
                        t_step_gpu_ms);
            } else
#endif
            {
                LOG_INF("eagle3 profile: prompt=%zu depth=%d beam=%d props=%d out=%zu total=%.3fms prefill=%.3fms logits=%.3fms score=%.3fms step=%.3fms calls(logits=%d,step=%d)\n",
                        prompt_tgt.size(),
                        max_depth,
                        beam_width,
                        max_proposals,
                        all_nodes.size(),
                        t_total_us / 1000.0,
                        t_prefill_us / 1000.0,
                        t_logits_us / 1000.0,
                        t_scoring_us / 1000.0,
                        t_step_us / 1000.0,
                        n_logits_calls,
                        n_step_calls);
            }

            LOG_INF("eagle3 profile root: prompt=%zu root_step=%.3fms\n",
                    prompt_tgt.size(),
                    t_root_step_us / 1000.0);

            for (size_t i = 0; i < depth_active_beams.size(); ++i) {
                const double select_ms = i < depth_select_ms.size() ? depth_select_ms[i] : 0.0;
                const double score_ms  = i < depth_score_ms.size()  ? depth_score_ms[i]  : 0.0;
                const double step_ms   = i < depth_step_ms.size()   ? depth_step_ms[i]   : 0.0;
                const int32_t expands  = i < depth_expansions.size() ? depth_expansions[i] : 0;
#if defined(GGML_USE_CUDA)
                if (profile_gpu) {
                    const double select_gpu_ms = i < depth_select_gpu_ms.size() ? depth_select_gpu_ms[i] : 0.0;
                    const double step_gpu_ms   = i < depth_step_gpu_ms.size()   ? depth_step_gpu_ms[i]   : 0.0;
                    LOG_INF("eagle3 profile depth=%zu active_beams=%d expansions=%d select=%.3fms score=%.3fms step=%.3fms gpu(select=%.3fms,step=%.3fms)\n",
                            i,
                            depth_active_beams[i],
                            expands,
                            select_ms,
                            score_ms,
                            step_ms,
                            select_gpu_ms,
                            step_gpu_ms);
                } else
#endif
                {
                    LOG_INF("eagle3 profile depth=%zu active_beams=%d expansions=%d select=%.3fms score=%.3fms step=%.3fms\n",
                            i,
                            depth_active_beams[i],
                            expands,
                            select_ms,
                            score_ms,
                            step_ms);
                }
            }
        }

        std::vector<int32_t> roots;
        std::vector<std::vector<int32_t>> children;

        last_tree_states.clear();

        for (const auto & entry : all_nodes) {
            const auto & seq = entry.tokens;
            int32_t parent = -1;
            for (size_t depth = 0; depth < seq.size(); ++depth) {
                const llama_token tok = seq[depth];
                int32_t node_idx = -1;
                if (parent < 0) {
                    for (int32_t idx : roots) {
                        if (last_tree.tokens[idx] == tok) {
                            node_idx = idx;
                            break;
                        }
                    }
                } else {
                    for (int32_t idx : children[parent]) {
                        if (last_tree.tokens[idx] == tok) {
                            node_idx = idx;
                            break;
                        }
                    }
                }

                if (node_idx < 0) {
                    node_idx = (int32_t) last_tree.tokens.size();
                    last_tree.tokens.push_back(tok);
                    last_tree.parents.push_back(parent);
                    last_tree.depths.push_back((int32_t) depth);
                    last_tree_states.emplace_back();
                    children.emplace_back();

                    if (parent < 0) {
                        roots.push_back(node_idx);
                    } else {
                        children[parent].push_back(node_idx);
                    }
                }

                llama_tokens prefix(seq.begin(), seq.begin() + (ptrdiff_t) depth + 1);
                auto it_state = prefix_states.find(prefix);
                if (it_state != prefix_states.end()) {
                    last_tree_states[(size_t) node_idx] = it_state->second;
                }

                parent = node_idx;
            }
        }

        common_speculative_tree_build_metadata(last_tree);

        draft_tokens = last_tree.tokens;
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    void accept_tokens(const llama_tokens & ids, llama_seq_id seq_id) override {
        if (seq_id != active_seq_id || ids.empty() || !has_last_root_state) {
            return;
        }

        std::vector<int32_t> roots;
        std::vector<std::vector<int32_t>> children(last_tree.tokens.size());
        roots.reserve(last_tree.tokens.size());

        for (size_t i = 0; i < last_tree.tokens.size(); ++i) {
            const int32_t parent = last_tree.parents[i];
            if (parent < 0) {
                roots.push_back((int32_t) i);
            } else if ((size_t) parent < children.size()) {
                children[(size_t) parent].push_back((int32_t) i);
            }
        }

        const auto find_token = [&](const std::vector<int32_t> & list, llama_token tok) -> int32_t {
            for (int32_t idx : list) {
                if (last_tree.tokens[(size_t) idx] == tok) {
                    return idx;
                }
            }
            return -1;
        };

        std::vector<int32_t> accepted_nodes;
        int32_t node = find_token(roots, ids[0]);
        while (node >= 0) {
            accepted_nodes.push_back(node);
            if (accepted_nodes.size() >= ids.size() - 1) {
                break;
            }
            const llama_token next_tok = ids[accepted_nodes.size()];
            node = find_token(children[(size_t) node], next_tok);
        }

        const size_t accepted_count = ids.size() > 0 ? ids.size() - 1 : 0;
        if (accepted_count > 0) {
            llama_tokens accepted_tokens(ids.begin(), ids.begin() + (ptrdiff_t) accepted_count);
            common_speculative_trace_mark_accepted(last_trace.proposal_tree, accepted_tokens);
        }

        size_t n_tokens = 0;
        std::vector<const ggml_tensor *> layer_tensors;
        if (!fetch_hidden_tensors(layer_tensors, n_tokens) || n_tokens == 0) {
            return;
        }

        const auto row_for_node = [&](int32_t node_idx) -> size_t {
            if (node_idx < 0) {
                return std::numeric_limits<size_t>::max();
            }
            const size_t idx = (size_t) node_idx;
            if (idx >= last_tree.tokens.size()) {
                return std::numeric_limits<size_t>::max();
            }
            if (last_tree.row_indices.size() == last_tree.tokens.size()) {
                return (size_t) last_tree.row_indices[idx];
            }
            return (size_t) last_tree.batch_start + idx;
        };

        if (accepted_nodes.empty()) {
            set_prefix_frontier(last_root_state, 0);
        } else {
            const int32_t deepest = accepted_nodes.back();
            if ((size_t) deepest >= last_tree_states.size()) {
                return;
            }
            const size_t row_idx = row_for_node(deepest);
            if (row_idx >= n_tokens) {
                return;
            }
            set_prefix_frontier(last_tree_states[(size_t) deepest], row_idx);
        }

        prefix_prompt_len += ids.size();
    }

    bool get_tree(common_speculative_tree & out) const override {
        out = last_tree;
        return true;
    }

    bool get_trace(common_speculative_trace & out) const override {
        out = last_trace;
        return true;
    }

    void set_tree(const common_speculative_tree & tree) override {
        if (tree.tokens != last_tree.tokens ||
            tree.parents != last_tree.parents ||
            tree.depths != last_tree.depths) {
            return;
        }

        last_tree.batch_start = tree.batch_start;
        last_tree.row_indices = tree.row_indices;
    }

private:
    bool fetch_hidden_tensors(
            std::vector<const ggml_tensor *> & tensors,
            size_t & n_tokens) const {
        tensors.clear();
        n_tokens = 0;

        for (int32_t layer_id : layer_ids) {
            size_t n_layer_tokens = 0;
            const ggml_tensor * data = llama_eagle3_get_hidden_capture(ctx_tgt, layer_id, &n_layer_tokens);
            if (!data || n_layer_tokens == 0) {
                return false;
            }
            if (tensors.empty()) {
                n_tokens = n_layer_tokens;
            } else if (n_layer_tokens != n_tokens) {
                return false;
            }
            tensors.push_back(data);
        }

        return !tensors.empty();
    }

    bool build_hidden_concat_host(
            const std::vector<const ggml_tensor *> & tensors,
            size_t token_idx,
            std::vector<float> & out) const {
        if (tensors.empty()) {
            return false;
        }

        out.resize((size_t) hidden_in_dim);
        size_t offset = 0;
        for (const ggml_tensor * t : tensors) {
            if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != target_hidden_size || token_idx >= (size_t) t->ne[1]) {
                return false;
            }
            const size_t n_bytes = (size_t) target_hidden_size * sizeof(float);
            ggml_backend_tensor_get(t, out.data() + offset, token_idx * (size_t) t->nb[1], n_bytes);
            offset += (size_t) target_hidden_size;
        }
        return true;
    }

    bool write_dump(llama_seq_id seq_id) {
        if (dump_dir.empty()) {
            return true;
        }
        if (!dump_pending) {
            return true;
        }

        namespace fs = std::filesystem;

        std::error_code ec;
        fs::create_directories(dump_dir, ec);
        if (ec) {
            LOG_ERR("%s: failed to create CASCADE_EAGLE_DUMP_DIR='%s': %s\n",
                    __func__, dump_dir.c_str(), ec.message().c_str());
            return false;
        }

        const fs::path dir = fs::path(dump_dir);

        // Core metadata
        const int32_t draft_vocab_size = model ? model->hparams.draft_vocab_size : 0;
        const size_t n_prompt = dump_prompt_tgt.size();
        const size_t n_steps  = dump_head_input_ids.size();

        if (n_prompt == 0 || dump_id_last < 0) {
            LOG_ERR("%s: dump missing prompt/id_last (n_prompt=%zu id_last=%d)\n",
                    __func__, n_prompt, (int) dump_id_last);
            return false;
        }
        if (n_steps != n_prompt) {
            LOG_WRN("%s: unexpected dump sizes (n_steps=%zu n_prompt=%zu)\n", __func__, n_steps, n_prompt);
        }

        // prompt + id_last
        std::vector<int32_t> prompt_i32(n_prompt);
        for (size_t i = 0; i < n_prompt; ++i) {
            prompt_i32[i] = (int32_t) dump_prompt_tgt[i];
        }
        const int32_t id_last_i32 = (int32_t) dump_id_last;

        std::vector<int32_t> layer_ids_i32(layer_ids.begin(), layer_ids.end());

        // Dump per-layer teacher hiddens (prompt_tgt only).
        std::vector<const ggml_tensor *> layer_tensors;
        size_t n_layer_tokens = 0;
        if (!fetch_hidden_tensors(layer_tensors, n_layer_tokens) || n_layer_tokens < n_prompt) {
            LOG_ERR("%s: failed to fetch teacher hidden ptrs for dump\n", __func__);
            return false;
        }

        if (!write_npy_i32(dir / "prompt_tgt.npy", prompt_i32.data(), {n_prompt})) {
            return false;
        }
        if (!write_npy_i32(dir / "id_last.npy", &id_last_i32, {1})) {
            return false;
        }
        if (!write_npy_i32(dir / "layer_ids.npy", layer_ids_i32.data(), {layer_ids_i32.size()})) {
            return false;
        }

        for (size_t li = 0; li < layer_tensors.size(); ++li) {
            const int32_t layer_id = layer_ids[li];
            std::vector<float> layer_host((size_t) n_prompt * (size_t) target_hidden_size);
            ggml_backend_tensor_get(layer_tensors[li], layer_host.data(), 0, layer_host.size() * sizeof(float));
            const fs::path p = dir / ("teacher_hidden_layer_" + std::to_string(layer_id) + ".npy");
            if (!write_npy_f32(p, layer_host.data(), {n_prompt, (size_t) target_hidden_size})) {
                return false;
            }
        }

        // Dump head inputs/outputs for the token-by-token run (prompt_tgt[1:] + id_last).
        if (!dump_teacher_hidden_by_step.empty()) {
            if (!write_npy_f32(dir / "teacher_hidden_concat_by_step.npy", dump_teacher_hidden_by_step.data(), {n_steps, (size_t) hidden_in_dim})) {
                return false;
            }
        }
        if (!dump_head_input_ids.empty()) {
            if (!write_npy_i32(dir / "head_input_ids.npy", dump_head_input_ids.data(), {n_steps})) {
                return false;
            }
        }
        if (!dump_head_embd_by_step.empty()) {
            if (!write_npy_f32(dir / "head_embd_by_step.npy", dump_head_embd_by_step.data(), {n_steps, (size_t) hidden_size})) {
                return false;
            }
        }
        if (!dump_head_embd_norm_by_step.empty()) {
            if (!write_npy_f32(dir / "head_embd_norm_by_step.npy", dump_head_embd_norm_by_step.data(), {n_steps, (size_t) hidden_size})) {
                return false;
            }
        }
        if (!dump_head_hidden_proj_by_step.empty()) {
            if (!write_npy_f32(dir / "head_hidden_proj_by_step.npy", dump_head_hidden_proj_by_step.data(), {n_steps, (size_t) hidden_size})) {
                return false;
            }
        }
        if (!dump_head_hidden_norm_by_step.empty()) {
            if (!write_npy_f32(dir / "head_hidden_norm_by_step.npy", dump_head_hidden_norm_by_step.data(), {n_steps, (size_t) hidden_size})) {
                return false;
            }
        }
        if (!dump_head_cat_by_step.empty()) {
            if (!write_npy_f32(dir / "head_cat_by_step.npy", dump_head_cat_by_step.data(), {n_steps, (size_t) hidden_size * 2})) {
                return false;
            }
        }
        if (!dump_head_q_by_step.empty()) {
            if (!write_npy_f32(dir / "head_q_by_step.npy", dump_head_q_by_step.data(), {n_steps, (size_t) hidden_size})) {
                return false;
            }
        }
        if (!dump_head_k_by_step.empty()) {
            const size_t kv_dim = (size_t) model->hparams.head_dim * model->hparams.num_kv_heads;
            if (!write_npy_f32(dir / "head_k_by_step.npy", dump_head_k_by_step.data(), {n_steps, kv_dim})) {
                return false;
            }
        }
        if (!dump_head_v_by_step.empty()) {
            const size_t kv_dim = (size_t) model->hparams.head_dim * model->hparams.num_kv_heads;
            if (!write_npy_f32(dir / "head_v_by_step.npy", dump_head_v_by_step.data(), {n_steps, kv_dim})) {
                return false;
            }
        }
        if (!dump_head_hidden_after_step.empty()) {
            if (!write_npy_f32(dir / "head_hidden_after_step.npy", dump_head_hidden_after_step.data(), {n_steps, (size_t) hidden_size})) {
                return false;
            }
        }

        if (!dump_root_logits_draft.empty()) {
            if ((int32_t) dump_root_logits_draft.size() != draft_vocab_size) {
                LOG_WRN("%s: root logits size mismatch: got=%zu expected=%d\n",
                        __func__, dump_root_logits_draft.size(), draft_vocab_size);
            }
            if (!write_npy_f32(dir / "head_root_logits_draft.npy", dump_root_logits_draft.data(), {(size_t) dump_root_logits_draft.size()})) {
                return false;
            }
        }

        // Meta file for convenience.
        {
            std::ofstream meta(dir / "meta.json", std::ios::binary);
            if (!meta) {
                LOG_ERR("%s: failed to write meta.json\n", __func__);
                return false;
            }

            meta
                << "{\n"
                << "  \"format\": \"cascade_eagle3_dump_v1\",\n"
                << "  \"seq_id\": " << (int) seq_id << ",\n"
                << "  \"prompt_tgt_len\": " << n_prompt << ",\n"
                << "  \"head_steps\": " << n_steps << ",\n"
                << "  \"hidden_in_dim\": " << hidden_in_dim << ",\n"
                << "  \"hidden_size\": " << hidden_size << ",\n"
                << "  \"target_hidden_size\": " << target_hidden_size << ",\n"
                << "  \"head_dim\": " << model->hparams.head_dim << ",\n"
                << "  \"num_heads\": " << model->hparams.num_heads << ",\n"
                << "  \"num_kv_heads\": " << model->hparams.num_kv_heads << ",\n"
                << "  \"draft_vocab_size\": " << draft_vocab_size << ",\n"
                << "  \"layer_ids\": [";
            for (size_t i = 0; i < layer_ids.size(); ++i) {
                meta << layer_ids[i];
                if (i + 1 < layer_ids.size()) {
                    meta << ", ";
                }
            }
            meta << "],\n";

            meta
                << "  \"files\": {\n"
                << "    \"prompt_tgt\": \"prompt_tgt.npy\",\n"
                << "    \"id_last\": \"id_last.npy\",\n"
                << "    \"layer_ids\": \"layer_ids.npy\",\n"
                << "    \"teacher_hidden_layer_*\": \"teacher_hidden_layer_*.npy\",\n"
                << "    \"teacher_hidden_concat_by_step\": \"teacher_hidden_concat_by_step.npy\",\n"
                << "    \"head_input_ids\": \"head_input_ids.npy\",\n"
                << "    \"head_embd_by_step\": \"head_embd_by_step.npy\",\n"
                << "    \"head_q_by_step\": \"head_q_by_step.npy\",\n"
                << "    \"head_k_by_step\": \"head_k_by_step.npy\",\n"
                << "    \"head_v_by_step\": \"head_v_by_step.npy\",\n"
                << "    \"head_hidden_after_step\": \"head_hidden_after_step.npy\",\n"
                << "    \"head_root_logits_draft\": \"head_root_logits_draft.npy\"\n"
                << "  }\n"
                << "}\n";
        }

        LOG_INF("%s: wrote EAGLE3 dump to '%s'\n", __func__, dump_dir.c_str());
        return true;
    }

    bool prefill_to(const llama_tokens & prompt_tgt, llama_seq_id seq_id) {
        GGML_UNUSED(seq_id);
        if (prompt_tgt.size() < prefix_prompt_len) {
            reset_prefix_cache();
        }

        if (prompt_tgt.size() == prefix_prompt_len && prefix_prompt_len > 0) {
            return true;
        }

        if (prompt_tgt.size() < 2) {
            prefix_prompt_len = prompt_tgt.size();
            prefix_tail_hidden_concat.clear();
            prefix_tail_capture_idx = 0;
            return true;
        }

        std::vector<const ggml_tensor *> layer_tensors;
        size_t n_tokens = 0;
        if (!fetch_hidden_tensors(layer_tensors, n_tokens)) {
            return false;
        }
        if (prefix_prompt_len == 0 && n_tokens < prompt_tgt.size()) {
            return false;
        }
        if (prefix_prompt_len > 0) {
            const size_t delta = prompt_tgt.size() - prefix_prompt_len;
            if (delta == 0) {
                prefix_tail_capture_idx = 0;
                return true;
            }
            if (prefix_tail_hidden_concat.size() != (size_t) hidden_in_dim || n_tokens < delta) {
                reset_prefix_cache();
                return false;
            }
        }

        if (prefix_prompt_len == 0) {
            if (const char * dump_path = std::getenv("CASCADE_EAGLE_DUMP")) {
                // Dump tokens + concatenated teacher hidden stream for external parity checks
                // (e.g., compare against the PyTorch reference implementation).
                std::vector<float> all_hidden;
                all_hidden.resize(prompt_tgt.size() * (size_t) hidden_in_dim);
                for (size_t i = 0; i < prompt_tgt.size(); ++i) {
                    if (!build_hidden_concat_host(layer_tensors, i, hidden_concat_buf)) {
                        return false;
                    }
                    std::memcpy(
                        all_hidden.data() + i * (size_t) hidden_in_dim,
                        hidden_concat_buf.data(),
                        (size_t) hidden_in_dim * sizeof(float));
                }

                std::ofstream out(dump_path, std::ios::binary);
                if (!out) {
                    LOG_ERR("%s: failed to open CASCADE_EAGLE_DUMP='%s'\n", __func__, dump_path);
                    return false;
                }

                const uint32_t n_tokens_u32 = (uint32_t) prompt_tgt.size();
                const uint32_t dim_u32      = (uint32_t) hidden_in_dim;

                out.write(reinterpret_cast<const char *>(&n_tokens_u32), sizeof(n_tokens_u32));
                out.write(reinterpret_cast<const char *>(&dim_u32),      sizeof(dim_u32));
                out.write(reinterpret_cast<const char *>(prompt_tgt.data()), prompt_tgt.size() * sizeof(prompt_tgt[0]));
                out.write(reinterpret_cast<const char *>(all_hidden.data()), all_hidden.size() * sizeof(all_hidden[0]));

                LOG_INF("%s: wrote CASCADE_EAGLE_DUMP='%s' (%u tokens, dim=%u)\n",
                        __func__, dump_path, n_tokens_u32, dim_u32);
            }
        }

        if (prefix_prompt_len == 0) {
            prefix_state = {};
        }

        const bool dump_steps = dump_pending && prefix_prompt_len == 0 && !dump_dir.empty();
        if (dump_steps) {
            dump_prompt_tgt = prompt_tgt;
            dump_head_input_ids.clear();
            dump_teacher_hidden_by_step.clear();
            dump_head_embd_by_step.clear();
            dump_head_embd_norm_by_step.clear();
            dump_head_hidden_proj_by_step.clear();
            dump_head_hidden_norm_by_step.clear();
            dump_head_cat_by_step.clear();
            dump_head_q_by_step.clear();
            dump_head_k_by_step.clear();
            dump_head_v_by_step.clear();
            dump_head_hidden_after_step.clear();

            const size_t n_steps_total = prompt_tgt.size(); // prompt_tgt[1:] + id_last (recorded later in draft())
            dump_head_input_ids.reserve(n_steps_total);
            dump_teacher_hidden_by_step.reserve(n_steps_total * (size_t) hidden_in_dim);
            dump_head_embd_by_step.reserve(n_steps_total * (size_t) hidden_size);
            dump_head_embd_norm_by_step.reserve(n_steps_total * (size_t) hidden_size);
            dump_head_hidden_proj_by_step.reserve(n_steps_total * (size_t) hidden_size);
            dump_head_hidden_norm_by_step.reserve(n_steps_total * (size_t) hidden_size);
            dump_head_cat_by_step.reserve(n_steps_total * (size_t) hidden_size * 2);
            dump_head_q_by_step.reserve(n_steps_total * (size_t) hidden_size);
            dump_head_k_by_step.reserve(n_steps_total * (size_t) model->hparams.head_dim * model->hparams.num_kv_heads);
            dump_head_v_by_step.reserve(n_steps_total * (size_t) model->hparams.head_dim * model->hparams.num_kv_heads);
            dump_head_hidden_after_step.reserve(n_steps_total * (size_t) hidden_size);
        }

        if (prefix_prompt_len > 0) {
            const size_t first_new = prefix_prompt_len;
            llama_eagle3_step_debug dbg;
            if (dump_steps) {
                dbg.embd        = &dump_head_embd_by_step;
                dbg.embd_norm   = &dump_head_embd_norm_by_step;
                dbg.hidden_proj = &dump_head_hidden_proj_by_step;
                dbg.hidden_norm = &dump_head_hidden_norm_by_step;
                dbg.cat         = &dump_head_cat_by_step;
                dbg.q           = &dump_head_q_by_step;
                dbg.k           = &dump_head_k_by_step;
                dbg.v           = &dump_head_v_by_step;

                dump_head_input_ids.push_back((int32_t) prompt_tgt[first_new]);
                dump_teacher_hidden_by_step.insert(
                    dump_teacher_hidden_by_step.end(),
                    prefix_tail_hidden_concat.begin(),
                    prefix_tail_hidden_concat.end());
            }

            if (!llama_eagle3_step(
                        *model,
                        rt,
                        prefix_state,
                        prefix_tail_hidden_concat.data(),
                        hidden_in_dim,
                        prompt_tgt[first_new],
                        nullptr,
                        dump_steps ? &dbg : nullptr)) {
                return false;
            }

            if (dump_steps) {
                dump_head_hidden_after_step.insert(
                    dump_head_hidden_after_step.end(),
                    prefix_state.hidden.begin(),
                    prefix_state.hidden.end());
            }
        }

        const size_t start = prefix_prompt_len > 0 ? prefix_prompt_len : 0;
        const size_t local_offset = prefix_prompt_len > 0 ? prefix_prompt_len : 0;
        for (size_t i = start; i + 1 < prompt_tgt.size(); ++i) {
            llama_eagle3_step_debug dbg;
            if (dump_steps) {
                const size_t local_idx = i - local_offset;
                if (!build_hidden_concat_host(layer_tensors, local_idx, hidden_concat_buf)) {
                    return false;
                }
                dbg.embd        = &dump_head_embd_by_step;
                dbg.embd_norm   = &dump_head_embd_norm_by_step;
                dbg.hidden_proj = &dump_head_hidden_proj_by_step;
                dbg.hidden_norm = &dump_head_hidden_norm_by_step;
                dbg.cat         = &dump_head_cat_by_step;
                dbg.q           = &dump_head_q_by_step;
                dbg.k           = &dump_head_k_by_step;
                dbg.v           = &dump_head_v_by_step;

                dump_head_input_ids.push_back((int32_t) prompt_tgt[i + 1]);
                dump_teacher_hidden_by_step.insert(
                    dump_teacher_hidden_by_step.end(),
                    hidden_concat_buf.begin(),
                    hidden_concat_buf.end());
            }

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

            if (dump_steps) {
                dump_head_hidden_after_step.insert(
                    dump_head_hidden_after_step.end(),
                    prefix_state.hidden.begin(),
                    prefix_state.hidden.end());
            }
        }

        const size_t tail_idx = prefix_prompt_len > 0
            ? (prompt_tgt.size() - prefix_prompt_len - 1)
            : (prompt_tgt.size() - 1);
        if (!build_hidden_concat_host(layer_tensors, tail_idx, prefix_tail_hidden_concat)) {
            return false;
        }
        prefix_tail_capture_idx = tail_idx;
        prefix_prompt_len = prompt_tgt.size();
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_state state;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_state state)
        : common_speculative_state(type), state(state) {}

    void begin(const llama_tokens & prompt, llama_seq_id seq_id) override {
        GGML_UNUSED(prompt);
        GGML_UNUSED(seq_id);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_seq_id seq_id,
            llama_tokens & result) override {
        result = common_ngram_simple_draft(state, prompt_tgt, id_last);
        GGML_UNUSED(params);
        GGML_UNUSED(seq_id);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt, llama_seq_id seq_id) override {
        GGML_UNUSED(prompt);
        GGML_UNUSED(seq_id);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_seq_id seq_id,
            llama_tokens & result) override {
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
        GGML_UNUSED(seq_id);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt, llama_seq_id seq_id) override {
        i_last = 0;
        GGML_UNUSED(seq_id);

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_seq_id seq_id,
            llama_tokens & result) override {
        GGML_UNUSED(params);
        GGML_UNUSED(seq_id);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt, llama_seq_id seq_id) override {
        GGML_UNUSED(prompt);
        GGML_UNUSED(seq_id);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_seq_id seq_id,
            llama_tokens & result) override {
        GGML_UNUSED(params);
        GGML_UNUSED(seq_id);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t check_rate = config.params.ngram_check_rate;
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, check_rate, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    if (params.model_dft) {
        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        const bool has_draft_model = params.has_dft();
        const bool want_draft = (params.type == COMMON_SPECULATIVE_TYPE_DRAFT || params.type == COMMON_SPECULATIVE_TYPE_NONE);
        const bool want_eagle3 = (params.type == COMMON_SPECULATIVE_TYPE_EAGLE3);

        bool has_draft = has_draft_model && want_draft;
        bool has_draft_eagle3 = has_draft_model && want_eagle3;

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            // shared instance for all speculative decoding contexts
            if (!params.ngram_mod) {
                params.ngram_mod = std::make_shared<common_ngram_mod>(params.ngram_size_n, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                        params.ngram_size_n, params.ngram_mod->size(),
                        (float)(params.ngram_mod->size_bytes())/1024/1024);

                if (params.ngram_size_n < 16) {
                    LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, params.ngram_size_n);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type, ctx_tgt, config.params));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;
                uint16_t check_rate       = ngram_map.check_rate;

                auto config_simple = common_ngram_simple_config{
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value,
                    /* .check_rate      = */ check_rate
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ common_ngram_simple_state(config_simple)
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        params.lookup_cache_static, params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls = */ std::move(impls)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->begin(prompt, seq_id);
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last,
        llama_seq_id seq_id) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            const int64_t t_start_us = impl->gen_perf ? ggml_time_us() : 0;

            impl->draft(params, prompt_tgt, id_last, seq_id, result);

            const int64_t t_now_us = impl->gen_perf ? ggml_time_us() : 0;

            impl->drafts_call_count++;
            impl->gen_duration_us += t_now_us - t_start_us; // accumulate duration for this implementation
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->drafts_call_count, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->drafts_generated_count++;
            impl->drafts_generated_tokens += result.size();

            break; // We have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

bool common_speculative_get_tree(common_speculative * spec, common_speculative_tree & out) {
    out.clear();
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return false;
    }

    return spec->curr_impl->get_tree(out);
}

void common_speculative_set_tree(common_speculative * spec, const common_speculative_tree & tree) {
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return;
    }

    spec->curr_impl->set_tree(tree);
}

bool common_speculative_get_trace(common_speculative * spec, common_speculative_trace & out) {
    out.clear();
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return false;
    }

    return spec->curr_impl->get_trace(out);
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    if (n_accepted > 0) {
        impl->drafts_accepted_count++;
        impl->drafts_accepted_tokens += n_accepted;
    }

    impl->accept(n_accepted);
}

void common_speculative_accept_tokens(common_speculative * spec, const llama_tokens & ids, llama_seq_id seq_id) {
    if (spec == nullptr || spec->curr_impl == nullptr || ids.empty()) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;
    impl->accept_tokens(ids, seq_id);
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->gen_duration_us / 1000.0;
            str_perf = ", dur = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        // TODO: report time for begin() and accept()
        LOG_INF("statistics %s: #calls = %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->drafts_call_count,
                impl->drafts_generated_count,
                impl->drafts_accepted_count,
                impl->drafts_generated_tokens,
                impl->drafts_accepted_tokens,
                str_perf.c_str());
    }
}
