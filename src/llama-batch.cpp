#include "llama-batch.h"

#include "llama-impl.h"
#include "llama-vocab.h"
#include "llama-memory.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>

llama_batch_allocr::llama_batch_allocr(uint32_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {
    const char * LLAMA_BATCH_DEBUG = getenv("LLAMA_BATCH_DEBUG");
    debug = LLAMA_BATCH_DEBUG ? atoi(LLAMA_BATCH_DEBUG) : 0;

    seq_pos.resize(LLAMA_MAX_SEQ);
    seq_cpl.resize(LLAMA_MAX_SEQ);
    for (auto & cur : seq_cpl) {
        cur.resize(LLAMA_MAX_SEQ);
    }

    seq_idx.resize(LLAMA_MAX_SEQ, -1);
}

bool llama_batch_allocr::init(
        const llama_batch & batch_inp,
        const llama_vocab & vocab,
        const llama_memory_i * memory,
        uint32_t n_embd,
        uint32_t n_seq_max,
        bool output_all) {
    clear();

    batch = batch_inp;

    this->vocab = &vocab;

    GGML_ASSERT(batch.n_tokens > 0);

    //
    // validate input batch
    //

    if (n_seq_max > LLAMA_MAX_SEQ) {
        LLAMA_LOG_ERROR("%s: n_seq_max = %d > %d\n", __func__, n_seq_max, LLAMA_MAX_SEQ);
        return false;
    }

    if (batch.token) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return false;
            }
        }
    }

    if (batch.seq_id) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                if (batch.seq_id && (batch.seq_id[i][s] < 0 || batch.seq_id[i][s] >= (llama_seq_id) n_seq_max)) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id[%d][%d] = %d >= %d\n", __func__, i, s, batch.seq_id[i][s], (llama_seq_id) n_seq_max);
                    return false;
                }
            }
        }
    }

    //
    // auto-generate missing fields
    //

    if (!batch.n_seq_id) {
        n_seq_id.resize(batch.n_tokens);
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            n_seq_id[i] = seq_id_0.size();
        }
        batch.n_seq_id = n_seq_id.data();
    }

    if (!batch.seq_id) {
        seq_id.resize(batch.n_tokens + 1);
        seq_id[batch.n_tokens] = NULL;
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            seq_id[i] = seq_id_0.data();
        }
        batch.seq_id = seq_id.data();
    }

    if (!batch.pos) {
        pos.resize(batch.n_tokens);

        // initialize the starting position for each sequence based on the positions in the memory
        llama_pos p0[LLAMA_MAX_SEQ];
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            if (!memory) {
                // if no memory -> start from 0
                p0[s] = 0;
            } else {
                p0[s] = memory->seq_pos_max(s) + 1;
            }
        }

        for (int32_t i = 0; i < batch.n_tokens; i++) {
            const llama_seq_id seq_id = batch.seq_id[i][0];

            pos[i] = p0[seq_id];

            // update the starting position for all sequences that are assigned to the this token
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                p0[seq_id] = pos[i] + 1;
            }
        }

        batch.pos = pos.data();
    }

    if (batch.kv_slot) {
        bool any_explicit = false;
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.kv_slot[i] >= 0) {
                any_explicit = true;
                continue;
            }
        }

        if (!any_explicit) {
            batch.kv_slot = nullptr;
        } else {
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                if (batch.kv_slot[i] < 0) {
                    LLAMA_LOG_ERROR("%s: batch has mixed explicit and implicit kv_slot entries\n", __func__);
                    return false;
                }
            }
        }
    }

    if (batch.kv_slot) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.kv_slot[i] < 0) {
                LLAMA_LOG_ERROR("%s: invalid kv_slot[%d] = %d\n", __func__, i, batch.kv_slot[i]);
                return false;
            }
        }
    }

    if (!batch.logits) {
        if (output_all) {
            // return the output for all tokens
            output.resize(batch.n_tokens, true);
        } else {
            // return the output only for the last token
            output.resize(batch.n_tokens, false);
            output[output.size() - 1] = true;
        }

        batch.logits = output.data();
    } else if (output_all) {
        bool warn = false;

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.logits[i] == 0) {
                warn = true;
            }
        }

        if (warn) {
            LLAMA_LOG_WARN("%s: embeddings required but some input tokens were not marked as outputs -> overriding\n", __func__);

            output.resize(batch.n_tokens, true);
            batch.logits = output.data();
        }
    }

    //
    // compute stats
    //

    this->n_embd    = n_embd;
    this->n_seq_max = n_seq_max;

    // count the outputs in this batch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        n_outputs += batch.logits[i] != 0;
    }

    has_cpl = false;

    // determine coupled sequences
    // these are pairs of sequences that have at least one token in the input batch that is assigned to both of them
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const llama_seq_id s0 = batch.seq_id[i][0];

        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            const llama_seq_id s1 = batch.seq_id[i][s];

            seq_pos[s1].insert(batch.pos[i]);

            if (s > 0) {
                // mark that sequence s1 is coupled to s0
                seq_cpl[s1][s0] = true;

                // note: tracking the other way around is not necessary for now
                //seq_cpl[s0][s1] = true;

                has_cpl = true;
            }
        }
    }

    // precompute the sequence sets for each token and determine the unique sequence ids that participate in the batch
    {
        seq_set_t seq_set_unq;

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            seq_set_t cur;
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cur        .set(seq_id);
                seq_set_unq.set(seq_id);
            }

            seq_set.push_back(cur);
            seq_set_map[cur].push_back(i);
        }

        for (uint32_t s = 0; s < n_seq_max; ++s) {
            if (seq_set_unq.test(s)) {
                seq_idx[s] = seq_id_unq.size();
                seq_id_unq.push_back(s);
            }
        }
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: input batch info:\n", __func__);

        llama_ubatch ubatch {
            /*.b_equal_seqs =*/ false,
            /*.n_tokens     =*/ (uint32_t) batch.n_tokens,
            /*.n_seq_tokens =*/ (uint32_t) 1,
            /*.n_seqs       =*/ (uint32_t) batch.n_tokens,
            /*.n_seqs_unq   =*/ (uint32_t) this->seq_id_unq.size(),
            /*.n_pos        =*/ n_pos_per_embd,
            /*.token        =*/ batch.token,
            /*.embd         =*/ batch.embd,
            /*.pos          =*/ batch.pos,
            /*.kv_slot      =*/ batch.kv_slot,
            /*.n_seq_id     =*/ batch.n_seq_id,
            /*.seq_id       =*/ batch.seq_id,
            /*.seq_id_unq   =*/ this->seq_id_unq.data(),
            /*.seq_idx      =*/ this->seq_idx.data(),
            /*.output       =*/ batch.logits,
            /*.rope_pos     =*/ nullptr,
            /*.data         =*/ {},
        };

        ubatch_print(ubatch, debug);

        LLAMA_LOG_DEBUG("%s:   seq       = [\n", __func__);
        for (int s0 = 0; s0 < (int) seq_pos.size(); ++s0) {
            if (seq_pos[s0].empty()) {
                continue;
            }

            std::stringstream ss;
            for (int s1 = 0; s1 < (int) seq_cpl[s0].size(); ++s1) {
                if (seq_cpl[s0][s1]) {
                    ss << s1 << " ";
                }
            }

            LLAMA_LOG_DEBUG("%s:  %4d: pos = [%4d, %4d], cpl = %s\n",
                    __func__, s0, seq_pos_min(s0), seq_pos_max(s0), ss.str().empty() ? "-" : ss.str().c_str());
        }
        LLAMA_LOG_DEBUG("%s:   ]\n", __func__);
    }

    //
    // consistency checks
    //

    if (n_pos_per_embd > 1) {
        // M-RoPE case: allow position to "jump" forward only (non-continuous positions are allowed)
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            if (seq_pos[s].empty()) {
                continue;
            }

            const llama_pos p0 = memory ? memory->seq_pos_max(s) : -1;

            if (batch.token) {
                if (p0 >= 0 && p0 >= seq_pos_min(s)) {
                    LLAMA_LOG_ERROR(
                            "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
                            " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X = %d\n"
                            " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
                            " for M-RoPE, it is required that the position satisfies: X < Y\n",
                            __func__, s, s, p0, s, seq_pos_min(s));

                    return false;
                }
            } else {
                // embedding inputs can have overlapping positions
                if (p0 >= 0 && p0 > seq_pos_min(s)) {
                    LLAMA_LOG_ERROR(
                            "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
                            " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X = %d\n"
                            " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
                            " for M-RoPE, it is required that the position satisfies: X <= Y\n",
                            __func__, s, s, p0, s, seq_pos_min(s));

                    return false;
                }
            }
        }
    } else {
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            if (seq_pos[s].empty()) {
                continue;
            }

            const llama_pos p0 = memory ? memory->seq_pos_max(s) : -1;

            if (p0 >= 0) {
                bool ok = true;

                if (seq_pos_min(s) != p0 + 1) {
                    ok = false;
                }

                if (!ok) {
                    LLAMA_LOG_ERROR(
                            "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
                            " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X = %d\n"
                            " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
                            " it is required that the sequence positions remain consecutive: Y = X + 1\n",
                            __func__, s, s, p0, s, seq_pos_min(s));

                    return false;
                }
            }

            if (seq_pos_max(s) - seq_pos_min(s) + 1 > (int) seq_pos[s].size()) {
                LLAMA_LOG_ERROR("%s: sequence %d positions are not continuous\n", __func__, s);
                return false;
            }
        }
    }

    if (memory) {
        for (uint32_t s0 = 0; s0 < n_seq_max; ++s0) {
            for (uint32_t s1 = 0; s1 < n_seq_max; ++s1) {
                if (seq_cpl[s0][s1]) {
                    if (memory->seq_pos_min(s0) != memory->seq_pos_min(s1) ||
                        memory->seq_pos_max(s0) != memory->seq_pos_max(s1)) {
                        LLAMA_LOG_ERROR("%s: sequence %d is coupled to %d in the input batch, but have divereged\n", __func__, s0, s1);
                        return false;
                    }
                }
            }
        }
    }

    // disallow partial sequence sub-sets:
    //
    // invalid:          x
    //            i: 0 1 2 ...
    // ---------------------------------------
    // seq_id[i][0]: 0 0 1
    // seq_id[i][1]: 1 1 2
    // seq_id[i][2]: 2
    //
    // disallow decreasing sequence positions:
    //
    // invalid:                  x
    //            i: 0 1 2 3 4 5 6 ...
    // ---------------------------------------
    //       pos[i]: 4 5 0 1 6 2 3
    // seq_id[i][0]: 0 0 1 1 0 1 0
    //
    {
        seq_set_t cur_seq_set[LLAMA_MAX_SEQ];
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            cur_seq_set[s].set();
        }

        llama_pos cur_seq_pos[LLAMA_MAX_SEQ];
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            cur_seq_pos[s] = -1;
        }

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const llama_pos pos = batch.pos[i];

            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cur_seq_set[seq_id] &= seq_set[i];

                if (cur_seq_set[seq_id].none()) {
                    LLAMA_LOG_ERROR("%s: sequence %d belongs to incompatible sequence sets (not allowed)\n", __func__, seq_id);
                    return false;
                }

                if (pos < cur_seq_pos[seq_id]) {
                    LLAMA_LOG_ERROR("%s: sequence %d positions are decreasing (not allowed)\n", __func__, seq_id);
                    return false;
                }
            }
        }
    }

    split_reset();

    return true;
}

llama_ubatch llama_batch_allocr::ubatch_reserve(uint32_t n_seq_tokens, uint32_t n_seqs) {
    const uint32_t n_tokens = n_seq_tokens*n_seqs;

    clear();
    split_reset();

    const int64_t n_pos_all = (int64_t) n_tokens*n_pos_per_embd;

    auto udata = std::make_shared<llama_ubatch::data_t>();

    udata->token     .resize(n_tokens);
    udata->embd      .clear();
    udata->pos       .resize(n_pos_all);
    udata->kv_slot   .resize(n_tokens, -1);
    udata->n_seq_id  .resize(n_tokens);
    udata->seq_id    .resize(n_tokens);
    udata->seq_id_unq.resize(0);
    udata->seq_idx   .resize(LLAMA_MAX_SEQ, -1);
    udata->output    .resize(n_tokens);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        udata->seq_idx[s] = s;
        udata->seq_id_unq.push_back(s);
    }

    llama_ubatch res {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.n_seqs_unq   =*/ n_seqs,
        /*.n_pos        =*/ n_pos_per_embd,

        /*.token        =*/ udata->token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ udata->pos.data(),
        /*.kv_slot      =*/ udata->kv_slot.data(),
        /*.n_seq_id     =*/ udata->n_seq_id.data(),
        /*.seq_id       =*/ udata->seq_id.data(),
        /*.seq_id_unq   =*/ udata->seq_id_unq.data(),
        /*.seq_idx      =*/ udata->seq_idx.data(),
        /*.output       =*/ udata->output.data(),
        /*.rope_pos     =*/ nullptr,
        /*.data         =*/ std::move(udata),
    };

    return res;
}

const llama_batch & llama_batch_allocr::get_batch() const {
    return batch;
}

uint32_t llama_batch_allocr::get_n_tokens() const {
    return batch.n_tokens;
}

uint32_t llama_batch_allocr::get_n_outputs() const {
    return n_outputs;
}

uint32_t llama_batch_allocr::get_n_used() const {
    return n_used;
}

std::vector<int32_t> & llama_batch_allocr::get_out_ids() {
    return out_ids;
}

llama_pos llama_batch_allocr::seq_pos_min(llama_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].begin();
}

llama_pos llama_batch_allocr::seq_pos_max(llama_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].rbegin();
}

void llama_batch_allocr::split_reset() {
    out_ids.clear();

    expand_processed_seqs.clear();
    expand_seen_outputs.clear();

    n_used = 0;

    used.clear();
    used.resize(get_n_tokens(), false);
}

llama_ubatch llama_batch_allocr::split_simple(uint32_t n_ubatch) {
    // find the first unused token
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }

    // we are done
    if (cur_idx >= used.size()) {
        return {};
    }

    std::vector<int32_t> idxs;

    while (true) {
        idxs.push_back(cur_idx);

        used[cur_idx] = true;
        ++n_used;

        ++cur_idx;

        if (cur_idx >= used.size()) {
            break;
        }

        if (idxs.size() >= n_ubatch) {
            break;
        }
    }

    return ubatch_add(idxs, idxs.size(), false);
}

llama_ubatch llama_batch_allocr::split_equal(uint32_t n_ubatch, bool sequential) {
    if (sequential && has_cpl) {
        return split_equal_expand(n_ubatch);
    }

    std::vector<seq_set_t> cur_seq_set;

    llama_seq_id last_seq_id = -1;

    // determine the non-overlapping sequence sets participating in this ubatch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (used[i]) {
            continue;
        }

        bool add = true;

        for (uint32_t s = 0; s < cur_seq_set.size(); ++s) {
            // no overlap with existing sequence sets:
            if (!(cur_seq_set[s] & seq_set[i]).none()) {
                add = false;
                break;
            }
        }

        // accept only increasing sequence ids
        if (sequential) {
            add = add && (cur_seq_set.empty() || batch.seq_id[i][0] == last_seq_id + 1);
        }

        if (add) {
            cur_seq_set.push_back(seq_set[i]);

            last_seq_id = batch.seq_id[i][0];

            if (cur_seq_set.size() > n_ubatch) {
                break;
            }
        }
    }

    const uint32_t n_seqs = cur_seq_set.size();

    // we are done
    if (n_seqs == 0) {
        return {};
    }

    // the current batch index of each sequence set
    std::vector<int32_t> cur_idx(n_seqs, 0);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        while (used[seq_set_map[cur_seq_set[s]][cur_idx[s]]]) {
            ++cur_idx[s];
        }
    }

    // the list of batch indices for each sequence set
    // at the end we will concat these to get the final ubatch
    std::vector<idx_vec_t> idxs_per_seq(n_seqs);

    while (true) {
        // we can only add new n_seq_tokens tokens if all the sequence sets have at least one more unused token and
        //   if we haven't reached n_ubatch
        bool can_expand = true;

        for (uint32_t s = 0; s < n_seqs; ++s) {
            if (cur_idx[s] >= (int32_t) seq_set_map[cur_seq_set[s]].size()) {
                can_expand = false;
                break;
            }
        }

        if (!can_expand) {
            break;
        }

        for (uint32_t s = 0; s < n_seqs; ++s) {
            const int32_t idx = seq_set_map[cur_seq_set[s]][cur_idx[s]];

            idxs_per_seq[s].push_back(idx);

            used[idx] = true;
            ++n_used;

            ++cur_idx[s];
        }

        if  ((idxs_per_seq[0].size() + 1)*n_seqs > n_ubatch) {
            break;
        }
    }

    // concat the per-sequence-set lists
    std::vector<int32_t> idxs;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        idxs.insert(idxs.end(), idxs_per_seq[s].begin(), idxs_per_seq[s].end());
    }

    return ubatch_add(idxs, n_seqs, true);
}

llama_ubatch llama_batch_allocr::split_equal_expand(uint32_t n_ubatch) {
    // For coupled batches (e.g., tree verification with hybrid models), expand each
    // coupled token into per-sequence copies. The result is an equal_seqs ubatch where
    // each token has n_seq_id=1. Shared tree nodes are recomputed once per branch.
    //
    // Called repeatedly — each call processes a group of sequences with equal token count.
    // State is tracked across calls via expand_processed_seqs and expand_seen_outputs.

    // 1. Build per-sequence token lists, excluding already-processed seq_ids
    std::map<llama_seq_id, std::vector<int32_t>> per_seq;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        for (int s = 0; s < batch.n_seq_id[i]; ++s) {
            const llama_seq_id sid = batch.seq_id[i][s];
            if (expand_processed_seqs.count(sid) == 0) {
                per_seq[sid].push_back(i);
            }
        }
    }

    if (per_seq.empty()) {
        return {};
    }

    // 2. Find the maximum per-sequence token count among remaining sequences.
    uint32_t n_seq_tokens = 0;
    for (const auto & kv : per_seq) {
        if ((uint32_t) kv.second.size() > n_seq_tokens) {
            n_seq_tokens = (uint32_t) kv.second.size();
        }
    }

    if (n_seq_tokens == 0) {
        return {};
    }

    // 3. Select sequences with the maximum token count, up to n_ubatch capacity
    std::vector<llama_seq_id> selected_seqs;
    for (const auto & kv : per_seq) {
        if ((uint32_t) kv.second.size() == n_seq_tokens) {
            selected_seqs.push_back(kv.first);
        }
        if ((uint32_t)(selected_seqs.size() + 1) * n_seq_tokens > n_ubatch) {
            break;
        }
    }

    if (selected_seqs.empty()) {
        return {};
    }

    const uint32_t n_seqs = selected_seqs.size();
    const uint32_t n_tokens_expanded = n_seqs * n_seq_tokens;

    LLAMA_LOG_DEBUG("%s: round with %u seqs x %u tokens = %u expanded (batch has %d tokens)\n",
            __func__, n_seqs, n_seq_tokens, n_tokens_expanded, batch.n_tokens);

    // 4. Build the expanded ubatch directly
    auto udata = std::make_shared<llama_ubatch::data_t>();

    const int64_t n_embd_all = batch.embd ? (int64_t) n_tokens_expanded * n_embd : 0;
    const int64_t n_pos_all  =              (int64_t) n_tokens_expanded * n_pos_per_embd;

    udata->token     .resize(n_tokens_expanded);
    udata->embd      .resize(n_embd_all);
    udata->pos       .resize(n_pos_all);
    udata->kv_slot   .resize(n_tokens_expanded, -1);
    udata->n_seq_id  .resize(n_tokens_expanded);
    udata->seq_id    .resize(n_tokens_expanded);
    udata->seq_id_unq.resize(0);
    udata->seq_idx   .resize(LLAMA_MAX_SEQ, -1);
    udata->output    .resize(n_tokens_expanded);

    udata->seq_id_data.reserve(n_tokens_expanded);

    seq_set_t seq_set_unq;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const llama_seq_id sid = selected_seqs[s];
        const auto & idxs = per_seq[sid];

        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            const int32_t orig_idx = idxs[t];

            if (batch.token) {
                udata->token[i] = batch.token[orig_idx];
            }

            if (batch.embd) {
                memcpy(udata->embd.data() + i * n_embd,
                       batch.embd + (int64_t) orig_idx * n_embd,
                       n_embd * sizeof(float));
            }

            for (size_t j = 0; j < (size_t) n_pos_per_embd; ++j) {
                size_t src_off = batch.token ? 0 : j * batch.n_tokens;
                udata->pos[j * n_tokens_expanded + i] = batch.pos[src_off + orig_idx];
            }

            if (batch.kv_slot) {
                udata->kv_slot[i] = batch.kv_slot[orig_idx];
            }

            // Each expanded token gets a single seq_id
            udata->n_seq_id[i] = 1;
            udata->seq_id_data.push_back(sid);
            seq_set_unq.set(sid);

            // Only output for the first occurrence of each original batch index (across all rounds)
            if (batch.logits[orig_idx] && expand_seen_outputs.count(orig_idx) == 0) {
                udata->output[i] = 1;
                out_ids.push_back(orig_idx);
                expand_seen_outputs.insert(orig_idx);
            } else {
                udata->output[i] = 0;
            }
        }
    }

    // 5. Track processed sequences and mark tokens as used when fully covered
    for (const auto & sid : selected_seqs) {
        expand_processed_seqs.insert(sid);
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (used[i]) {
            continue;
        }
        // A token is fully used when all its seq_ids have been processed
        bool all_processed = true;
        for (int s = 0; s < batch.n_seq_id[i]; ++s) {
            if (expand_processed_seqs.count(batch.seq_id[i][s]) == 0) {
                all_processed = false;
                break;
            }
        }
        if (all_processed) {
            used[i] = true;
            ++n_used;
        }
    }

    // 6. Set up seq_id pointers
    llama_seq_id * seq_id_ptr = udata->seq_id_data.data();
    for (uint32_t i = 0; i < n_tokens_expanded; ++i) {
        udata->seq_id[i] = seq_id_ptr;
        seq_id_ptr += 1; // n_seq_id is always 1
    }

    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (seq_set_unq.test(s)) {
            udata->seq_idx[s] = udata->seq_id_unq.size();
            udata->seq_id_unq.push_back(s);
        }
    }

    llama_ubatch res {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ n_tokens_expanded,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.n_seqs_unq   =*/ (uint32_t) udata->seq_id_unq.size(),
        /*.n_pos        =*/ n_pos_per_embd,

        /*.token        =*/ batch.token ? udata->token.data() : nullptr,
        /*.embd         =*/ batch.embd ? udata->embd.data() : nullptr,
        /*.pos          =*/ udata->pos.data(),
        /*.kv_slot      =*/ batch.kv_slot ? udata->kv_slot.data() : nullptr,
        /*.n_seq_id     =*/ udata->n_seq_id.data(),
        /*.seq_id       =*/ udata->seq_id.data(),
        /*.seq_id_unq   =*/ udata->seq_id_unq.data(),
        /*.seq_idx      =*/ udata->seq_idx.data(),
        /*.output       =*/ udata->output.data(),
        /*.rope_pos     =*/ nullptr,
        /*.data         =*/ std::move(udata),
    };

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: added expanded ubatch (coupled → per-seq) to split:\n", __func__);
        ubatch_print(res, debug);
    }

    return res;
}

llama_ubatch llama_batch_allocr::split_seq(uint32_t n_ubatch) {
    // find the first unused token
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }

    // we are done
    if (cur_idx >= used.size()) {
        return {};
    }

    // this is the starting sequence set
    // we allow adding tokens only if their sequence set is a subset of the current sequence set
    auto cur_seq_set = seq_set[cur_idx];

    std::vector<int32_t> idxs;

    while (true) {
        idxs.push_back(cur_idx);

        used[cur_idx] = true;
        ++n_used;

        if (idxs.size() >= n_ubatch) {
            break;
        }

        do {
            ++cur_idx;
        } while (cur_idx < get_n_tokens() && (used[cur_idx] || ((cur_seq_set & seq_set[cur_idx]) != seq_set[cur_idx])));

        if (cur_idx == get_n_tokens()) {
            break;
        }

        cur_seq_set = seq_set[cur_idx];
    }

    return ubatch_add(idxs, 1, true);
}

void llama_batch_allocr::clear() {
    n_outputs = 0;

    batch = {};

    pos       .clear();
    n_seq_id  .clear();
    kv_slot   .clear();
    seq_id    .clear();
    seq_id_unq.clear();
    output    .clear();

    for (auto & cur : seq_pos) {
        cur.clear();
    }

    for (auto & cur : seq_cpl) {
        std::fill(cur.begin(), cur.end(), false);
    }

    seq_set.clear();

    seq_set_map.clear();

    std::fill(seq_idx.begin(), seq_idx.end(), -1);
}

llama_ubatch llama_batch_allocr::ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs) {
    const uint32_t n_tokens = idxs.size();

    assert(n_tokens%n_seqs == 0);

    auto udata = std::make_shared<llama_ubatch::data_t>();

    const int64_t n_embd_all = batch.embd ? (int64_t) n_tokens*n_embd : 0;
    const int64_t n_pos_all  =              (int64_t) n_tokens*n_pos_per_embd;

    udata->token     .resize(n_tokens);
    udata->embd      .resize(n_embd_all);
    udata->pos       .resize(n_pos_all);
    udata->kv_slot   .resize(n_tokens, -1);
    udata->n_seq_id  .resize(n_tokens);
    udata->seq_id    .resize(n_tokens);
    udata->seq_id_unq.resize(0);
    udata->seq_idx   .resize(LLAMA_MAX_SEQ, -1);
    udata->output    .resize(n_tokens);

    udata->seq_id_data.reserve(n_tokens);

    seq_set_t seq_set_unq;

    for (size_t i = 0; i < idxs.size(); ++i) {
        if (batch.token) {
            udata->token[i] = batch.token[idxs[i]];
        }

        if (batch.embd) {
            memcpy(udata->embd.data() + i*n_embd, batch.embd + (int64_t) idxs[i]*n_embd, n_embd*sizeof(float));
        }

        for (size_t j = 0; j < (size_t)n_pos_per_embd; ++j) {
            // if we are using M-RoPE
            //     if the current batch is text, we need to broadcast the same position across all RoPE sections
            //     otherwise, the input batch is image embeddings, we copy the positions as-is
            // if we are not using M-RoPE, there is only one position per token (this loop runs only once)
            size_t src_off = batch.token ? 0 : j*batch.n_tokens;
            udata->pos[j*n_tokens + i] = batch.pos[src_off + idxs[i]];
        }

        udata->n_seq_id[i] = batch.n_seq_id[idxs[i]];
        if (batch.kv_slot) {
            udata->kv_slot[i] = batch.kv_slot[idxs[i]];
        }
        udata->output[i]   = batch.logits[idxs[i]];

        for (int s = 0; s < udata->n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = batch.seq_id[idxs[i]][s];

            udata->seq_id_data.push_back(seq_id);
            seq_set_unq.set(seq_id);
        }

        if (udata->output[i]) {
            out_ids.push_back(idxs[i]);
        }
    }

    llama_seq_id * seq_id_ptr = udata->seq_id_data.data();
    for (size_t i = 0; i < idxs.size(); ++i) {
        udata->seq_id[i] = seq_id_ptr;
        seq_id_ptr += udata->n_seq_id[i];
    }

    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (seq_set_unq.test(s)) {
            udata->seq_idx[s] = udata->seq_id_unq.size();
            udata->seq_id_unq.push_back(s);
        }
    }

    llama_ubatch res {
        /*.b_equal_seqs =*/ equal_seqs,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_tokens/n_seqs,
        /*.n_seqs       =*/ n_seqs,
        /*.n_seqs_unq   =*/ (uint32_t) udata->seq_id_unq.size(),
        /*.n_pos        =*/ n_pos_per_embd,

        /*.token        =*/ batch.token ? udata->token.data() : nullptr,
        /*.embd         =*/ batch.embd ? udata->embd.data() : nullptr,
        /*.pos          =*/ udata->pos.data(),
        /*.kv_slot      =*/ batch.kv_slot ? udata->kv_slot.data() : nullptr,
        /*.n_seq_id     =*/ udata->n_seq_id.data(),
        /*.seq_id       =*/ udata->seq_id.data(),
        /*.seq_id_unq   =*/ udata->seq_id_unq.data(),
        /*.seq_idx      =*/ udata->seq_idx.data(),
        /*.output       =*/ udata->output.data(),
        /*.rope_pos     =*/ nullptr,
        /*.data         =*/ std::move(udata),
    };

    // propagate rope_pos override if set
    if (!rope_pos_override.empty()) {
        auto & rdata = res.data; // shared_ptr still accessible via res
        rdata->rope_pos.resize(n_pos_all);
        for (size_t i = 0; i < idxs.size(); ++i) {
            for (size_t j = 0; j < (size_t)n_pos_per_embd; ++j) {
                rdata->rope_pos[j*n_tokens + i] = rope_pos_override[idxs[i]];
            }
        }
        res.rope_pos = rdata->rope_pos.data();
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: added ubatch to split:\n", __func__);

        ubatch_print(res, debug);
    }

    return res;
}

void llama_batch_allocr::ubatch_print(const llama_ubatch & ubatch, int debug) {
    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s:   equal_seqs   = %d\n", __func__, ubatch.equal_seqs());
        LLAMA_LOG_DEBUG("%s:   n_tokens     = %d\n", __func__, ubatch.n_tokens);
        LLAMA_LOG_DEBUG("%s:   n_seq_tokens = %d\n", __func__, ubatch.n_seq_tokens);
        LLAMA_LOG_DEBUG("%s:   n_seqs       = %d\n", __func__, ubatch.n_seqs);
        LLAMA_LOG_DEBUG("%s:   n_seqs_unq   = %d\n", __func__, ubatch.n_seqs_unq);

        std::stringstream ss_seq_id_unq;
        std::stringstream ss_seq_idx;

        ss_seq_id_unq << "[ ";
        ss_seq_idx << "[";

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            ss_seq_id_unq << ubatch.seq_id_unq[s] << " ";
        }

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (ubatch.seq_idx[s] >= 0) {
                ss_seq_idx << ubatch.seq_idx[s]%10;
            } else {
                ss_seq_idx << ".";
            }
        }

        ss_seq_id_unq << "]";
        ss_seq_idx    << "]";

        LLAMA_LOG_DEBUG("%s:   token      = %p\n", __func__, (void *) ubatch.token);
        LLAMA_LOG_DEBUG("%s:   embd       = %p\n", __func__, (void *) ubatch.embd);
        LLAMA_LOG_DEBUG("%s:   pos        = %p\n", __func__, (void *) ubatch.pos);
        LLAMA_LOG_DEBUG("%s:   kv_slot    = %p\n", __func__, (void *) ubatch.kv_slot);
        LLAMA_LOG_DEBUG("%s:   n_seq_id   = %p\n", __func__, (void *) ubatch.n_seq_id);
        LLAMA_LOG_DEBUG("%s:   seq_id     = %p\n", __func__, (void *) ubatch.seq_id);
        LLAMA_LOG_DEBUG("%s:   seq_id_unq = %s\n", __func__, ss_seq_id_unq.str().c_str());
        LLAMA_LOG_DEBUG("%s:   seq_idx    = %s\n", __func__, ss_seq_idx.str().c_str());
        LLAMA_LOG_DEBUG("%s:   output     = %p\n", __func__, (void *) ubatch.output);
        LLAMA_LOG_DEBUG("%s:   n_outputs  = %d\n", __func__, n_outputs);

        if (debug > 1) {
            int seq_id_max = 0;
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                    for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                        seq_id_max = std::max(seq_id_max, ubatch.seq_id[i][s]);
                    }
                }
            }
            ++seq_id_max;

            LLAMA_LOG_DEBUG("%s:   token     = [\n", __func__);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                std::vector<int8_t> seq_id(seq_id_max);

                for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                    seq_id[ubatch.seq_id[i][s]] = 1;
                }

                std::stringstream ss;
                for (int s = 0; s < seq_id_max; ++s) {
                    if (seq_id[s]) {
                        ss << s%10;
                    } else {
                        ss << ".";
                    }
                }

                if (ubatch.token) {
                    LLAMA_LOG_DEBUG("%s:  %4d: id = %6d (%16s), pos = %4d, kv_slot = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                            __func__, i, ubatch.token[i], vocab->token_to_piece(ubatch.token[i]).c_str(),
                            ubatch.pos[i], ubatch.kv_slot ? ubatch.kv_slot[i] : -1, ubatch.n_seq_id[i], ss.str().c_str(), ubatch.output[i]);
                } else {
                    LLAMA_LOG_DEBUG("%s:  %4d: [embd], pos = %4d, kv_slot = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                            __func__, i, ubatch.pos[i], ubatch.kv_slot ? ubatch.kv_slot[i] : -1, ubatch.n_seq_id[i], ss.str().c_str(), ubatch.output[i]);
                }
            }
            LLAMA_LOG_DEBUG("%s:   ]\n", __func__);
        }
    }
}

void llama_batch_allocr::set_rope_pos_override(const llama_pos * data, uint32_t n) {
    rope_pos_override.assign(data, data + n);
}

void llama_batch_allocr::clear_rope_pos_override() {
    rope_pos_override.clear();
}

//
// interface implementation
//

struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens) {
    return {
        /*n_tokens =*/ n_tokens,
        /*tokens   =*/ tokens,
        /*embd     =*/ nullptr,
        /*pos      =*/ nullptr,
        /*kv_slot  =*/ nullptr,
        /*n_seq_id =*/ nullptr,
        /*seq_id   =*/ nullptr,
        /*logits   =*/ nullptr,
    };
}

struct llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    llama_batch batch = {
        /*n_tokens =*/ 0,
        /*tokens   =*/ nullptr,
        /*embd     =*/ nullptr,
        /*pos      =*/ nullptr,
        /*kv_slot  =*/ nullptr,
        /*n_seq_id =*/ nullptr,
        /*seq_id   =*/ nullptr,
        /*logits   =*/ nullptr,
    };

    if (embd) {
        batch.embd = (float *) malloc(sizeof(float) * n_tokens_alloc * embd);
    } else {
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens_alloc);
    }

    batch.pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens_alloc);
    batch.kv_slot  = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens_alloc);
    batch.n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens_alloc);
    batch.seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * (n_tokens_alloc + 1));
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch.seq_id[n_tokens_alloc] = nullptr;

    batch.logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens_alloc);
    std::fill_n(batch.kv_slot, n_tokens_alloc, -1);

    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.embd)     free(batch.embd);
    if (batch.pos)      free(batch.pos);
    if (batch.kv_slot)  free(batch.kv_slot);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i] != nullptr; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}
