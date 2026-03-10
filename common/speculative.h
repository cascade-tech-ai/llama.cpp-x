#pragma once
// AI-GENERATED: This file was modified with AI assistance for an experimental fork.
// DO NOT SUBMIT upstream unless rewritten or exhaustively reviewed by a human.

#include "llama.h"
#include "common.h"

#include <vector>

struct common_speculative_tree {
    uint32_t batch_start = 1; // index of the first draft token in the target batch
    std::vector<llama_token> tokens;
    std::vector<int32_t> parents; // -1 for root nodes
    std::vector<int32_t> depths;
    std::vector<uint32_t> row_indices; // optional explicit target batch row per logical node
    std::vector<int32_t> first_child;  // compact child chain, -1 if none
    std::vector<int32_t> next_sibling; // compact sibling chain, -1 if none
    std::vector<uint32_t> leaf_masks;  // descendant leaf bitmask for each node
    uint32_t leaf_count = 0;

    void clear() {
        tokens.clear();
        parents.clear();
        depths.clear();
        row_indices.clear();
        first_child.clear();
        next_sibling.clear();
        leaf_masks.clear();
        leaf_count = 0;
        batch_start = 1;
    }

    size_t size() const {
        return tokens.size();
    }
};

struct common_speculative_trace_node {
    llama_token token = LLAMA_TOKEN_NULL;
    float prob = 0.0f;
    float cum_prob = 0.0f;
};

struct common_speculative_trace {
    std::vector<llama_tokens> proposal_paths;
    std::vector<std::vector<common_speculative_trace_node>> proposal_graph;

    void clear() {
        proposal_paths.clear();
        proposal_graph.clear();
    }
};

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt, llama_seq_id seq_id = 0);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_seq_id  seq_id = 0);

// retrieve the latest speculative tree (if the current implementation supports it)
bool common_speculative_get_tree(common_speculative * spec, common_speculative_tree & out);

// retrieve the latest speculative proposal trace, if available
bool common_speculative_get_trace(common_speculative * spec, common_speculative_trace & out);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// informs the speculative decoder of the exact accepted token sequence for the most recent tree pass
void common_speculative_accept_tokens(common_speculative * spec, const llama_tokens & ids, llama_seq_id seq_id = 0);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);
