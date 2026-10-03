// Octopus Hybrid AI Engine -- tokenizer layer.
//
// Loads a vocabulary straight out of a GGUF file (no hard-coded tables). For
// byte-level BPE vocabularies (tokenizer.ggml.model = "gpt2" with a
// tokenizer.ggml.merges array) the encoder implements the exact GPT-2
// byte-pair algorithm, so its output can be compared token-for-token against
// the reference `.inp`/`.out` pairs shipped by llama.cpp.
//
// Honest scope:
//   * Byte-level BPE is exact and validated against those fixtures.
//   * SentencePiece-style vocabularies (model = "llama"/"spm", no merges) fall
//     back to a greedy longest-match encoder, which is NOT the reference
//     tokenizer; the fallback is reported as degraded and its measured match
//     rate against any available fixture is reported instead of claimed.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"
#include "octopus/gguf.hpp"

#include <map>
#include <memory>
#include <span>

namespace oct { class Module; }

namespace oct::tokenizer {

struct BpeTable;   // opaque merge-rank table, defined in the implementation

struct Vocab {
    std::string model;                       // "gpt2", "llama", "bert", ...
    std::string pre;                         // tokenizer.ggml.pre when present
    std::vector<std::string> tokens;         // id -> piece
    std::vector<std::string> merges;         // "A B" in rank order
    std::map<std::string, uint32_t> index;   // piece -> id
    uint32_t bos_id = UINT32_MAX, eos_id = UINT32_MAX, pad_id = UINT32_MAX;
    bool exact_bpe = false;                  // true only for the exact algorithm path

    // Merge ranks are built lazily on first use and shared by copies of this
    // vocabulary. They are keyed to the token list above, so a re-used pointer
    // address can never return another vocabulary's ranks.
    mutable std::shared_ptr<const BpeTable> ranks;

    size_t size() const { return tokens.size(); }
    static Outcome<Vocab> from_gguf(const gguf::ModelInfo& info);
    static Outcome<Vocab> load(const std::string& path);
};

struct Encoding {
    std::vector<uint32_t> ids;
    std::string method;      // "bpe.byte_level.exact" | "greedy.longest_match.degraded"
    bool        degraded = false;
    std::string warnings;
};

Outcome<Encoding> encode(const Vocab& v, std::string_view text);
Outcome<std::string> decode(const Vocab& v, std::span<const uint32_t> ids);

// Comparison against the upstream fixture format: `.inp` segments separated by
// "\n__ggml_vocab_test__\n" and one line of expected ids per segment in `.out`.
struct ReferenceCheck {
    size_t cases = 0;
    size_t exact_cases = 0;
    size_t token_matches = 0;
    size_t token_total = 0;
    std::vector<std::pair<std::string, double>> per_case;   // debug aid (first 8)
    double case_match_rate() const { return cases ? double(exact_cases) / double(cases) : 0.0; }
    double token_match_rate() const { return token_total ? double(token_matches) / double(token_total) : 0.0; }
    Json to_json() const;
};

Outcome<ReferenceCheck> check_against_fixture(const Vocab& v, const std::string& inp_path,
                                              const std::string& out_path);

std::shared_ptr<oct::Module> make_tokenizer_module();

}  // namespace oct::tokenizer
