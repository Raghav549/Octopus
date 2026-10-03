// Tokenizer tests: GGUF vocabulary loading and exact comparison against the
// reference fixtures shipped with llama.cpp (`.inp` / `.out` pairs).
// SPDX-License-Identifier: MIT
#include "harness.hpp"

#include "octopus/tokenizer.hpp"

#include <fstream>
#include <string>
#include <vector>

using namespace oct;

namespace {

std::string model_dir() {
    // The llama.cpp checkout is part of the source tree; when it is absent the
    // fixture-based tests skip rather than silently passing.
    const char* candidates[] = {
        "third_party/llama.cpp/models/",
        "../third_party/llama.cpp/models/",
        "../../third_party/llama.cpp/models/",
    };
    for (const char* c : candidates) {
        std::ifstream probe(std::string(c) + "ggml-vocab-gpt-2.gguf");
        if (probe) return c;
    }
    return {};
}

}  // namespace

OCT_TEST(tokenizer, reads_vocabulary_from_the_gguf_file) {
    const std::string dir = model_dir();
    if (dir.empty()) OCT_SKIP("llama.cpp vocabulary fixtures not present");
    auto vocab = tokenizer::Vocab::load(dir + "ggml-vocab-gpt-2.gguf");
    OCT_CHECK(bool(vocab));
    if (!vocab) OCT_SKIP("cannot load fixture: " + vocab.status.message);
    OCT_CHECK(vocab->model == "gpt2");
    OCT_CHECK(vocab->size() >= 32000);
    OCT_CHECK(vocab->index.count("the") == 1);
    OCT_CHECK(vocab->exact_bpe);          // merges present -> exact algorithm path
    OCT_CHECK(vocab->pre.empty() || vocab->pre == "gpt-2" || vocab->pre == "gpt2");
}

OCT_TEST(tokenizer, gpt2_matches_the_reference_fixture_exactly) {
    const std::string dir = model_dir();
    if (dir.empty()) OCT_SKIP("llama.cpp vocabulary fixtures not present");
    const std::string base = dir + "ggml-vocab-gpt-2.gguf";
    auto vocab = tokenizer::Vocab::load(base);
    OCT_CHECK(bool(vocab));
    if (!vocab) OCT_SKIP("cannot load fixture: " + vocab.status.message);
    auto rc = tokenizer::check_against_fixture(*vocab, base + ".inp", base + ".out");
    OCT_CHECK(bool(rc));
    if (!rc) OCT_SKIP("fixture unavailable: " + rc.status.message);
    OCT_NOTE("gpt2 fixture: cases=" << rc->cases << " exact=" << rc->exact_cases
             << " tokens=" << rc->token_matches << "/" << rc->token_total);
    OCT_CHECK(rc->cases >= 40);
    // Byte-level BPE plus the faithful GPT-2 pre-tokenizer must reproduce the
    // reference ids: this is an equality claim, not a similarity threshold.
    OCT_EQ(rc->exact_cases, rc->cases);
    OCT_EQ(rc->token_matches, rc->token_total);
    OCT_NEAR(rc->case_match_rate(), 1.0, 0.0);
    OCT_NEAR(rc->token_match_rate(), 1.0, 0.0);
}

OCT_TEST(tokenizer, degraded_paths_are_labelled_and_measured) {
    const std::string dir = model_dir();
    if (dir.empty()) OCT_SKIP("llama.cpp vocabulary fixtures not present");
    const std::string base = dir + "ggml-vocab-deepseek-llm.gguf";
    auto vocab = tokenizer::Vocab::load(base);
    OCT_CHECK(bool(vocab));
    if (!vocab) OCT_SKIP("cannot load fixture: " + vocab.status.message);
    OCT_CHECK(!vocab->pre.empty());       // this vocabulary overrides the pre-tokenizer
    auto enc = tokenizer::encode(*vocab, "hello world");
    OCT_CHECK(bool(enc));
    if (enc) {
        // The encoder must say that it is not the reference implementation.
        OCT_CHECK(enc->degraded);
        OCT_CHECK(enc->method.find("degraded") != std::string::npos);
        OCT_CHECK(!enc->ids.empty());
    }
    auto rc = tokenizer::check_against_fixture(*vocab, base + ".inp", base + ".out");
    OCT_CHECK(bool(rc));                  // the measurement itself must run
    if (rc) {
        OCT_NOTE("deepseek-llm fixture: case_match_rate=" << rc->case_match_rate()
                 << " token_match_rate=" << rc->token_match_rate());
        OCT_CHECK(rc->cases > 0);
        OCT_CHECK(rc->case_match_rate() >= 0.0 && rc->case_match_rate() <= 1.0);
        // Honesty gate: an unsupported pre-tokenizer cannot score a perfect match.
        OCT_CHECK(rc->case_match_rate() < 1.0);
    }
}

OCT_TEST(tokenizer, decode_round_trips_through_the_vocabulary) {
    tokenizer::Vocab v;
    v.model = "gpt2";
    v.tokens = {"a", "b", "ab", "c"};
    v.merges = {"a b"};
    for (size_t i = 0; i < v.tokens.size(); ++i) v.index.emplace(v.tokens[i], uint32_t(i));
    v.exact_bpe = true;

    auto enc = tokenizer::encode(v, "abab");
    OCT_CHECK(bool(enc));
    if (enc) {
        OCT_EQ(enc->ids.size(), size_t(2));
        OCT_EQ(enc->ids[0], uint32_t(2));
        OCT_EQ(enc->ids[1], uint32_t(2));
        auto back = tokenizer::decode(v, enc->ids);
        OCT_CHECK(bool(back));
        if (back) OCT_CHECK(*back == "abab");
    }
    const std::vector<uint32_t> bogus{99};
    auto out_of_range = tokenizer::decode(v, bogus);
    OCT_CHECK(!out_of_range);             // unknown ids are rejected, not substituted
}
