// Octopus Hybrid AI Engine -- synthetic tiny-model generator.
//
// Writes a complete, loadable llama-architecture GGUF file whose weights are
// deterministic pseudo-random numbers. Its purpose is to let the *real*
// inference path (llama.cpp load -> tokenize -> decode -> sample ->
// detokenize) be exercised offline on any machine, with no downloaded weights.
//
// HONESTY CONTRACT: a model produced here emits nonsense. It demonstrates that
// the execution path runs; it says nothing about model quality, and the file
// carries metadata (`octopus.weights = synthetic-pseudo-random`) so that no
// downstream consumer can mistake it for a trained model.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <string>

namespace oct::synthetic {

struct ModelSpec {
    int64_t n_layer = 2;
    int64_t n_embd = 64;
    int64_t n_head = 4;
    int64_t n_head_kv = 4;
    int64_t n_ff = 128;
    int64_t n_ctx = 128;
    uint64_t seed = 1;
    double weight_scale = 0.02;
};

struct WrittenModel {
    std::string path;
    int64_t layers = 0, embd = 0, heads = 0, ff = 0;
    uint32_t vocab = 0;
    size_t bytes = 0;
    std::string sha256;
    Json to_json() const;
};

// Vocabulary layout produced: 256 byte-level BPE tokens (GPT-2 byte table),
// then <unk>, <s>, </s>.
Outcome<WrittenModel> write_model(const std::string& path, const ModelSpec& spec = {});

}  // namespace oct::synthetic
