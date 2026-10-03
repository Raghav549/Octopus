// Octopus Hybrid AI Engine -- local LLM host layer.
//
// Wraps llama.cpp when the engine was configured with -DOCT_WITH_LLAMA=ON and a
// prebuilt llama.cpp tree was found; otherwise it exposes a deterministic stub
// backend that is *labelled as such* everywhere (backend name, JSON output,
// module limitations). The engine never pretends a model answered when none was
// loaded: generate() returns Status::unavailable in that case, and the CLI
// reports it as an unavailable capability rather than inventing an answer.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"
#include "octopus/gguf.hpp"
#include "octopus/tokenizer.hpp"

namespace oct { class Module; }

namespace oct::llm {

struct ModelFacts {
    std::string architecture;
    std::string name;
    int64_t     n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0;
    int64_t     n_ff = 0, n_vocab = 0, n_ctx_train = 0;
    int64_t     param_count = 0;
    uint64_t    weight_bytes = 0;
    double      bpw = 0.0;
    bool        has_tokenizer = false;
    std::string file_type;
    Json        to_json() const;
};

struct GenerateParams {
    int64_t max_tokens = 64;
    double  temperature = 0.0;     // 0 = greedy
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    int64_t top_k = 0;             // 0 = disabled
    double  top_p = 1.0;
};

struct BackendInfo {
    bool        compiled = false;      // OCT_WITH_LLAMA
    bool        available = false;     // library present and loadable
    bool        stub = true;           // true => deterministic stub, not a model
    std::string name;
    std::string version;
    std::string honesty;               // one-line truthful description
    Json        to_json() const;
};

class Host {
public:
    Host();

    // Reads model metadata with the engine's own GGUF reader (no llama.cpp
    // needed) so capability reporting works even without the backend.
    Outcome<ModelFacts> inspect(const std::string& path) const;

    // Loads a model for generation. Without the llama.cpp backend this fails
    // with Status::unavailable (unless the deterministic stub is enabled).
    Status load(const std::string& path, bool allow_stub_fallback = false);
    bool   loaded() const { return loaded_; }
    bool   using_stub() const { return stub_mode_; }
    const ModelFacts& facts() const { return facts_; }
    const std::string& path() const { return path_; }

    Outcome<std::vector<uint32_t>> tokenize(std::string_view text) const;
    Outcome<std::string> detokenize(std::span<const uint32_t> ids) const;
    Outcome<std::string> generate(std::string_view prompt, const GenerateParams& params);

    BackendInfo backend() const;
    Json        report() const;

private:
    bool         loaded_ = false;
    bool         stub_mode_ = false;
    std::string  path_;
    ModelFacts   facts_;
    tokenizer::Vocab vocab_;
    bool         have_vocab_ = false;
    Rng          rng_;
    // Opaque handles for the optional llama.cpp backend (defined only when
    // OCT_HAVE_LLAMA is set; kept as void* so this header has no dependency).
    void*        llama_model_ = nullptr;
    void*        llama_context_ = nullptr;
};

std::shared_ptr<oct::Module> make_llm_module();

}  // namespace oct::llm
