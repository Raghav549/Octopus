// Octopus Hybrid AI Engine -- local LLM host layer.
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// Optional C++ stub fallbacks (`allow_stub_fallback`, `"stub-answer{...}"`)
// are strictly banned. GGUF metadata inspection and vocabulary tokenization
// use the engine's native GGUF reader and BPE tokenizer, while live token
// generation requires the native llama.cpp inference backend. When the
// llama.cpp backend or hardware is absent on the build host, live inference
// blocks return Status::hardware_skip ("UNSUPPORTED_HARDWARE_SKIP").
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
    int64_t  max_tokens = 64;
    double   temperature = 0.0;     // 0 = greedy
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    int64_t  top_k = 0;             // 0 = disabled
    double   top_p = 1.0;
};

struct BackendInfo {
    bool        compiled = false;
    bool        available = false;
    bool        hardware_skipped = false;
    std::string name;
    std::string version;
    std::string honesty;
    Json        to_json() const;
};

class Host {
public:
    Host();
    ~Host();
    Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;

    // Reads model metadata with the engine's own GGUF reader.
    Outcome<ModelFacts> inspect(const std::string& path) const;

    // Loads a model for generation. Without the llama.cpp backend this returns
    // Status::hardware_skip ("UNSUPPORTED_HARDWARE_SKIP"); C++ stub fallbacks
    // are strictly banned.
    Status load(const std::string& path);
    bool   loaded() const { return loaded_; }
    const ModelFacts& facts() const { return facts_; }
    const std::string& path() const { return path_; }

    Outcome<std::vector<uint32_t>> tokenize(std::string_view text) const;
    Outcome<std::string> detokenize(std::span<const uint32_t> ids) const;
    Outcome<std::string> generate(std::string_view prompt, const GenerateParams& params);

    BackendInfo backend() const;
    Json        report() const;

private:
    bool             loaded_ = false;
    std::string      path_;
    ModelFacts       facts_;
    tokenizer::Vocab vocab_;
    bool             have_vocab_ = false;
    void*            llama_model_ = nullptr;
    void*            llama_context_ = nullptr;
};

std::shared_ptr<oct::Module> make_llm_module();

}  // namespace oct::llm
