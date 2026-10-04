// Octopus Hybrid AI Engine -- LLM host (llama.cpp bridge, zero stub fallbacks).
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// All C++ stub fallbacks (`allow_stub_fallback`, `"stub-answer{...}"`) have
// been purged. When the prebuilt llama.cpp runtime is absent on the build host,
// live inference blocks explicitly report UNSUPPORTED_HARDWARE_SKIP.
// SPDX-License-Identifier: MIT
#include "octopus/llm.hpp"

#include "octopus/module.hpp"

#include <climits>
#include <cmath>
#include <sstream>

#if defined(OCT_HAVE_LLAMA)
#include <llama.h>
#endif

namespace oct::llm {

Json ModelFacts::to_json() const {
    Json j;
    j.begin_object();
    j.field("architecture", architecture);
    j.field("name", name);
    j.field("n_layer", n_layer);
    j.field("n_embd", n_embd);
    j.field("n_head", n_head);
    j.field("n_head_kv", n_head_kv);
    j.field("n_ff", n_ff);
    j.field("n_vocab", n_vocab);
    j.field("n_ctx_train", n_ctx_train);
    j.field("parameter_count", param_count);
    j.field("weight_bytes", weight_bytes);
    j.field("bits_per_weight", bpw);
    j.field("has_tokenizer", has_tokenizer);
    j.field("file_type", file_type);
    j.end_object();
    return j;
}

Json BackendInfo::to_json() const {
    Json j;
    j.begin_object();
    j.field("compiled", compiled);
    j.field("available", available);
    j.field("hardware_skipped", hardware_skipped);
    j.field("name", name);
    j.field("version", version);
    j.field("honesty", honesty);
    j.end_object();
    return j;
}

Host::Host() = default;

Host::~Host() {
#if defined(OCT_HAVE_LLAMA)
    if (llama_context_) llama_free(static_cast<llama_context*>(llama_context_));
    if (llama_model_)   llama_model_free(static_cast<llama_model*>(llama_model_));
#endif
}

BackendInfo Host::backend() const {
    BackendInfo b;
#if defined(OCT_HAVE_LLAMA)
    b.compiled = true;
    b.available = true;
    b.hardware_skipped = false;
    b.name = "llama.cpp";
    b.version = "b11371";
    b.honesty = loaded_
        ? "llama.cpp static backend linked and a model is loaded; generation runs natively on this machine"
        : "llama.cpp static backend linked; supply a real GGUF model file to generate tokens";
#else
    b.compiled = false;
    b.available = false;
    b.hardware_skipped = true;
    b.name = "UNSUPPORTED_HARDWARE_SKIP";
    b.version = kVersionString;
    b.honesty = "UNSUPPORTED_HARDWARE_SKIP: llama.cpp runtime absent on build host; C++ stub fallback is strictly banned";
#endif
    return b;
}

Outcome<ModelFacts> Host::inspect(const std::string& path) const {
    auto info = gguf::read_model_info(path);
    if (!info) return info.status;
    const auto facts = info->facts();
    ModelFacts f;
    f.architecture = facts.architecture;
    f.name = facts.name;
    f.n_layer = facts.n_layer;
    f.n_embd = facts.n_embd;
    f.n_head = facts.n_head;
    f.n_head_kv = facts.n_head_kv;
    f.n_ff = facts.n_ff;
    f.n_vocab = facts.n_vocab;
    f.n_ctx_train = facts.n_ctx_train;
    f.param_count = facts.param_count;
    f.weight_bytes = facts.weight_bytes;
    f.bpw = facts.bpw;
    f.has_tokenizer = facts.has_tokenizer;
    if (auto ft = info->get_str("general.file_type")) f.file_type = *ft;
    return f;
}

Status Host::load(const std::string& path) {
    auto f = inspect(path);
    if (!f) return f.status;
    facts_ = *f;
    path_ = path;
    auto vocab = tokenizer::Vocab::load(path);
    if (vocab) {
        vocab_ = *vocab;
        have_vocab_ = true;
    }
#if defined(OCT_HAVE_LLAMA)
    llama_backend_init();
    auto params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(path.c_str(), params);
    if (model) {
        llama_model_ = model;
        auto cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        llama_context_ = llama_init_from_model(model, cparams);
        if (llama_context_) {
            loaded_ = true;
            return Status::ok();
        }
        llama_model_free(model);
        llama_model_ = nullptr;
    }
    return Status::unavailable("llm: llama.cpp failed to load '" + path + "'");
#else
    return Status::hardware_skip(
        "UNSUPPORTED_HARDWARE_SKIP: llama.cpp inference backend is absent on this host; "
        "C++ stub fallback is strictly banned");
#endif
}

Outcome<std::vector<uint32_t>> Host::tokenize(std::string_view text) const {
    if (!have_vocab_) return Status::unavailable("llm: no vocabulary loaded");
    auto enc = tokenizer::encode(vocab_, text);
    if (!enc) return enc.status;
    return enc->ids;
}

Outcome<std::string> Host::detokenize(std::span<const uint32_t> ids) const {
    if (!have_vocab_) return Status::unavailable("llm: no vocabulary loaded");
    return tokenizer::decode(vocab_, ids);
}

#if defined(OCT_HAVE_LLAMA)
namespace {

std::string llama_generate_impl(llama_model* model, llama_context* ctx, const std::string& prompt,
                                const GenerateParams& p, std::string* error) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        *error = "model has no vocabulary";
        return {};
    }

    int32_t needed = llama_tokenize(vocab, prompt.c_str(), int32_t(prompt.size()), nullptr, 0, true,
                                    true);
    if (needed == INT32_MIN) {
        *error = "prompt is too long to tokenise";
        return {};
    }
    if (needed < 0) needed = -needed;
    std::vector<llama_token> tokens(size_t(std::max(1, needed)));
    const int32_t n_tok = llama_tokenize(vocab, prompt.c_str(), int32_t(prompt.size()),
                                         tokens.data(), int32_t(tokens.size()), true, true);
    if (n_tok < 0) {
        *error = "tokenisation failed (needed " + std::to_string(-n_tok) + " tokens)";
        return {};
    }
    tokens.resize(size_t(n_tok));
    if (tokens.empty()) {
        *error = "prompt produced no tokens";
        return {};
    }
    if (int64_t(tokens.size()) >= int64_t(llama_n_ctx(ctx))) {
        *error = "prompt does not fit in the context (" + std::to_string(tokens.size()) + " >= " +
                 std::to_string(llama_n_ctx(ctx)) + ")";
        return {};
    }

    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!chain) {
        *error = "could not create a sampler chain";
        return {};
    }
    if (p.temperature <= 0.0) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        if (p.top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(int32_t(p.top_k)));
        if (p.top_p < 1.0)
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(float(p.top_p), 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(float(p.temperature)));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(uint32_t(p.seed)));
    }

    if (llama_memory_t mem = llama_get_memory(ctx)) llama_memory_clear(mem, /*data=*/true);

    llama_batch batch = llama_batch_get_one(tokens.data(), int32_t(tokens.size()));
    if (llama_decode(ctx, batch) != 0) {
        llama_sampler_free(chain);
        *error = "prefill decode failed (prompt may not fit the context)";
        return {};
    }

    std::string out;
    for (int64_t i = 0; i < p.max_tokens; ++i) {
        const llama_token id = llama_sampler_sample(chain, ctx, -1);
        if (id == llama_vocab_eos(vocab)) break;
        char buf[256];
        int32_t n = llama_token_to_piece(vocab, id, buf, int32_t(sizeof(buf)), 0, true);
        if (n < 0) {
            std::string big(size_t(-n), '\0');
            n = llama_token_to_piece(vocab, id, big.data(), int32_t(big.size()), 0, true);
            if (n < 0) break;
            out.append(big.data(), size_t(n));
        } else {
            out.append(buf, size_t(n));
        }
        llama_token next = id;
        batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) {
            *error = "decode failed at token " + std::to_string(i);
            break;
        }
    }
    llama_sampler_free(chain);
    return out;
}

}  // namespace
#endif

Outcome<std::string> Host::generate(std::string_view prompt, const GenerateParams& params) {
#if defined(OCT_HAVE_LLAMA)
    if (!loaded_ || !llama_model_ || !llama_context_)
        return Status::unavailable("llm: no model loaded; call load() first");
    std::string error;
    const std::string text = llama_generate_impl(static_cast<llama_model*>(llama_model_),
                                                 static_cast<llama_context*>(llama_context_),
                                                 std::string(prompt), params, &error);
    if (text.empty())
        return Status::degraded("llm: generation produced no tokens" +
                                (error.empty() ? std::string() : std::string(" (") + error + ")"));
    return text;
#else
    (void)prompt;
    (void)params;
    return Status::hardware_skip(
        "UNSUPPORTED_HARDWARE_SKIP: llama.cpp inference backend is absent on this host; "
        "C++ stub fallback is strictly banned");
#endif
}

Json Host::report() const {
    Json j;
    j.begin_object();
    j.field("loaded", loaded_);
    if (!path_.empty()) j.field("model_path", path_);
    j.key("model_facts");
    j.value(facts_.to_json().str());
    j.key("backend");
    j.value(backend().to_json().str());
    j.end_object();
    return j;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class LlmModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "llm.host";
        i.version = "2.0.0";
        i.language = std::string("C++20 GGUF + ") +
                     (raw_llama_compiled() ? "llama.cpp" : "UNSUPPORTED_HARDWARE_SKIP (no stub fallback)");
        i.role = "local SLM host: GGUF inspection, tokenization, native llama.cpp generation (zero stubs)";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"llm.inspect", "llm.load", "llm.tokenize", "llm.generate", "llm.report"};
        i.limitations = raw_llama_compiled()
            ? std::vector<std::string>{
                  "generation requires a real GGUF model file; stub fallbacks are banned",
                  "quantised kernels come from llama.cpp"}
            : std::vector<std::string>{
                  "UNSUPPORTED_HARDWARE_SKIP: llama.cpp runtime is absent on this build host; "
                  "C++ stub fallbacks are banned",
                  "GGUF metadata and BPE vocabulary inspection work natively without stubs"};
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }
    static bool raw_llama_compiled() {
#if defined(OCT_HAVE_LLAMA)
        return true;
#else
        return false;
#endif
    }
    std::string describe() const override {
        return "Loads GGUF metadata locally and runs native llama.cpp inference (or reports "
               "UNSUPPORTED_HARDWARE_SKIP when absent; C++ stub fallbacks are banned).";
    }
    Status self_check() override {
        Host host;
        const BackendInfo b = host.backend();
        if (b.to_json().str().empty()) return Status::internal("llm: empty backend report");
        const std::string path = "/tmp/octopus_llm_selftest.gguf";
        gguf::Writer w;
        w.set_str("general.architecture", "llama");
        w.set_str("general.name", "selftest");
        w.set_u32("llama.block_count", 2);
        w.set_u32("llama.embedding_length", 16);
        w.set_u32("llama.attention.head_count", 2);
        w.set_u32("llama.vocab_size", 4);
        w.set_str_array("tokenizer.ggml.tokens", {"a", "b", "ab", " "});
        w.set_str("tokenizer.ggml.model", "llama");
        w.add_tensor("token_embd.weight", {16, 4}, gguf::GgmlType::F32, std::vector<uint8_t>(16 * 4 * 4, 0));
        Status wr = w.write_file(path);
        if (!wr) return Status::internal("llm: could not write the fixture: " + wr.message);
        auto facts = host.inspect(path);
        if (!facts) return Status::internal("llm: inspect failed: " + facts.status.message);
        if (facts->n_layer != 2 || facts->n_vocab != 4)
            return Status::internal("llm: model facts do not match the fixture");
#if !defined(OCT_HAVE_LLAMA)
        Status ld = host.load(path);
        if (!ld.is_hardware_skip())
            return Status::internal("llm: expected UNSUPPORTED_HARDWARE_SKIP when llama.cpp absent");
        auto g = host.generate("hello", GenerateParams{});
        if (!g.status.is_hardware_skip())
            return Status::internal("llm: expected UNSUPPORTED_HARDWARE_SKIP from generate()");
#endif
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_llm_module() { return std::make_shared<LlmModule>(); }

}  // namespace oct::llm
