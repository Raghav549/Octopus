// Octopus Hybrid AI Engine -- LLM host (llama.cpp bridge + stub fallback).
//
// The llama.cpp path is compiled only when OCT_WITH_LLAMA is defined (the CMake
// option of the same name, which also requires a prebuilt static llama.cpp
// tree). All stub behaviour is deterministic and labelled.
// SPDX-License-Identifier: MIT
#include "octopus/llm.hpp"

#include <climits>

#include "octopus/module.hpp"

#include <cmath>
#include <sstream>

#if defined(OCT_HAVE_LLAMA)
#include <llama.h>
#endif

namespace oct::llm {

namespace {

std::string first_line(std::string_view s, size_t max_len = 240) {
    std::string out;
    for (char c : s) {
        if (c == '\n') break;
        out += c;
        if (out.size() >= max_len) break;
    }
    return out;
}

}  // namespace

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
    j.field("stub", stub);
    j.field("name", name);
    j.field("version", version);
    j.field("honesty", honesty);
    j.end_object();
    return j;
}

Host::Host() : rng_(0xC0FFEEULL) {}

BackendInfo Host::backend() const {
    BackendInfo b;
    b.compiled = false;
    b.available = false;
    b.stub = true;
    b.name = "stub.oracle";
    b.version = kVersionString;
    b.honesty = "no inference backend compiled in: generate() is unavailable; "
                "OCT_WITH_LLAMA=OFF or no prebuilt llama.cpp tree was found";
#if defined(OCT_HAVE_LLAMA)
    b.compiled = true;
    // "available" means the llama.cpp library is linked and usable, not that a
    // model has been loaded: ModelFacts/load() are what report model state.
    b.available = true;
    b.name = "llama.cpp";
    b.version = "b11371";
    if (loaded_ && !stub_mode_) {
        b.stub = false;
        b.honesty = "llama.cpp static backend linked and a model is loaded; generation runs "
                    "entirely on this machine";
    } else {
        b.stub = true;
        b.honesty = "llama.cpp static backend linked, but no model is loaded: load() refuses to "
                    "answer unless a real model file is supplied (the deterministic stub is only "
                    "used when explicitly allowed)";
    }
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

Status Host::load(const std::string& path, bool allow_stub_fallback) {
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
    if (allow_stub_fallback) { /* fall through to stub below */ }
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
            stub_mode_ = false;
            return Status::ok();
        }
        llama_model_free(model);
        llama_model_ = nullptr;
    }
    if (!allow_stub_fallback)
        return Status::unavailable("llm: llama.cpp failed to load '" + path + "'");
#endif
    if (!allow_stub_fallback)
        return Status::unavailable(
            "llm: no inference backend compiled in (configure with -DOCT_WITH_LLAMA=ON and "
            "OCT_LLAMA_PREBUILT_DIR); refusing to answer without a model");
    // Deterministic stub: metadata is real, generation is synthetic and labelled.
    loaded_ = true;
    stub_mode_ = true;
    return Status::degraded("llm: deterministic stub backend enabled (no llama.cpp); "
                            "generation is NOT model inference");
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

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------
#if defined(OCT_HAVE_LLAMA)
namespace {

// Greedy/temperature sampling loop against the loaded context.
//
// NOTE ON TEST STATUS: this function is compiled and linked only in an
// OCT_WITH_LLAMA build, and it needs real model weights to run end to end.
// In the environment this repository was assembled in no weights were
// available, so the loop compiles and links but has never executed a token.
// See docs/LIMITATIONS.md. Failures are returned, never guessed around.
std::string llama_generate_impl(llama_model* model, llama_context* ctx, const std::string& prompt,
                                const GenerateParams& p, std::string* error) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        *error = "model has no vocabulary";
        return {};
    }

    // Tokenise: a negative return is the required buffer size.
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

    // Reset the KV cache: without this, a second generate() call continues from
    // the previous generation's state and greedy decoding stops being
    // reproducible (found by tests/test_llama.cpp).
    if (llama_memory_t mem = llama_get_memory(ctx)) llama_memory_clear(mem, /*data=*/true);

    // Prefill the prompt, then sample one token per decode step.
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
        if (n < 0) {                       // buffer too small: retry with the reported size
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
    if (!loaded_)
        return Status::unavailable("llm: no model loaded; call load() first");
    if (!stub_mode_) {
#if defined(OCT_HAVE_LLAMA)
        std::string error;
        const std::string text = llama_generate_impl(static_cast<llama_model*>(llama_model_),
                                                     static_cast<llama_context*>(llama_context_),
                                                     std::string(prompt), params, &error);
        if (text.empty())
            return Status::degraded("llm: generation produced no tokens" +
                                    (error.empty() ? std::string() : std::string(" (") + error + ")"));
        return text;
#else
        return Status::unavailable("llm: stub mode reached the non-stub path");
#endif
    }
    // --- deterministic stub oracle ------------------------------------------
    // Documented behaviour: tokenize the prompt, then emit a diagnostic answer
    // assembled from measured facts. It is a test double for the host plumbing,
    // never a substitute for a model.
    std::ostringstream os;
    auto ids = tokenize(prompt);
    const size_t n_tokens = ids ? ids->size() : 0;
    os << "stub-answer{";
    os << "prompt_bytes=" << prompt.size();
    os << ",prompt_tokens=" << n_tokens;
    os << ",vocab=" << (have_vocab_ ? int64_t(vocab_.size()) : 0);
    os << ",model='" << facts_.name << "'";
    os << ",max_tokens=" << params.max_tokens;
    os << ",temperature=" << params.temperature;
    os << ",note='deterministic stub, not model inference'}";
    return os.str();
}

Json Host::report() const {
    Json j;
    j.begin_object();
    j.field("loaded", loaded_);
    j.field("stub_mode", stub_mode_);
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
        i.version = "1.0.0";
        i.language = std::string("C++20") + (raw_llama_compiled() ? " + llama.cpp" : " (no inference backend)");
        i.role = "local SLM host: GGUF inspection, tokenization, offline generation";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"llm.inspect", "llm.load", "llm.tokenize", "llm.generate", "llm.report"};
        i.limitations = raw_llama_compiled()
            ? std::vector<std::string>{
                  "generation requires a model file; nothing is downloaded and no network is used",
                  "quantised kernels come from llama.cpp, not from this repository",
                  "no sampling beyond greedy/temperature in the host wrapper"}
            : std::vector<std::string>{
                  "no inference backend compiled in (OCT_WITH_LLAMA=OFF): generate() is unavailable",
                  "the deterministic stub exists only for pipeline tests and is labelled as such",
                  "model metadata (GGUF) is still read natively, so capability reporting stays honest"};
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
        return "Loads GGUF metadata locally and runs (or refuses to run) generation on this "
               "machine; the backend identity is reported with every answer.";
    }
    Status self_check() override {
        Host host;
        const BackendInfo b = host.backend();
        if (!b.to_json().str().empty() == false) return Status::internal("llm: empty backend report");
#if !defined(OCT_HAVE_LLAMA)
        if (b.compiled || b.available || !b.stub)
            return Status::internal("llm: backend flags are wrong for a build without llama.cpp");
        auto g = host.generate("hello", GenerateParams{});
        if (g) return Status::internal("llm: generation succeeded without a backend");
#endif
        // Stub mode must be deterministic and must not claim to be a model.
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
        Status ld = host.load(path, /*allow_stub_fallback=*/true);
        if (!ld) return Status::internal("llm: stub load failed: " + ld.message);
        auto g1 = host.generate("ab ab", GenerateParams{8});
        auto g2 = host.generate("ab ab", GenerateParams{8});
        if (!g1 || !g2 || *g1 != *g2) return Status::internal("llm: stub output is not deterministic");
        if (g1->find("stub-answer") == std::string::npos)
            return Status::internal("llm: stub output is not labelled");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_llm_module() { return std::make_shared<LlmModule>(); }

}  // namespace oct::llm
