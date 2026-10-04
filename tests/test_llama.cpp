// llama.cpp integration & hardware-skip tests.
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// Optional C++ stub fallbacks (`allow_stub_fallback`, `"stub-answer{...}"`)
// are banned. When the native llama.cpp backend is absent, live inference
// blocks report `UNSUPPORTED_HARDWARE_SKIP`.
// SPDX-License-Identifier: MIT
#include "harness.hpp"

#include "octopus/llm.hpp"
#include "octopus/synthetic_model.hpp"

#include <fstream>
#include <string>

using namespace oct;

namespace {
std::string fixture(const char* name) {
    const char* candidates[] = {
        "third_party/llama.cpp/models/",
        "../third_party/llama.cpp/models/",
        "../../third_party/llama.cpp/models/",
    };
    for (const char* c : candidates) {
        std::ifstream probe(std::string(c) + name);
        if (probe) return std::string(c) + name;
    }
    return {};
}
}  // namespace

OCT_TEST(llm, hardware_skip_gate_bans_stub_fallback) {
    llm::Host host;
    const llm::BackendInfo b = host.backend();
    if (!b.compiled) {
        OCT_CHECK(b.hardware_skipped);
        OCT_CHECK(b.name == "UNSUPPORTED_HARDWARE_SKIP");
        auto g = host.generate("hello", llm::GenerateParams{});
        OCT_CHECK(!g);
        OCT_CHECK(g.status.is_hardware_skip());
    } else {
        OCT_CHECK(b.available);
        OCT_CHECK(!b.hardware_skipped);
    }
}

OCT_TEST_REQ_LLAMA(llm, backend_is_linked_and_honest) {
    const llm::BackendInfo b = llm::Host{}.backend();
    OCT_NOTE("backend=" << b.to_json().str());
    OCT_CHECK(b.compiled);
    OCT_CHECK(b.available);
    OCT_CHECK(!b.hardware_skipped);
    OCT_CHECK(b.name.find("llama.cpp") != std::string::npos);
    OCT_CHECK(!b.honesty.empty());
}

OCT_TEST_REQ_LLAMA(llm, metadata_only_gguf_is_refused_not_guessed) {
    const std::string path = fixture("ggml-vocab-gpt-2.gguf");
    if (path.empty()) OCT_HARDWARE_SKIP("vocabulary fixture not present");
    llm::Host host;
    auto facts = host.inspect(path);
    OCT_CHECK(bool(facts));
    Status st = host.load(path);
    OCT_CHECK(!st.is_ok());
    OCT_CHECK(!host.loaded());
    llm::GenerateParams p;
    auto text = host.generate("hello", p);
    OCT_CHECK(!text);
}

OCT_TEST_REQ_LLAMA(llm, synthetic_model_end_to_end_generation_is_deterministic) {
    const std::string path = "/tmp/octopus_synthetic_model.gguf";
    synthetic::ModelSpec spec;
    spec.n_layer = 2;
    spec.n_embd = 64;
    spec.n_head = 4;
    spec.n_ff = 128;
    auto written = synthetic::write_model(path, spec);
    OCT_CHECK(bool(written));
    if (!written) OCT_HARDWARE_SKIP("cannot write the synthetic model: " + written.status.message);

    llm::Host host;
    Status loaded = host.load(path);
    OCT_CHECK_MSG(loaded.is_ok(), "load failed: " << loaded.message);
    if (!loaded.is_ok()) OCT_HARDWARE_SKIP("llama.cpp could not load the synthetic model: " + loaded.message);

    const llm::BackendInfo b = host.backend();
    OCT_CHECK(b.compiled);
    OCT_CHECK(b.available);
    OCT_CHECK(!b.hardware_skipped);

    auto ids = host.tokenize("hello world");
    OCT_CHECK(bool(ids));
    if (ids) OCT_CHECK(ids->size() >= 2);

    llm::GenerateParams p;
    p.max_tokens = 8;
    p.temperature = 0.0;
    auto first = host.generate("hello world", p);
    auto second = host.generate("hello world", p);
    OCT_CHECK(bool(first));
    OCT_CHECK(bool(second));
    if (first && second) {
        OCT_CHECK(!first->empty());
        OCT_CHECK(*first == *second);
    }
}
