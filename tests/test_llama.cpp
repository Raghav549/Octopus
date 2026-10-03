// llama.cpp integration tests.
//
// These tests run only in a build configured with -DOCT_WITH_LLAMA=ON and a
// prebuilt llama.cpp (they SKIP with a printed reason otherwise, via the
// harness's require_llama flag). In this repository snapshot no model weights
// are available, so the tests assert what can be verified without weights:
// that the backend is genuinely linked, that a metadata-only GGUF is refused
// rather than loaded, and that generation without a model fails closed.
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

OCT_TEST_REQ_LLAMA(llm, backend_is_linked_and_honest) {
    const llm::BackendInfo b = llm::Host{}.backend();
    OCT_NOTE("backend=" << b.to_json().str());
    OCT_CHECK(b.compiled);                 // OCT_HAVE_LLAMA really is defined
    OCT_CHECK(b.available);                // the linked library is usable
    OCT_CHECK(b.name.find("llama.cpp") != std::string::npos);
    OCT_CHECK(!b.honesty.empty());
    // No model is loaded yet in this snapshot, so answers must still be stub-
    // labelled; after a real load() the flag flips (asserted below).
    OCT_CHECK(b.stub);
}

OCT_TEST_REQ_LLAMA(llm, metadata_only_gguf_is_refused_not_guessed) {
    const std::string path = fixture("ggml-vocab-gpt-2.gguf");
    if (path.empty()) OCT_SKIP("vocabulary fixture not present");
    llm::Host host;
    // The metadata reader works without a backend.
    auto facts = host.inspect(path);
    OCT_CHECK(bool(facts));
    // But llama.cpp cannot execute a vocabulary-only file: no tensors.
    Status st = host.load(path, /*allow_stub_fallback=*/false);
    OCT_CHECK(!st.is_ok());
    OCT_CHECK(!host.loaded());
    llm::GenerateParams p;
    auto text = host.generate("hello", p);
    OCT_CHECK(!text);                      // nothing loaded -> no answer, ever
}

OCT_TEST_REQ_LLAMA(llm, missing_model_fails_and_stub_stays_labelled) {
    llm::Host host;
    Status missing = host.load("/nonexistent/model.gguf", false);
    OCT_CHECK(!missing.is_ok());

    const std::string path = fixture("ggml-vocab-gpt-2.gguf");
    if (path.empty()) OCT_SKIP("vocabulary fixture not present");
    Status stub = host.load(path, /*allow_stub_fallback=*/true);
    OCT_CHECK(stub.is_ok());
    if (!stub.is_ok()) OCT_SKIP("stub fallback refused: " + stub.message);
    llm::GenerateParams p;
    p.max_tokens = 4;
    auto text = host.generate("hello", p);
    OCT_CHECK(bool(text));
    if (text) {
        // Even with llama.cpp linked, a fallback answer must be labelled.
        OCT_CHECK(text->find("stub") != std::string::npos);
        OCT_CHECK(text->find("not model inference") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// The real thing: load a model, prefill, sample, detokenize. The model is the
// synthetic tiny llama written by include/octopus/synthetic_model.hpp, so this
// runs offline with no downloaded weights. The generated text is nonsense by
// construction (pseudo-random weights); what is asserted is that the pipeline
// executes, that greedy decoding is deterministic, and that the backend reports
// the truth about what produced the tokens.
// ---------------------------------------------------------------------------
OCT_TEST_REQ_LLAMA(llm, synthetic_model_end_to_end_generation_is_deterministic) {
    const std::string path = "/tmp/octopus_synthetic_model.gguf";
    synthetic::ModelSpec spec;
    spec.n_layer = 2;
    spec.n_embd = 64;
    spec.n_head = 4;
    spec.n_ff = 128;
    auto written = synthetic::write_model(path, spec);
    OCT_CHECK(bool(written));
    if (!written) OCT_SKIP("cannot write the synthetic model: " + written.status.message);
    OCT_NOTE("synthetic model: " << written->to_json().str());

    llm::Host host;
    Status loaded = host.load(path, /*allow_stub_fallback=*/false);
    OCT_CHECK_MSG(loaded.is_ok(), "load failed: " << loaded.message);
    if (!loaded.is_ok()) OCT_SKIP("llama.cpp could not load the synthetic model: " + loaded.message);

    const llm::BackendInfo b = host.backend();
    OCT_CHECK(b.compiled);
    OCT_CHECK(b.available);
    OCT_CHECK(!b.stub);                      // real inference, not the stub oracle

    auto ids = host.tokenize("hello world");
    OCT_CHECK(bool(ids));
    if (ids) OCT_CHECK(ids->size() >= 2);

    llm::GenerateParams p;
    p.max_tokens = 8;
    p.temperature = 0.0;                     // greedy: must be reproducible
    auto first = host.generate("hello world", p);
    auto second = host.generate("hello world", p);
    OCT_CHECK(bool(first));
    OCT_CHECK(bool(second));
    if (first) {
        OCT_CHECK(!first->empty());
        OCT_NOTE("greedy sample (synthetic weights, expected nonsense): " << first->size() << " bytes");
    }
    if (first && second) OCT_CHECK(*first == *second);

    // The metadata reader sees the file as a real model, and the synthetic
    // provenance is visible rather than hidden.
    auto facts = host.inspect(path);
    OCT_CHECK(bool(facts));
    if (facts) {
        OCT_EQ(facts->n_layer, int64_t(2));
        OCT_EQ(facts->n_embd, int64_t(64));
        OCT_CHECK(facts->has_tokenizer);
        OCT_CHECK(facts->n_vocab >= 256);
        OCT_CHECK(facts->bpw > 30.0 && facts->bpw < 33.0);   // F32 weights
    }
}
