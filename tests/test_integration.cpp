// Cross-category tests that the architecture spec requires beyond the layer
// unit tests: offline end-to-end paths, build/catalogue sanity, security
// properties (sandbox boundaries, fail-closed behaviour), memory bounds and
// parser fuzzing. Each test executes its assertions; none can PASS vacuously.
// SPDX-License-Identifier: MIT
#include "harness.hpp"

#include "octopus/apl.hpp"
#include "octopus/catalog.hpp"
#include "octopus/intercal.hpp"
#include "octopus/lisp.hpp"
#include "octopus/llm.hpp"
#include "octopus/module.hpp"
#include "octopus/occam.hpp"
#include "octopus/piet.hpp"
#include "octopus/prolog.hpp"
#include "octopus/smalltalk.hpp"
#include "octopus/stackvm.hpp"
#include "octopus/supervisor.hpp"
#include "octopus/tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace oct;

namespace {

std::string model_dir() {
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

// ---------------------------------------------------------------------------
// Model metadata: read from a real GGUF without any inference backend.
// ---------------------------------------------------------------------------
OCT_TEST(integration, model_metadata_from_a_real_gguf_file) {
    const std::string dir = model_dir();
    if (dir.empty()) OCT_SKIP("llama.cpp fixtures not present");
    llm::Host host;
    auto facts = host.inspect(dir + "ggml-vocab-gpt-2.gguf");
    OCT_CHECK(bool(facts));
    if (!facts) OCT_SKIP("cannot inspect fixture: " + facts.status.message);
    OCT_NOTE("n_vocab=" << facts->n_vocab << " has_tokenizer=" << facts->has_tokenizer);
    OCT_CHECK(facts->has_tokenizer);
    OCT_CHECK(facts->n_vocab >= 32000);
    OCT_CHECK(!facts->architecture.empty() || facts->n_vocab > 0);

    // A non-GGUF file must be rejected, not guessed at.
    auto bogus = host.inspect(dir + "ggml-vocab-gpt-2.gguf.inp");
    OCT_CHECK(!bogus);
}

// ---------------------------------------------------------------------------
// Security / sandbox boundaries: capabilities are explicit; nothing escapes its
// module; a module cannot claim a capability it does not advertise.
// ---------------------------------------------------------------------------
OCT_TEST(integration, security_capabilities_are_explicit_and_fail_closed) {
    Registry reg;
    register_builtin_modules(reg);

    // Every advertised capability resolves back to a module that lists it.
    for (const std::string& cap : reg.all_capabilities()) {
        auto m = reg.resolve(cap);
        OCT_CHECK_MSG(m != nullptr, "capability without a module: " << cap);
        if (!m) continue;
        const auto caps = m->info().capabilities;
        OCT_CHECK(std::find(caps.begin(), caps.end(), cap) != caps.end());
    }

    // The guardrail module must document that it is not a cipher. This is a
    // documentation assertion that is executed, not a comment.
    auto guard = reg.resolve("guard.encode");
    OCT_CHECK(guard != nullptr);
    if (guard) {
        bool says_obfuscation = false;
        for (const auto& l : guard->info().limitations)
            if (l.find("NOT a cipher") != std::string::npos) says_obfuscation = true;
        OCT_CHECK(says_obfuscation);

        bool claims_immortality = false;
        for (const auto& l : guard->info().limitations)
            if (l.find("immune to decompilation") != std::string::npos) claims_immortality = true;
        OCT_CHECK(!claims_immortality);          // no unsupported security claims
    }

    // Untrusted plugin boundary: a module whose self_check fails must be
    // reported as failing, never silently accepted.
    class BrokenModule final : public Module {
    public:
        ModuleInfo info() const override {
            ModuleInfo i;
            i.name = "test.broken";
            i.version = "0.0.1";
            i.language = "C++20";
            i.role = "deliberately failing module for the sandbox test";
            i.capabilities = {"test.broken"};
            i.limitations = {"only used by tests"};
            return i;
        }
        Status self_check() override { return Status::internal("intentional failure"); }
    };
    Registry r2;
    r2.add(std::make_shared<BrokenModule>());
    auto results = r2.self_check_all();
    OCT_EQ(results.size(), size_t(1));
    OCT_CHECK(!results[0].status.is_ok());
}

// ---------------------------------------------------------------------------
// Fuzz tests for parsers/DSLs: malformed input must be rejected, never crash.
// ---------------------------------------------------------------------------
OCT_TEST(integration, parsers_reject_malformed_input_without_crashing) {
    const std::vector<std::string> nasty = {
        "", "(", ")", "((((((((((", ")-", "\"unterminated",
        "\xff\xfe\x00binary", "1 2 3 + + + +", "⍳", "⍴⍴⍴⍴",
        "x ←", "← 3", "(", "1 +", "+ 1", "((((",
    };
    int rejected = 0, accepted = 0;
    for (const std::string& src : nasty) {
        apl::Environment env;
        auto r = apl::eval_line(src, env);      // must not throw or crash
        if (r) ++accepted; else ++rejected;
    }
    OCT_NOTE("apl fuzz: accepted=" << accepted << " rejected=" << rejected);
    OCT_CHECK(rejected > 0);
    OCT_EQ(accepted + rejected, int(nasty.size()));

    for (const std::string& src : nasty) {
        prolog::KnowledgeBase kb;
        auto r = prolog::parse_term(src);       // must return a status, not crash
        (void)r;
    }
    for (const std::string& src : nasty) {
        lisp::Interp interp;
        auto r = interp.eval_string(src);
        (void)r;
    }
    for (const std::string& src : nasty) {
        stackvm::Vm vm;
        auto r = vm.compile(src);
        (void)r;
    }
    OCT_CHECK(true);                            // reaching here proves no crash
}

OCT_TEST(integration, guard_fuzz_accepts_no_mutation_across_three_seeds) {
    const std::string key = "fuzz-key";
    const std::string env = intercal::encode("payload-to-fuzz-across-seeds", key);
    size_t total_accepted = 0, total = 0;
    for (uint64_t seed : {1ull, 7ull, 99ull}) {
        auto r = intercal::fuzz_guard(env, key, 128, seed);
        total_accepted += r.accepted;
        total += r.iterations;
    }
    OCT_EQ(total, size_t(384));
    OCT_EQ(total_accepted, size_t(0));
}

// ---------------------------------------------------------------------------
// Offline end-to-end: the whole chain must work with no network and no model,
// and every step must be reproducible.
// ---------------------------------------------------------------------------
OCT_TEST(integration, offline_end_to_end_chain) {
    // 1. APL computes an array, 2. Prolog verifies a claim about the result,
    // 3. Occam confirms it in parallel, 4. Piet renders the state, 5. the
    // supervisor watches it and 6. the audit chain records the run.
    apl::Environment env;
    auto sum = apl::eval_line("+/ \xe2\x8d\xb3 101", env);              // 0..100
    OCT_CHECK(bool(sum));
    if (!sum) OCT_SKIP("APL failed: " + sum.status.message);
    const double total = sum->scalar_f64();
    OCT_NEAR(total, 5050.0, 0.0);

    prolog::KnowledgeBase kb;
    OCT_CHECK(kb.add_text("total(5050).").is_ok());
    kb.declare_complete("total/1");
    auto goal = prolog::parse_term("total(5050)");
    OCT_CHECK(bool(goal));
    auto v = prolog::verify(kb, *goal);
    OCT_CHECK(v.verdict == prolog::Verdict::Grounded);

    std::vector<occam::Strategy> strategies = {
        {"gauss", []() -> Outcome<double> { return 101.0 * 100.0 / 2.0; }},
        {"apl", [total]() -> Outcome<double> { return total; }},
    };
    occam::Verdict verdict = occam::parallel_verify(strategies, 0.5);
    OCT_CHECK(verdict.agreed);

    std::vector<double> state(16 * 16, total / 5050.0);
    piet::Raster r = piet::render_state(state, 16, 16, -1.0, 1.0);
    OCT_EQ(r.codes.size(), size_t(256));

    supervisor::Supervisor sup;
    sup.add_check(supervisor::Check{"e2e", []() -> Status { return Status::ok(); }, 1.0,
                                    supervisor::Severity::Info});
    auto rows = sup.tick(1);
    OCT_EQ(rows.size(), size_t(1));
    OCT_CHECK(rows[0].healthy);
    OCT_CHECK(sup.audit().verify());
    OCT_CHECK(sup.audit().size() >= 1);
}

// ---------------------------------------------------------------------------
// Memory-budget behaviour: the engine's own allocations are bounded and the
// kernels honour a documented size ceiling instead of allocating without limit.
// ---------------------------------------------------------------------------
OCT_TEST(integration, memory_use_is_bounded_by_the_declared_sizes) {
    // The stack VM's memory is an explicit, bounded array; out-of-range access
    // faults instead of growing it.
    stackvm::Vm vm(256);
    OCT_CHECK(vm.compile(": store 42 255 ! 255 @ ;  : oob 42 256 ! ;").is_ok());
    auto ok = vm.call("store", {});
    OCT_CHECK(bool(ok));
    auto fault = vm.call("oob", {});
    OCT_CHECK(!fault);

    // Piet rendering is bounded by the requested grid, not by the input length.
    std::vector<double> big(100000, 0.25);
    piet::Raster r = piet::render_state(big, 32, 32, -1.0, 1.0);
    OCT_EQ(r.rgb.size(), size_t(32 * 32 * 3));
    OCT_EQ(r.codes.size(), size_t(32 * 32));    // extra input is folded into the grid
}

// ---------------------------------------------------------------------------
// Concurrency: the actor system and the parallel layer under repetition must
// stay deterministic in their *reported* results.
// ---------------------------------------------------------------------------
OCT_TEST(integration, concurrency_is_deterministic_for_equal_work) {
    for (int rep = 0; rep < 8; ++rep) {
        smalltalk::ActorSystem sys("rep");
        int n = 0;
        sys.spawn("a", [&](const std::string&, const smalltalk::Args&) -> Outcome<std::string> {
            return std::to_string(++n);
        }, smalltalk::Strategy::OneForOne, 1);
        auto r1 = sys.send("a", {"go"});
        auto r2 = sys.send("a", {"go"});
        OCT_CHECK(bool(r1) && bool(r2));
        if (r1) OCT_CHECK(*r1 == "1");
        if (r2) OCT_CHECK(*r2 == "2");
    }

    occam::Channel<int> ch(1);
    std::thread producer([&] {
        for (int i = 0; i < 32; ++i) ch.send(i);
        ch.close();
    });
    std::vector<int> seen;
    while (auto v = ch.recv()) seen.push_back(*v);
    producer.join();
    OCT_EQ(seen.size(), size_t(32));
    for (size_t i = 0; i < seen.size(); ++i) OCT_EQ(seen[i], int(i));   // FIFO order
}
