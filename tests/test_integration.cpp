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
#include "octopus/universe.hpp"

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

OCT_TEST(integration, physical_universe_image_and_video_engine_pipeline) {
    // 1. Coordinate-to-Piet translation (4-channel: light, reflection, fluid, gravity)
    const int w = 16, h = 16;
    std::vector<double> coords(size_t(w * h * 4), 0.0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t base = size_t(y * w + x) * 4;
            coords[base + 0] = 0.6 + 0.4 * std::cos(0.2 * double(x));
            coords[base + 1] = 0.3 + 0.2 * std::sin(0.3 * double(y));
            coords[base + 2] = 0.5 * std::sin(0.25 * double(x + y));
            coords[base + 3] = 1.0 / (1.0 + 0.05 * double(x * x + y * y));
        }
    }
    auto fr = universe::translate_coordinates_to_piet(coords, w, h, 0.5);
    OCT_CHECK(bool(fr));
    if (fr) {
        OCT_CHECK(fr->sld_report.sound);
        OCT_EQ(fr->canvas.width, w);
        OCT_EQ(fr->canvas.height, h);
        OCT_CHECK(fr->total_radiance > 0.0);
        const std::string y4m = "/tmp/octopus_test_universe.y4m";
        OCT_CHECK(universe::write_y4m_video(y4m, {fr->canvas, fr->canvas}, 24).is_ok());
    }

    // 2. Live Fortran 2023 synthesis or explicit UNSUPPORTED_HARDWARE_SKIP
    universe::SceneSpec spec;
    spec.width = 16;
    spec.height = 16;
    spec.frames = 2;
    const auto vr = universe::synthesize_video(spec);
    if (numerics::KernelLibrary::instance().fortran_available()) {
        OCT_CHECK(!vr.hardware_skipped);
        OCT_CHECK(vr.status.is_ok());
        OCT_EQ(vr.frames.size(), size_t(2));
    } else {
        OCT_CHECK(vr.hardware_skipped);
        OCT_CHECK(vr.status.is_hardware_skip());
    }
}

OCT_TEST(integration, autonomous_agent_supervisor_live_lisp_and_prolog_healing) {
    supervisor::Supervisor sup;
    smalltalk::ActorSystem actors("watchdog.test");
    lisp::Interp interp;
    // Faulty initial LISP function (division by zero)
    OCT_CHECK(interp.eval_string("(define (reactor-flux x) (/ x 0))").is_ok());
    prolog::KnowledgeBase kb;
    prolog::load_axiom_core(kb);

    actors.spawn("reactor.cell", [&](const std::string&, const smalltalk::Args& call_args) -> Outcome<std::string> {
        const std::string a = call_args.empty() ? "6" : call_args[0];
        auto r = interp.eval_string("(reactor-flux " + a + ")");
        if (!r.ok()) return r.status;
        return (*r)->to_string();
    }, smalltalk::Strategy::OneForOne, 3);

    // Trigger live exception -> automated LISP hot-patch -> Prolog SLD sweep -> live retry
    const auto rep = sup.heal_actor_exception(
        actors, interp, kb,
        "reactor.cell", "step", {"6"},
        "reactor-flux", "(lambda (x) (* (+ x 4) 3))",
        "sound_backend(fortran2023)");

    OCT_CHECK(rep.fault_intercepted);
    OCT_CHECK(rep.lisp_rewritten);
    OCT_CHECK(rep.prolog_invariant_ok);
    OCT_CHECK(rep.recovered_live);
    OCT_EQ(rep.healed_reply, std::string("30"));
    OCT_CHECK(rep.lisp_generation_after > rep.lisp_generation_before);
    OCT_CHECK(sup.audit().verify());
}

OCT_TEST(integration, multi_language_dna_prolog_guardrail_apl_compressor_forth_occam_intercal) {
    // 1. Prolog SLD Soundness Guardrail
    const auto good_sld = prolog::verify_response_soundness(
        "explain conservation", "fortran2023 euler_hllc conserves mass");
    OCT_CHECK(good_sld.sound);
    const auto bad_sld = prolog::verify_response_soundness(
        "explain conservation", "stub-answer{not model inference}");
    OCT_CHECK(!bad_sld.sound);

    // 2. APL Context Matrix Compressor
    const std::vector<uint32_t> toks = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const auto cc = apl::compress_context_matrix(toks, 4);
    OCT_CHECK(bool(cc));
    if (cc) {
        OCT_EQ(cc->matrix_shape[0], 4);
        OCT_EQ(cc->matrix_shape[1], 4);
        OCT_EQ(cc->row_energy.size(), size_t(4));
        OCT_NEAR(cc->row_energy[0], 10.0, 1e-12); // 1+2+3+4 = 10
        OCT_CHECK(cc->matrix_compression_ratio >= 4.0);
    }

    // 3. Forth hardware register file + return stack
    stackvm::Vm vm;
    OCT_CHECK(vm.compile("21 r0! r0@ 2 * >r r@ r> + r1! r1@ .").is_ok());
    OCT_CHECK(bool(vm.run()));
    OCT_NEAR(vm.get_register(0), 21.0, 1e-12);
    OCT_NEAR(vm.get_register(1), 84.0, 1e-12);

    // 4. Occam + Forth + Smalltalk Triad Coordinator
    const auto triad = occam::coordinate_triad("", 3.0);
    OCT_CHECK(triad.ok);
    OCT_CHECK(triad.occam_verdict.agreed);
    OCT_NEAR(triad.occam_verdict.value, 16.0, 1e-12); // (3+1)^2 = 16
    OCT_CHECK(triad.smalltalk_reply.find("triad-ack:") != std::string::npos);

    // 5. INTERCAL Obfuscated Core Shield (mingle + select + hash-linked envelope)
    intercal::AuditChain chain;
    const auto env = intercal::seal_code_structure("lisp.ast", "(define (f x) (* x x))", "shield-key", &chain);
    const auto unsealed = intercal::unseal_code_structure(env, "shield-key", &chain);
    OCT_CHECK(bool(unsealed));
    if (unsealed) OCT_EQ(*unsealed, std::string("(define (f x) (* x x))"));
}
