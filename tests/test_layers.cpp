// Tests for the runtime/guardrail layers: Smalltalk actors, Occam parallel
// reasoning, INTERCAL guardrail + audit chain, Piet visualisation, the LLM host
// and the supervisor. Numeric correctness lives in test_numerics.cpp.
// SPDX-License-Identifier: MIT
#include "harness.hpp"

#include "octopus/intercal.hpp"
#include "octopus/llm.hpp"
#include "octopus/occam.hpp"
#include "octopus/piet.hpp"
#include "octopus/prolog.hpp"
#include "octopus/smalltalk.hpp"
#include "octopus/supervisor.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>
#include <vector>

using namespace oct;

namespace {

std::string read_head(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    std::string out(n, '\0');
    f.read(&out[0], std::streamsize(n));
    out.resize(size_t(f.gcount()));
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Smalltalk actors
// ---------------------------------------------------------------------------
OCT_TEST(layers, smalltalk_contains_faults_and_restarts) {
    smalltalk::ActorSystem sys("test");
    int counter = 0;
    sys.spawn("counter", [&](const std::string& sel, const smalltalk::Args&) -> Outcome<std::string> {
        if (sel == "boom") throw OctError("boom");
        return std::to_string(++counter);
    }, smalltalk::Strategy::OneForOne, 2);

    auto first = sys.send("counter", {"inc"});
    OCT_CHECK(bool(first));
    if (first) OCT_CHECK(*first == "1");

    auto fault = sys.send("counter", {"boom"});
    OCT_CHECK(!fault);                                    // contained, not propagated
    OCT_EQ(sys.info("counter").restarts, 1);
    OCT_EQ(sys.info("counter").faults, 1ull);
    OCT_CHECK(sys.healthy());                             // restarted, still up

    auto second = sys.send("counter", {"inc"});
    OCT_CHECK(bool(second));
    if (second) OCT_CHECK(*second == "2");
    OCT_CHECK(sys.events().size() >= 4);

    // Budget exhaustion: a permanently failing actor stops, the system degrades.
    smalltalk::ActorSystem budget("budget");
    budget.spawn("bad", [](const std::string&, const smalltalk::Args&) -> Outcome<std::string> {
        return Status::internal("always fails");
    }, smalltalk::Strategy::OneForOne, 1);
    budget.send("bad", {"x"});
    budget.send("bad", {"x"});
    OCT_CHECK(!budget.healthy());
    auto after_stop = budget.send("bad", {"x"});
    OCT_CHECK(!after_stop);
}

// ---------------------------------------------------------------------------
// Occam parallel reasoning
// ---------------------------------------------------------------------------
OCT_TEST(layers, occam_accepts_only_agreeing_strategies) {
    auto partial = [](int64_t steps) {
        return [steps]() -> Outcome<double> {
            double s = 0.0;
            for (int64_t k = 0; k < steps; ++k) s += (k % 2 ? -1.0 : 1.0) / double(2 * k + 1);
            return 4.0 * s;
        };
    };
    std::vector<occam::Strategy> agree = {
        {"a", partial(20000)},
        {"b", partial(20000)},
        {"c", partial(20000)},
    };
    occam::Verdict v = occam::parallel_verify(agree, 1e-12);
    OCT_CHECK(v.agreed);
    OCT_NEAR(v.value, 3.14159, 1e-2);
    OCT_CHECK(!v.winner.empty());
    OCT_CHECK(v.wall_seconds > 0.0);
    OCT_CHECK(v.serial_seconds > 0.0);
    // speedup is measured, not assumed: it may be below 1 for a tiny workload.
    OCT_NEAR(v.speedup, v.serial_seconds / v.wall_seconds, 1e-9);
    OCT_EQ(v.results.size(), size_t(3));

    std::vector<occam::Strategy> disagree = {
        {"a", []() -> Outcome<double> { return 1.0; }},
        {"b", []() -> Outcome<double> { return 2.0; }},
    };
    occam::Verdict bad = occam::parallel_verify(disagree, 1e-9);
    OCT_CHECK(!bad.agreed);                               // fail closed
    OCT_CHECK(bad.max_spread >= 0.5);

    // A strategy that errors means no verdict at all.
    std::vector<occam::Strategy> errored = {
        {"ok", []() -> Outcome<double> { return 1.0; }},
        {"err", []() -> Outcome<double> { return Status::internal("nope"); }},
    };
    OCT_CHECK(!occam::parallel_verify(errored, 1.0).agreed);
}

OCT_TEST(layers, occam_channel_and_par_map) {
    occam::Channel<int> ch(4);
    std::thread producer([&] {
        for (int i = 0; i < 100; ++i) {
            if (!ch.send(i)) break;
        }
        ch.close();
    });
    long long total = 0;
    while (auto v = ch.recv()) total += *v;
    producer.join();
    OCT_EQ(total, 4950LL);
    OCT_CHECK(ch.closed());

    auto mapped = occam::par_map<int>({1, 2, 3, 4, 5, 6},
                                      [](int x) -> Outcome<int> { return x * x; });
    OCT_EQ(mapped.size(), size_t(6));
    OCT_CHECK(mapped[0] && *mapped[0] == 1);
    OCT_CHECK(mapped[5] && *mapped[5] == 36);             // order preserved
}

// ---------------------------------------------------------------------------
// INTERCAL guardrail + audit chain
// ---------------------------------------------------------------------------
OCT_TEST(layers, intercal_guard_rejects_tampering) {
    const std::string key = "unit-test-key";
    const std::string plaintext = "the quick brown fox jumps over 13 lazy dogs";
    const std::string env = intercal::encode(plaintext, key);
    OCT_CHECK(env.size() == plaintext.size() + 4 + intercal::kNonceBytes + intercal::kTagBytes);
    OCT_CHECK(env.substr(0, 4) == std::string(intercal::kMagic, 4));

    auto back = intercal::decode(env, key);
    OCT_CHECK(bool(back));
    if (back) OCT_CHECK(*back == plaintext);
    OCT_CHECK(!intercal::decode(env, "wrong-key"));
    OCT_CHECK(!intercal::decode(env.substr(0, env.size() - 4), key));   // truncated

    intercal::FuzzReport f = intercal::fuzz_guard(env, key, 256, 42);
    OCT_EQ(f.iterations, size_t(256));
    OCT_EQ(f.accepted, size_t(0));                        // every mutation detected
    OCT_EQ(f.rejected, size_t(256));
}

OCT_TEST(layers, intercal_audit_chain_detects_edits) {
    intercal::AuditChain chain;
    chain.append("boot", "engine");
    chain.append("kernel.run", "sod1d");
    chain.append("shutdown", "clean");
    OCT_EQ(chain.size(), size_t(3));
    OCT_CHECK(chain.verify());

    intercal::AuditChain tampered;
    size_t i = 0;
    for (const intercal::AuditEntry& e : chain.entries()) {
        intercal::AuditEntry copy = e;
        if (i == 1) copy.payload_hash = std::string(64, '0');
        tampered.append_raw(copy);
        ++i;
    }
    size_t broken = 999;
    OCT_CHECK(!tampered.verify(&broken));
    OCT_EQ(broken, size_t(1));

    // Re-ordering entries is also detected (prev_hash linkage).
    intercal::AuditChain reordered;
    reordered.append_raw(chain.entries()[1]);
    reordered.append_raw(chain.entries()[0]);
    OCT_CHECK(!reordered.verify());
}

// ---------------------------------------------------------------------------
// Piet visualisation
// ---------------------------------------------------------------------------
OCT_TEST(layers, piet_quantisation_is_bounded) {
    OCT_EQ(piet::palette().size(), size_t(20));
    std::vector<double> state;
    for (int i = 0; i < 101; ++i) state.push_back(-1.0 + 2.0 * double(i) / 100.0);
    const std::vector<int> codes = piet::quantise(state, -1.0, 1.0);
    OCT_EQ(codes.size(), state.size());
    const std::vector<double> back = piet::dequantise(codes, -1.0, 1.0);

    double max_err = 0.0;
    for (size_t i = 0; i < state.size(); ++i) max_err = std::max(max_err, std::fabs(back[i] - state[i]));
    OCT_NOTE("piet max quantisation error=" << max_err);
    OCT_CHECK(max_err <= 0.12);                           // documented palette step bound
    for (int c : codes) OCT_CHECK(c >= 0 && c < 20);
}

OCT_TEST(layers, piet_renders_a_valid_png_and_marks_invalid_states) {
    std::vector<double> state(64 * 64);
    for (size_t i = 0; i < state.size(); ++i)
        state[i] = std::sin(double(i) * 0.1) * 0.5;
    state[0] = std::numeric_limits<double>::quiet_NaN();

    piet::Raster r = piet::render_state(state, 64, 64, -1.0, 1.0);
    OCT_EQ(r.width, 64);
    OCT_EQ(r.height, 64);
    OCT_EQ(r.rgb.size(), size_t(64 * 64 * 3));
    OCT_EQ(r.codes.size(), size_t(64 * 64));
    OCT_EQ(r.codes[0], 18);                               // NaN -> black (invalid marker)

    const std::string path = "/tmp/octopus_piet_test.png";
    Status w = r.write_png(path);
    OCT_CHECK(w.is_ok());
    if (w.is_ok()) {
        const std::string head = read_head(path, 26);
        OCT_CHECK(head.size() >= 26);
        OCT_CHECK(head.substr(1, 3) == "PNG");
        // IHDR width/height, big-endian at offsets 16 and 20.
        auto be32 = [&](size_t off) {
            return (uint32_t(uint8_t(head[off])) << 24) | (uint32_t(uint8_t(head[off + 1])) << 16) |
                   (uint32_t(uint8_t(head[off + 2])) << 8) | uint32_t(uint8_t(head[off + 3]));
        };
        OCT_EQ(be32(16), uint32_t(64));
        OCT_EQ(be32(20), uint32_t(64));
    }
    // The fingerprint is a function of the pixels alone.
    OCT_CHECK(r.fingerprint() == piet::render_state(state, 64, 64, -1.0, 1.0).fingerprint());
    OCT_CHECK(r.fingerprint().size() == 64);
}

// ---------------------------------------------------------------------------
// LLM host: honest capability reporting
// ---------------------------------------------------------------------------
OCT_TEST(layers, llm_host_reports_what_it_can_and_cannot_do) {
    llm::Host host;
    const llm::BackendInfo b = host.backend();
    OCT_NOTE("backend=" << b.to_json().str());
    OCT_CHECK(!b.name.empty());
    OCT_CHECK(!b.honesty.empty());
    // In a build without llama.cpp the host must say so rather than pretend.
    if (!b.compiled) {
        OCT_CHECK(!b.available);
        OCT_CHECK(b.stub);
    }

    // A missing model is refused outright: even the stub needs real metadata.
    Status missing = host.load("/nonexistent/model.gguf", true);
    OCT_CHECK(!missing.is_ok());

    // With a real (vocabulary) GGUF and the stub explicitly allowed, the host
    // loads metadata-only and labels every answer as not-model-inference.
    const char* fixture = "third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf";
    if (!std::ifstream(fixture).good()) OCT_SKIP("vocabulary fixture not present");
    Status stub = host.load(fixture, true);
    OCT_CHECK(host.loaded());
    if (host.loaded()) {
        llm::GenerateParams p;
        p.max_tokens = 8;
        auto text = host.generate("hello", p);
        OCT_CHECK(bool(text));
        if (text) {
            OCT_CHECK(text->find("stub") != std::string::npos);
            OCT_CHECK(text->find("not model inference") != std::string::npos);
        }
    }
}

// ---------------------------------------------------------------------------
// Supervisor: supervised recovery
// ---------------------------------------------------------------------------
OCT_TEST(layers, supervisor_restarts_on_failure_and_records_audit) {
    supervisor::Supervisor sup;
    int calls = 0;
    int restarts = 0;
    sup.add_check(supervisor::Check{
        "flaky",
        [&]() -> Status {
            ++calls;
            return calls <= 2 ? Status::internal("transient") : Status::ok();
        },
        0.0, supervisor::Severity::Warn});
    sup.set_restart([&]() -> Status {
        ++restarts;
        return Status::ok();
    });

    auto first = sup.tick(2);                             // tolerated: one failure < threshold
    OCT_EQ(first.size(), size_t(1));
    OCT_CHECK(first[0].action == supervisor::Action::None);
    OCT_CHECK(sup.healthy());

    auto second = sup.tick(2);                            // second failure -> restart
    OCT_CHECK(!second.empty());
    OCT_CHECK(second[0].action == supervisor::Action::Restart);
    OCT_EQ(restarts, 1);
    OCT_EQ(sup.restarts(), int64_t(1));
    OCT_CHECK(sup.audit().verify());
    OCT_CHECK(sup.audit().size() >= 2);

    // The report is structured JSON with named actions, not booleans.
    const std::string report = sup.report().str();
    OCT_CHECK(report.find("\"action\":\"restart\"") != std::string::npos);
    OCT_CHECK(report.find("\"audit\":{") != std::string::npos);
}

OCT_TEST(layers, supervisor_watchdog_and_prolog_invariant) {
    supervisor::Supervisor sup;
    sup.add_check(supervisor::Check{
        "slow",
        []() -> Status {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return Status::ok();
        },
        0.005, supervisor::Severity::Info});              // 5 ms deadline, must trip
    auto rows = sup.tick(1);
    OCT_EQ(rows.size(), size_t(1));
    OCT_CHECK(rows[0].timed_out);

    prolog::KnowledgeBase kb;
    OCT_CHECK(kb.add_text("limit(1000).").is_ok());
    auto goal = prolog::parse_term("limit(1000)");
    OCT_CHECK(bool(goal));
    supervisor::InvariantCheck inv;
    inv.name = "reaction-limit";
    inv.goal = *goal;
    inv.severity = supervisor::Severity::Critical;
    sup.add_invariant(kb, inv);

    auto rows2 = sup.tick(1);
    OCT_CHECK(!rows2.empty());
    OCT_CHECK(rows2.back().healthy);                      // grounded invariant holds

    prolog::KnowledgeBase kb2;
    OCT_CHECK(kb2.add_text("limit(1).").is_ok());
    kb2.declare_complete("limit/1");                     // only then can a miss refute
    auto goal2 = prolog::parse_term("limit(1000)");
    OCT_CHECK(bool(goal2));
    supervisor::InvariantCheck inv2;
    inv2.name = "broken-invariant";
    inv2.goal = *goal2;
    inv2.severity = supervisor::Severity::Critical;
    sup.add_invariant(kb2, inv2);
    auto rows3 = sup.tick(1);
    OCT_CHECK(!rows3.back().healthy);
    OCT_CHECK(rows3.back().action != supervisor::Action::None);
}
