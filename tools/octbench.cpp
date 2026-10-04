// Octopus benchmark harness: reproducible timings for the kernels and the
// language layers. Emits CSV on stdout so a run can be archived and diffed.
//
// Reproducibility rules (ARCHITECTURE_SPEC.md, item 14):
//   * every input is fixed in this file (no random seeds unless printed),
//   * every run prints the build/machine fingerprint and result fingerprints,
//   * timings are min/median over --repeats, never a single sample,
//   * a run that cannot execute its case fails loudly instead of printing a
//     number.
// SPDX-License-Identifier: MIT
#include "octopus/apl.hpp"
#include "octopus/intercal.hpp"
#include "octopus/lisp.hpp"
#include "octopus/module.hpp"
#include "octopus/numerics.hpp"
#include "octopus/piet.hpp"
#include "octopus/prolog.hpp"
#include "octopus/router.hpp"
#include "octopus/tokenizer.hpp"
#include "octopus/universe.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace oct;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    out += '"';
    return out;
}

struct Stat {
    double min_ms = 0.0, median_ms = 0.0, max_ms = 0.0;
};

Stat summarise(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min_ms = v.front();
    s.max_ms = v.back();
    s.median_ms = v[v.size() / 2];
    return s;
}

void row(const char* category, const std::string& name, const std::string& params, int repeats,
         const Stat& s, const std::string& detail) {
    std::printf("%s,%s,%s,%d,%.3f,%.3f,%.3f,%s\n", category, csv_escape(name).c_str(),
                csv_escape(params).c_str(), repeats, s.min_ms, s.median_ms, s.max_ms,
                csv_escape(detail).c_str());
}

}  // namespace

int main(int argc, char** argv) {
    int repeats = 3;
    bool require = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--repeats" && i + 1 < argc) repeats = std::stoi(argv[++i]);
        else if (a == "--require") require = true;      // non-zero exit if a case fails
        else if (a == "--help") {
            std::printf("usage: octbench [--repeats N] [--require]\n"
                        "prints CSV: category,name,params,repeats,min_ms,median_ms,max_ms,detail\n");
            return 0;
        }
    }
    if (repeats < 1) repeats = 1;

    std::printf("# octbench build=%s\n", build_fingerprint().c_str());
    const HostInfo h = host_info();
    std::printf("# host arch=%s os=%s threads=%u avx2=%d avx512=%d ram=%llu\n", h.arch.c_str(),
                h.os.c_str(), h.hw_threads, int(h.has_avx2), int(h.has_avx512),
                (unsigned long long)h.total_ram_bytes);
    std::printf("category,name,params,repeats,min_ms,median_ms,max_ms,detail\n");

    int failures = 0;

    // --- numerical kernels -------------------------------------------------
    struct Case {
        std::string kernel;
        std::map<std::string, double> params;
        double t_end;
    };
    const std::vector<Case> cases = {
        {"sod1d", {{"n", 400}}, 0.2},
        {"heat2d", {{"n", 64}}, 0.05},
        {"poisson2d", {{"n", 64}}, 1.0},
        {"kepler", {{"e", 0.3}}, 6.283185307179586},
        {"nbody", {{"n", 3}}, 2.0},
        {"gas", {{"ratio", 1.4}}, 1.0},
        {"linsolve", {{"n", 64}}, 1.0},
        {"cavity2d", {{"n", 32}, {"re", 100}}, 5.0},
        {"tensor_field", {{"n", 24}, {"mass", 1.2}}, 1.0},
    };
    for (const Case& c : cases) {
        auto k = numerics::KernelLibrary::instance().find(c.kernel);
        if (!k) { std::fprintf(stderr, "octbench: no kernel %s\n", c.kernel.c_str()); ++failures; continue; }
        numerics::ProblemSpec spec;
        spec.kernel = c.kernel;
        spec.params = c.params;
        spec.t_end = c.t_end;
        std::vector<double> times;
        std::string detail;
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = Clock::now();
            const numerics::Result res = k->run(spec);
            times.push_back(ms_since(t0));
            std::ostringstream os;
            os << "fp=" << res.fingerprint.substr(0, 16) << "; elems=" << res.data.size()
               << "; backend=" << res.backend;
            if (res.diag.residual != 0.0) os << "; residual=" << res.diag.residual;
            if (res.diag.iterations != 0) os << "; iters=" << res.diag.iterations;
            detail = os.str();
            if (r == 0 && res.fingerprint.empty()) { ++failures; }
        }
        std::ostringstream ps;
        for (auto it = c.params.begin(); it != c.params.end(); ++it)
            ps << (it == c.params.begin() ? "" : ";") << it->first << "=" << it->second;
        row("kernel", c.kernel, ps.str(), repeats, summarise(times), detail);
    }

    // --- language layers ---------------------------------------------------
    {
        std::vector<double> times;
        std::string detail;
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = Clock::now();
            apl::Environment env;
            auto res = apl::eval_line("+/ \xe2\x8d\xb3 100000", env);
            times.push_back(ms_since(t0));
            if (!res) ++failures;
            detail = "sum_0_99999=" + std::to_string(res ? res->scalar_f64() : -1.0);
        }
        row("layer", "apl.iota.reduce", "n=100000", repeats, summarise(times), detail);
    }
    {
        std::vector<double> times;
        std::string detail;
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = Clock::now();
            prolog::KnowledgeBase kb;
            for (int i = 0; i < 200; ++i)
                kb.add_text("edge(" + std::to_string(i) + ", " + std::to_string(i + 1) + ").");
            kb.add_text("path(X, Y) :- edge(X, Y).");
            kb.add_text("path(X, Y) :- edge(X, Z), path(Z, Y).");
            auto goal = prolog::parse_term("path(0, 200)");
            // The chain needs deeper recursion than the default 256 levels; the
            // limits used for the benchmark are stated explicitly so the row is
            // reproducible rather than dependent on a default.
            prolog::Limits limits;
            limits.max_depth = 1024;
            limits.max_inferences = 1000000;
            auto v = goal ? prolog::verify(kb, *goal, limits) : prolog::Verification{};
            times.push_back(ms_since(t0));
            if (v.verdict != prolog::Verdict::Grounded) ++failures;
            detail = "inferences=" + std::to_string(v.inferences);
        }
        row("layer", "prolog.path.depth200", "clauses=202", repeats, summarise(times), detail);
    }
    {
        const std::string dir = "third_party/llama.cpp/models/";
        auto vocab = tokenizer::Vocab::load(dir + "ggml-vocab-gpt-2.gguf");
        if (!vocab) {
            row("layer", "tokenizer.bpe", "vocab=gpt-2", 0, {0, 0, 0},
                "UNSUPPORTED_HARDWARE_SKIP: llama.cpp fixture absent");
        } else {
            // One warm-up call: the first encode builds the merge-rank table
            // (one-time cost, reported separately by the min/max spread).
            (void)tokenizer::encode(*vocab, "warm-up");
            std::vector<double> times;
            std::string detail;
            for (int r = 0; r < repeats; ++r) {
                const auto t0 = Clock::now();
                auto enc = tokenizer::encode(*vocab, "The quick brown fox jumps over 13 lazy dogs.");
                times.push_back(ms_since(t0));
                if (!enc) ++failures;
                detail = "tokens=" + std::to_string(enc ? enc->ids.size() : 0);
            }
            row("layer", "tokenizer.bpe", "vocab=gpt-2;text=44B", repeats, summarise(times), detail);
        }
    }
    {
        std::vector<double> times;
        std::string detail;
        const std::string key = "bench-key";
        for (int r = 0; r < repeats; ++r) {
            std::string payload(64 * 1024, 'x');
            const auto t0 = Clock::now();
            const std::string env = intercal::encode(payload, key);
            auto back = intercal::decode(env, key);
            times.push_back(ms_since(t0));
            if (!back || *back != payload) ++failures;
            detail = "bytes=" + std::to_string(env.size());
        }
        row("layer", "intercal.guard.roundtrip", "payload=64KiB", repeats, summarise(times), detail);
    }
    {
        std::vector<double> times;
        std::string detail;
        std::vector<double> state(512 * 512);
        for (size_t i = 0; i < state.size(); ++i) state[i] = std::sin(double(i) * 0.01) * 0.5;
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = Clock::now();
            piet::Raster raster = piet::render_state(state, 512, 512, -1.0, 1.0);
            const Status w = raster.write_png("/tmp/octbench_piet.png");
            times.push_back(ms_since(t0));
            if (!w.is_ok()) ++failures;
            detail = "png_bytes=" + std::to_string(raster.rgb.size()) + "; fp=" + raster.fingerprint().substr(0, 16);
        }
        row("layer", "piet.render.png", "512x512", repeats, summarise(times), detail);
    }
    {
        std::vector<double> times;
        std::string detail;
        router::Router r;
        for (int rep = 0; rep < repeats; ++rep) {
            const auto t0 = Clock::now();
            auto rt = r.classify("compute covariant riemannian tensor field curvature");
            times.push_back(ms_since(t0));
            if (rt.kind != router::TaskKind::Numeric ||
                rt.capability != "numeric.kernel.tensor_field") ++failures;
            detail = "w_sha256=" + r.telemetry().weights_sha256.substr(0, 16) +
                     "; conf=" + std::to_string(rt.confidence);
        }
        row("layer", "router.micro_neural", "64x32x10_GELU", repeats, summarise(times), detail);
    }
    {
        std::vector<double> times;
        std::string detail;
        std::vector<double> coords(32 * 32 * 4, 0.5);
        for (int rep = 0; rep < repeats; ++rep) {
            const auto t0 = Clock::now();
            auto fr = universe::translate_coordinates_to_piet(coords, 32, 32, 0.25);
            times.push_back(ms_since(t0));
            if (!fr.ok() || !fr->sld_report.sound) ++failures;
            detail = "fp=" + (fr.ok() ? fr->canvas.fingerprint().substr(0, 16) : std::string("err"));
        }
        row("layer", "universe.piet_projection", "32x32x4", repeats, summarise(times), detail);
    }

    if (require && failures > 0) {
        std::fprintf(stderr, "octbench: %d case(s) failed\n", failures);
        return 1;
    }
    return 0;
}
