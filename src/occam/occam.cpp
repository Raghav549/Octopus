// Octopus Hybrid AI Engine -- Occam-style parallel reasoning (implementation).
// SPDX-License-Identifier: MIT
#include "octopus/occam.hpp"

#include "octopus/module.hpp"

#include <algorithm>
#include <cmath>

namespace oct::occam {

Json Verdict::to_json() const {
    Json j;
    j.begin_object();
    j.field("agreed", agreed);
    j.field("value", value);
    j.field("tolerance", tolerance);
    j.field("max_spread", max_spread);
    j.field("winner", winner);
    j.field("wall_seconds", wall_seconds);
    j.field("serial_seconds", serial_seconds);
    j.field("speedup", speedup);
    j.begin_array("results");
    for (const auto& r : results) {
        j.begin_object();
        j.field("strategy", r.first);
        j.field("value", r.second);
        j.end_object();
    }
    j.end_array();
    j.begin_array("durations");
    for (double d : durations) j.value(d);
    j.end_array();
    j.end_object();
    return j;
}

Verdict parallel_verify(const std::vector<Strategy>& strategies, double tolerance) {
    Verdict v;
    v.tolerance = tolerance;
    if (strategies.empty()) return v;

    const size_t n = strategies.size();
    std::vector<Outcome<double>> values(n);
    std::vector<double> durations(n, 0.0);

    const auto t0 = Clock::now();
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        threads.emplace_back([&, i] {
            const auto s0 = Clock::now();
            try {
                values[i] = strategies[i].run();
            } catch (const std::exception& e) {
                values[i] = Status::internal(std::string("strategy threw: ") + e.what());
            }
            durations[i] = seconds_since(s0);
        });
    }
    for (auto& t : threads) t.join();
    v.wall_seconds = seconds_since(t0);
    v.durations = durations;
    for (double d : durations) v.serial_seconds += d;
    v.speedup = v.wall_seconds > 0.0 ? v.serial_seconds / v.wall_seconds : 0.0;

    std::vector<double> good;
    for (size_t i = 0; i < n; ++i) {
        if (values[i]) {
            good.push_back(*values[i]);
            v.results.emplace_back(strategies[i].name, *values[i]);
        }
    }
    if (good.size() != n) return v;                  // some strategy failed: no verdict

    std::vector<double> sorted = good;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    double spread = 0.0;
    size_t winner = 0;
    for (size_t i = 0; i < n; ++i) {
        spread = std::max(spread, std::fabs(*values[i] - median));
        if (durations[i] < durations[winner]) winner = i;
    }
    v.value = median;
    v.max_spread = spread;
    v.winner = strategies[winner].name;
    v.agreed = spread <= tolerance;
    return v;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class OccamModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "occam.parallel";
        i.version = "0.1.0";
        i.language = "Occam-style CSP (C++20 threads)";
        i.role = "Parallel reasoning: independent strategies run concurrently and must agree "
                 "within a declared tolerance; disagreement fails closed.";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"parallel.strategies", "parallel.channel", "parallel.par_map",
                          "parallel.consensus", "parallel.timing"};
        i.limitations = {
            "threads are OS threads: the layer measures real wall-clock speedup and reports it, "
            "it does not claim linear scaling",
            "consensus is a tolerance check on doubles, not a proof; strategies must be "
            "genuinely independent for the check to mean anything",
            "no work stealing, no distributed execution, no GPU offload",
            "one shared address space: a strategy that crashes the process is not contained "
            "(use the actor runtime for that)"};
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }

    std::string describe() const override {
        return "Runs independent strategies on hardware threads, accepts the median only if "
               "every strategy agrees within tolerance, and reports measured speedup.";
    }

    Status self_check() override {
        // Three different partial-sum strategies for pi; agreement is the check.
        auto partial = [](int64_t steps) {
            return [steps]() -> Outcome<double> {
                double s = 0.0;
                for (int64_t k = 0; k < steps; ++k)
                    s += (k % 2 == 0 ? 1.0 : -1.0) / double(2 * k + 1);
                return 4.0 * s;
            };
        };
        std::vector<Strategy> strategies = {
            {"leibniz.1e6", partial(1000000)},
            {"leibniz.1e6b", partial(1000000)},
            {"euler.4e5", []() -> Outcome<double> {
                 double s = 0.0;
                 for (int64_t k = 0; k < 400000; ++k) s += 1.0 / double((2 * k + 1) * (2 * k + 1));
                 return std::sqrt(8.0 * s);
             }},
        };
        Verdict ok = parallel_verify(strategies, 5e-3);
        if (!ok.agreed) return Status::internal("occam: agreeing strategies disagreed");
        if (std::fabs(ok.value - 3.14159265358979) > 5e-3)
            return Status::internal("occam: pi strategies are wrong");
        if (!(ok.speedup > 0.0)) return Status::internal("occam: speedup was not measured");

        // Disagreeing strategies must fail closed.
        std::vector<Strategy> bad = {
            {"a", []() -> Outcome<double> { return 1.0; }},
            {"b", []() -> Outcome<double> { return 2.0; }},
        };
        if (parallel_verify(bad, 1e-9).agreed)
            return Status::internal("occam: disagreement was accepted");

        // Channel semantics: bounded send/recv across threads, close wakes waiters.
        Channel<int> ch(2);
        std::thread producer([&] {
            for (int i = 0; i < 10; ++i) ch.send(i);
            ch.close();
        });
        int sum = 0;
        while (auto v = ch.recv()) sum += *v;
        producer.join();
        if (sum != 45) return Status::internal("occam: channel lost messages");

        auto mapped = par_map<int>({1, 2, 3, 4, 5}, [](int x) -> Outcome<int> { return x * x; });
        int total = 0;
        for (auto& m : mapped) {
            if (!m) return Status::internal("occam: par_map produced an error");
            total += *m;
        }
        if (total != 55) return Status::internal("occam: par_map result wrong");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_occam_module() {
    return std::make_shared<OccamModule>();
}

}  // namespace oct::occam
