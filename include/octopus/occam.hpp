// Octopus Hybrid AI Engine -- Occam-style parallel reasoning layer.
//
// Role in the engine (ARCHITECTURE_SPEC.md): run independent reasoning
// strategies concurrently and only accept a result when they agree inside a
// declared tolerance. Disagreement is not averaged away, it fails closed.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

namespace oct { class Module; }

namespace oct::occam {

// A bounded, closeable CSP channel. send() blocks while the buffer is full,
// recv() blocks while it is empty; closing wakes every waiter.
template <class T>
class Channel {
public:
    explicit Channel(size_t capacity = 1) : cap_(capacity == 0 ? 1 : capacity) {}

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    bool send(T v) {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock, [&] { return closed_ || q_.size() < cap_; });
        if (closed_) return false;
        q_.push(std::move(v));
        not_empty_.notify_one();
        return true;
    }

    std::optional<T> recv() {
        std::unique_lock<std::mutex> lock(mu_);
        not_empty_.wait(lock, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return std::nullopt;      // closed and drained
        T v = std::move(q_.front());
        q_.pop();
        not_full_.notify_one();
        return v;
    }

    std::optional<T> try_recv() {
        std::lock_guard<std::mutex> lock(mu_);
        if (q_.empty()) return std::nullopt;
        T v = std::move(q_.front());
        q_.pop();
        not_full_.notify_one();
        return v;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mu_);
        return closed_;
    }
    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return q_.size();
    }

private:
    mutable std::mutex      mu_;
    std::condition_variable not_empty_, not_full_;
    std::queue<T>           q_;
    size_t                  cap_;
    bool                    closed_ = false;
};

// Named strategy: a pure function of nothing (the caller closes over its inputs)
// returning a result or an error.
struct Strategy {
    std::string                        name;
    std::function<Outcome<double>()>   run;
};

struct Verdict {
    bool                     agreed = false;
    double                   value = 0.0;
    double                   tolerance = 0.0;
    double                   max_spread = 0.0;      // max |value - winner|
    std::string              winner;                 // fastest strategy
    std::vector<std::pair<std::string, double>> results;
    std::vector<double>      durations;              // seconds, parallel run
    double                   wall_seconds = 0.0;
    double                   serial_seconds = 0.0;   // sum of strategy durations
    double                   speedup = 0.0;          // serial / wall (measured)
    Json to_json() const;
};

// Run all strategies concurrently on hardware threads. A single result is
// accepted only if every strategy that produced one lies within `tolerance` of
// the median value; otherwise `agreed` is false and the reasons are reported.
Verdict parallel_verify(const std::vector<Strategy>& strategies, double tolerance);

// Map fn over inputs in parallel; results keep input order.
template <class T, class F>
std::vector<Outcome<T>> par_map(const std::vector<T>& inputs, F fn) {
    std::vector<Outcome<T>> out(inputs.size());
    std::vector<std::thread> threads;
    const size_t n = inputs.size();
    const size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t chunks = std::min(n, hw);
    for (size_t c = 0; c < chunks; ++c) {
        threads.emplace_back([&, c] {
            for (size_t i = c; i < n; i += chunks) out[i] = fn(inputs[i]);
        });
    }
    for (auto& t : threads) t.join();
    return out;
}

// Simultaneous Occam (parallel CSP channels) + Forth (low-level stack &
// hardware register file) + Smalltalk (dynamic message-passing actor system)
// triad coordinator.
struct TriadExecution {
    bool                  ok = false;
    Verdict               occam_verdict;
    double                forth_stack_top = 0.0;
    std::vector<double>   forth_registers;
    std::string           smalltalk_reply;
    int64_t               channel_messages = 0;
    Json                  to_json() const;
};

TriadExecution coordinate_triad(std::string_view forth_program, double seed_value = 3.0);

std::shared_ptr<oct::Module> make_occam_module();

}  // namespace oct::occam
