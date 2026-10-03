// Octopus Hybrid AI Engine -- capability router.
//
// The router is the engine's front door: it decides which language-DNA module a
// request belongs to and then executes it. Two rules keep it honest:
//
//   1. Classification is explicit about its evidence. An explicit prefix
//      ("kernel:kepler e=0.3") is a declaration, not a guess, and is reported
//      with confidence 1.0 and the reason "explicit prefix". A heuristic guess
//      says so, quotes the hint it matched, and carries a lower confidence.
//   2. A request that cannot be routed, or that routes to a module which cannot
//      serve it (no model loaded, no strategy set), fails closed with a Status.
//      The router never fabricates an answer.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"
#include "octopus/module.hpp"
#include "octopus/prolog.hpp"

#include <string>
#include <vector>

namespace oct::router {

enum class TaskKind : uint8_t {
    Numeric,     // Fortran-compatible numerical kernels
    Array,       // APL-inspired array IR
    Symbolic,    // Lisp-style AST/data layer
    Logic,       // Prolog-style verification
    Parallel,    // Occam/CSP task graph
    LowLevel,    // Forth-style stack VM
    Actor,       // Smalltalk-style message objects
    Guard,       // INTERCAL-style compatibility/fuzz guardrail
    Visual,      // Piet canvas IR
    Language,    // quantized SLM through llama.cpp
    Unknown,     // refused
};

const char* to_string(TaskKind k) noexcept;
TaskKind    task_kind_from_string(std::string_view s) noexcept;

struct Route {
    TaskKind    kind = TaskKind::Unknown;
    std::string module;       // module name the request is dispatched to
    std::string capability;   // dotted capability it resolves through
    std::string reason;       // why this route was chosen
    double      confidence = 0.0;   // 1.0 only for an explicit declaration
    bool        explicit_route = false;
    Json to_json() const;
};

struct RouterOptions {
    std::string model_path;          // for Language tasks; empty = no model
    bool        allow_stub = false;  // permit the labelled deterministic stub
    std::string output_path;         // for Visual tasks (PNG)
};

class Router {
public:
    Router(const Registry* registry = nullptr, const prolog::KnowledgeBase* kb = nullptr);

    // Classification only: no side effects, no execution.
    Route classify(std::string_view request) const;

    // Classify, then execute through the owning module. The returned JSON always
    // contains the route (so a caller can audit *why* an answer came from a
    // given module) and the module's own result.
    Outcome<Json> execute(std::string_view request, const RouterOptions& options = {}) const;

    // Exposed so tests can drive the individual executors.
    Outcome<Json> execute_route(const Route& route, std::string_view payload,
                                const RouterOptions& options = {}) const;

    std::vector<std::string> example_requests() const;

private:
    const Registry*                registry_;
    const prolog::KnowledgeBase*   kb_;
};

std::shared_ptr<oct::Module> make_router_module();

}  // namespace oct::router
