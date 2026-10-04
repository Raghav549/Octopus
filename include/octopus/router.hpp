// Octopus Hybrid AI Engine -- Inline Micro-Neural Capability Router.
//
// Replaces static rule-based classification with an inline 2-layer GELU
// micro-neural network router (64-D prompt embedding -> 32-D hidden layer with
// LayerNorm & GELU -> 10-class task head + 9-class Fortran kernel head) that:
//   1. Adaptively learns high-confidence vector representations of prompt
//      strings via analytic backpropagation (cross-entropy + momentum SGD).
//   2. Modulates routing logits using real-time execution speed EMAs measured
//      across every runtime language module.
//   3. Seals its learned weight matrices inside an INTERCAL tamper-evident
//      structural envelope and renders its neural synapse pathways onto the
//      20-colour Piet visual canvas.
//   4. Rejects out-of-distribution (OOD) / unroutable inputs via logit-energy
//      thresholding, never fabricating answers.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"
#include "octopus/intercal.hpp"
#include "octopus/module.hpp"
#include "octopus/piet.hpp"
#include "octopus/prolog.hpp"

#include <array>
#include <mutex>
#include <string>
#include <vector>

namespace oct::router {

enum class TaskKind : uint8_t {
    Numeric,     // Fortran 2023 numerical physics kernels
    Array,       // APL array IR & context matrix compressor
    Symbolic,    // LISP autonomous self-modifying AST layer
    Logic,       // Prolog SLD chronological backtracking guardrail
    Parallel,    // Occam CSP parallel reasoning
    LowLevel,    // Forth stack + hardware register VM
    Actor,       // Smalltalk live-cell message objects
    Guard,       // INTERCAL obfuscated core shield
    Visual,      // Piet visual canvas logic engine
    Language,    // Quantized SLM through llama.cpp
    Unknown,     // Refused / out-of-distribution
};

const char* to_string(TaskKind k) noexcept;
TaskKind    task_kind_from_string(std::string_view s) noexcept;

struct Route {
    TaskKind            kind = TaskKind::Unknown;
    std::string         module;       // module name the request is dispatched to
    std::string         capability;   // dotted capability it resolves through
    std::string         reason;       // neural receptor evidence or explicit prefix
    double              confidence = 0.0;   // 1.0 for explicit prefix; softmax p in (0,1) for neural route
    bool                explicit_route = false;
    double              execution_speed_ema_ms = 0.0;
    double              logit_energy = 0.0;
    std::vector<double> embedding_preview;  // first 8 dimensions of the 64-D prompt vector
    Json to_json() const;
};

struct RouterOptions {
    std::string model_path;          // for Language tasks; empty = no model
    std::string output_path;         // for Visual tasks (PNG/PPM)
};

struct NeuralTelemetry {
    int64_t                  training_steps = 0;
    double                   last_loss = 0.0;
    double                   curriculum_accuracy = 0.0;
    std::string              weights_sha256;
    std::array<double, 10>   module_speed_ema_ms{};
    std::array<int64_t, 10>  module_dispatches{};
    Json                     to_json() const;
};

class Router {
public:
    static constexpr size_t kEmbedDim   = 64;
    static constexpr size_t kHiddenDim  = 32;
    static constexpr size_t kNumClasses = 10;
    static constexpr size_t kNumKernels = 9;

    Router(const Registry* registry = nullptr, const prolog::KnowledgeBase* kb = nullptr);

    // Compute 64-D normalized neural feature embedding for any prompt string.
    std::array<double, kEmbedDim> embed(std::string_view request,
                                        std::string* top_feature_out = nullptr,
                                        int* matched_kernel_out = nullptr) const;

    // Neural classification: embeds prompt, runs 2-layer GELU forward pass
    // modulated by real-time module execution speed EMAs, and applies OOD energy gating.
    Route classify(std::string_view request) const;

    // Online learning step: updates neural router weights via backpropagation
    // on (prompt, target_kind) and optionally updates execution speed EMA.
    double learn(std::string_view prompt, TaskKind target_kind,
                 double execution_seconds = -1.0, double learning_rate = 0.08);

    // Record real-time execution latency for a task module.
    void record_execution_speed(TaskKind kind, double execution_seconds) const;

    // Classify, execute through the owning module, measure real-time execution
    // speed, and reinforce the neural router online.
    Outcome<Json> execute(std::string_view request, const RouterOptions& options = {}) const;

    // Exposed so tests can drive individual executors.
    Outcome<Json> execute_route(const Route& route, std::string_view payload,
                                const RouterOptions& options = {}) const;

    NeuralTelemetry telemetry() const;

    // Seal the learned neural weight tensors inside an INTERCAL core shield envelope.
    intercal::ShieldEnvelope seal_weights(std::string_view key = "octopus-neural-router",
                                          intercal::AuditChain* chain = nullptr) const;

    // Render the neural router's synapse weights & hidden activations for a prompt
    // onto a 20-colour Piet visual canvas.
    piet::Raster render_synapses(std::string_view prompt, int width = 32, int height = 16) const;

    std::vector<std::string> example_requests() const;

private:
    void init_and_pretrain_network();

    const Registry*              registry_;
    const prolog::KnowledgeBase* kb_;

    mutable std::mutex                             mu_;
    // Layer 1: kEmbedDim (64) -> kHiddenDim (32)
    mutable std::array<double, kHiddenDim * kEmbedDim> w1_{};
    mutable std::array<double, kHiddenDim>             b1_{};
    mutable std::array<double, kHiddenDim * kEmbedDim> v_w1_{};
    // Layer 2 (task head): kHiddenDim (32) -> kNumClasses (10)
    mutable std::array<double, kNumClasses * kHiddenDim> w2_{};
    mutable std::array<double, kNumClasses>              b2_{};
    mutable std::array<double, kNumClasses * kHiddenDim> v_w2_{};
    // Layer 2b (physics kernel sub-head): kHiddenDim (32) -> kNumKernels (9)
    mutable std::array<double, kNumKernels * kHiddenDim> wk_{};
    mutable std::array<double, kNumKernels>              bk_{};

    // Real-time execution speed EMAs (in milliseconds) per TaskKind (0..9)
    mutable std::array<double, kNumClasses>  speed_ema_ms_{};
    mutable std::array<int64_t, kNumClasses> dispatches_{};
    mutable int64_t                          training_steps_ = 0;
    mutable double                           last_loss_ = 0.0;
    mutable double                           curriculum_accuracy_ = 0.0;
};

std::shared_ptr<oct::Module> make_router_module();

}  // namespace oct::router
