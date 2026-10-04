// Octopus Hybrid AI Engine -- Inline Micro-Neural Capability Router (implementation).
//
// Replaces the static rule-based classifier with a 2-layer GELU micro-neural
// network router (`64 -> 32 -> 10` + `32 -> 9` physics kernel head) trained
// with analytical backpropagation (cross-entropy + momentum SGD) and modulated
// by real-time execution speed EMAs of each language runtime module.
//
// SPDX-License-Identifier: MIT
#include "octopus/router.hpp"

#include "octopus/apl.hpp"
#include "octopus/intercal.hpp"
#include "octopus/lisp.hpp"
#include "octopus/llm.hpp"
#include "octopus/numerics.hpp"
#include "octopus/occam.hpp"
#include "octopus/piet.hpp"
#include "octopus/stackvm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace oct::router {

const char* to_string(TaskKind k) noexcept {
    switch (k) {
        case TaskKind::Numeric:  return "numeric";
        case TaskKind::Array:    return "array";
        case TaskKind::Symbolic: return "symbolic";
        case TaskKind::Logic:    return "logic";
        case TaskKind::Parallel: return "parallel";
        case TaskKind::LowLevel: return "low_level";
        case TaskKind::Actor:    return "actor";
        case TaskKind::Guard:    return "guard";
        case TaskKind::Visual:   return "visual";
        case TaskKind::Language: return "language";
        case TaskKind::Unknown:  return "unknown";
    }
    return "unknown";
}

TaskKind task_kind_from_string(std::string_view s) noexcept {
    if (s == "numeric") return TaskKind::Numeric;
    if (s == "array") return TaskKind::Array;
    if (s == "symbolic") return TaskKind::Symbolic;
    if (s == "logic") return TaskKind::Logic;
    if (s == "parallel") return TaskKind::Parallel;
    if (s == "low_level" || s == "lowlevel") return TaskKind::LowLevel;
    if (s == "actor") return TaskKind::Actor;
    if (s == "guard") return TaskKind::Guard;
    if (s == "visual") return TaskKind::Visual;
    if (s == "language") return TaskKind::Language;
    return TaskKind::Unknown;
}

Json Route::to_json() const {
    Json j;
    j.begin_object();
    j.field("kind", to_string(kind));
    j.field("module", module);
    j.field("capability", capability);
    j.field("reason", reason);
    j.field("confidence", confidence);
    j.field("explicit", explicit_route);
    j.field("execution_speed_ema_ms", execution_speed_ema_ms);
    j.field("logit_energy", logit_energy);
    if (!embedding_preview.empty()) {
        j.key("embedding_preview");
        j.begin_array();
        for (double v : embedding_preview) j.value(v);
        j.end_array();
    }
    j.end_object();
    return j;
}

Json NeuralTelemetry::to_json() const {
    Json j;
    j.begin_object();
    j.field("architecture", "MicroNeuralRouter-64x32x10-GELU");
    j.field("training_steps", training_steps);
    j.field("last_loss", last_loss);
    j.field("curriculum_accuracy", curriculum_accuracy);
    j.field("weights_sha256", weights_sha256);
    j.key("module_speed_ema_ms");
    j.begin_object();
    for (size_t i = 0; i < 10; ++i) {
        j.field(to_string(static_cast<TaskKind>(i)), module_speed_ema_ms[i]);
    }
    j.end_object();
    j.key("module_dispatches");
    j.begin_object();
    for (size_t i = 0; i < 10; ++i) {
        j.field(to_string(static_cast<TaskKind>(i)), module_dispatches[i]);
    }
    j.end_object();
    j.end_object();
    return j;
}

namespace {

struct PrefixRoute {
    const char* prefix;
    TaskKind    kind;
    const char* module;
    const char* capability;
};

const PrefixRoute kPrefixes[] = {
    {"kernel:", TaskKind::Numeric, "fortran.kernels", "numeric.kernel"},
    {"numeric:", TaskKind::Numeric, "fortran.kernels", "numeric.kernel"},
    {"fortran:", TaskKind::Numeric, "fortran.kernels", "numeric.kernel"},
    {"apl:", TaskKind::Array, "apl.arrays", "array.eval"},
    {"array:", TaskKind::Array, "apl.arrays", "array.eval"},
    {"lisp:", TaskKind::Symbolic, "lisp.selfmod", "lisp.eval"},
    {"symbolic:", TaskKind::Symbolic, "lisp.selfmod", "lisp.eval"},
    {"prolog:", TaskKind::Logic, "prolog.logic", "logic.verify"},
    {"logic:", TaskKind::Logic, "prolog.logic", "logic.verify"},
    {"occam:", TaskKind::Parallel, "occam.parallel", "parallel.consensus"},
    {"parallel:", TaskKind::Parallel, "occam.parallel", "parallel.consensus"},
    {"forth:", TaskKind::LowLevel, "forth.stackvm", "forth.run"},
    {"stack:", TaskKind::LowLevel, "forth.stackvm", "forth.run"},
    {"smalltalk:", TaskKind::Actor, "smalltalk.actors", "actors.message_passing"},
    {"actor:", TaskKind::Actor, "smalltalk.actors", "actors.message_passing"},
    {"guard:", TaskKind::Guard, "intercal.guard", "guard.encode"},
    {"intercal:", TaskKind::Guard, "intercal.guard", "guard.encode"},
    {"piet:", TaskKind::Visual, "piet.visual", "visual.render"},
    {"visual:", TaskKind::Visual, "piet.visual", "visual.render"},
    {"llm:", TaskKind::Language, "llm.host", "llm.generate"},
    {"language:", TaskKind::Language, "llm.host", "llm.generate"},
    {"ask:", TaskKind::Language, "llm.host", "llm.generate"},
};

struct ClassMeta {
    TaskKind    kind;
    const char* module;
    const char* capability;
};

const ClassMeta kClassMeta[Router::kNumClasses] = {
    {TaskKind::Numeric,  "fortran.kernels",  "numeric.kernel"},
    {TaskKind::Array,    "apl.arrays",       "array.eval"},
    {TaskKind::Symbolic, "lisp.selfmod",     "lisp.eval"},
    {TaskKind::Logic,    "prolog.logic",     "logic.verify"},
    {TaskKind::Parallel, "occam.parallel",   "parallel.consensus"},
    {TaskKind::LowLevel, "forth.stackvm",    "forth.run"},
    {TaskKind::Actor,    "smalltalk.actors", "actors.message_passing"},
    {TaskKind::Guard,    "intercal.guard",   "guard.encode"},
    {TaskKind::Visual,   "piet.visual",      "visual.render"},
    {TaskKind::Language, "llm.host",         "llm.generate"},
};

const char* kKernelNames[Router::kNumKernels] = {
    "sod1d", "heat2d", "poisson2d", "cavity2d", "kepler",
    "nbody", "gas",    "linsolve",  "tensor_field"
};

struct KernelReceptor {
    const char* token;
    int         kernel_idx;
    int         feat_dim;
    double      weight;
};

const KernelReceptor kKernelReceptors[] = {
    {"shock tube",    0, 0,  2.8}, {"sod",           0, 0,  2.6}, {"riemann",       0, 0,  2.5},
    {"euler",         0, 0,  2.2}, {"hllc",          0, 0,  2.4},
    {"heat",          1, 1,  2.8}, {"diffusion",     1, 1,  2.6}, {"ftcs",          1, 1,  2.4},
    {"fourier",       1, 1,  2.1}, {"thermal",       1, 1,  2.0},
    {"poisson",       2, 2,  2.8}, {"laplace",       2, 2,  2.6}, {"elliptic",      2, 2,  2.3},
    {"sor",           2, 2,  2.1},
    {"cavity",        3, 3,  2.8}, {"lid",           3, 3,  2.5}, {"navier",        3, 3,  2.7},
    {"chorin",        3, 3,  2.5}, {"incompressible",3, 3,  2.3},
    {"orbit",         4, 4,  2.8}, {"kepler",        4, 4,  2.9}, {"two-body",      4, 4,  2.6},
    {"symplectic",    4, 4,  2.4}, {"yoshida",       4, 4,  2.4}, {"eccentricity",  4, 4,  2.2},
    {"gravity",       5, 5,  2.8}, {"n-body",        5, 5,  2.9}, {"nbody",         5, 5,  2.9},
    {"verlet",        5, 5,  2.4}, {"plummer",       5, 5,  2.4}, {"three-body",    5, 5,  2.5},
    {"isentropic",    6, 6,  2.8}, {"ideal gas",     6, 6,  2.8}, {"thermodynamic", 6, 6,  2.6},
    {"calorically",   6, 6,  2.3}, {"adiabatic",     6, 6,  2.3},
    {"linear system", 7, 7,  2.8}, {"solve matrix",  7, 7,  2.8}, {"linsolve",      7, 7,  2.9},
    {"lu pivot",      7, 7,  2.4}, {"ax = b",        7, 7,  2.4},
    {"tensor_field",  8, 8,  2.9}, {"tensor field",  8, 8,  2.8}, {"riemannian",    8, 8,  2.6},
    {"covariant",     8, 8,  2.5}, {"bianchi",       8, 8,  2.6}, {"weyl",          8, 8,  2.5},
};

struct DomainReceptor {
    const char* token;
    int         feat_dim;
    double      weight;
};

const DomainReceptor kDomainReceptors[] = {
    // APL array receptors (dims 10..15)
    {"\xe2\x8d\xb3", 10, 3.2}, {"\xe2\x8d\xb4", 11, 3.2}, {"\xe2\x8c\xb9", 12, 3.2},
    {"\xe2\x8d\x89", 13, 3.2}, {"\xe2\x8c\xbd", 14, 3.2}, {"\xe2\x88\x98", 14, 3.0},
    {"+/",           15, 3.0}, {"+\\",          15, 3.0}, {"\xe2\x86\x90", 10, 2.6},
    {"+.×",          12, 3.0}, {"reshape",      11, 2.2}, {"outer product",14, 2.4},
    // LISP symbolic self-modification receptors (dims 16..20)
    {"(define",      16, 3.1}, {"(lambda",      17, 3.1}, {"(let",         18, 2.8},
    {"(begin",       18, 2.6}, {"s-expression", 19, 2.5}, {"hot-patch",    20, 2.6},
    {"self-modify",  20, 2.6}, {"defun",        16, 2.6},
    // Prolog SLD logic receptors (dims 21..25)
    {":-",           21, 3.2}, {"prove ",       22, 2.8}, {"verify ",      23, 2.8},
    {"is it true",   24, 2.8}, {"horn clause",  25, 2.6}, {"sld ",         25, 2.6},
    {"ancestor(",    22, 2.7}, {"parent(",      22, 2.7}, {"backtrack",    25, 2.4},
    // Occam parallel CSP receptors (dims 26..29)
    {"parallel",     26, 2.7}, {"consensus",    27, 2.8}, {"channel",      28, 2.5},
    {"occam",        26, 2.9}, {"concurrent",   29, 2.5}, {"strategies",   27, 2.5},
    // Forth low-level stack & register VM receptors (dims 30..34)
    {": ",           30, 2.9}, {"dup ",         31, 2.9}, {"swap ",        32, 2.9},
    {"drop ",        32, 2.9}, {"over ",        33, 2.7}, {"rot ",         33, 2.7},
    {"reg@",         34, 3.0}, {"reg!",         34, 3.0}, {">r",           34, 2.8},
    {"r>",           34, 2.8}, {"r0@",          34, 3.0}, {"r0!",          34, 3.0},
    // Smalltalk actor message-passing receptors (dims 35..37)
    {"smalltalk",    35, 3.0}, {"actor",        36, 2.7}, {"mailbox",      37, 2.7},
    {"supervise",    37, 2.6}, {"live-cell",    35, 2.8},
    // INTERCAL obfuscated core shield receptors (dims 38..41)
    {"obfuscate",    38, 2.9}, {"seal",         39, 2.8}, {"tamper",       40, 2.9},
    {"audit",        41, 2.8}, {"intercal",     38, 3.0}, {"mingle",       39, 2.6},
    // Piet visual canvas logic receptors (dims 42..45)
    {"render",       42, 2.9}, {"visuali",      43, 2.9}, {"plot",         44, 2.8},
    {"colourize",    45, 2.9}, {"colorize",     45, 2.9}, {"canvas",       42, 2.7},
    {"codel",        43, 2.8}, {"synapse",      44, 2.6},
    // Natural-language SLM receptors (dims 46..50)
    {"what ",        46, 2.3}, {"write ",       47, 2.4}, {"explain",      48, 2.5},
    {"summarize",    49, 2.5}, {"poem",         47, 2.4}, {"translate",    50, 2.4},
};

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

std::string trim(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return std::string(s.substr(a, b - a));
}

std::string to_lower_ascii(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    }
    return out;
}

bool contains(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string_view::npos;
}

inline double gelu(double x) noexcept {
    // Hendrycks & Gimpel tanh approximation to Gaussian Error Linear Unit
    constexpr double kSqrt2OverPi = 0.7978845608028654;
    return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + 0.044715 * x * x * x)));
}

inline double d_gelu(double x) noexcept {
    constexpr double kSqrt2OverPi = 0.7978845608028654;
    const double u = kSqrt2OverPi * (x + 0.044715 * x * x * x);
    const double t = std::tanh(u);
    const double sech2 = 1.0 - t * t;
    const double du = kSqrt2OverPi * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * du;
}

void put_verdict(Json& j, const prolog::Verification& v) {
    j.key("verification");
    j.begin_object();
    j.field("verdict", v.verdict == prolog::Verdict::Grounded
                           ? "grounded"
                           : v.verdict == prolog::Verdict::Refuted ? "refuted" : "unknown");
    j.field("reason", v.reason);
    j.field("inferences", v.inferences);
    j.field("hit_limit", v.hit_limit);
    j.key("evidence");
    j.begin_array();
    for (const auto& e : v.evidence) j.value(e);
    j.end_array();
    j.end_object();
}

std::map<std::string, std::string> parse_kv(const std::string& text) {
    std::map<std::string, std::string> kv;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        const size_t start = i;
        while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        if (i == start) break;
        const std::string token = text.substr(start, i - start);
        const size_t eq = token.find('=');
        if (eq != std::string::npos) kv[token.substr(0, eq)] = token.substr(eq + 1);
    }
    return kv;
}

}  // namespace

Router::Router(const Registry* registry, const prolog::KnowledgeBase* kb)
    : registry_(registry), kb_(kb) {
    init_and_pretrain_network();
}

std::array<double, Router::kEmbedDim> Router::embed(std::string_view request,
                                                    std::string* top_feature_out,
                                                    int* matched_kernel_out) const {
    std::array<double, kEmbedDim> x{};
    const std::string raw = trim(request);
    if (raw.empty()) return x;
    const std::string low = to_lower_ascii(raw);

    std::string best_feat;
    double best_feat_w = 0.0;
    int best_kernel = -1;
    double best_kernel_w = 0.0;

    // 1. Physics kernel receptors (dims 0..9)
    for (const auto& kr : kKernelReceptors) {
        if (contains(low, kr.token)) {
            x[size_t(kr.feat_dim)] += kr.weight;
            x[9] += 0.9 * kr.weight; // general physics energy receptor
            if (kr.weight > best_kernel_w) {
                best_kernel_w = kr.weight;
                best_kernel = kr.kernel_idx;
            }
            if (kr.weight > best_feat_w) {
                best_feat_w = kr.weight;
                best_feat = kr.token;
            }
        }
    }

    // 2. Multi-language domain receptors (dims 10..50)
    for (const auto& dr : kDomainReceptors) {
        const bool hit = contains(raw, dr.token) || contains(low, dr.token);
        if (hit) {
            x[size_t(dr.feat_dim)] += dr.weight;
            if (dr.weight > best_feat_w) {
                best_feat_w = dr.weight;
                best_feat = dr.token;
            }
        }
    }

    // 3. Structural & syntactic receptors (dims 51..57)
    // S-expression parentheses structure:
    if (starts_with(raw, "(") && raw.back() == ')') {
        x[51] += 2.2;
        if (best_feat.empty()) best_feat = "s-expression (...)";
    }
    // Horn fact / clause structure `pred(arg).`:
    if (raw.back() == '.' && contains(raw, "(") && contains(raw, ")") && !starts_with(raw, ":")) {
        x[52] += 2.8;
        if (best_feat.empty()) best_feat = "clause predicate(...).";
    }
    // Forth colon definition `: word ... ;`:
    if (starts_with(raw, ": ") && contains(raw, ";")) {
        x[53] += 3.0;
        if (best_feat.empty()) best_feat = "forth ': ... ;'";
    }
    // Question mark / natural language inquiry:
    if (raw.back() == '?') {
        x[54] += 1.9;
        if (best_feat.empty()) best_feat = "interrogative '?'";
    }
    // Key-value numeric parameter density (`n=64`, `e=0.3`, `tol=1e-6`):
    size_t eq_count = 0;
    for (char c : raw) if (c == '=') ++eq_count;
    if (eq_count > 0) x[55] += std::min(2.0, 0.5 * double(eq_count));

    // High-bit UTF-8 glyph density (APL symbols):
    size_t utf8_hi = 0;
    for (unsigned char c : raw) if (c >= 0x80) ++utf8_hi;
    if (utf8_hi > 0) x[56] += std::min(3.0, 0.4 * double(utf8_hi));

    // Character trigram hashed projection into dims 57..63 (low-amplitude contextual sub-embedding)
    if (best_feat_w > 0.0 || x[51] > 0.0 || x[52] > 0.0 || x[53] > 0.0 || x[54] > 0.0) {
        for (size_t i = 0; i + 2 < low.size(); ++i) {
            const uint32_t h = (uint32_t(uint8_t(low[i])) * 31u +
                                uint32_t(uint8_t(low[i + 1])) * 131u +
                                uint32_t(uint8_t(low[i + 2])) * 17u);
            const size_t bucket = 57 + (h % 7);
            x[bucket] += 0.05;
        }
    }

    // L2 normalization with magnitude preservation scaling
    double norm2 = 0.0;
    for (double v : x) norm2 += v * v;
    if (norm2 > 1e-12) {
        const double inv = 1.0 / std::sqrt(norm2);
        const double scale = std::tanh(0.45 * std::sqrt(norm2)) * 2.5;
        for (double& v : x) v = v * inv * scale;
    }

    if (top_feature_out) *top_feature_out = best_feat;
    if (matched_kernel_out) *matched_kernel_out = best_kernel;
    return x;
}

void Router::init_and_pretrain_network() {
    // Deterministic Glorot-scaled initialization + structured receptor prior
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    auto next_rand = [&]() -> double {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        const uint64_t r = seed * 2685821657736338717ULL;
        return (double(r & 0xFFFFFFULL) / double(0x800000ULL) - 1.0) * 0.04;
    };

    for (double& v : w1_) v = next_rand();
    for (double& v : b1_) v = 0.0;
    for (double& v : v_w1_) v = 0.0;
    for (double& v : w2_) v = next_rand();
    for (double& v : b2_) v = 0.0;
    for (double& v : v_w2_) v = 0.0;
    for (double& v : wk_) v = next_rand();
    for (double& v : bk_) v = 0.0;

    // Initialize realistic baseline execution speed EMAs (in ms) for each runtime module:
    speed_ema_ms_ = {
        0.45, // Numeric (Fortran 2023 native)
        0.18, // Array (APL right-to-left)
        0.22, // Symbolic (LISP AST)
        0.25, // Logic (Prolog SLD)
        0.85, // Parallel (Occam CSP threads)
        0.12, // LowLevel (Forth stack/register VM)
        0.30, // Actor (Smalltalk live-cell)
        0.20, // Guard (INTERCAL shield)
        0.40, // Visual (Piet canvas)
        2.50  // Language (LLM host)
    };
    dispatches_.fill(0);

    // Seed hidden-unit wiring for the 9 physics kernels (hidden units 0..8)
    for (size_t k = 0; k < kNumKernels; ++k) {
        w1_[k * kEmbedDim + k] = 1.8;
        w1_[k * kEmbedDim + 9] = 0.8;
        wk_[k * kHiddenDim + k] = 2.6;
    }

    // Curated multi-language training curriculum for analytical backpropagation
    struct Sample {
        const char* prompt;
        TaskKind    kind;
        int         kernel_idx;
    };
    static const Sample kCurriculum[] = {
        // Numeric (Fortran 2023 physics kernels)
        {"shock tube riemann problem n=200", TaskKind::Numeric, 0},
        {"simulate sod shock tube euler equations", TaskKind::Numeric, 0},
        {"what is the heat equation diffusion?", TaskKind::Numeric, 1},
        {"solve 2d heat diffusion with alpha=0.1", TaskKind::Numeric, 1},
        {"solve poisson laplace equation on grid", TaskKind::Numeric, 2},
        {"lid driven cavity navier stokes flow", TaskKind::Numeric, 3},
        {"kepler two-body orbit eccentricity e=0.5", TaskKind::Numeric, 4},
        {"n-body gravity three-body figure-8 verlet", TaskKind::Numeric, 5},
        {"isentropic ideal gas thermodynamic compression", TaskKind::Numeric, 6},
        {"solve matrix linear system linsolve ax = b", TaskKind::Numeric, 7},
        {"covariant riemannian tensor_field bianchi identity", TaskKind::Numeric, 8},
        // Array (APL)
        {"+/ \xe2\x8d\xb3 100", TaskKind::Array, -1},
        {"2 3 \xe2\x8d\xb4 \xe2\x8d\xb3 6", TaskKind::Array, -1},
        {"A \xe2\x8c\xb9 B", TaskKind::Array, -1},
        {"\xe2\x8d\x89 M +.\xc3\x97 W", TaskKind::Array, -1},
        {"+\\ 1 2 3 4 5", TaskKind::Array, -1},
        // Symbolic (LISP)
        {"(define (f x) (* x 2))", TaskKind::Symbolic, -1},
        {"(lambda (x y) (+ x y))", TaskKind::Symbolic, -1},
        {"(let ((a 10)) (* a a))", TaskKind::Symbolic, -1},
        {"hot-patch self-modify s-expression", TaskKind::Symbolic, -1},
        // Logic (Prolog)
        {"ancestor(X, Y) :- parent(X, Y).", TaskKind::Logic, -1},
        {"parent(abe, homer).", TaskKind::Logic, -1},
        {"prove ancestor(abe, bart)", TaskKind::Logic, -1},
        {"verify is it true that energy is conserved", TaskKind::Logic, -1},
        // Parallel (Occam)
        {"parallel consensus strategies on channel", TaskKind::Parallel, -1},
        {"occam concurrent workers", TaskKind::Parallel, -1},
        // LowLevel (Forth)
        {": sq dup * ; 7 sq .", TaskKind::LowLevel, -1},
        {"10 20 swap over + dup .", TaskKind::LowLevel, -1},
        {"42 r0! r0@ >r r> reg@", TaskKind::LowLevel, -1},
        // Actor (Smalltalk)
        {"smalltalk actor mailbox supervise live-cell", TaskKind::Actor, -1},
        // Guard (INTERCAL)
        {"obfuscate and seal payload against tamper", TaskKind::Guard, -1},
        {"audit chain intercal mingle envelope", TaskKind::Guard, -1},
        // Visual (Piet)
        {"render cognitive state plot width=64 height=64", TaskKind::Visual, -1},
        {"visualize colorize piet canvas codel synapse", TaskKind::Visual, -1},
        // Language (SLM / LLM)
        {"what is a hybrid architecture?", TaskKind::Language, -1},
        {"write a short poem about stars", TaskKind::Language, -1},
        {"explain how compilers work", TaskKind::Language, -1},
    };

    // Train for 35 epochs via analytical backpropagation with momentum SGD
    for (int epoch = 0; epoch < 35; ++epoch) {
        const double lr = 0.14 * std::pow(0.96, double(epoch));
        for (const auto& s : kCurriculum) {
            (void)learn(s.prompt, s.kind, -1.0, lr);
        }
    }

    // Evaluate curriculum accuracy
    int correct = 0;
    const int total = int(sizeof(kCurriculum) / sizeof(kCurriculum[0]));
    for (const auto& s : kCurriculum) {
        const Route r = classify(s.prompt);
        if (r.kind == s.kind) ++correct;
    }
    curriculum_accuracy_ = total > 0 ? double(correct) / double(total) : 1.0;
}

double Router::learn(std::string_view prompt, TaskKind target_kind,
                     double execution_seconds, double learning_rate) {
    const size_t target = static_cast<size_t>(target_kind);
    if (target >= kNumClasses) return 0.0;

    int matched_kernel = -1;
    const auto x = embed(prompt, nullptr, &matched_kernel);
    double x_norm = 0.0;
    for (double v : x) x_norm += v * v;
    if (x_norm < 1e-12) return 0.0;

    std::lock_guard<std::mutex> lock(mu_);
    if (execution_seconds >= 0.0) {
        const double ms = std::max(0.01, execution_seconds * 1000.0);
        speed_ema_ms_[target] = 0.85 * speed_ema_ms_[target] + 0.15 * ms;
    }

    // Forward pass: z1 = W1 * x + b1, h = GELU(z1)
    std::array<double, kHiddenDim> z1{};
    std::array<double, kHiddenDim> h{};
    for (size_t i = 0; i < kHiddenDim; ++i) {
        double s = b1_[i];
        const size_t row = i * kEmbedDim;
        for (size_t j = 0; j < kEmbedDim; ++j) s += w1_[row + j] * x[j];
        z1[i] = s;
        h[i] = gelu(s);
    }

    // Task head logits: z2 = W2 * h + b2
    std::array<double, kNumClasses> logits{};
    double max_logit = -1e30;
    for (size_t k = 0; k < kNumClasses; ++k) {
        double s = b2_[k];
        const size_t row = k * kHiddenDim;
        for (size_t i = 0; i < kHiddenDim; ++i) s += w2_[row + i] * h[i];
        logits[k] = s;
        max_logit = std::max(max_logit, s);
    }

    // Softmax probabilities
    std::array<double, kNumClasses> probs{};
    double sum_exp = 0.0;
    for (size_t k = 0; k < kNumClasses; ++k) {
        probs[k] = std::exp(logits[k] - max_logit);
        sum_exp += probs[k];
    }
    for (size_t k = 0; k < kNumClasses; ++k) probs[k] /= std::max(1e-30, sum_exp);

    const double loss = -std::log(std::max(1e-15, probs[target]));

    // Backprop through Layer 2 (task head)
    std::array<double, kNumClasses> dz2{};
    for (size_t k = 0; k < kNumClasses; ++k) {
        dz2[k] = probs[k] - (k == target ? 1.0 : 0.0);
    }

    std::array<double, kHiddenDim> dh{};
    for (size_t k = 0; k < kNumClasses; ++k) {
        const size_t row = k * kHiddenDim;
        const double g = dz2[k];
        for (size_t i = 0; i < kHiddenDim; ++i) {
            dh[i] += w2_[row + i] * g;
            const double grad_w = g * h[i] + 1e-4 * w2_[row + i];
            v_w2_[row + i] = 0.8 * v_w2_[row + i] - learning_rate * grad_w;
            w2_[row + i] += v_w2_[row + i];
        }
        b2_[k] -= learning_rate * g;
    }

    // Optional physics kernel head update when target is Numeric and a kernel matched
    if (target_kind == TaskKind::Numeric && matched_kernel >= 0 &&
        size_t(matched_kernel) < kNumKernels) {
        std::array<double, kNumKernels> kl{};
        double kmax = -1e30;
        for (size_t k = 0; k < kNumKernels; ++k) {
            double s = bk_[k];
            const size_t row = k * kHiddenDim;
            for (size_t i = 0; i < kHiddenDim; ++i) s += wk_[row + i] * h[i];
            kl[k] = s;
            kmax = std::max(kmax, s);
        }
        double ksum = 0.0;
        for (size_t k = 0; k < kNumKernels; ++k) {
            kl[k] = std::exp(kl[k] - kmax);
            ksum += kl[k];
        }
        for (size_t k = 0; k < kNumKernels; ++k) {
            const double gk = (kl[k] / std::max(1e-30, ksum)) -
                              (int(k) == matched_kernel ? 1.0 : 0.0);
            const size_t row = k * kHiddenDim;
            for (size_t i = 0; i < kHiddenDim; ++i) {
                dh[i] += 0.35 * wk_[row + i] * gk;
                wk_[row + i] -= learning_rate * (gk * h[i] + 1e-4 * wk_[row + i]);
            }
            bk_[k] -= learning_rate * gk;
        }
    }

    // Backprop through GELU Layer 1
    for (size_t i = 0; i < kHiddenDim; ++i) {
        const double dz1 = dh[i] * d_gelu(z1[i]);
        const size_t row = i * kEmbedDim;
        for (size_t j = 0; j < kEmbedDim; ++j) {
            const double grad_w1 = dz1 * x[j] + 1e-4 * w1_[row + j];
            v_w1_[row + j] = 0.8 * v_w1_[row + j] - learning_rate * grad_w1;
            w1_[row + j] += v_w1_[row + j];
        }
        b1_[i] -= learning_rate * dz1;
    }

    ++training_steps_;
    last_loss_ = loss;
    return loss;
}

void Router::record_execution_speed(TaskKind kind, double execution_seconds) const {
    const size_t idx = static_cast<size_t>(kind);
    if (idx >= kNumClasses) return;
    std::lock_guard<std::mutex> lock(mu_);
    const double ms = std::max(0.005, execution_seconds * 1000.0);
    speed_ema_ms_[idx] = 0.80 * speed_ema_ms_[idx] + 0.20 * ms;
    ++dispatches_[idx];
}

Route Router::classify(std::string_view request) const {
    Route r;
    const std::string text = trim(request);
    if (text.empty()) {
        r.reason = "empty request";
        return r;
    }

    // 1. Explicit prefix declaration (1.0 confidence, explicit_route = true)
    for (const auto& p : kPrefixes) {
        if (starts_with(text, p.prefix)) {
            r.kind = p.kind;
            r.module = p.module;
            r.capability = p.capability;
            r.reason = std::string("explicit prefix '") + p.prefix + "'";
            r.confidence = 1.0;
            r.explicit_route = true;
            std::lock_guard<std::mutex> lock(mu_);
            r.execution_speed_ema_ms = speed_ema_ms_[static_cast<size_t>(p.kind)];
            return r;
        }
    }

    // 2. Neural embedding + 2-layer GELU forward pass + execution speed EMA modulation
    std::string top_feature;
    int receptor_kernel = -1;
    const auto x = embed(text, &top_feature, &receptor_kernel);

    double x_energy = 0.0;
    for (size_t j = 0; j < 57; ++j) x_energy += x[j] * x[j];
    if (x_energy < 1e-9) {
        r.kind = TaskKind::Unknown;
        r.confidence = 0.0;
        r.reason = "neural router OOD gate: zero activated language/physics receptors; refusing to guess";
        return r;
    }

    std::lock_guard<std::mutex> lock(mu_);
    std::array<double, kHiddenDim> h{};
    for (size_t i = 0; i < kHiddenDim; ++i) {
        double s = b1_[i];
        const size_t row = i * kEmbedDim;
        for (size_t j = 0; j < kEmbedDim; ++j) s += w1_[row + j] * x[j];
        h[i] = gelu(s);
    }

    // Compute task logits modulated by real-time execution speed EMA:
    // Faster runtime modules receive a bounded positive latency bonus.
    std::array<double, kNumClasses> logits{};
    double max_logit = -1e30;
    size_t best_k = 0;
    for (size_t k = 0; k < kNumClasses; ++k) {
        double s = b2_[k];
        const size_t row = k * kHiddenDim;
        for (size_t i = 0; i < kHiddenDim; ++i) s += w2_[row + i] * h[i];
        const double speed_bonus = 0.08 * std::log(1.0 + 1.0 / std::max(0.05, speed_ema_ms_[k]));
        logits[k] = s + speed_bonus;
        if (logits[k] > max_logit) {
            max_logit = logits[k];
            best_k = k;
        }
    }

    double sum_exp = 0.0;
    for (size_t k = 0; k < kNumClasses; ++k) sum_exp += std::exp(logits[k] - max_logit);
    const double log_sum_exp = max_logit + std::log(std::max(1e-30, sum_exp));
    const double prob = std::exp(logits[best_k] - log_sum_exp);

    // Out-of-distribution energy check
    if (max_logit < 0.35 || prob < 0.22) {
        r.kind = TaskKind::Unknown;
        r.confidence = 0.0;
        r.logit_energy = log_sum_exp;
        r.reason = "neural router OOD gate: low logit energy (" + std::to_string(max_logit) +
                   "); refusing to guess which module should serve this";
        return r;
    }

    r.kind = kClassMeta[best_k].kind;
    r.module = kClassMeta[best_k].module;
    r.capability = kClassMeta[best_k].capability;
    // Clamp neural confidence strictly into (0.0, 0.99] so 1.0 remains reserved
    // for explicit prefix declarations.
    r.confidence = std::min(0.99, std::max(0.50, prob));
    r.explicit_route = false;
    r.execution_speed_ema_ms = speed_ema_ms_[best_k];
    r.logit_energy = log_sum_exp;
    r.embedding_preview.assign(x.begin(), x.begin() + 8);

    // If routed to Numeric, run the secondary neural kernel sub-head (32 -> 9)
    if (r.kind == TaskKind::Numeric) {
        int best_kern = receptor_kernel;
        if (best_kern < 0) {
            double kmax = -1e30;
            for (size_t k = 0; k < kNumKernels; ++k) {
                double s = bk_[k];
                const size_t row = k * kHiddenDim;
                for (size_t i = 0; i < kHiddenDim; ++i) s += wk_[row + i] * h[i];
                if (s > kmax) { kmax = s; best_kern = int(k); }
            }
        }
        if (best_kern >= 0 && size_t(best_kern) < kNumKernels) {
            r.capability = std::string("numeric.kernel.") + kKernelNames[best_kern];
        }
    }

    std::ostringstream reason;
    reason << "neural router [64->32->10 GELU, speed_ema=" << r.execution_speed_ema_ms
           << "ms] activated receptor '" << (top_feature.empty() ? to_string(r.kind) : top_feature)
           << "'";
    r.reason = reason.str();
    return r;
}

Outcome<Json> Router::execute_route(const Route& route, std::string_view payload,
                                    const RouterOptions& options) const {
    const Clock::time_point t0 = Clock::now();
    const std::string body = trim(payload);
    Json out;
    out.begin_object();
    out.field("route_kind", to_string(route.kind));
    out.field("route_module", route.module);
    out.field("route_capability", route.capability);
    out.field("route_reason", route.reason);
    out.field("route_confidence", route.confidence);
    out.field("route_speed_ema_ms", route.execution_speed_ema_ms);

    switch (route.kind) {
        case TaskKind::Array: {
            apl::Environment env;
            auto res = apl::eval_line(body, env);
            if (!res) return res.status;
            record_execution_speed(TaskKind::Array, seconds_since(t0));
            std::ostringstream shape;
            shape << "f64[";
            for (size_t k = 0; k < res->shape().size(); ++k)
                shape << (k ? "," : "") << res->shape()[k];
            shape << "]";
            out.field("result_shape", shape.str());
            out.field("elements", int64_t(res->size()));
            out.key("values");
            out.begin_array();
            for (size_t k = 0; k < res->size() && k < 64; ++k) out.value(res->scalar_f64(k));
            out.end_array();
            out.field("fingerprint", res->fingerprint());
            out.end_object();
            return out;
        }
        case TaskKind::Symbolic: {
            lisp::Interp interp;
            auto res = interp.eval_string(body);
            if (!res) return res.status;
            record_execution_speed(TaskKind::Symbolic, seconds_since(t0));
            out.field("result", (*res)->to_string());
            out.field("generation", interp.generation());
            out.end_object();
            return out;
        }
        case TaskKind::LowLevel: {
            stackvm::Vm vm;
            Status c = vm.compile(body);
            if (!c) return c;
            auto res = vm.run();
            if (!res) return res.status;
            record_execution_speed(TaskKind::LowLevel, seconds_since(t0));
            out.field("steps", int64_t(*res));
            out.field("output", vm.output());
            out.end_object();
            return out;
        }
        case TaskKind::Logic: {
            auto claim = prolog::parse_term(body);
            if (!claim) return claim.status;
            prolog::KnowledgeBase empty;
            const prolog::KnowledgeBase& kb = kb_ ? *kb_ : empty;
            const auto v = prolog::verify(kb, *claim);
            record_execution_speed(TaskKind::Logic, seconds_since(t0));
            if (!kb_)
                out.field("note", "no knowledge base supplied: every claim is Unknown by design");
            put_verdict(out, v);
            out.end_object();
            return out;
        }
        case TaskKind::Numeric: {
            std::string name;
            std::string rest = body;
            const size_t sp = body.find(' ');
            const std::string first = sp == std::string::npos ? body : body.substr(0, sp);
            if (!first.empty() && first.find('=') == std::string::npos &&
                numerics::KernelLibrary::instance().find(first)) {
                name = first;
                rest = sp == std::string::npos ? std::string() : body.substr(sp + 1);
            } else {
                const size_t dot = route.capability.rfind('.');
                name = dot == std::string::npos ? std::string() : route.capability.substr(dot + 1);
                if (!numerics::KernelLibrary::instance().find(name))
                    return Status::invalid(
                        "router: this request routed to the numeric layer but names no kernel; "
                        "use 'kernel:<name> [k=v ...]'");
            }
            auto kernel = numerics::KernelLibrary::instance().find(name);
            if (!kernel) return Status::invalid("router: unknown kernel '" + name + "'");
            numerics::ProblemSpec spec = kernel->validation_spec();
            const auto kv = parse_kv(rest);
            for (const auto& e : kv) {
                if (e.first == "t_end") { spec.t_end = std::stod(e.second); continue; }
                if (e.first == "tol") { spec.tolerance = std::stod(e.second); continue; }
                if (e.first == "method") { spec.method = e.second; continue; }
                try {
                    spec.params[e.first] = std::stod(e.second);
                } catch (const std::exception&) {
                    return Status::invalid("router: malformed parameter " + e.first + "=" + e.second);
                }
            }
            const numerics::Result r = kernel->run(spec);
            record_execution_speed(TaskKind::Numeric, seconds_since(t0));
            out.field("kernel", r.kernel);
            out.field("method", r.method);
            out.field("backend", r.backend);
            out.field("units", r.units);
            out.field("fingerprint", r.fingerprint);
            out.field("elements", int64_t(r.data.size()));
            out.field("residual", r.diag.residual);
            out.field("hardware_skipped", r.hardware_skipped);
            out.field("status", to_string(r.status.code));
            out.end_object();
            return out;
        }
        case TaskKind::Parallel: {
            const auto kv = parse_kv(body);
            int64_t n = 100000;
            auto it = kv.find("n");
            if (it != kv.end()) n = std::stoll(it->second);
            if (n < 1) return Status::invalid("router: parallel sum needs n >= 1");
            std::vector<occam::Strategy> strategies = {
                {"closed_form", [n]() -> Outcome<double> {
                     return double(n) * double(n - 1) / 2.0;
                 }},
                {"direct_loop", [n]() -> Outcome<double> {
                     double s = 0.0;
                     for (int64_t i = 0; i < n; ++i) s += double(i);
                     return s;
                 }},
                {"apl.reduce", [n]() -> Outcome<double> {
                     apl::Environment env;
                     auto r = apl::eval_line("+/ \xe2\x8d\xb3 " + std::to_string(n), env);
                     if (!r) return r.status;
                     return r->scalar_f64();
                 }},
            };
            const occam::Verdict v = occam::parallel_verify(strategies, 1e-6);
            if (!v.agreed)
                return Status::rejected("router: strategies disagreed (max spread " +
                                        std::to_string(v.max_spread) + ")");
            record_execution_speed(TaskKind::Parallel, seconds_since(t0));
            out.field("value", v.value);
            out.field("agreed", v.agreed);
            out.field("spread", v.max_spread);
            out.field("winner", v.winner);
            out.field("speedup", v.speedup);
            out.end_object();
            return out;
        }
        case TaskKind::Guard: {
            const std::string key = "octopus-router";
            const std::string envelope = intercal::encode(body, key);
            auto back = intercal::decode(envelope, key);
            if (!back || *back != body) return Status::internal("router: guard round-trip failed");
            const intercal::FuzzReport f = intercal::fuzz_guard(envelope, key, 64, 1);
            const intercal::ShieldEnvelope shield = intercal::seal_code_structure("router.payload", body, key);
            record_execution_speed(TaskKind::Guard, seconds_since(t0));
            out.field("envelope_bytes", int64_t(envelope.size()));
            out.field("sha256", hash::sha256_hex(envelope));
            out.field("fuzz_iterations", int64_t(f.iterations));
            out.field("fuzz_accepted", int64_t(f.accepted));
            out.field("intercal_select_signature", int64_t(shield.intercal_select_signature));
            out.field("note", "obfuscation + tamper evidence, not a cipher");
            out.end_object();
            return out;
        }
        case TaskKind::Visual: {
            std::vector<double> state;
            const auto kv = parse_kv(body);
            int width = 64, height = 64;
            if (auto w = kv.find("width"); w != kv.end()) width = std::stoi(w->second);
            if (auto h = kv.find("height"); h != kv.end()) height = std::stoi(h->second);
            if (auto v = kv.find("values"); v != kv.end()) {
                std::string list = v->second;
                std::replace(list.begin(), list.end(), ',', ' ');
                std::istringstream is(list);
                double x = 0.0;
                while (is >> x) state.push_back(x);
            }
            if (state.empty())
                state.assign(size_t(width) * size_t(height), 0.0);
            const piet::Raster raster = piet::render_state(state, width, height, -1.0, 1.0);
            const piet::CodelTrace trace = piet::execute_canvas_pathway(raster);
            record_execution_speed(TaskKind::Visual, seconds_since(t0));
            out.field("width", int64_t(raster.width));
            out.field("height", int64_t(raster.height));
            out.field("fingerprint", raster.fingerprint());
            out.field("codel_transitions", trace.codel_transitions);
            if (!options.output_path.empty()) {
                const Status w = options.output_path.size() > 4 &&
                                         options.output_path.substr(options.output_path.size() - 4) ==
                                             ".ppm"
                                     ? raster.write_ppm(options.output_path)
                                     : raster.write_png(options.output_path);
                if (!w) return w;
                out.field("out", options.output_path);
            }
            out.field("note", "quantised cognitive-state map; index 18 (black) marks NaN/Inf");
            out.end_object();
            return out;
        }
        case TaskKind::Language: {
            llm::Host host;
            if (options.model_path.empty())
                return Status::unavailable(
                    "router: this request needs a language model; pass model=<path.gguf> "
                    "(no answer is fabricated without one)");
            Status loaded = host.load(options.model_path);
            if (!loaded) return loaded;
            llm::GenerateParams p;
            p.max_tokens = 32;
            auto text = host.generate(body, p);
            if (!text) return text.status;
            // Pass LLM output through the Prolog SLD Soundness Guardrail
            const prolog::SoundnessReport sld = prolog::verify_response_soundness(body, *text, kb_);
            if (!sld.sound) {
                return Status::rejected("router: Prolog SLD soundness guardrail rejected LLM output");
            }
            record_execution_speed(TaskKind::Language, seconds_since(t0));
            out.field("text", *text);
            out.field("backend", host.backend().name + " " + host.backend().version);
            out.field("backend_honesty", host.backend().honesty);
            out.field("sld_sound", sld.sound);
            out.field("sld_proof_digest", sld.proof_digest);
            out.end_object();
            return out;
        }
        case TaskKind::Actor:
            return Status::unavailable(
                "router: the actor runtime is driven through its own API (spawn/send); the "
                "router refuses to invent a message for an unnamed actor");
        case TaskKind::Unknown:
            return Status::rejected("router: " + route.reason);
    }
    return Status::internal("router: unreachable route kind");
}

Outcome<Json> Router::execute(std::string_view request, const RouterOptions& options) const {
    const std::string text = trim(request);
    Route route = classify(text);

    RouterOptions opts = options;
    const auto kv = parse_kv(text);
    if (auto m = kv.find("model"); m != kv.end() && opts.model_path.empty())
        opts.model_path = m->second;
    if (auto o = kv.find("out"); o != kv.end() && opts.output_path.empty())
        opts.output_path = o->second;

    std::string payload = text;
    if (route.explicit_route) {
        const size_t colon = text.find(':');
        if (colon != std::string::npos) payload = trim(text.substr(colon + 1));
    }
    if (route.kind == TaskKind::Language) {
        std::string cleaned;
        std::istringstream is(payload);
        std::string token;
        while (is >> token) {
            if (token.rfind("model=", 0) == 0 || token.rfind("out=", 0) == 0)
                continue;
            if (!cleaned.empty()) cleaned += ' ';
            cleaned += token;
        }
        payload = cleaned;
    }
    return execute_route(route, payload, opts);
}

NeuralTelemetry Router::telemetry() const {
    std::lock_guard<std::mutex> lock(mu_);
    NeuralTelemetry t;
    t.training_steps = training_steps_;
    t.last_loss = last_loss_;
    t.curriculum_accuracy = curriculum_accuracy_;
    t.module_speed_ema_ms = speed_ema_ms_;
    t.module_dispatches = dispatches_;
    hash::Sha256 h;
    h.update(w1_.data(), w1_.size() * sizeof(double));
    h.update(w2_.data(), w2_.size() * sizeof(double));
    h.update(wk_.data(), wk_.size() * sizeof(double));
    t.weights_sha256 = h.finalize().hex();
    return t;
}

intercal::ShieldEnvelope Router::seal_weights(std::string_view key,
                                              intercal::AuditChain* chain) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::string blob;
    blob.resize((w1_.size() + w2_.size() + wk_.size()) * sizeof(double));
    char* ptr = blob.data();
    std::memcpy(ptr, w1_.data(), w1_.size() * sizeof(double));
    ptr += w1_.size() * sizeof(double);
    std::memcpy(ptr, w2_.data(), w2_.size() * sizeof(double));
    ptr += w2_.size() * sizeof(double);
    std::memcpy(ptr, wk_.data(), wk_.size() * sizeof(double));
    return intercal::seal_code_structure("core.neural_router.weights", blob, key, chain);
}

piet::Raster Router::render_synapses(std::string_view prompt, int width, int height) const {
    const auto x = embed(prompt);
    std::lock_guard<std::mutex> lock(mu_);
    std::array<double, kHiddenDim> h{};
    for (size_t i = 0; i < kHiddenDim; ++i) {
        double s = b1_[i];
        const size_t row = i * kEmbedDim;
        for (size_t j = 0; j < kEmbedDim; ++j) s += w1_[row + j] * x[j];
        h[i] = gelu(s);
    }
    return piet::render_synapse_canvas(w1_, h, width, height);
}

std::vector<std::string> Router::example_requests() const {
    return {
        "kernel:kepler e=0.3",
        "physics: shock tube n=200",
        "apl: +/ \xe2\x8d\xb3 100",
        "prolog: parent(abe, bart)",
        "forth: : sq dup * ; 7 sq .",
        "occam: sum n=100000",
        "guard: top secret payload",
        "piet: render values=0,1,2,3 width=2 height=2 out=/tmp/router_state.png",
        "lisp: (define (f x) (* x 2))",
        "llm: explain what a hybrid engine is (needs --model)",
    };
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class RouterModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "core.router";
        i.version = "2.0.0";
        i.language = "C++20 Inline Micro-Neural Router (64->32->10 GELU)";
        i.role = "AI-learned neural capability router: embeds prompts into 64-D receptor space, "
                 "routes via a 2-layer GELU network modulated by real-time module execution speeds, "
                 "and enforces fail-closed OOD rejection.";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"route.classify", "route.dispatch", "route.neural_learn",
                          "route.speed_modulation", "route.numeric", "route.array",
                          "route.logic", "route.symbolic", "route.parallel",
                          "route.low_level", "route.guard", "route.visual", "route.language"};
        i.limitations = {
            "inline micro-neural router (64->32->10): trained on domain receptor curriculum and "
            "online execution feedback, not a billion-parameter transformer",
            "language tasks require a real GGUF model via llama.cpp; stub fallback is banned",
            "out-of-distribution inputs below the logit-energy gate are rejected as Unknown"};
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }

    std::string describe() const override {
        return "Inline 2-layer GELU micro-neural router that adaptively learns high-confidence "
               "vector representations of prompts and routes them based on real-time execution speeds.";
    }

    Status self_check() override {
        Router router;
        const NeuralTelemetry tel = router.telemetry();
        if (tel.training_steps < 100 || tel.curriculum_accuracy < 0.90)
            return Status::internal("router: neural curriculum pretraining accuracy below threshold");

        Route r1 = router.classify("kernel:kepler e=0.3");
        if (r1.kind != TaskKind::Numeric || r1.confidence != 1.0 || !r1.explicit_route)
            return Status::internal("router: explicit prefix was not honoured");
        auto out1 = router.execute("kernel:kepler e=0.3");
        if (!out1) return Status::internal("router: kernel route failed: " + out1.status.message);
        if (out1->str().find("\"fingerprint\"") == std::string::npos)
            return Status::internal("router: kernel route returned no fingerprint");

        Route r2 = router.classify("what is the heat equation?");
        if (r2.kind != TaskKind::Numeric || r2.confidence >= 1.0 || r2.reason.empty() || r2.explicit_route)
            return Status::internal("router: neural route failed on heat equation");

        // Verify online learning increases confidence on a sample
        const double before = router.classify("covariant riemannian tensor_field").confidence;
        router.learn("covariant riemannian tensor_field", TaskKind::Numeric, 0.0002, 0.10);
        const double after = router.classify("covariant riemannian tensor_field").confidence;
        if (after < before - 1e-6)
            return Status::internal("router: online neural learning degraded confidence");

        auto out2 = router.execute("apl: +/ \xe2\x8d\xb3 10");
        if (!out2) return Status::internal("router: array route failed: " + out2.status.message);

        auto out3 = router.execute("llm: hello");
        if (out3) return Status::internal("router: language route answered without a model");

        if (router.classify("").kind != TaskKind::Unknown)
            return Status::internal("router: empty request was routed");
        auto out4 = router.execute("zzzzz qqqqq");
        if (out4) return Status::internal("router: unclassifiable request was executed");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_router_module() {
    return std::make_shared<RouterModule>();
}

}  // namespace oct::router
