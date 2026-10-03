// Octopus Hybrid AI Engine -- capability router (implementation).
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

// An explicit prefix is a declaration by the caller, not a guess.
const PrefixRoute kPrefixes[] = {
    {"kernel:", TaskKind::Numeric, "fortran.kernels", "numeric.kernel"},
    {"numeric:", TaskKind::Numeric, "fortran.kernels", "numeric.kernel"},
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

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

std::string trim(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return std::string(s.substr(a, b - a));
}

bool contains(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string_view::npos;
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

}  // namespace

Router::Router(const Registry* registry, const prolog::KnowledgeBase* kb)
    : registry_(registry), kb_(kb) {}

Route Router::classify(std::string_view request) const {
    Route r;
    const std::string text = trim(request);
    if (text.empty()) {
        r.reason = "empty request";
        return r;
    }

    // 1. explicit prefix
    for (const auto& p : kPrefixes) {
        if (starts_with(text, p.prefix)) {
            r.kind = p.kind;
            r.module = p.module;
            r.capability = p.capability;
            r.reason = std::string("explicit prefix '") + p.prefix + "'";
            r.confidence = 1.0;
            r.explicit_route = true;
            return r;
        }
    }

    // 2. structured hints (heuristic; the matched hint is reported verbatim)
    auto heuristic = [&](TaskKind kind, std::string module, std::string capability,
                         std::string reason, double confidence) {
        r.kind = kind;
        r.module = std::move(module);
        r.capability = std::move(capability);
        r.reason = std::move(reason);
        r.confidence = confidence;
        return r;
    };

    // APL glyphs are unambiguous: no other layer uses them.
    static const char* kGlyphs[] = {"\xe2\x8d\xb3", "\xe2\x8d\xb4", "\xe2\x8c\xb9",
                                    "\xe2\x8d\x89", "\xe2\x8c\xbd", "\xe2\x88\x98",
                                    "+/", "+\\"};
    for (const char* g : kGlyphs)
        if (contains(text, g))
            return heuristic(TaskKind::Array, "apl.arrays", "array.eval",
                             std::string("request contains the APL glyph '") + g + "'", 0.95);

    // Physics/maths vocabulary maps to validated kernels.
    struct KernelHint { const char* word; const char* kernel; };
    static const KernelHint kKernelHints[] = {
        {"shock tube", "sod1d"}, {"sod", "sod1d"}, {"riemann", "sod1d"},
        {"heat", "heat2d"},      {"diffusion", "heat2d"},
        {"poisson", "poisson2d"},{"laplace", "poisson2d"},
        {"cavity", "cavity2d"},  {"lid", "cavity2d"},  {"navier", "cavity2d"},
        {"orbit", "kepler"},     {"kepler", "kepler"}, {"two-body", "kepler"},
        {"gravity", "nbody"},    {"n-body", "nbody"},  {"nbody", "nbody"},
        {"isentropic", "gas"},   {"ideal gas", "gas"}, {"thermodynamic", "gas"},
        {"linear system", "linsolve"}, {"solve matrix", "linsolve"},
    };
    for (const auto& h : kKernelHints) {
        if (contains(text, h.word))
            return heuristic(TaskKind::Numeric, "fortran.kernels",
                             std::string("numeric.kernel.") + h.kernel,
                             std::string("matched kernel keyword '") + h.word + "'", 0.7);
    }

    // Logic: clause syntax or a verification verb.
    if (contains(text, ":-") || (text.back() == '.' && contains(text, "(")) ||
        contains(text, "prove ") || contains(text, "verify ") || contains(text, "is it true"))
        return heuristic(TaskKind::Logic, "prolog.logic", "logic.verify",
                         "request looks like a clause or a verification request", 0.6);

    // Forth-ish tokens.
    if (starts_with(text, ": ") || contains(text, "dup ") || contains(text, "swap ") ||
        contains(text, "drop "))
        return heuristic(TaskKind::LowLevel, "forth.stackvm", "forth.run",
                         "request contains Forth-style words", 0.7);

    // Visualisation.
    if (contains(text, "render") || contains(text, "visuali") || contains(text, "plot") ||
        contains(text, "colourize") || contains(text, "colorize"))
        return heuristic(TaskKind::Visual, "piet.visual", "visual.render",
                         "request asks for a visualisation", 0.6);

    // Guardrail / tamper evidence.
    if (contains(text, "obfuscate") || contains(text, "seal") || contains(text, "tamper") ||
        contains(text, "audit"))
        return heuristic(TaskKind::Guard, "intercal.guard", "guard.encode",
                         "request asks for the guardrail", 0.6);

    // Symbolic.
    if (starts_with(text, "(") && contains(text, "define"))
        return heuristic(TaskKind::Symbolic, "lisp.selfmod", "lisp.eval",
                         "request looks like a LISP definition", 0.6);

    // Anything left that reads like a question goes to the language model -- and
    // is refused downstream if no model is loaded.
    if (!text.empty() && (text.back() == '?' || contains(text, "what ") ||
                          contains(text, "write ") || contains(text, "explain")))
        return heuristic(TaskKind::Language, "llm.host", "llm.generate",
                         "natural-language request: needs a loaded SLM", 0.5);

    r.reason = "no rule matched; refusing to guess which module should serve this";
    return r;
}

namespace {

// Parses "key=value" pairs out of a request body.
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

Outcome<Json> Router::execute_route(const Route& route, std::string_view payload,
                                    const RouterOptions& options) const {
    const std::string body = trim(payload);
    Json out;
    out.begin_object();
    // The route is echoed with the answer so a caller can audit why this module
    // produced it (Json has no raw-insert, so the fields are written directly).
    out.field("route_kind", to_string(route.kind));
    out.field("route_module", route.module);
    out.field("route_capability", route.capability);
    out.field("route_reason", route.reason);
    out.field("route_confidence", route.confidence);

    switch (route.kind) {
        case TaskKind::Array: {
            apl::Environment env;
            auto res = apl::eval_line(body, env);
            if (!res) return res.status;
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
            out.field("steps", int64_t(*res));
            out.field("output", vm.output());
            out.end_object();
            return out;
        }
        case TaskKind::Logic: {
            // "prolog: <claim>" verifies against the caller's knowledge base (if
            // one was supplied); without one, the claim can only be reported as
            // Unknown, which is what happens here.
            auto claim = prolog::parse_term(body);
            if (!claim) return claim.status;
            prolog::KnowledgeBase empty;
            const prolog::KnowledgeBase& kb = kb_ ? *kb_ : empty;
            const auto v = prolog::verify(kb, *claim);
            if (!kb_)
                out.field("note", "no knowledge base supplied: every claim is Unknown by design");
            put_verdict(out, v);
            out.end_object();
            return out;
        }
        case TaskKind::Numeric: {
            // Two accepted forms: "kernel:<name> [k=v ...]" (the name is the
            // first token and must be a real kernel), or free text whose route
            // already named the kernel through its capability.
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
            out.field("kernel", r.kernel);
            out.field("method", r.method);
            out.field("backend", r.backend);
            out.field("units", r.units);
            out.field("fingerprint", r.fingerprint);
            out.field("elements", int64_t(r.data.size()));
            out.field("residual", r.diag.residual);
            out.end_object();
            return out;
        }
        case TaskKind::Parallel: {
            // A well-defined consensus task: sum of 0..n-1 by three independent
            // strategies (closed form, direct loop, APL reduce). Disagreement
            // fails closed, which is the point of the layer.
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
            out.field("envelope_bytes", int64_t(envelope.size()));
            out.field("sha256", hash::sha256_hex(envelope));
            out.field("fuzz_iterations", int64_t(f.iterations));
            out.field("fuzz_accepted", int64_t(f.accepted));
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
            out.field("width", int64_t(raster.width));
            out.field("height", int64_t(raster.height));
            out.field("fingerprint", raster.fingerprint());
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
            Status loaded = host.load(options.model_path, options.allow_stub);
            if (!loaded) return loaded;
            llm::GenerateParams p;
            p.max_tokens = 32;
            auto text = host.generate(body, p);
            if (!text) return text.status;
            out.field("text", *text);
            out.field("backend", host.backend().name + " " + host.backend().version);
            out.field("backend_honesty", host.backend().honesty);
            out.field("stub", host.backend().stub);
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

    // Allow "model=<path>" to be carried on any language request.
    RouterOptions opts = options;
    const auto kv = parse_kv(text);
    if (auto m = kv.find("model"); m != kv.end() && opts.model_path.empty())
        opts.model_path = m->second;
    if (auto s = kv.find("stub"); s != kv.end()) opts.allow_stub = (s->second == "true" || s->second == "1");
    if (auto o = kv.find("out"); o != kv.end() && opts.output_path.empty())
        opts.output_path = o->second;

    // Strip an explicit prefix before handing the payload to the module.
    std::string payload = text;
    if (route.explicit_route) {
        const size_t colon = text.find(':');
        if (colon != std::string::npos) payload = trim(text.substr(colon + 1));
    }
    if (route.kind == TaskKind::Language) {
        // Remove router-only directives so the model sees the question itself.
        std::string cleaned;
        std::istringstream is(payload);
        std::string token;
        while (is >> token) {
            if (token.rfind("model=", 0) == 0 || token.rfind("stub=", 0) == 0 ||
                token.rfind("out=", 0) == 0)
                continue;
            if (!cleaned.empty()) cleaned += ' ';
            cleaned += token;
        }
        payload = cleaned;
    }
    return execute_route(route, payload, opts);
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
        i.version = "0.1.0";
        i.language = "C++20";
        i.role = "Capability router: classifies a request by task type and dispatches it to the "
                 "owning language-DNA module, with explicit evidence and fail-closed behaviour.";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"route.classify", "route.dispatch", "route.numeric", "route.array",
                          "route.logic",   "route.symbolic",  "route.parallel",
                          "route.low_level", "route.guard",  "route.visual", "route.language"};
        i.limitations = {
            "classification of free text is heuristic: only an explicit prefix is treated as a "
            "declaration (confidence 1.0); every other route reports the hint it matched and a "
            "lower confidence, and an unmatched request is refused rather than guessed",
            "the router does not chain modules: verification of a generated answer against a "
            "knowledge base must be requested explicitly (for example prolog:<claim>)",
            "language tasks need a model path; without one the router fails closed and never "
            "substitutes the deterministic stub unless allow_stub is set",
            "no learned classifier and no planner: dispatch cannot decompose a compound "
            "request into a task graph"};
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }

    std::string describe() const override {
        return "Front door of the engine: every request is classified with stated evidence and "
               "dispatched to the module that owns it.";
    }

    Status self_check() override {
        Router router;
        // Explicit routes classify with full confidence.
        Route r1 = router.classify("kernel:kepler e=0.3");
        if (r1.kind != TaskKind::Numeric || r1.confidence != 1.0 || !r1.explicit_route)
            return Status::internal("router: explicit prefix was not honoured");
        auto out1 = router.execute("kernel:kepler e=0.3");
        if (!out1) return Status::internal("router: kernel route failed: " + out1.status.message);
        if (out1->str().find("\"fingerprint\"") == std::string::npos)
            return Status::internal("router: kernel route returned no fingerprint");

        // Heuristic routes state their evidence.
        Route r2 = router.classify("what is the heat equation?");
        if (r2.confidence >= 1.0 || r2.reason.empty() || r2.explicit_route)
            return Status::internal("router: heuristic route claimed certainty");

        // Array route really evaluates.
        auto out2 = router.execute("apl: +/ \xe2\x8d\xb3 10");
        if (!out2) return Status::internal("router: array route failed: " + out2.status.message);

        // Language without a model must fail closed.
        auto out3 = router.execute("llm: hello");
        if (out3) return Status::internal("router: language route answered without a model");

        // Unroutable requests are refused.
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
