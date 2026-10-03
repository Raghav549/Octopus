// Capability router tests: classification evidence, dispatch to the owning
// layer, execution through that layer, and fail-closed behaviour.
// SPDX-License-Identifier: MIT
#include "harness.hpp"

#include "octopus/router.hpp"

#include <string>
#include <vector>

using namespace oct;

OCT_TEST(router, classify_reports_its_evidence) {
    router::Router r;

    // An explicit prefix is a declaration: full confidence, explicitly marked.
    const std::vector<std::pair<std::string, router::TaskKind>> explicit_routes = {
        {"kernel:kepler e=0.3", router::TaskKind::Numeric},
        {"apl: +/ \xe2\x8d\xb3 10", router::TaskKind::Array},
        {"prolog: parent(a, b)", router::TaskKind::Logic},
        {"forth: : sq dup * ; 3 sq .", router::TaskKind::LowLevel},
        {"lisp: (define (f x) x)", router::TaskKind::Symbolic},
        {"guard: payload", router::TaskKind::Guard},
        {"piet: render", router::TaskKind::Visual},
        {"occam: sum n=10", router::TaskKind::Parallel},
        {"llm: hello", router::TaskKind::Language},
    };
    for (const auto& c : explicit_routes) {
        const router::Route route = r.classify(c.first);
        OCT_CHECK_MSG(route.kind == c.second, "wrong kind for " << c.first);
        OCT_CHECK_MSG(route.explicit_route, "not marked explicit: " << c.first);
        OCT_CHECK_MSG(route.confidence == 1.0, "explicit route not certain: " << c.first);
        OCT_CHECK_MSG(!route.module.empty(), "no module for " << c.first);
    }

    // A heuristic route must admit that it is a guess and quote its evidence.
    const router::Route heat = r.classify("what is the heat equation?");
    OCT_CHECK(heat.kind == router::TaskKind::Numeric);
    OCT_CHECK(!heat.explicit_route);
    OCT_CHECK(heat.confidence < 1.0);
    OCT_CHECK(heat.reason.find("heat") != std::string::npos);
    OCT_CHECK(heat.capability.find("heat2d") != std::string::npos);

    // And an unmatched request is refused, not guessed at.
    const router::Route nothing = r.classify("zzzzz qqqqq");
    OCT_CHECK(nothing.kind == router::TaskKind::Unknown);
    OCT_CHECK(nothing.confidence == 0.0);
    OCT_CHECK(!nothing.reason.empty());
    OCT_CHECK(r.classify("").kind == router::TaskKind::Unknown);
}

OCT_TEST(router, dispatches_every_layer_and_executes_it) {
    router::Router r;

    // Array layer: sum of 0..10 via APL reduce.
    auto apl_out = r.execute("apl: +/ \xe2\x8d\xb3 11");
    OCT_CHECK(bool(apl_out));
    if (apl_out) {
        OCT_NOTE("apl route: " << apl_out->str());
        OCT_CHECK(apl_out->str().find("\"values\":[55]") != std::string::npos);
        OCT_CHECK(apl_out->str().find("\"fingerprint\"") != std::string::npos);
    }

    // Numeric layer: a validated kernel with its metadata attached.
    auto num_out = r.execute("kernel:poisson2d n=16");
    OCT_CHECK(bool(num_out));
    if (num_out) {
        const std::string s = num_out->str();
        OCT_CHECK(s.find("\"kernel\":\"poisson2d\"") != std::string::npos);
        OCT_CHECK(s.find("\"backend\"") != std::string::npos);
        OCT_CHECK(s.find("\"fingerprint\"") != std::string::npos);
    }

    // Stack VM.
    auto forth_out = r.execute("forth: : sq dup * ; 9 sq .");
    OCT_CHECK(bool(forth_out));
    if (forth_out) {
        OCT_CHECK(forth_out->str().find("\"output\":\"81 \"") != std::string::npos);
        OCT_CHECK(forth_out->str().find("\"steps\"") != std::string::npos);
    }

    // Symbolic layer.
    auto lisp_out = r.execute("lisp: (define (f x) (* x 3)) (f 7)");
    OCT_CHECK(bool(lisp_out));
    if (lisp_out) OCT_CHECK(lisp_out->str().find("\"result\":\"21\"") != std::string::npos);

    // Guardrail: round-trip plus fuzz, and the honest wording.
    auto guard_out = r.execute("guard: router test payload");
    OCT_CHECK(bool(guard_out));
    if (guard_out) {
        const std::string s = guard_out->str();
        OCT_CHECK(s.find("\"fuzz_accepted\":0") != std::string::npos);
        OCT_CHECK(s.find("not a cipher") != std::string::npos);
    }

    // Parallel consensus across three independent strategies.
    auto par_out = r.execute("occam: sum n=1000");
    OCT_CHECK(bool(par_out));
    if (par_out) {
        const std::string s = par_out->str();
        OCT_CHECK(s.find("\"agreed\":true") != std::string::npos);
        OCT_CHECK(s.find("\"value\":499500") != std::string::npos);
    }

    // Visual layer: renders and reports a fingerprint.
    auto vis_out = r.execute("piet: render values=0,1,2,3 width=2 height=2 out=/tmp/router_test.png");
    OCT_CHECK(bool(vis_out));
    if (vis_out) {
        OCT_CHECK(vis_out->str().find("\"fingerprint\"") != std::string::npos);
        OCT_CHECK(vis_out->str().find("\"out\":\"/tmp/router_test.png\"") != std::string::npos);
    }

    // Logic layer without a knowledge base is honest about being open-world.
    auto logic_out = r.execute("prolog: parent(abe, bart)");
    OCT_CHECK(bool(logic_out));
    if (logic_out) OCT_CHECK(logic_out->str().find("\"verdict\":\"unknown\"") != std::string::npos);
}

OCT_TEST(router, verifies_claims_against_a_knowledge_base) {
    prolog::KnowledgeBase kb;
    OCT_CHECK(kb.add_text("parent(abe, homer).").is_ok());
    OCT_CHECK(kb.add_text("parent(homer, bart).").is_ok());
    kb.declare_complete("parent/2");
    router::Router r(nullptr, &kb);

    auto yes = r.execute("prolog: parent(abe, homer)");
    OCT_CHECK(bool(yes));
    if (yes) {
        OCT_CHECK(yes->str().find("\"verdict\":\"grounded\"") != std::string::npos);
        OCT_CHECK(yes->str().find("\"evidence\"") != std::string::npos);
    }
    auto no = r.execute("prolog: parent(abe, bart)");
    OCT_CHECK(bool(no));
    if (no) OCT_CHECK(no->str().find("\"verdict\":\"refuted\"") != std::string::npos);
    auto open = r.execute("prolog: wears(bart, fedora)");
    OCT_CHECK(bool(open));
    if (open) OCT_CHECK(open->str().find("\"verdict\":\"unknown\"") != std::string::npos);
}

OCT_TEST(router, fails_closed_when_a_layer_cannot_serve) {
    router::Router r;
    // Language without a model: refused, never stubbed implicitly.
    auto lang = r.execute("llm: write me a poem");
    OCT_CHECK(!lang);
    if (!lang) OCT_CHECK(lang.status.code == Code::Unavailable);

    // Actor runtime: the router will not invent a message for an unnamed actor.
    auto actor = r.execute("actor: do something");
    OCT_CHECK(!actor);

    // Unknown requests are rejected.
    auto unknown = r.execute("zzzzz qqqqq");
    OCT_CHECK(!unknown);
    if (!unknown) OCT_CHECK(unknown.status.code == Code::Rejected);

    // A numeric route that names no kernel is refused rather than defaulted.
    auto bad_kernel = r.execute("kernel:nosuchkernel n=4");
    OCT_CHECK(!bad_kernel);
}
