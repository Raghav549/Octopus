// Tests for the language-DNA layers: APL, Prolog, LISP, Forth-style stack VM.
// Every assertion is executed: a test cannot report PASS without running one.
// SPDX-License-Identifier: MIT
#include "harness.hpp"

#include "octopus/apl.hpp"
#include "octopus/lisp.hpp"
#include "octopus/prolog.hpp"
#include "octopus/stackvm.hpp"

#include <cmath>

using namespace oct;

// ---------------------------------------------------------------------------
// APL
// ---------------------------------------------------------------------------
OCT_TEST(languages, apl_iota_reduce_and_reshape) {
    apl::Environment env;
    auto sum = apl::eval_line("+/ \xe2\x8d\xb3 100", env);            // +/ iota 100
    OCT_CHECK(bool(sum));
    OCT_EQ(sum->size(), size_t(1));
    OCT_NEAR(sum->scalar_f64(), 4950.0, 0.0);

    auto reshape = apl::eval_line("x \xe2\x86\x90 2 3 \xe2\x8d\xb4 \xe2\x8d\xb3 6", env);
    OCT_CHECK(bool(reshape));
    OCT_EQ(reshape->shape().size(), size_t(2));
    OCT_EQ(reshape->shape()[0], int64_t(2));
    OCT_EQ(reshape->shape()[1], int64_t(3));
    OCT_NEAR(reshape->scalar_f64(5), 5.0, 0.0);

    auto shape = apl::eval_line("\xe2\x8d\xb4 x", env);                // monadic shape
    OCT_CHECK(bool(shape));
    OCT_EQ(shape->size(), size_t(2));
    OCT_NEAR(shape->scalar_f64(0), 2.0, 0.0);
    OCT_NEAR(shape->scalar_f64(1), 3.0, 0.0);

    auto transpose = apl::eval_line("\xe2\x8d\x89 x", env);            // transpose
    OCT_CHECK(bool(transpose));
    OCT_EQ(transpose->shape()[0], int64_t(3));
    OCT_EQ(transpose->shape()[1], int64_t(2));
    OCT_NEAR(transpose->scalar_f64(0), 0.0, 0.0);
    OCT_NEAR(transpose->scalar_f64(1), 3.0, 0.0);
}

OCT_TEST(languages, apl_scan_and_products_match_hand_computation) {
    apl::Environment env;
    auto scan = apl::eval_line("+\\ 1 2 3 4", env);                    // scan sum
    OCT_CHECK(bool(scan));
    OCT_EQ(scan->size(), size_t(4));
    OCT_NEAR(scan->scalar_f64(0), 1.0, 0.0);
    OCT_NEAR(scan->scalar_f64(1), 3.0, 0.0);
    OCT_NEAR(scan->scalar_f64(2), 6.0, 0.0);
    OCT_NEAR(scan->scalar_f64(3), 10.0, 0.0);

    // (2 2 rho iota 4) +.x (2 2 rho iota 4) = [[2,3],[6,11]]
    auto inner = apl::eval_line(
        "(2 2 \xe2\x8d\xb4 \xe2\x8d\xb3 4) +.\xc3\x97 (2 2 \xe2\x8d\xb4 \xe2\x8d\xb3 4)", env);
    OCT_CHECK(bool(inner));
    if (inner) {
        OCT_EQ(inner->size(), size_t(4));
        OCT_NEAR(inner->scalar_f64(0), 2.0, 0.0);
        OCT_NEAR(inner->scalar_f64(1), 3.0, 0.0);
        OCT_NEAR(inner->scalar_f64(2), 6.0, 0.0);
        OCT_NEAR(inner->scalar_f64(3), 11.0, 0.0);
    }

    // Outer product: 1 2 (outer) x 3 4 5 = [[3,4,5],[6,8,10]]
    auto outer = apl::eval_line("1 2 \xe2\x88\x98.\xc3\x97 3 4 5", env);
    OCT_CHECK(bool(outer));
    if (outer) {
        OCT_EQ(outer->shape().size(), size_t(2));
        OCT_EQ(outer->shape()[0], int64_t(2));
        OCT_EQ(outer->shape()[1], int64_t(3));
        OCT_NEAR(outer->scalar_f64(0), 3.0, 0.0);
        OCT_NEAR(outer->scalar_f64(2), 5.0, 0.0);
        OCT_NEAR(outer->scalar_f64(3), 6.0, 0.0);
        OCT_NEAR(outer->scalar_f64(5), 10.0, 0.0);
    }
}

OCT_TEST(languages, apl_solve_satisfies_the_linear_system) {
    // (2 2 rho 3 1 1 2) solve 5 5  ->  x = [1,2]
    apl::Environment env;
    auto x = apl::eval_line("(2 2 \xe2\x8d\xb4 3 1 1 2) \xe2\x8c\xb9 5 5", env);
    OCT_CHECK(bool(x));
    if (!x) OCT_SKIP("APL solve unavailable: " + x.status.message);
    OCT_EQ(x->size(), size_t(2));
    OCT_NEAR(x->scalar_f64(0), 1.0, 1e-12);
    OCT_NEAR(x->scalar_f64(1), 2.0, 1e-12);
    // Residual check, computed independently of the solve: 3*1 + 1*2 = 5.
    const double r0 = 3.0 * x->scalar_f64(0) + 1.0 * x->scalar_f64(1) - 5.0;
    const double r1 = 1.0 * x->scalar_f64(0) + 2.0 * x->scalar_f64(1) - 5.0;
    OCT_NEAR(r0, 0.0, 1e-12);
    OCT_NEAR(r1, 0.0, 1e-12);
}

OCT_TEST(languages, apl_rejects_malformed_programs_without_guessing) {
    apl::Environment env;
    OCT_CHECK(!apl::eval_line("+", env));
    OCT_CHECK(!apl::eval_line("1 2 3 + 1 2", env));           // shape mismatch
    OCT_CHECK(!apl::eval_line("nonexistent_variable", env));
    auto div0 = apl::eval_line("1 \xc3\xb7 0", env);
    // IEEE semantics: 1/0 is +inf, a value rather than an error; require it to be
    // reported as non-finite instead of being silently clamped.
    if (div0) OCT_CHECK(!std::isfinite(div0->scalar_f64(0)));
}

// ---------------------------------------------------------------------------
// Prolog
// ---------------------------------------------------------------------------
OCT_TEST(languages, prolog_grounds_only_with_a_derivation) {
    prolog::KnowledgeBase kb;
    OCT_CHECK(kb.add_text("parent(homer, bart).").is_ok());
    OCT_CHECK(kb.add_text("parent(homer, lisa).").is_ok());
    OCT_CHECK(kb.add_text("parent(abe, homer).").is_ok());
    OCT_CHECK(kb.add_text("ancestor(X, Y) :- parent(X, Y).").is_ok());
    OCT_CHECK(kb.add_text("ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).").is_ok());
    kb.declare_complete("parent/2");

    auto goal = prolog::parse_term("ancestor(abe, bart)");
    OCT_CHECK(bool(goal));
    auto v = prolog::verify(kb, *goal);
    OCT_CHECK(v.verdict == prolog::Verdict::Grounded);
    OCT_CHECK(!v.evidence.empty());          // grounded *with* a proof trace
    OCT_CHECK(v.inferences > 0);

    auto false_claim = prolog::parse_term("parent(abe, bart)");
    OCT_CHECK(bool(false_claim));
    auto r = prolog::verify(kb, *false_claim);
    OCT_CHECK(r.verdict == prolog::Verdict::Refuted);      // only for complete predicates

    auto unknown_claim = prolog::parse_term("wears(bart, fedora)");
    OCT_CHECK(bool(unknown_claim));
    auto u = prolog::verify(kb, *unknown_claim);
    OCT_CHECK(u.verdict == prolog::Verdict::Unknown);      // open world: never "false"
    OCT_CHECK(u.reason.find("open world") != std::string::npos);
}

OCT_TEST(languages, prolog_arithmetic_and_bounded_search) {
    prolog::KnowledgeBase kb;
    auto expr = prolog::parse_term("X is 2 + 3 * 4 - 1");
    OCT_CHECK(bool(expr));
    auto sol = kb.solve(*expr);
    OCT_CHECK(sol.proved);
    OCT_EQ(sol.answer.bindings.count("X"), size_t(1));
    if (sol.answer.bindings.count("X"))
        OCT_NEAR(sol.answer.bindings["X"]->num, 13.0, 0.0);

    // A truncated search must report Unknown and say that it hit the limit.
    prolog::KnowledgeBase loop;
    OCT_CHECK(loop.add_text("step(X) :- step(X).").is_ok());
    auto infinite = prolog::parse_term("step(a)");
    OCT_CHECK(bool(infinite));
    prolog::Limits tight;
    tight.max_inferences = 500;
    auto t = prolog::verify(loop, *infinite, tight);
    OCT_CHECK(t.verdict == prolog::Verdict::Unknown);
    OCT_CHECK(t.hit_limit);
}

// ---------------------------------------------------------------------------
// LISP
// ---------------------------------------------------------------------------
OCT_TEST(languages, lisp_recursion_and_higher_order_functions) {
    lisp::Interp interp;
    OCT_CHECK(interp.eval_string(
        "(define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))").is_ok());
    auto f = interp.eval_string("(fib 20)");
    OCT_CHECK(bool(f));
    if (f) OCT_NEAR((*f)->num, 6765.0, 0.0);

    auto mapped = interp.eval_string("(map (lambda (x) (* x x)) (list 1 2 3 4))");
    OCT_CHECK(bool(mapped));
    if (mapped) OCT_CHECK((*mapped)->to_string() == "(1 4 9 16)");

    OCT_CHECK(!interp.eval_string("(undefined-function 1 2)"));   // errors are values
}

OCT_TEST(languages, lisp_live_patch_is_reversible) {
    lisp::Interp interp;
    OCT_CHECK(interp.eval_string("(define (f x) (* x 2))").is_ok());
    auto before = interp.eval_string("(f 21)");
    OCT_CHECK(bool(before));
    if (before) OCT_NEAR((*before)->num, 42.0, 0.0);

    auto generation = interp.patch("f", "(define (f x) (+ x 1))");
    OCT_CHECK(bool(generation));
    if (generation) OCT_EQ(*generation, int64_t(1));
    auto patched = interp.eval_string("(f 41)");
    OCT_CHECK(bool(patched));
    if (patched) OCT_NEAR((*patched)->num, 42.0, 0.0);
    OCT_EQ(interp.patch_log().size(), size_t(1));

    // Rollback restores the previous definition set exactly.
    OCT_CHECK(interp.rollback(0).is_ok());
    auto restored = interp.eval_string("(f 21)");
    OCT_CHECK(bool(restored));
    if (restored) OCT_NEAR((*restored)->num, 42.0, 0.0);
    auto patched_gone = interp.eval_string("(f 41)");
    OCT_CHECK(bool(patched_gone));
    if (patched_gone) OCT_NEAR((*patched_gone)->num, 82.0, 0.0);  // back to x*2
}

// ---------------------------------------------------------------------------
// Forth-style stack VM
// ---------------------------------------------------------------------------
OCT_TEST(languages, stackvm_words_arithmetic_and_control_flow) {
    stackvm::Vm vm;
    OCT_CHECK(vm.compile(": sq dup * ;  : cube dup sq * ;").is_ok());
    auto sq = vm.call("sq", {12.0});
    OCT_CHECK(bool(sq));
    if (sq) OCT_NEAR(*sq, 144.0, 0.0);
    auto cube = vm.call("cube", {3.0});
    OCT_CHECK(bool(cube));
    if (cube) OCT_NEAR(*cube, 27.0, 0.0);

    stackvm::Vm control;
    OCT_CHECK(control.compile(": fact dup 1 > if dup 1 - fact * else drop 1 then ;").is_ok());
    auto f = control.call("fact", {6.0});
    OCT_CHECK(bool(f));
    if (f) OCT_NEAR(*f, 720.0, 0.0);
}

OCT_TEST(languages, stackvm_memory_and_faults) {
    stackvm::Vm vm;
    // Forth store order is ( value address -- ); 42 0 ! then 0 @ must read 42.
    OCT_CHECK(vm.compile(": store1 42 0 !  0 @ ;  : bad 999999 @ ;").is_ok());
    auto ok = vm.call("store1", {});
    OCT_CHECK(bool(ok));
    if (ok) OCT_NEAR(*ok, 42.0, 0.0);

    auto fault = vm.call("bad", {});
    OCT_CHECK(!fault);                                  // out of range is a fault...
    if (!fault) OCT_CHECK(fault.status.message.find("memory fault") != std::string::npos);

    OCT_CHECK(vm.compile(": boom 1 0 / ;").is_ok());
    auto div = vm.call("boom", {});
    OCT_CHECK(!div);                                    // ... and so is division by zero
}
