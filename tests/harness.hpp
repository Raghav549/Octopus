// Octopus test harness. A test that does not execute its assertion cannot
// report PASS: every PASS implies at least one checked assertion ran.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cstdio>

namespace octest {

using TestFn = void (*)();

struct Case {
    std::string suite;
    std::string name;
    TestFn      fn;
    bool        require_llama = false;
    bool        require_fortran = false;
};

std::vector<Case>& registry();

struct Registrar {
    Registrar(const char* suite, const char* name, TestFn fn, bool llama = false, bool fortran = false) {
        registry().push_back({suite, name, fn, llama, fortran});
    }
};

// Per-test bookkeeping.
struct Ctx {
    int checks = 0;
    int failures = 0;
    std::string current;
    std::vector<std::string> messages;
};
Ctx& ctx();

void fail(const std::string& msg, const char* file, int line);
void note(const std::string& msg);

struct SkipTest { std::string reason; };

int run_all(const std::string& filter, bool json_output, bool list_only);

// Records which optional backends were compiled into this build (drives the
// per-test skip decision for OCT_TEST_REQ_LLAMA / OCT_TEST_REQ_FORTRAN).
void set_capabilities(bool llama_linked, bool fortran_linked);

}  // namespace octest

#define OCT_TEST(SUITE, NAME)                                                  \
    static void SUITE##_##NAME##_body();                                       \
    static ::octest::Registrar SUITE##_##NAME##_reg(#SUITE, #NAME,             \
                                                    &SUITE##_##NAME##_body);   \
    static void SUITE##_##NAME##_body()

#define OCT_TEST_REQ_LLAMA(SUITE, NAME)                                        \
    static void SUITE##_##NAME##_body();                                       \
    static ::octest::Registrar SUITE##_##NAME##_reg(#SUITE, #NAME,             \
                                                    &SUITE##_##NAME##_body,    \
                                                    true, false);              \
    static void SUITE##_##NAME##_body()

#define OCT_TEST_REQ_FORTRAN(SUITE, NAME)                                      \
    static void SUITE##_##NAME##_body();                                       \
    static ::octest::Registrar SUITE##_##NAME##_reg(#SUITE, #NAME,             \
                                                    &SUITE##_##NAME##_body,    \
                                                    false, true);              \
    static void SUITE##_##NAME##_body()

#define OCT_CHECK(COND)                                                        \
    do {                                                                       \
        ::octest::ctx().checks++;                                              \
        if (!(COND)) ::octest::fail("CHECK failed: " #COND, __FILE__, __LINE__); \
    } while (0)

#define OCT_CHECK_MSG(COND, MSG)                                               \
    do {                                                                       \
        ::octest::ctx().checks++;                                              \
        if (!(COND)) {                                                         \
            std::ostringstream oss_;                                           \
            oss_ << "CHECK failed: " #COND " -- " << MSG;                      \
            ::octest::fail(oss_.str(), __FILE__, __LINE__);                    \
        }                                                                      \
    } while (0)

#define OCT_EQ(A, B)                                                           \
    do {                                                                       \
        ::octest::ctx().checks++;                                              \
        auto va_ = (A);                                                        \
        auto vb_ = (B);                                                        \
        if (!(va_ == vb_)) {                                                   \
            std::ostringstream oss_;                                           \
            oss_ << "EQ failed: " #A " == " #B " (" << va_ << " vs " << vb_ << ")"; \
            ::octest::fail(oss_.str(), __FILE__, __LINE__);                    \
        }                                                                      \
    } while (0)

#define OCT_NEAR(A, B, TOL)                                                    \
    do {                                                                       \
        ::octest::ctx().checks++;                                              \
        double va_ = double(A);                                                \
        double vb_ = double(B);                                                \
        if (!(std::fabs(va_ - vb_) <= (TOL))) {                                \
            std::ostringstream oss_;                                           \
            oss_ << "NEAR failed: " #A " ~= " #B " |diff|=" << std::fabs(va_ - vb_) \
                 << " > " << (TOL);                                            \
            ::octest::fail(oss_.str(), __FILE__, __LINE__);                    \
        }                                                                      \
    } while (0)

#define OCT_THROWS(EXPR)                                                       \
    do {                                                                       \
        ::octest::ctx().checks++;                                              \
        bool threw_ = false;                                                   \
        try { EXPR; } catch (...) { threw_ = true; }                           \
        if (!threw_) ::octest::fail("expected exception: " #EXPR, __FILE__, __LINE__); \
    } while (0)

#define OCT_NOTE(MSG)                                                          \
    do {                                                                       \
        std::ostringstream oss_;                                               \
        oss_ << MSG;                                                           \
        ::octest::note(oss_.str());                                            \
    } while (0)

#define OCT_SKIP(REASON) throw ::octest::SkipTest{REASON}
