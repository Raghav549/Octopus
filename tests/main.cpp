// Octopus test runner.
// SPDX-License-Identifier: MIT
#include "harness.hpp"

#include "octopus/core.hpp"

#include <algorithm>
#include <fstream>
#include <cstring>

namespace octest {

std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

Ctx& ctx() {
    static Ctx c;
    return c;
}

void fail(const std::string& msg, const char* file, int line) {
    Ctx& c = ctx();
    c.failures++;
    std::ostringstream os;
    os << file << ":" << line << ": " << msg;
    c.messages.push_back(os.str());
}

void note(const std::string& msg) { ctx().messages.push_back("note: " + msg); }

static bool g_have_llama = false;
static bool g_have_fortran = false;

void set_capabilities(bool llama, bool fortran) {
    g_have_llama = llama;
    g_have_fortran = fortran;
}

int run_all(const std::string& filter, bool json_output, bool list_only) {
    auto& cases = registry();
    std::stable_sort(cases.begin(), cases.end(), [](const Case& a, const Case& b) {
        if (a.suite != b.suite) return a.suite < b.suite;
        return a.name < b.name;
    });

    if (list_only) {
        for (const auto& c : cases) std::cout << c.suite << "." << c.name << "\n";
        return 0;
    }

    int passed = 0, failed = 0, skipped = 0, total_checks = 0;
    std::vector<std::pair<std::string, std::vector<std::string>>> failures;

    for (const auto& c : cases) {
        std::string full = c.suite + "." + c.name;
        if (!filter.empty() && full.find(filter) == std::string::npos) continue;

        Ctx& cc = ctx();
        cc.checks = 0;
        cc.failures = 0;
        cc.current = full;
        cc.messages.clear();

        std::string skip_reason;
        if (c.require_llama && !g_have_llama) skip_reason = "llama.cpp not available in this build";
        else if (c.require_fortran && !g_have_fortran) skip_reason = "Fortran backend not compiled";

        if (!skip_reason.empty()) {
            skipped++;
            std::cout << "[ SKIP ] " << full << " (" << skip_reason << ")\n";
            continue;
        }

        bool crashed = false;
        std::string error;
        try {
            c.fn();
        } catch (const SkipTest& s) {
            skipped++;
            std::cout << "[ SKIP ] " << full << " (" << s.reason << ")\n";
            continue;
        } catch (const std::exception& e) {
            crashed = true;
            error = e.what();
        } catch (...) {
            crashed = true;
            error = "unknown exception";
        }

        total_checks += cc.checks;

        // A PASS requires >= 1 executed assertion and zero failures.
        bool ok = !crashed && cc.failures == 0 && cc.checks > 0;
        if (crashed) {
            std::ostringstream os;
            os << "uncaught exception: " << error;
            cc.messages.push_back(os.str());
        } else if (cc.checks == 0 && cc.failures == 0) {
            cc.messages.push_back("no assertions executed -- refusing to report PASS");
        }

        if (ok) {
            passed++;
            if (!json_output) {
                std::cout << "[ PASS ] " << full << " (" << cc.checks << " checks)";
                for (const auto& m : cc.messages) std::cout << "\n         " << m;
                std::cout << "\n";
            }
        } else {
            failed++;
            failures.emplace_back(full, cc.messages);
            std::cout << "[ FAIL ] " << full << " (" << cc.checks << " checks)\n";
            for (const auto& m : cc.messages) std::cout << "         " << m << "\n";
        }
    }

    if (json_output) {
        oct::Json j;
        j.begin_object();
        j.field("passed", passed);
        j.field("failed", failed);
        j.field("skipped", skipped);
        j.field("assertions", total_checks);
        j.field("llama_available", g_have_llama);
        j.field("fortran_available", g_have_fortran);
        j.key("failures");
        j.begin_array();
        for (const auto& f : failures) {
            j.begin_object();
            j.field("test", f.first);
            for (const auto& m : f.second) j.field("msg", m);
            j.end_object();
        }
        j.end_array();
        j.end_object();
        std::cout << j.take() << "\n";
    }

    std::cout << "\n== " << passed << " passed, " << failed << " failed, " << skipped
              << " skipped, " << total_checks << " assertions ==\n";
    return failed == 0 ? 0 : 1;
}

}  // namespace octest

int main(int argc, char** argv) {
    std::string filter;
    bool json = false, list = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--json") json = true;
        else if (a == "--list") list = true;
        else if (a.rfind("--filter=", 0) == 0) filter = a.substr(9);
        else if (a == "--filter" && i + 1 < argc) filter = argv[++i];
        else if (a == "-h" || a == "--help") {
            std::cout << "usage: oct_tests [--filter=SUBSTR] [--json] [--list]\n";
            return 0;
        }
    }
    oct::set_log_level(oct::LogLevel::Error);
    return octest::run_all(filter, json, list);
}
