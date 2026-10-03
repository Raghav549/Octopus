// Octopus Hybrid AI Engine -- command line interface.
//
// Every command is offline, deterministic where the underlying component is
// deterministic, and prints machine-readable JSON on request. Commands that
// cannot run (no model, no Fortran compiler, no llama.cpp) fail with an
// explicit status instead of inventing a result -- that is the honesty contract
// the architecture spec requires.
// SPDX-License-Identifier: MIT
#include "octopus/apl.hpp"
#include "octopus/catalog.hpp"
#include "octopus/intercal.hpp"
#include "octopus/lisp.hpp"
#include "octopus/llm.hpp"
#include "octopus/numerics.hpp"
#include "octopus/piet.hpp"
#include "octopus/prolog.hpp"
#include "octopus/stackvm.hpp"
#include "octopus/supervisor.hpp"
#include "octopus/tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using namespace oct;

namespace {

int g_exit = 0;

void fail(const Status& s) {
    std::cout << "{\"status\":\"" << to_string(s.code) << "\",\"message\":"
              << "\"" << s.message << "\"}\n";
    g_exit = 2;
}

std::vector<std::string> args_from(int argc, char** argv, int start) {
    std::vector<std::string> out;
    for (int i = start; i < argc; ++i) out.emplace_back(argv[i]);
    return out;
}

std::map<std::string, std::string> parse_flags(const std::vector<std::string>& args) {
    std::map<std::string, std::string> flags;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i].rfind("--", 0) == 0) {
            std::string key = args[i].substr(2);
            std::string value = "true";
            if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) value = args[++i];
            else if (key.find('=') != std::string::npos) {
                value = key.substr(key.find('=') + 1);
                key = key.substr(0, key.find('='));
            }
            flags[key] = value;
        }
    }
    return flags;
}

void print_json(const Json& j) { std::cout << j.str() << "\n"; }

int cmd_version() {
    Json j;
    j.begin_object();
    j.field("engine", std::string_view(kEngineName));
    j.field("version", std::string_view(kVersionString));
    const HostInfo h = host_info();
    j.field("arch", h.arch);
    j.field("os", h.os);
    j.field("threads", int64_t(h.hw_threads));
    j.field("avx2", h.has_avx2);
    j.field("avx512", h.has_avx512);
    j.key("llama");
    j.value(llm::Host{}.backend().to_json().str());
    j.field("fortran", numerics::KernelLibrary::instance().fortran_backend_id());
    j.end_object();
    print_json(j);
    return 0;
}

int cmd_modules(bool json) {
    Registry reg;
    register_builtin_modules(reg);
    if (json) {
        Json j;
        j.begin_object();
        j.field("count", int64_t(reg.size()));
        j.key("modules");
        j.begin_array();
        // Built directly into the document: nesting a module's JSON as a string
        // would force every consumer to parse twice.
        for (const auto& info : reg.inventory()) {
            j.begin_object();
            j.field("name", info.name);
            j.field("version", info.version);
            j.field("language", info.language);
            j.field("role", info.role);
            j.field("trust", trust_name(info.trust));
            j.field("compiled_in", info.compiled_in);
            j.field("build_id", info.build_id);
            j.key("capabilities");
            j.begin_array();
            for (const auto& c : info.capabilities) j.value(c);
            j.end_array();
            j.key("limitations");
            j.begin_array();
            for (const auto& l : info.limitations) j.value(l);
            j.end_array();
            j.end_object();
        }
        j.end_array();
        j.end_object();
        print_json(j);
    } else {
        for (const auto& info : reg.inventory())
            std::cout << info.name << "  " << info.version << "  [" << info.language << "]  "
                      << info.capabilities.size() << " capabilities\n    " << info.role << "\n";
    }
    return 0;
}

int cmd_selftest(bool json) {
    Registry reg;
    register_builtin_modules(reg);
    const auto results = reg.self_check_all();
    size_t failed = 0;
    Json j;
    j.begin_object();
    j.key("modules");
    j.begin_array();
    for (const auto& r : results) {
        j.begin_object();
        j.field("module", r.module);
        j.field("status", to_string(r.status.code));
        if (!r.status.message.empty()) j.field("message", r.status.message);
        j.end_object();
        if (!r.status.is_ok()) ++failed;
    }
    j.end_array();
    j.field("modules_total", int64_t(results.size()));
    j.field("modules_failed", int64_t(failed));
    j.end_object();
    if (json) {
        print_json(j);
    } else {
        for (const auto& r : results)
            std::cout << (r.status.is_ok() ? "[ ok ] " : "[FAIL] ") << r.module
                      << (r.status.message.empty() ? "" : "  -- " + r.status.message) << "\n";
        std::cout << results.size() - failed << "/" << results.size() << " modules passed\n";
    }
    if (failed) g_exit = 1;
    return 0;
}

int cmd_kernels(const std::vector<std::string>& args, bool json) {
    const auto& lib = numerics::KernelLibrary::instance();
    const std::string sub = args.empty() ? "list" : args[0];
    if (sub == "list") {
        Json j;
        j.begin_object();
        j.key("kernels");
        j.begin_array();
        for (const auto& line : lib.describe_all()) j.value(line);
        j.end_array();
        j.field("fortran_available", lib.fortran_available());
        j.field("fortran_compiler", lib.fortran_compiler());
        j.end_object();
        print_json(j);
        return 0;
    }
    if (args.size() < 2) {
        fail(Status::invalid("usage: octopus kernels run|validate <name> [k=v ...]"));
        return 0;
    }
    const std::string name = args[1];
    auto kernel = lib.find(name);
    if (!kernel) {
        fail(Status::invalid("unknown kernel: " + name));
        return 0;
    }
    // Start from the kernel's canonical validation parameters, so a bare
    // `kernels run <name>` exercises a meaningful case instead of the generic
    // defaults (which are not valid for every kernel).
    numerics::ProblemSpec spec = kernel->validation_spec();
    for (size_t i = 2; i < args.size(); ++i) {
        const size_t eq = args[i].find('=');
        if (eq == std::string::npos) continue;
        try {
            spec.params[args[i].substr(0, eq)] = std::stod(args[i].substr(eq + 1));
        } catch (...) {
            fail(Status::invalid("malformed parameter: " + args[i]));
            return 0;
        }
    }
    if (sub == "validate") {
        const auto v = kernel->validate(spec);
        Json j = v.to_json();
        print_json(j);
        if (!v.accepted()) g_exit = 1;
        return 0;
    }
    const auto r = kernel->run(spec);
    print_json(r.to_json(/*include_data=*/json));
    return 0;
}

int cmd_apl(const std::vector<std::string>& args) {
    if (args.empty()) {
        fail(Status::invalid("usage: octopus apl \"<expression>\""));
        return 0;
    }
    apl::Environment env;
    apl::EvalTrace trace;
    auto r = apl::eval_line(args[0], env, &trace);
    if (!r) {
        fail(r.status);
        return 0;
    }
    Json j;
    j.begin_object();
    j.field("result", r->describe());
    j.field("fingerprint", r->fingerprint());
    j.field("primitives_applied", trace.primitive_count);
    j.key("values");
    j.begin_array();
    for (size_t i = 0; i < std::min<size_t>(r->size(), 16); ++i)
        j.value(r->scalar_f64(i), 17);
    j.end_array();
    j.end_object();
    print_json(j);
    return 0;
}

int cmd_prolog(const std::vector<std::string>& args) {
    const auto flags = parse_flags(args);
    auto file = flags.find("file");
    auto query = flags.find("query");
    if (query == flags.end()) {
        fail(Status::invalid("usage: octopus prolog --file kb.pl --query \"goal\""));
        return 0;
    }
    prolog::KnowledgeBase kb;
    if (file != flags.end()) {
        std::ifstream in(file->second);
        if (!in) {
            fail(Status::unavailable("cannot open knowledge base: " + file->second));
            return 0;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            Status st = kb.add_text(line);
            if (!st) {
                fail(Status::invalid("while reading " + file->second + ": " + st.message));
                return 0;
            }
        }
    }
    // Claims about the engine itself are facts, not opinions: the CLI can add
    // them with --fact "text".
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--fact" && i + 1 < args.size()) {
            Status st = kb.add_text(args[i + 1]);
            if (!st) { fail(st); return 0; }
        }
    }
    if (auto complete = flags.find("complete"); complete != flags.end()) {
        std::istringstream is(complete->second);
        std::string p;
        while (std::getline(is, p, ',')) if (!p.empty()) kb.declare_complete(p);
    }
    auto goal = prolog::parse_term(query->second);
    if (!goal) {
        fail(goal.status);
        return 0;
    }
    prolog::Limits limits;
    limits.max_solutions = flags.count("all") ? 32 : 1;
    const auto v = prolog::verify(kb, *goal, limits);
    print_json(v.to_json());
    if (v.verdict == prolog::Verdict::Refuted) g_exit = 1;
    return 0;
}

int cmd_lisp(const std::vector<std::string>& args) {
    if (args.empty()) {
        fail(Status::invalid("usage: octopus lisp \"<expression>\""));
        return 0;
    }
    lisp::Interp interp;
    auto r = interp.eval_string(args[0]);
    if (!r) {
        fail(r.status);
        return 0;
    }
    Json j;
    j.begin_object();
    j.field("result", (*r)->to_string());
    j.field("generation", interp.generation());
    j.end_object();
    print_json(j);
    return 0;
}

int cmd_forth(const std::vector<std::string>& args) {
    if (args.empty()) {
        fail(Status::invalid("usage: octopus forth \"<program>\""));
        return 0;
    }
    stackvm::Vm vm;
    Status st = vm.compile(args[0]);
    if (!st) { fail(st); return 0; }
    auto steps = vm.run();
    if (!steps) { fail(steps.status); return 0; }
    Json j;
    j.begin_object();
    j.field("steps", *steps);
    j.field("output", vm.output());
    j.key("stack");
    j.begin_array();
    for (double v : vm.data_stack()) j.value(v, 17);
    j.end_array();
    j.end_object();
    print_json(j);
    return 0;
}

int cmd_tokenizer(const std::vector<std::string>& args) {
    const auto flags = parse_flags(args);
    auto model = flags.find("model");
    if (model == flags.end()) {
        fail(Status::invalid("usage: octopus tokenizer --model <vocab.gguf> [--text 'x'] "
                             "[--fixture <vocab.gguf>] [--ids '1 2 3']"));
        return 0;
    }
    auto vocab = tokenizer::Vocab::load(model->second);
    if (!vocab) { fail(vocab.status); return 0; }
    if (auto fixture = flags.find("fixture"); fixture != flags.end()) {
        auto rc = tokenizer::check_against_fixture(*vocab, fixture->second + ".inp",
                                                   fixture->second + ".out");
        if (!rc) { fail(rc.status); return 0; }
        Json j = rc->to_json();
        print_json(j);
        return 0;
    }
    if (auto text = flags.find("text"); text != flags.end()) {
        auto enc = tokenizer::encode(*vocab, text->second);
        if (!enc) { fail(enc.status); return 0; }
        Json j;
        j.begin_object();
        j.field("method", enc->method);
        j.field("degraded", enc->degraded);
        if (!enc->warnings.empty()) j.field("warnings", enc->warnings);
        j.field("count", int64_t(enc->ids.size()));
        j.key("ids");
        j.begin_array();
        for (uint32_t id : enc->ids) j.value(int64_t(id));
        j.end_array();
        auto back = tokenizer::decode(*vocab, enc->ids);
        if (back) j.field("roundtrip", *back == text->second ? "exact" : "different");
        j.end_object();
        print_json(j);
        return 0;
    }
    Json j;
    j.begin_object();
    j.field("model", vocab->model);
    j.field("tokens", int64_t(vocab->size()));
    j.field("merges", int64_t(vocab->merges.size()));
    j.field("exact_bpe", vocab->exact_bpe);
    j.end_object();
    print_json(j);
    return 0;
}

int cmd_llm(const std::vector<std::string>& args) {
    const auto flags = parse_flags(args);
    llm::Host host;
    auto model = flags.find("model");
    if (model == flags.end()) {
        print_json(host.backend().to_json());
        return 0;
    }
    auto facts = host.inspect(model->second);
    if (!facts) { fail(facts.status); return 0; }
    if (auto prompt = flags.find("prompt"); prompt != flags.end()) {
        llm::GenerateParams p;
        if (auto mt = flags.find("max-tokens"); mt != flags.end()) p.max_tokens = std::stoll(mt->second);
        Status ld = host.load(model->second, /*allow_stub_fallback=*/flags.count("allow-stub") > 0);
        if (!ld) { fail(ld); return 0; }
        auto text = host.generate(prompt->second, p);
        if (!text) { fail(text.status); return 0; }
        Json j;
        j.begin_object();
        j.field("text", *text);
        j.key("backend");
        j.value(host.backend().to_json().str());
        j.end_object();
        print_json(j);
        return 0;
    }
    print_json(facts->to_json());
    return 0;
}

int cmd_guard(const std::vector<std::string>& args) {
    const auto flags = parse_flags(args);
    const std::string sub = args.empty() ? std::string() : args[0];
    auto key = flags.find("key");
    if (key == flags.end()) {
        fail(Status::invalid("usage: octopus guard encode|decode|fuzz --key K --in F [--out F]"));
        return 0;
    }
    auto read_file = [](const std::string& path) -> Outcome<std::string> {
        std::ifstream in(path, std::ios::binary);
        if (!in) return Status::unavailable("cannot open " + path);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    auto write_file = [](const std::string& path, const std::string& data) -> Status {
        std::ofstream f(path, std::ios::binary);
        if (!f) return Status::unavailable("cannot write " + path);
        f.write(data.data(), std::streamsize(data.size()));
        return Status::ok();
    };
    auto in_path = flags.find("in");
    if (in_path == flags.end()) { fail(Status::invalid("guard needs --in")); return 0; }
    auto data = read_file(in_path->second);
    if (!data) { fail(data.status); return 0; }

    if (sub == "fuzz") {
        const int64_t trials = flags.count("trials") ? std::stoll(flags.at("trials")) : 1000;
        const intercal::FuzzReport r =
            intercal::fuzz_guard(*data, key->second, size_t(trials), 1);
        print_json(r.to_json());
        if (r.accepted != 0) g_exit = 1;      // any accepted mutation is a failure
        return 0;
    }
    if (sub == "encode") {
        const std::string blob = intercal::encode(*data, key->second);
        const auto out = flags.find("out");
        if (out == flags.end()) { fail(Status::invalid("guard encode needs --out")); return 0; }
        Status st = write_file(out->second, blob);
        if (!st) { fail(st); return 0; }
        Json j;
        j.begin_object();
        j.field("bytes", int64_t(blob.size()));
        j.field("out", out->second);
        j.field("note", "obfuscation + tamper evidence, not a cipher");
        j.end_object();
        print_json(j);
        return 0;
    }
    if (sub == "decode") {
        auto pt = intercal::decode(*data, key->second);
        if (!pt) { fail(pt.status); return 0; }
        const auto out = flags.find("out");
        Status st = write_file(out == flags.end() ? std::string("/dev/stdout") : out->second, *pt);
        if (!st) { fail(st); return 0; }
        return 0;
    }
    fail(Status::invalid("unknown guard subcommand: " + sub));
    return 0;
}

int cmd_piet(const std::vector<std::string>& args) {
    const auto flags = parse_flags(args);
    auto in = flags.find("in");
    auto out = flags.find("out");
    if (in == flags.end() || out == flags.end()) {
        fail(Status::invalid("usage: octopus piet --in state.txt --out state.png [--width W --height H]"));
        return 0;
    }
    std::ifstream f(in->second);
    std::vector<double> values;
    double v = 0.0;
    while (f >> v) values.push_back(v);
    if (values.empty()) { fail(Status::invalid("piet: no numbers in " + in->second)); return 0; }
    const int w = flags.count("width") ? std::stoi(flags.at("width")) : 64;
    const int h = flags.count("height") ? std::stoi(flags.at("height")) : 64;
    double lo = -1.0, hi = 1.0;
    if (flags.count("lo")) lo = std::stod(flags.at("lo"));
    if (flags.count("hi")) hi = std::stod(flags.at("hi"));
    auto r = piet::render_state(values, w, h, lo, hi);
    Status w1 = out->second.size() > 4 && out->second.substr(out->second.size() - 4) == ".ppm"
                    ? r.write_ppm(out->second)
                    : r.write_png(out->second);
    if (!w1) { fail(w1); return 0; }
    Json j;
    j.begin_object();
    j.field("out", out->second);
    j.field("width", w);
    j.field("height", h);
    j.field("sha256", r.fingerprint());
    j.key("legend");
    j.value(piet::palette_legend().str());
    j.end_object();
    print_json(j);
    return 0;
}

int cmd_supervise(bool json) {
    supervisor::Supervisor sup;
    sup.add_check(supervisor::Check{"kernels.validate", [] {
                                        for (const auto& kernel_name :
                                             numerics::KernelLibrary::instance().names()) {
                                            auto k = numerics::KernelLibrary::instance().find(kernel_name);
                                            if (!k) return Status::internal("kernel vanished: " + kernel_name);
                                            const numerics::Validation v = k->validate(k->validation_spec());
                                            if (!v.accepted())
                                                return Status::internal("kernel '" + kernel_name +
                                                                        "' failed validation: " + v.notes);
                                        }
                                        return Status::ok();
                                    }, 60.0, supervisor::Severity::Critical});
    sup.add_check(supervisor::Check{"layers.self_check", [] {
                                        Registry reg;
                                        register_builtin_modules(reg);
                                        for (const auto& r : reg.self_check_all())
                                            if (!r.status.is_ok())
                                                return Status::internal(r.module + ": " +
                                                                        r.status.message);
                                        return Status::ok();
                                    }, 300.0, supervisor::Severity::Critical});
    const auto rows = sup.tick(1);
    (void)rows;
    const Json report = sup.report();
    if (json) print_json(report);
    else {
        std::cout << "supervisor healthy=" << (sup.healthy() ? "yes" : "no")
                  << " restarts=" << sup.restarts() << "\n"
                  << report.str() << "\n";
    }
    return 0;
}

int cmd_doctor() {
    Json j;
    j.begin_object();
    const auto& lib = numerics::KernelLibrary::instance();
    j.field("fortran_available", lib.fortran_available());
    j.field("fortran_compiler", lib.fortran_compiler());
    j.key("llama");
    j.value(llm::Host{}.backend().to_json().str());
    const HostInfo h = host_info();
    j.field("threads", int64_t(h.hw_threads));
    j.field("arch", h.arch);
    j.field("os", h.os);
    j.key("kernels");
    j.begin_array();
    for (const auto& n : lib.names()) j.value(n);
    j.end_array();
    j.end_object();
    print_json(j);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = args_from(argc, argv, 1);
    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        std::cout
            << "octopus " << kVersionString << " -- offline hybrid AI engine\n\n"
            << "usage: octopus <command> [options]\n\n"
            << "  version                     build + host + backend report (JSON)\n"
            << "  doctor                      capability report for this machine\n"
            << "  modules [--json]            capability inventory of every module\n"
            << "  selftest [--json]           run every module's self-check\n"
            << "  kernels list|run|validate   numerical kernel library\n"
            << "  apl \"expr\"                  evaluate an APL expression\n"
            << "  prolog --file kb.pl --query \"goal\" [--complete p/2]\n"
            << "  lisp \"expr\"                 evaluate a LISP expression\n"
            << "  forth \"program\"             run a Forth-style program\n"
            << "  tokenizer --model v.gguf [--text 'x' | --fixture v.gguf]\n"
            << "  llm [--model m.gguf [--prompt 'p' [--max-tokens N] [--allow-stub]]]\n"
            << "  guard encode|decode|fuzz --key K --in F [--out F] [--trials N]\n"
            << "  piet --in state.txt --out state.png [--width W --height H]\n"
            << "  supervise [--json]          run health checks and print the report\n";
        return 0;
    }
    const bool json = std::find(args.begin(), args.end(), std::string("--json")) != args.end();
    const std::string cmd = args[0];
    if (cmd == "version") return cmd_version();
    if (cmd == "doctor") return cmd_doctor();
    if (cmd == "modules") return cmd_modules(json);
    if (cmd == "selftest") return cmd_selftest(json);
    if (cmd == "kernels") return cmd_kernels(args_from(argc, argv, 2), json);
    if (cmd == "apl") return cmd_apl(args_from(argc, argv, 2));
    if (cmd == "prolog") return cmd_prolog(args_from(argc, argv, 2));
    if (cmd == "lisp") return cmd_lisp(args_from(argc, argv, 2));
    if (cmd == "forth") return cmd_forth(args_from(argc, argv, 2));
    if (cmd == "tokenizer") return cmd_tokenizer(args_from(argc, argv, 2));
    if (cmd == "llm") return cmd_llm(args_from(argc, argv, 2));
    if (cmd == "guard") return cmd_guard(args_from(argc, argv, 2));
    if (cmd == "piet") return cmd_piet(args_from(argc, argv, 2));
    if (cmd == "supervise") return cmd_supervise(json);
    fail(Status::invalid("unknown command: " + cmd + " (try --help)"));
    return g_exit;
}
