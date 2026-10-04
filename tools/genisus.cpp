// Genisus / Octopus Unified Hybrid AI Engine -- CLI & Multi-Dimensional
// Local Canvas Dashboard Interface.
//
// Executes deterministic, non-hallucinatory physical-universe computing across
// all強制 (forced) multi-language DNA bindings:
//   - Fortran 2023 (Absolute Physics Core: fluids, Kepler/N-body, tensor fields)
//   - LISP (Autonomous Evolution Engine: live AST hot-patching & rollback)
//   - Prolog (Deterministic Soundness Guardrail: SLD chronological backtracking)
//   - APL (Array Codec & Context Matrix Compressor: right-to-left evaluation)
//   - Occam, Forth, & Smalltalk (CSP parallel threads, register VM, live actors)
//   - INTERCAL (The Obfuscated Core Shield: bit-mingled hash-linked envelope)
//   - Piet (Visual Canvas Logic Engine: 20-colour codel & synapse rasterizer)
//
// SPDX-License-Identifier: MIT

#include "octopus/apl.hpp"
#include "octopus/catalog.hpp"
#include "octopus/intercal.hpp"
#include "octopus/lisp.hpp"
#include "octopus/llm.hpp"
#include "octopus/numerics.hpp"
#include "octopus/occam.hpp"
#include "octopus/piet.hpp"
#include "octopus/prolog.hpp"
#include "octopus/router.hpp"
#include "octopus/smalltalk.hpp"
#include "octopus/stackvm.hpp"
#include "octopus/supervisor.hpp"
#include "octopus/tokenizer.hpp"
#include "octopus/universe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#if defined(__unix__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace oct;

namespace {

int g_exit = 0;

void fail(const Status& s) {
    std::cout << "{\"status\":\"" << to_string(s.code) << "\",\"message\":"
              << "\"" << s.message << "\"}\n";
    g_exit = s.is_hardware_skip() ? 0 : 2;
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
            if (key.find('=') != std::string::npos) {
                value = key.substr(key.find('=') + 1);
                key = key.substr(0, key.find('='));
            } else if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) {
                value = args[++i];
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
    j.field("cli", "genisus");
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
    size_t skipped = 0;
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
        if (r.status.is_hardware_skip()) {
            ++skipped;
        } else if (!r.status.is_ok()) {
            ++failed;
        }
    }
    j.end_array();
    j.field("modules_total", int64_t(results.size()));
    j.field("modules_hardware_skipped", int64_t(skipped));
    j.field("modules_failed", int64_t(failed));
    j.end_object();
    if (json) {
        print_json(j);
    } else {
        for (const auto& r : results) {
            const char* tag = r.status.is_ok() ? "[ ok ] "
                            : r.status.is_hardware_skip() ? "[UNSUPPORTED_HARDWARE_SKIP] "
                            : "[FAIL] ";
            std::cout << tag << r.module
                      << (r.status.message.empty() ? "" : "  -- " + r.status.message) << "\n";
        }
        std::cout << (results.size() - failed) << "/" << results.size()
                  << " modules operational (" << skipped << " hardware-skipped)\n";
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
        fail(Status::invalid("usage: genisus kernels run|validate <name> [k=v ...]"));
        return 0;
    }
    const std::string name = args[1];
    auto kernel = lib.find(name);
    if (!kernel) {
        fail(Status::invalid("unknown kernel: " + name));
        return 0;
    }
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
        if (!v.hardware_skipped && !v.accepted()) g_exit = 1;
        return 0;
    }
    const auto r = kernel->run(spec);
    print_json(r.to_json(/*include_data=*/json));
    return 0;
}

int cmd_apl(const std::vector<std::string>& args) {
    if (args.empty()) {
        fail(Status::invalid("usage: genisus apl \"<expression>\""));
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
        fail(Status::invalid("usage: genisus prolog --file kb.pl --query \"goal\""));
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
        fail(Status::invalid("usage: genisus lisp \"<expression>\""));
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
        fail(Status::invalid("usage: genisus forth \"<program>\""));
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
    j.key("registers");
    j.begin_array();
    for (size_t i = 0; i < 4; ++i) j.value(vm.get_register(i), 17);
    j.end_array();
    j.end_object();
    print_json(j);
    return 0;
}

int cmd_tokenizer(const std::vector<std::string>& args) {
    const auto flags = parse_flags(args);
    auto model = flags.find("model");
    if (model == flags.end()) {
        fail(Status::invalid("usage: genisus tokenizer --model <vocab.gguf> [--text 'x'] "
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
        Status ld = host.load(model->second);
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
        fail(Status::invalid("usage: genisus guard encode|decode|fuzz --key K --in F [--out F]"));
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
        if (r.accepted != 0) g_exit = 1;
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
        fail(Status::invalid("usage: genisus piet --in state.txt --out state.png [--width W --height H]"));
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

int cmd_ask(const std::vector<std::string>& args) {
    const auto flags = parse_flags(args);
    std::string request;
    for (const auto& a : args) {
        if (a.rfind("--", 0) != 0) { request = a; break; }
    }
    if (request.empty()) {
        fail(Status::invalid("usage: genisus ask \"<request>\" [--model M.gguf] [--out F]"));
        return 0;
    }
    if (flags.count("model")) request += " model=" + flags.at("model");
    if (flags.count("out")) request += " out=" + flags.at("out");
    router::Router r;
    router::RouterOptions opts;
    if (flags.count("model")) opts.model_path = flags.at("model");
    if (flags.count("out")) opts.output_path = flags.at("out");
    auto out = r.execute(request, opts);
    if (!out) {
        fail(out.status);
        return 0;
    }
    print_json(*out);
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
                                            if (!v.hardware_skipped && !v.accepted())
                                                return Status::internal("kernel '" + kernel_name +
                                                                        "' failed validation: " + v.notes);
                                        }
                                        return Status::ok();
                                    }, 60.0, supervisor::Severity::Critical});
    sup.add_check(supervisor::Check{"layers.self_check", [] {
                                        Registry reg;
                                        register_builtin_modules(reg);
                                        for (const auto& r : reg.self_check_all())
                                            if (!r.status.is_ok() && !r.status.is_hardware_skip())
                                                return Status::internal(r.module + ": " +
                                                                        r.status.message);
                                        return Status::ok();
                                    }, 300.0, supervisor::Severity::Critical});

    // Also run a live Autonomous Agent Supervisor healing cycle (Smalltalk + LISP + Prolog)
    smalltalk::ActorSystem actors("watchdog.cell");
    lisp::Interp interp;
    (void)interp.eval_string("(define (cell-fn x) (/ 100 0))"); // faulty initial binding
    prolog::KnowledgeBase kb;
    prolog::load_axiom_core(kb);
    actors.spawn("physics.watchdog", [&](const std::string&, const smalltalk::Args& call_args) -> Outcome<std::string> {
        const std::string arg = call_args.empty() ? "4" : call_args[0];
        auto r = interp.eval_string("(cell-fn " + arg + ")");
        if (!r.ok()) return r.status;
        return (*r)->to_string();
    }, smalltalk::Strategy::OneForOne, 3);

    auto heal_rep = sup.heal_actor_exception(
        actors, interp, kb,
        "physics.watchdog", "eval", {"4"},
        "cell-fn", "(lambda (x) (/ 100 (+ x 1)))",
        "sound_backend(fortran2023)");
    (void)heal_rep;

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

int cmd_universe(const std::vector<std::string>& args) {
    const auto flags = parse_flags(args);
    universe::SceneSpec spec;
    if (flags.count("width"))    spec.width = std::stoi(flags.at("width"));
    if (flags.count("height"))   spec.height = std::stoi(flags.at("height"));
    if (flags.count("frames"))   spec.frames = std::stoi(flags.at("frames"));
    if (flags.count("time"))     spec.time_t = std::stod(flags.at("time"));
    if (flags.count("reynolds")) spec.reynolds = std::stod(flags.at("reynolds"));
    if (flags.count("out"))      spec.output_path = flags.at("out");

    if (spec.frames > 1 ||
        (spec.output_path.size() > 4 &&
         spec.output_path.substr(spec.output_path.size() - 4) == ".y4m")) {
        const universe::VideoResult vr = universe::synthesize_video(spec);
        print_json(vr.to_json());
        return 0;
    }
    const universe::FrameResult fr = universe::synthesize_frame(spec);
    print_json(fr.to_json());
    return 0;
}

int cmd_heal() {
    supervisor::Supervisor sup;
    smalltalk::ActorSystem actors("autonomous.watchdog");
    lisp::Interp interp;
    (void)interp.eval_string("(define (flux-step x) (/ x 0))");
    prolog::KnowledgeBase kb;
    prolog::load_axiom_core(kb);

    actors.spawn("flux.cell", [&](const std::string&, const smalltalk::Args& call_args) -> Outcome<std::string> {
        const std::string arg = call_args.empty() ? "9" : call_args[0];
        auto r = interp.eval_string("(flux-step " + arg + ")");
        if (!r.ok()) return r.status;
        return (*r)->to_string();
    }, smalltalk::Strategy::OneForOne, 3);

    const auto rep = sup.heal_actor_exception(
        actors, interp, kb,
        "flux.cell", "compute", {"9"},
        "flux-step", "(lambda (x) (* (+ x 1) 2.5))",
        "sound_backend(fortran2023)");
    print_json(rep.to_json());
    return rep.recovered_live ? 0 : 1;
}

int cmd_triad(const std::vector<std::string>& args) {
    const auto flags = parse_flags(args);
    const std::string prog = flags.count("forth") ? flags.at("forth") : "";
    const double seed = flags.count("seed") ? std::stod(flags.at("seed")) : 3.0;
    const auto res = occam::coordinate_triad(prog, seed);
    print_json(res.to_json());
    return res.ok ? 0 : 1;
}

// Build a deterministic 4-channel coordinate matrix for the Piet Canvas
// visual preview when inspecting the multi-dimensional dashboard.
piet::Raster build_dashboard_canvas(int w, int h, double t, router::Router& router) {
    if (numerics::fortran_bridge::available()) {
        universe::SceneSpec spec;
        spec.width = w;
        spec.height = h;
        spec.time_t = t;
        auto fr = universe::synthesize_frame(spec);
        if (!fr.hardware_skipped && fr.status.is_ok()) return fr.canvas;
    }
    return router.render_synapses("covariant riemannian tensor_field kepler shock tube", w, h);
}

std::string render_ansi_canvas(const piet::Raster& r) {
    std::ostringstream os;
    // Render 2 vertical pixels per terminal character row using UTF-8 upper half block '▀'
    for (int y = 0; y + 1 < r.height; y += 2) {
        os << "  │";
        for (int x = 0; x < r.width; ++x) {
            const size_t i_top = size_t(y * r.width + x) * 3;
            const size_t i_bot = size_t((y + 1) * r.width + x) * 3;
            os << "\033[38;2;" << int(r.rgb[i_top + 0]) << ";" << int(r.rgb[i_top + 1]) << ";" << int(r.rgb[i_top + 2])
               << ";48;2;" << int(r.rgb[i_bot + 0]) << ";" << int(r.rgb[i_bot + 1]) << ";" << int(r.rgb[i_bot + 2])
               << "m\xe2\x96\x80\033[0m";
        }
        os << "│\n";
    }
    return os.str();
}

Json build_dashboard_state_json() {
    router::Router r;
    const auto tel = r.telemetry();
    const auto shield = r.seal_weights();
    const piet::Raster canvas = build_dashboard_canvas(24, 12, 0.35, r);
    const piet::CodelTrace ctrace = piet::execute_canvas_pathway(canvas, 128);
    const occam::TriadExecution triad = occam::coordinate_triad("", 4.0);

    // Run live Autonomous Agent Supervisor cycle
    supervisor::Supervisor sup;
    smalltalk::ActorSystem actors("dashboard.watchdog");
    lisp::Interp interp;
    (void)interp.eval_string("(define ( dash-step x ) (/ x 0))");
    prolog::KnowledgeBase kb;
    prolog::load_axiom_core(kb);
    actors.spawn("dash.cell", [&](const std::string&, const smalltalk::Args& a) -> Outcome<std::string> {
        auto v = interp.eval_string("(dash-step " + (a.empty() ? std::string("7") : a[0]) + ")");
        if (!v.ok()) return v.status;
        return (*v)->to_string();
    });
    const auto heal = sup.heal_actor_exception(
        actors, interp, kb, "dash.cell", "step", {"7"},
        "dash-step", "(lambda (x) (+ (* x x) 1))", "sound_backend(fortran2023)");

    // Run APL context matrix compression
    const std::vector<uint32_t> sample_tokens = {
        11, 24, 37, 42, 19, 88, 63, 71, 15, 29, 44, 52, 91, 103, 67, 84
    };
    const auto apl_cc = apl::compress_context_matrix(sample_tokens, 4);

    Json j;
    j.begin_object();
    j.field("cli", "genisus");
    j.field("engine", std::string_view(kEngineName));
    j.field("version", std::string_view(kVersionString));
    j.field("fortran_available", numerics::KernelLibrary::instance().fortran_available());
    j.field("fortran_backend", numerics::KernelLibrary::instance().fortran_backend_id());
    j.key("neural_router");
    j.raw_json(tel.to_json().str());
    j.key("intercal_shield");
    j.raw_json(shield.to_json().str());
    j.key("autonomous_watchdog");
    j.raw_json(heal.to_json().str());
    j.key("triad_coordinator");
    j.raw_json(triad.to_json().str());
    if (apl_cc.ok()) {
        j.key("apl_context_compression");
        j.raw_json(apl_cc->to_json().str());
    }
    j.key("piet_canvas");
    j.begin_object();
    j.field("width", int64_t(canvas.width));
    j.field("height", int64_t(canvas.height));
    j.field("fingerprint", canvas.fingerprint());
    j.field("codel_transitions", ctrace.codel_transitions);
    j.key("codes");
    j.begin_array();
    for (int c : canvas.codes) j.value(int64_t(c));
    j.end_array();
    j.end_object();
    j.end_object();
    return j;
}

std::string dashboard_html() {
    return R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GENISUS // Multi-Dimensional Physical-Universe Canvas Dashboard</title>
<style>
  :root {
    --bg: #080b12;
    --panel: #101726;
    --border: #1e2d4a;
    --accent: #38bdf8;
    --accent2: #a855f7;
    --ok: #22c55e;
    --warn: #f59e0b;
    --text: #e2e8f0;
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px;
    background: radial-gradient(circle at top right, #111c35, var(--bg));
    color: var(--text);
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 20px; border: 1px solid var(--border);
    background: var(--panel); border-radius: 10px; margin-bottom: 18px;
  }
  h1 { margin: 0; font-size: 1.25rem; letter-spacing: 0.06em; color: var(--accent); }
  .badge {
    padding: 4px 10px; border-radius: 999px; font-size: 0.75rem;
    background: rgba(56, 189, 248, 0.14); border: 1px solid var(--accent); color: var(--accent);
  }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 18px;
  }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }
  .card h2 {
    margin: 0 0 12px 0; font-size: 0.95rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--accent);
    border-bottom: 1px solid var(--border); padding-bottom: 8px;
  }
  canvas {
    width: 100%; height: 220px; image-rendering: pixelated;
    border: 1px solid var(--border); border-radius: 6px; background: #000;
  }
  .kv { display: flex; justify-content: space-between; font-size: 0.82rem; padding: 5px 0; border-bottom: 1px dashed rgba(148,163,184,0.15); }
  .kv span:first-child { color: var(--muted); }
  .controls { display: flex; gap: 8px; margin-top: 10px; }
  input, button {
    font-family: inherit; font-size: 0.82rem; padding: 8px 12px;
    border-radius: 6px; border: 1px solid var(--border);
    background: #0b111e; color: var(--text);
  }
  input { flex: 1; }
  button {
    background: linear-gradient(135deg, #0284c7, #7c3aed);
    border: none; cursor: pointer; font-weight: 600;
  }
  pre {
    background: #060911; border: 1px solid var(--border); border-radius: 6px;
    padding: 10px; font-size: 0.76rem; overflow-x: auto; max-height: 200px;
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>GENISUS // OCTOPUS HYBRID AI ENGINE</h1>
    <div style="font-size:0.78rem;color:var(--muted);margin-top:4px;">
      Forced Multi-Language DNA Regime: Fortran 2023 • LISP • Prolog • APL • Occam • Forth • Smalltalk • INTERCAL • Piet
    </div>
  </div>
  <div class="badge" id="status-badge">INITIALIZING...</div>
</header>

<div class="grid">
  <div class="card">
    <h2>Piet 20-Colour Visual Canvas &amp; Physical Universe Projection</h2>
    <canvas id="piet-canvas" width="24" height="12"></canvas>
    <div id="piet-meta" style="margin-top:8px;"></div>
  </div>

  <div class="card">
    <h2>Inline Micro-Neural Router (64 &rarr; 32 &rarr; 10 GELU)</h2>
    <div id="router-meta"></div>
    <div class="controls">
      <input id="prompt-input" value="what is the heat equation diffusion?" placeholder="Enter prompt to classify &amp; route..."/>
      <button onclick="runRoute()">Route</button>
    </div>
    <pre id="route-out">Click "Route" to inspect 64-D receptor embedding &amp; speed-modulated dispatch.</pre>
  </div>

  <div class="card">
    <h2>Autonomous Agent Supervisor (Smalltalk + LISP + Prolog)</h2>
    <div id="watchdog-meta"></div>
    <pre id="watchdog-json"></pre>
  </div>

  <div class="card">
    <h2>APL Context Matrix Compressor &amp; Occam/Forth/Smalltalk Triad</h2>
    <div id="triad-meta"></div>
    <pre id="triad-json"></pre>
  </div>
</div>

<script>
const PIET_PALETTE = [
  "#ffc0c0","#ff0000","#c00000","#ffffc0","#ffff00","#c0c000",
  "#c0ffc0","#00ff00","#00c000","#c0ffff","#00ffff","#00c0c0",
  "#c0c0ff","#0000ff","#0000c0","#ffc0ff","#ff00ff","#c000c0",
  "#000000","#ffffff"
];

function drawPiet(canvasData) {
  const cv = document.getElementById('piet-canvas');
  cv.width = canvasData.width;
  cv.height = canvasData.height;
  const ctx = cv.getContext('2d');
  for (let y = 0; y < canvasData.height; y++) {
    for (let x = 0; x < canvasData.width; x++) {
      const idx = canvasData.codes[y * canvasData.width + x] || 0;
      ctx.fillStyle = PIET_PALETTE[idx] || "#000";
      ctx.fillRect(x, y, 1, 1);
    }
  }
}

async function refreshState() {
  const res = await fetch('/api/state');
  const d = await res.json();
  document.getElementById('status-badge').textContent =
    d.fortran_available ? "FORTRAN 2023 NATIVE" : "UNSUPPORTED_HARDWARE_SKIP GUARD ACTIVE";
  drawPiet(d.piet_canvas);
  document.getElementById('piet-meta').innerHTML = `
    <div class="kv"><span>Canvas Geometry</span><span>${d.piet_canvas.width}x${d.piet_canvas.height} codels</span></div>
    <div class="kv"><span>Codel Transitions Executed</span><span>${d.piet_canvas.codel_transitions}</span></div>
    <div class="kv"><span>Canvas SHA-256</span><span>${d.piet_canvas.fingerprint.slice(0,20)}...</span></div>
    <div class="kv"><span>Fortran Physics Core</span><span>${d.fortran_backend}</span></div>
  `;
  const nr = d.neural_router;
  document.getElementById('router-meta').innerHTML = `
    <div class="kv"><span>Architecture</span><span>${nr.architecture}</span></div>
    <div class="kv"><span>Curriculum Accuracy</span><span>${(nr.curriculum_accuracy*100).toFixed(1)}% (${nr.training_steps} steps)</span></div>
    <div class="kv"><span>INTERCAL Shield Select Sig</span><span>0x${d.intercal_shield.intercal_select_signature.toString(16)}</span></div>
    <div class="kv"><span>Weights SHA-256</span><span>${nr.weights_sha256.slice(0,20)}...</span></div>
  `;
  const wd = d.autonomous_watchdog;
  document.getElementById('watchdog-meta').innerHTML = `
    <div class="kv"><span>Fault Intercepted Live</span><span>${wd.fault_intercepted}</span></div>
    <div class="kv"><span>LISP Hot-Patch Generation</span><span>gen ${wd.lisp_generation_before} &rarr; gen ${wd.lisp_generation_after}</span></div>
    <div class="kv"><span>Prolog SLD Invariant Verdict</span><span>${wd.prolog_verdict}</span></div>
    <div class="kv"><span>Healed Actor Reply</span><span>${wd.healed_reply}</span></div>
  `;
  document.getElementById('watchdog-json').textContent = JSON.stringify(wd, null, 2);

  const tr = d.triad_coordinator;
  const cc = d.apl_context_compression || {};
  document.getElementById('triad-meta').innerHTML = `
    <div class="kv"><span>Occam CSP Consensus</span><span>${tr.occam_verdict.agreed} (val=${tr.occam_verdict.value})</span></div>
    <div class="kv"><span>Forth Register Stack Top</span><span>${tr.forth_stack_top}</span></div>
    <div class="kv"><span>Smalltalk Actor Reply</span><span>${tr.smalltalk_reply}</span></div>
    <div class="kv"><span>APL Matrix Compression Ratio</span><span>${(cc.matrix_compression_ratio||0).toFixed(2)}x</span></div>
  `;
  document.getElementById('triad-json').textContent = JSON.stringify({triad: tr, apl_compression: cc}, null, 2);
}

async function runRoute() {
  const q = document.getElementById('prompt-input').value;
  const res = await fetch('/api/route?q=' + encodeURIComponent(q));
  const d = await res.json();
  document.getElementById('route-out').textContent = JSON.stringify(d, null, 2);
}

refreshState();
</script>
</body>
</html>)HTML";
}

std::string url_decode(std::string_view in) {
    std::string out;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '+' ) {
            out.push_back(' ');
        } else if (in[i] == '%' && i + 2 < in.size()) {
            char hex[3] = {in[i + 1], in[i + 2], 0};
            out.push_back(static_cast<char>(std::strtol(hex, nullptr, 16)));
            i += 2;
        } else {
            out.push_back(in[i]);
        }
    }
    return out;
}

int serve_dashboard_http(int port) {
#if defined(__unix__) || defined(__APPLE__)
    const int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) {
        fail(Status::io_error("dashboard: socket() failed"));
        return 1;
    }
    int opt = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY); // 0.0.0.0
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(srv);
        fail(Status::io_error("dashboard: bind() failed on port " + std::to_string(port)));
        return 1;
    }
    if (::listen(srv, 16) < 0) {
        ::close(srv);
        fail(Status::io_error("dashboard: listen() failed"));
        return 1;
    }

    std::cout << "GENISUS Multi-Dimensional Canvas Dashboard listening on http://0.0.0.0:"
              << port << "\n" << std::flush;

    router::Router live_router;
    while (true) {
        sockaddr_in cli{};
        socklen_t cli_len = sizeof(cli);
        const int fd = ::accept(srv, reinterpret_cast<sockaddr*>(&cli), &cli_len);
        if (fd < 0) continue;

        char buf[4096] = {0};
        const ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
        std::string req(buf, n > 0 ? size_t(n) : 0);

        std::string path = "/";
        const size_t sp1 = req.find(' ');
        if (sp1 != std::string::npos) {
            const size_t sp2 = req.find(' ', sp1 + 1);
            if (sp2 != std::string::npos) path = req.substr(sp1 + 1, sp2 - sp1 - 1);
        }

        std::string content_type = "application/json; charset=utf-8";
        std::string body;
        if (path.rfind("/api/state", 0) == 0) {
            body = build_dashboard_state_json().str();
        } else if (path.rfind("/api/route", 0) == 0) {
            std::string q = "what is the heat equation?";
            const size_t qpos = path.find("q=");
            if (qpos != std::string::npos) q = url_decode(path.substr(qpos + 2));
            const auto route = live_router.classify(q);
            body = route.to_json().str();
        } else {
            content_type = "text/html; charset=utf-8";
            body = dashboard_html();
        }

        std::ostringstream resp;
        resp << "HTTP/1.1 200 OK\r\n"
             << "Content-Type: " << content_type << "\r\n"
             << "Access-Control-Allow-Origin: *\r\n"
             << "Content-Length: " << body.size() << "\r\n"
             << "Connection: close\r\n\r\n"
             << body;
        const std::string out = resp.str();
        (void)::write(fd, out.data(), out.size());
        ::close(fd);
    }
    ::close(srv);
    return 0;
#else
    (void)port;
    fail(Status::hardware_skip("UNSUPPORTED_HARDWARE_SKIP: POSIX sockets unavailable on this host"));
    return 0;
#endif
}

int cmd_dashboard(const std::vector<std::string>& args, bool json) {
    const auto flags = parse_flags(args);
    if (flags.count("port") || flags.count("serve")) {
        const int port = flags.count("port") ? std::stoi(flags.at("port")) : 8080;
        return serve_dashboard_http(port);
    }
    const Json state = build_dashboard_state_json();
    if (json) {
        print_json(state);
        return 0;
    }

    router::Router r;
    const auto tel = r.telemetry();
    const piet::Raster canvas = build_dashboard_canvas(36, 12, 0.35, r);
    const piet::CodelTrace trace = piet::execute_canvas_pathway(canvas, 128);

    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║  GENISUS // MULTI-DIMENSIONAL PHYSICAL-UNIVERSE CANVAS DASHBOARD         ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n"
              << "  [Fortran 2023 Core]  " << numerics::KernelLibrary::instance().fortran_backend_id() << "\n"
              << "  [Neural Router]      64->32->10 GELU | steps=" << tel.training_steps
              << " | accuracy=" << (tel.curriculum_accuracy * 100.0) << "%\n"
              << "  [Piet Canvas Logic]  " << canvas.width << "x" << canvas.height
              << " codels | transitions=" << trace.codel_transitions
              << " | sha256=" << canvas.fingerprint().substr(0, 16) << "...\n"
              << "  ┌────────────────────────────────────┐\n"
              << render_ansi_canvas(canvas)
              << "  └────────────────────────────────────┘\n"
              << "  Tip: run `genisus dashboard --port 8080` to launch the live HTTP canvas UI.\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = args_from(argc, argv, 1);
    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        std::cout
            << "genisus (" << kEngineName << " " << kVersionString
            << ") -- deterministic physical-universe hybrid AI engine\n\n"
            << "usage: genisus <command> [options]\n\n"
            << "  dashboard [--port 8080|--json] multi-dimensional terminal & HTTP canvas dashboard\n"
            << "  universe [--width W --height H --frames N --out F.png|.y4m]\n"
            << "                              Fortran 2023 -> APL -> Prolog -> Piet visual engine\n"
            << "  heal                        run Smalltalk + LISP + Prolog autonomous self-healing\n"
            << "  triad [--forth P --seed S]  coordinate Occam CSP + Forth registers + Smalltalk\n"
            << "  version                     build + host + backend report (JSON)\n"
            << "  doctor                      capability report for this machine\n"
            << "  modules [--json]            capability inventory of every module\n"
            << "  selftest [--json]           run every module's self-check\n"
            << "  kernels list|run|validate   Fortran 2023 numerical kernel library\n"
            << "  apl \"expr\"                  evaluate an APL expression\n"
            << "  prolog --file kb.pl --query \"goal\" [--complete p/2]\n"
            << "  lisp \"expr\"                 evaluate a LISP expression\n"
            << "  forth \"program\"             run a Forth stack + register VM program\n"
            << "  tokenizer --model v.gguf [--text 'x' | --fixture v.gguf]\n"
            << "  llm [--model m.gguf [--prompt 'p' [--max-tokens N]]]\n"
            << "  guard encode|decode|fuzz --key K --in F [--out F] [--trials N]\n"
            << "  piet --in state.txt --out state.png [--width W --height H]\n"
            << "  ask \"<request>\"             classify via micro-neural router and execute\n"
            << "  supervise [--json]          run health checks, invariants & autonomous watchdog\n";
        return 0;
    }
    const bool json = std::find(args.begin(), args.end(), std::string("--json")) != args.end();
    const std::string cmd = args[0];
    if (cmd == "dashboard") return cmd_dashboard(args_from(argc, argv, 2), json);
    if (cmd == "universe")  return cmd_universe(args_from(argc, argv, 2));
    if (cmd == "heal" || cmd == "watchdog") return cmd_heal();
    if (cmd == "triad")     return cmd_triad(args_from(argc, argv, 2));
    if (cmd == "version")   return cmd_version();
    if (cmd == "doctor")    return cmd_doctor();
    if (cmd == "modules")   return cmd_modules(json);
    if (cmd == "selftest")  return cmd_selftest(json);
    if (cmd == "kernels")   return cmd_kernels(args_from(argc, argv, 2), json);
    if (cmd == "apl")       return cmd_apl(args_from(argc, argv, 2));
    if (cmd == "prolog")    return cmd_prolog(args_from(argc, argv, 2));
    if (cmd == "lisp")      return cmd_lisp(args_from(argc, argv, 2));
    if (cmd == "forth")     return cmd_forth(args_from(argc, argv, 2));
    if (cmd == "tokenizer") return cmd_tokenizer(args_from(argc, argv, 2));
    if (cmd == "llm")       return cmd_llm(args_from(argc, argv, 2));
    if (cmd == "guard")     return cmd_guard(args_from(argc, argv, 2));
    if (cmd == "piet")      return cmd_piet(args_from(argc, argv, 2));
    if (cmd == "supervise") return cmd_supervise(json);
    if (cmd == "ask" || cmd == "route") return cmd_ask(args_from(argc, argv, 2));
    fail(Status::invalid("unknown command: " + cmd + " (try --help)"));
    return g_exit;
}
