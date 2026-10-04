// Octopus Hybrid AI Engine -- Forth-style stack VM.
// SPDX-License-Identifier: MIT
#include "octopus/stackvm.hpp"

#include "octopus/module.hpp"

#include <cmath>
#include <sstream>

namespace oct::stackvm {

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw OctError(msg);
}

std::vector<std::string> tokenize(std::string_view s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        if (std::isspace(static_cast<unsigned char>(s[i]))) { ++i; continue; }
        if (s[i] == '\\') {   // line comment
            while (i < s.size() && s[i] != '\n') ++i;
            continue;
        }
        if (s[i] == '(') {    // ( comment )
            while (i < s.size() && s[i] != ')') ++i;
            if (i < s.size()) ++i;
            continue;
        }
        size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        out.push_back(std::string(s.substr(i, j - i)));
        i = j;
    }
    return out;
}

}  // namespace

Vm::Vm(size_t memory_cells) : memory_(memory_cells, 0.0) {}

const Word* Vm::find(std::string_view name) const {
    for (const auto& w : words_)
        if (w.name == name) return &w;
    return nullptr;
}

void Vm::reset() {
    stack_.clear();
    return_stack_.clear();
    registers_.fill(0.0);
    output_.clear();
    std::fill(memory_.begin(), memory_.end(), 0.0);
    main_.clear();
}

Status Vm::compile(std::string_view source) {
    const std::vector<std::string> toks = tokenize(source);
    std::vector<Instruction>* target = &main_;
    std::string defining;
    std::vector<Instruction> body;

    auto emit = [&](Instruction in) {
        if (!defining.empty()) body.push_back(std::move(in));
        else                    target->push_back(std::move(in));
    };

    try {
        for (size_t i = 0; i < toks.size(); ++i) {
            const std::string& t = toks[i];
            if (t == ":" ) {
                require(defining.empty(), "vm: nested definition");
                require(i + 1 < toks.size(), "vm: missing word name");
                defining = toks[++i];
                body.clear();
                continue;
            }
            if (t == ";") {
                require(!defining.empty(), "vm: ';' outside a definition");
                Word w;
                w.name = defining;
                w.code = body;
                // Replace an existing definition (live redefinition is allowed
                // and auditable: the old body is dropped only after parsing).
                for (auto& existing : words_)
                    if (existing.name == defining) existing = w;
                if (!find(defining)) words_.push_back(w);
                else {
                    for (auto& existing : words_)
                        if (existing.name == defining) existing = w;
                }
                defining.clear();
                body.clear();
                continue;
            }
            if (t == "if") { Instruction in; in.kind = Instruction::Kind::Jz; emit(in); continue; }
            if (t == "else") { Instruction in; in.kind = Instruction::Kind::Jmp; emit(in); continue; }
            if (t == "then") { Instruction in; in.kind = Instruction::Kind::Ret; in.address = -2; emit(in); continue; }

            Instruction in;
            // number?
            bool numeric = !t.empty();
            bool has_digit = false;
            size_t k = (t[0] == '-' || t[0] == '+') ? 1 : 0;
            if (k >= t.size()) numeric = false;
            for (size_t j = 0; j < t.size() && numeric; ++j) {
                if (std::isdigit(static_cast<unsigned char>(t[j]))) has_digit = true;
                else if (j >= k && t[j] != '.' && t[j] != 'e' && t[j] != 'E' && t[j] != '-' && t[j] != '+')
                    numeric = false;
            }
            numeric = numeric && has_digit;
            if (numeric && find(t) == nullptr) {
                in.kind = Instruction::Kind::Push;
                in.value = std::stod(t);
                emit(std::move(in));
                continue;
            }
            in.kind = Instruction::Kind::Op;
            in.name = t;
            emit(std::move(in));
        }
        require(defining.empty(), "vm: unterminated definition");
        Status resolved = resolve_control_flow();
        if (!resolved) return resolved;
        if (!find("run")) {
            Word w;
            w.name = "run";
            w.code = main_;
            words_.push_back(w);
        }
        return Status::ok();
    } catch (const std::exception& e) {
        return Status::invalid(e.what());
    }
}

Status Vm::execute(const std::vector<Instruction>& code, int64_t& steps, int64_t limit, int depth) {
    if (depth > 256) return Status::invalid("vm: call depth limit exceeded");
    try {
        for (size_t pc = 0; pc < code.size(); ++pc) {
            if (++steps > limit) return Status::invalid("vm: step limit exceeded");
            const Instruction& in = code[pc];
            switch (in.kind) {
                case Instruction::Kind::Push:
                    require(stack_.size() < stack_limit_, "vm: stack overflow");
                    stack_.push_back(in.value);
                    break;
                case Instruction::Kind::Print: {
                    require(!stack_.empty(), "vm: print on an empty stack");
                    std::ostringstream os;
                    const double v = stack_.back();
                    stack_.pop_back();
                    if (v == std::floor(v) && std::fabs(v) < 1e15) os << (long long)v;
                    else os << v;
                    output_ += os.str() + " ";
                    break;
                }
                case Instruction::Kind::Jmp:
                    require(in.address >= 0 && size_t(in.address) <= code.size(), "vm: bad jump target");
                    pc = size_t(in.address) - 1;
                    break;
                case Instruction::Kind::Jz: {
                    require(!stack_.empty(), "vm: if on an empty stack");
                    const double v = stack_.back();
                    stack_.pop_back();
                    if (v == 0.0) {
                        require(in.address >= 0 && size_t(in.address) <= code.size(), "vm: bad if target");
                        pc = size_t(in.address) - 1;
                    }
                    break;
                }
                case Instruction::Kind::Load: {
                    require(!stack_.empty(), "vm: load on an empty stack");
                    const double addr = stack_.back();
                    stack_.pop_back();
                    require(addr >= 0 && size_t(addr) < memory_.size(), "vm: memory fault (load)");
                    stack_.push_back(memory_[size_t(addr)]);
                    break;
                }
                case Instruction::Kind::Store: {
                    require(stack_.size() >= 2, "vm: store needs an address and a value");
                    const double addr = stack_.back();
                    stack_.pop_back();
                    const double value = stack_.back();
                    stack_.pop_back();
                    require(addr >= 0 && size_t(addr) < memory_.size(), "vm: memory fault (store)");
                    memory_[size_t(addr)] = value;
                    break;
                }
                case Instruction::Kind::Call:
                case Instruction::Kind::Op: {
                    const std::string& op = in.name;
                    // Low-level stack register & return-stack interactions:
                    // `>r`, `r>`, `r@`, `reg@`, `reg!`, and `r0@`..`r15@` / `r0!`..`r15!`
                    if (op == ">r") {
                        require(!stack_.empty(), "vm: >r on empty stack");
                        require(return_stack_.size() < stack_limit_, "vm: return stack overflow");
                        return_stack_.push_back(stack_.back());
                        stack_.pop_back();
                        break;
                    }
                    if (op == "r>") {
                        require(!return_stack_.empty(), "vm: r> on empty return stack");
                        require(stack_.size() < stack_limit_, "vm: stack overflow");
                        stack_.push_back(return_stack_.back());
                        return_stack_.pop_back();
                        break;
                    }
                    if (op == "r@") {
                        require(!return_stack_.empty(), "vm: r@ on empty return stack");
                        require(stack_.size() < stack_limit_, "vm: stack overflow");
                        stack_.push_back(return_stack_.back());
                        break;
                    }
                    if (op == "reg@") {
                        require(!stack_.empty(), "vm: reg@ on empty stack");
                        const double idx = stack_.back();
                        stack_.pop_back();
                        require(idx >= 0.0 && size_t(idx) < kNumRegisters, "vm: register index out of bounds");
                        stack_.push_back(registers_[size_t(idx)]);
                        break;
                    }
                    if (op == "reg!") {
                        require(stack_.size() >= 2, "vm: reg! needs value and register index");
                        const double idx = stack_.back();
                        stack_.pop_back();
                        const double val = stack_.back();
                        stack_.pop_back();
                        require(idx >= 0.0 && size_t(idx) < kNumRegisters, "vm: register index out of bounds");
                        registers_[size_t(idx)] = val;
                        break;
                    }
                    if (op.size() >= 3 && op[0] == 'r' && (op.back() == '@' || op.back() == '!')) {
                        bool digits = true;
                        for (size_t k = 1; k + 1 < op.size(); ++k)
                            if (!std::isdigit(static_cast<unsigned char>(op[k]))) digits = false;
                        if (digits) {
                            const size_t r_idx = size_t(std::stoul(op.substr(1, op.size() - 2)));
                            require(r_idx < kNumRegisters, "vm: register r" + std::to_string(r_idx) + " out of bounds");
                            if (op.back() == '@') {
                                require(stack_.size() < stack_limit_, "vm: stack overflow");
                                stack_.push_back(registers_[r_idx]);
                            } else {
                                require(!stack_.empty(), "vm: " + op + " on empty stack");
                                registers_[r_idx] = stack_.back();
                                stack_.pop_back();
                            }
                            break;
                        }
                    }
                    // primitives first
                    if (op == "+" || op == "-" || op == "*" || op == "/" || op == "mod" ||
                        op == "dup" || op == "drop" || op == "swap" || op == "over" || op == "rot" ||
                        op == "." || op == "@" || op == "!" || op == "<" || op == ">" || op == "=" ||
                        op == "sin" || op == "sqrt" || op == "negate" || op == "abs" || op == "min" || op == "max") {
                        if (op == "dup" || op == "drop" || op == "swap" || op == "over" || op == "rot") {
                            if (op == "dup") { require(!stack_.empty(), "vm: dup on empty"); stack_.push_back(stack_.back()); }
                            else if (op == "drop") { require(!stack_.empty(), "vm: drop on empty"); stack_.pop_back(); }
                            else if (op == "swap") {
                                require(stack_.size() >= 2, "vm: swap needs 2");
                                std::swap(stack_[stack_.size() - 1], stack_[stack_.size() - 2]);
                            } else if (op == "over") {
                                require(stack_.size() >= 2, "vm: over needs 2");
                                stack_.push_back(stack_[stack_.size() - 2]);
                            } else {
                                require(stack_.size() >= 3, "vm: rot needs 3");
                                const double a = stack_[stack_.size() - 3];
                                stack_[stack_.size() - 3] = stack_[stack_.size() - 2];
                                stack_[stack_.size() - 2] = stack_[stack_.size() - 1];
                                stack_[stack_.size() - 1] = a;
                            }
                            break;
                        }
                        // Memory/print faults must propagate: a nested execute()
                        // status that is dropped would silently swallow the error.
                        if (op == ".") {
                            Instruction p; p.kind = Instruction::Kind::Print;
                            Status s = execute({p}, steps, limit, depth);
                            if (!s) return s;
                            break;
                        }
                        if (op == "@") {
                            Instruction l; l.kind = Instruction::Kind::Load;
                            Status s = execute({l}, steps, limit, depth);
                            if (!s) return s;
                            break;
                        }
                        if (op == "!") {
                            Instruction st; st.kind = Instruction::Kind::Store;
                            Status s = execute({st}, steps, limit, depth);
                            if (!s) return s;
                            break;
                        }
                        if (op == "sin" || op == "sqrt" || op == "negate" || op == "abs") {
                            require(!stack_.empty(), "vm: unary op on an empty stack");
                            const double a = stack_.back();
                            stack_.pop_back();
                            if (op == "sin") stack_.push_back(std::sin(a));
                            else if (op == "sqrt") {
                                require(a >= 0.0, "vm: sqrt of a negative value");
                                stack_.push_back(std::sqrt(a));
                            } else if (op == "negate") stack_.push_back(-a);
                            else stack_.push_back(std::fabs(a));
                            break;
                        }
                        require(stack_.size() >= 2, "vm: binary op needs 2 operands");
                        const double b = stack_.back();
                        stack_.pop_back();
                        const double a = stack_.back();
                        stack_.pop_back();
                        if (op == "+") stack_.push_back(a + b);
                        else if (op == "-") stack_.push_back(a - b);
                        else if (op == "*") stack_.push_back(a * b);
                        else if (op == "/") {
                            require(b != 0.0, "vm: division by zero");
                            stack_.push_back(a / b);
                        } else if (op == "mod") {
                            require(b != 0.0, "vm: mod by zero");
                            stack_.push_back(std::fmod(a, b));
                        } else if (op == "<") stack_.push_back(a < b ? 1.0 : 0.0);
                        else if (op == ">") stack_.push_back(a > b ? 1.0 : 0.0);
                        else if (op == "=") stack_.push_back(a == b ? 1.0 : 0.0);
                        else if (op == "min") stack_.push_back(std::min(a, b));
                        else stack_.push_back(std::max(a, b));
                        break;
                    }
                    const Word* w = find(op);
                    require(w != nullptr, "vm: undefined word '" + op + "'");
                    return_stack_.push_back(double(pc));
                    Status st = execute(w->code, steps, limit, depth + 1);
                    return_stack_.pop_back();
                    if (!st) return st;
                    break;
                }
                case Instruction::Kind::Ret:
                    if (in.address == -2) break;   // 'then' marker: no-op
                    return Status::ok();
            }
        }
        return Status::ok();
    } catch (const std::exception& e) {
        return Status::invalid(e.what());
    }
}

Outcome<double> Vm::call(std::string_view word, std::vector<double> args) {
    const Word* w = find(word);
    if (!w) return Status::invalid("vm: undefined word '" + std::string(word) + "'");
    for (double a : args) stack_.push_back(a);
    int64_t steps = 0;
    Status st = execute(w->code, steps, 1000000, 0);
    if (!st) return st;
    if (stack_.empty()) return Status::invalid("vm: word returned no value");
    const double v = stack_.back();
    stack_.pop_back();
    return v;
}

// Resolve structured control flow. Addresses are absolute instruction indices;
// Jz pops a falsy flag and jumps, Jmp always jumps, 'then' closes the construct.
Status Vm::resolve_control_flow() {
    for (auto& w : words_) {
        std::vector<Instruction> code = w.code;
        std::vector<std::pair<size_t, size_t>> open_ifs;   // (jz index, else index or SIZE_MAX)
        for (size_t i = 0; i < code.size(); ++i) {
            if (code[i].kind == Instruction::Kind::Jz) {
                open_ifs.emplace_back(i, SIZE_MAX);
            } else if (code[i].kind == Instruction::Kind::Jmp) {
                if (open_ifs.empty())
                    return Status::invalid("vm: 'else' without a matching 'if' in " + w.name);
                if (open_ifs.back().second != SIZE_MAX)
                    return Status::invalid("vm: two 'else' branches for one 'if' in " + w.name);
                open_ifs.back().second = i;
            } else if (code[i].kind == Instruction::Kind::Ret && code[i].address == -2) {
                if (open_ifs.empty())
                    return Status::invalid("vm: 'then' without a matching 'if' in " + w.name);
                const std::pair<size_t, size_t> top = open_ifs.back();
                open_ifs.pop_back();
                code[top.first].address = (top.second == SIZE_MAX) ? int64_t(i) : int64_t(top.second) + 1;
                if (top.second != SIZE_MAX) code[top.second].address = int64_t(i) + 1;
            }
        }
        if (!open_ifs.empty()) return Status::invalid("vm: unbalanced if/then in " + w.name);
        w.code = std::move(code);
    }
    return Status::ok();
}

Outcome<int64_t> Vm::run(int64_t step_limit) {
    Word* entry = nullptr;
    for (auto& w : words_) if (w.name == "run") entry = &w;
    if (!entry) return Status::invalid("vm: no top-level program");
    int64_t steps = 0;
    Status st = execute(entry->code, steps, step_limit, 0);
    if (!st) return st;
    return steps;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class StackVmModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "forth.stackvm";
        i.version = "1.0.0";
        i.language = "Forth-style stack VM";
        i.role = "deterministic numeric bytecode back-end and post-processing";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"vm.compile", "vm.run", "vm.call", "vm.memory"};
        i.limitations = {
            "host VM only: no raw registers, no MMIO, no privilege escalation",
            "double-precision cells only; no separate data/code space or threading",
            "structured if/else/then only; do/loop and exceptions are not implemented",
            "execution is single-threaded and bounded by a step limit",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }
    std::string describe() const override {
        return "Forth-style stack machine with a bounds-checked memory array: fast, "
               "deterministic, and safe against its own programs.";
    }
    Status self_check() override {
        Vm vm;
        Status st = vm.compile(": sq dup * ;  : cube dup sq * ;  7 sq 3 cube . .");
        if (!st) return st;
        auto steps = vm.run();
        if (!steps) return Status::internal(steps.status.message);
        auto sq = vm.call("sq", {12.0});
        if (!sq || *sq != 144.0) return Status::internal("vm: sq(12) != 144");
        Vm vm2;
        if (!vm2.compile(": fact dup 1 > if dup 1 - fact * else drop 1 then ; 5 fact")) {}
        auto f = vm2.call("fact", {5.0});
        if (!f || *f != 120.0) return Status::internal("vm: fact(5) != 120");
        // memory fault must be a Status, not a crash
        Vm vm3;
        if (!vm3.compile(": bad 999999 @ ;")) return Status::internal("vm: compile bad word failed");
        auto bad = vm3.call("bad", {});
        if (bad) return Status::internal("vm: out-of-range load did not fault");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_stackvm_module() { return std::make_shared<StackVmModule>(); }

}  // namespace oct::stackvm
