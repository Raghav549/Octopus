// Octopus Hybrid AI Engine -- Forth-style stack VM ("direct layer").
//
// Role in the engine: a small, fully deterministic stack machine used for
// byte-level post-processing and as the reference target when the engine emits
// numeric programs. Honest scope: this is a *host* VM. It has no raw register
// access, no privilege escalation and no MMIO -- such things require a platform
// driver (see docs/LIMITATIONS.md). Its memory is a bounds-checked array and
// every fault is a Status, never a crash.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <variant>

namespace oct { class Module; }

namespace oct::stackvm {

using Op = std::variant<double, std::string>;   // literal or word name

struct Instruction {
    enum class Kind : uint8_t { Push, Op, Jmp, Jz, Call, Ret, Load, Store, Print };
    Kind        kind = Kind::Push;
    double      value = 0.0;
    std::string name;
    int64_t     address = -1;
};

struct Word {
    std::string name;
    std::vector<Instruction> code;
    bool        primitive = false;
};

class Vm {
public:
    explicit Vm(size_t memory_cells = 4096);

    // Compile Forth-ish text: `: sq dup * ;  5 sq .`
    Status compile(std::string_view source);
    // Run the top-level program (definitions are not executed).
    Outcome<int64_t> run(int64_t step_limit = 1000000);

    const std::vector<Word>& dictionary() const { return words_; }
    const std::vector<double>& data_stack() const { return stack_; }
    std::string output() const { return output_; }
    std::vector<double>& memory() { return memory_; }
    void reset();

    // Direct call of a defined word (used by tests and the CLI).
    Outcome<double> call(std::string_view word, std::vector<double> args = {});

private:
    Status execute(const std::vector<Instruction>& code, int64_t& steps, int64_t limit, int depth);
    Status resolve_control_flow();
    const Word* find(std::string_view name) const;

    std::vector<double>        stack_;
    std::vector<double>        return_stack_;
    std::vector<double>        memory_;
    std::vector<Word>          words_;
    std::vector<Instruction>   main_;
    std::string                output_;
    size_t                     stack_limit_ = 4096;
};

std::shared_ptr<oct::Module> make_stackvm_module();

}  // namespace oct::stackvm
