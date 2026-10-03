// Octopus Hybrid AI Engine -- LISP-style self-modifying layer.
//
// Role in the engine (docs/DESIGN.md): recursive self-improvement and *live
// patching* of engine behaviour. A patch replaces the definition of a named
// LISP function inside the interpreter's global environment; it never rewrites
// machine code and it is always reversible.
//
// Supervised-recovery contract:
//   * Every patch is recorded in an ordered log with a generation number.
//   * rollback(generation) restores the previous definition set exactly.
//   * The interpreter cannot call exit/abort and every error is a Status, so a
//     faulty patch degrades to an error value rather than taking the process
//     down (see the Smalltalk supervisor for the restart policy).
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <map>
#include <memory>

namespace oct { class Module; }

namespace oct::lisp {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
    enum class Kind : uint8_t { Nil, Bool, Num, Str, Symbol, List, Builtin, Closure };
    Kind                 kind = Kind::Nil;
    bool                 boolean = false;
    double               num = 0.0;
    std::string          text;        // string payload or symbol name
    std::vector<ValuePtr> items;      // list elements / closure body
    // Closure: params in `items` is not used; see `params`.
    std::vector<std::string> params;

    static ValuePtr nil();
    static ValuePtr boolean_(bool b);
    static ValuePtr number(double v);
    static ValuePtr string(std::string s);
    static ValuePtr symbol(std::string s);
    static ValuePtr list(std::vector<ValuePtr> items);
    std::string to_string() const;
    bool truthy() const { return !(kind == Kind::Bool && !boolean); }
};

// Reader for a useful Scheme/LISP subset: numbers, strings, symbols, quote,
// lists, dotted pairs are not supported (documented limitation).
Outcome<ValuePtr> read(std::string_view source);

class Interp {
public:
    Interp();

    Outcome<ValuePtr> eval(ValuePtr expr);
    Outcome<ValuePtr> eval_string(std::string_view src);

    // --- live patching (the self-modification surface) ---------------------
    struct Patch {
        int64_t     generation = 0;
        std::string name;
        std::string source;
        bool        rollback = false;   // true for the entry that undid a patch
    };

    // Evaluate `source`, which must define a function, and install it, replacing
    // any previous definition. Returns the new generation.
    Outcome<int64_t> patch(const std::string& name, std::string_view source);
    // Undo every patch with generation > target (0 = original state).
    Status rollback(int64_t target_generation);
    const std::vector<Patch>& patch_log() const { return log_; }
    int64_t generation() const { return generation_; }

    // Look up a global (function or value) without evaluating it.
    ValuePtr lookup(const std::string& name) const;

private:
    struct Frame {
        std::map<std::string, ValuePtr> vars;
        std::shared_ptr<Frame>          parent;
        ValuePtr find(const std::string& n) const {
            for (const Frame* f = this; f; f = f->parent.get()) {
                auto it = f->vars.find(n);
                if (it != f->vars.end()) return it->second;
            }
            return nullptr;
        }
    };

    Outcome<ValuePtr> eval_in(ValuePtr expr, std::shared_ptr<Frame> frame, int depth);
    Outcome<ValuePtr> apply(ValuePtr fn, std::vector<ValuePtr> args, int depth);

    std::shared_ptr<Frame>        global_;
    // Snapshot of the LISP-level (closure) definitions after each generation.
    std::vector<std::map<std::string, ValuePtr>> snapshots_;
    std::map<std::string, ValuePtr> builtins_;
    std::vector<Patch>            log_;
    int64_t                       generation_ = 0;
    int                           max_depth_ = 512;
};

std::shared_ptr<oct::Module> make_lisp_module();

}  // namespace oct::lisp
