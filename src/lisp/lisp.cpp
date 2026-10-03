// Octopus Hybrid AI Engine -- LISP-style layer (reader, evaluator, live patches).
// SPDX-License-Identifier: MIT
#include "octopus/lisp.hpp"

#include "octopus/module.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>

namespace oct::lisp {

ValuePtr Value::nil() { return std::make_shared<Value>(); }
ValuePtr Value::boolean_(bool b) {
    auto v = std::make_shared<Value>();
    v->kind = Kind::Bool;
    v->boolean = b;
    return v;
}
ValuePtr Value::number(double x) {
    auto v = std::make_shared<Value>();
    v->kind = Kind::Num;
    v->num = x;
    return v;
}
ValuePtr Value::string(std::string s) {
    auto v = std::make_shared<Value>();
    v->kind = Kind::Str;
    v->text = std::move(s);
    return v;
}
ValuePtr Value::symbol(std::string s) {
    auto v = std::make_shared<Value>();
    v->kind = Kind::Symbol;
    v->text = std::move(s);
    return v;
}
ValuePtr Value::list(std::vector<ValuePtr> l) {
    auto v = std::make_shared<Value>();
    v->kind = Kind::List;
    v->items = std::move(l);
    return v;
}

std::string Value::to_string() const {
    switch (kind) {
        case Kind::Nil: return "nil";
        case Kind::Bool: return boolean ? "#t" : "#f";
        case Kind::Num: {
            char b[40];
            if (num == std::floor(num) && std::fabs(num) < 1e15) std::snprintf(b, sizeof(b), "%.0f", num);
            else std::snprintf(b, sizeof(b), "%.10g", num);
            return b;
        }
        case Kind::Str: return "\"" + text + "\"";
        case Kind::Symbol: return text;
        case Kind::Builtin: return "<builtin:" + text + ">";
        case Kind::Closure: return "<lambda>";
        case Kind::List: {
            std::string out = "(";
            for (size_t i = 0; i < items.size(); ++i) {
                if (i) out += " ";
                out += items[i]->to_string();
            }
            return out + ")";
        }
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------
namespace {

class Reader {
public:
    explicit Reader(std::string_view s) : s_(s) {}

    // Number of bytes consumed by the last parse() call (used to iterate over
    // several top-level forms).
    size_t position() const { return i_; }

    ValuePtr parse() {
        skip();
        if (i_ >= s_.size()) throw OctError("lisp: unexpected end of input");
        const char c = s_[i_];
        if (c == '(') {
            ++i_;
            std::vector<ValuePtr> items;
            for (;;) {
                skip();
                if (i_ >= s_.size()) throw OctError("lisp: unterminated list");
                if (s_[i_] == ')') { ++i_; break; }
                items.push_back(parse());
            }
            return Value::list(std::move(items));
        }
        if (c == ')') throw OctError("lisp: unexpected ')'");
        if (c == '\'') {
            ++i_;
            return Value::list({Value::symbol("quote"), parse()});
        }
        if (c == '"') {
            ++i_;
            std::string out;
            while (i_ < s_.size() && s_[i_] != '"') {
                if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
                    ++i_;
                    switch (s_[i_]) {
                        case 'n': out += '\n'; break;
                        case 't': out += '\t'; break;
                        default: out += s_[i_]; break;
                    }
                } else {
                    out += s_[i_];
                }
                ++i_;
            }
            if (i_ >= s_.size()) throw OctError("lisp: unterminated string");
            ++i_;
            return Value::string(out);
        }
        // atom: number or symbol
        std::string token;
        while (i_ < s_.size() && !std::isspace(static_cast<unsigned char>(s_[i_])) &&
               s_[i_] != '(' && s_[i_] != ')')
            token += s_[i_++];
        if (token.empty()) throw OctError("lisp: empty token");
        bool numeric = true;
        size_t k = 0;
        if (token[0] == '-' || token[0] == '+') k = 1;
        if (k >= token.size()) numeric = false;
        for (size_t j = k; j < token.size() && numeric; ++j)
            if (!std::isdigit(static_cast<unsigned char>(token[j])) && token[j] != '.' &&
                token[j] != 'e' && token[j] != 'E' && token[j] != '-' && token[j] != '+')
                numeric = false;
        if (numeric && token != "-" && token != "+") {
            try {
                return Value::number(std::stod(token));
            } catch (...) {
                return Value::symbol(token);
            }
        }
        return Value::symbol(token);
    }
    bool at_end() { skip(); return i_ >= s_.size(); }

private:
    void skip() {
        while (i_ < s_.size()) {
            if (std::isspace(static_cast<unsigned char>(s_[i_]))) { ++i_; continue; }
            if (s_[i_] == ';') {   // comment to end of line
                while (i_ < s_.size() && s_[i_] != '\n') ++i_;
                continue;
            }
            break;
        }
    }
    std::string_view s_;
    size_t i_ = 0;
};

bool is_symbol(const ValuePtr& v, const char* name) {
    return v && v->kind == Value::Kind::Symbol && v->text == name;
}

double as_num(const ValuePtr& v) {
    if (!v || v->kind != Value::Kind::Num) throw OctError("lisp: expected a number, got " + v->to_string());
    return v->num;
}

}  // namespace

Outcome<ValuePtr> read(std::string_view source) {
    try {
        Reader r(source);
        return r.parse();
    } catch (const std::exception& e) {
        return Status::invalid(e.what());
    }
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------
namespace {

ValuePtr make_builtin(const std::string& name) {
    auto v = std::make_shared<Value>();
    v->kind = Value::Kind::Builtin;
    v->text = name;
    return v;
}

const char* const kBuiltinNames[] = {
    "+", "-", "*", "/", "mod", "abs", "sqrt", "min", "max", "=", "<", ">", "<=", ">=",
    "equal?", "eq?", "not", "list", "car", "cdr", "cons", "append", "length", "null?",
    "number?", "symbol?", "list?", "map", "apply", "error"
};

}  // namespace

Interp::Interp() {
    global_ = std::make_shared<Frame>();
    global_->vars["nil"] = Value::nil();
    global_->vars["#t"] = Value::boolean_(true);
    global_->vars["#f"] = Value::boolean_(false);
    global_->vars["pi"] = Value::number(3.14159265358979323846);
    global_->vars["e"] = Value::number(2.71828182845904523536);
    for (const char* n : kBuiltinNames) global_->vars[n] = make_builtin(n);
    snapshots_.push_back({});   // generation 0: no user definitions
}

ValuePtr Interp::lookup(const std::string& name) const {
    return global_->find(name);
}

Outcome<ValuePtr> Interp::eval_string(std::string_view src) {
    // A source string may hold several top-level forms (a script); each is
    // evaluated in order and the value of the last one is returned. A single
    // form behaves exactly as before. Errors stop evaluation and propagate.
    ValuePtr last;
    std::string_view rest = src;
    bool evaluated_any = false;
    for (;;) {
        // Skip whitespace between forms.
        size_t i = 0;
        while (i < rest.size() && std::isspace(static_cast<unsigned char>(rest[i]))) ++i;
        rest.remove_prefix(i);
        if (rest.empty()) break;
        Reader reader(rest);
        ValuePtr form;
        try {
            form = reader.parse();
        } catch (const std::exception& e) {
            return Status::invalid(e.what());
        }
        auto value = eval(form);
        if (!value) return value.status;
        last = *value;
        evaluated_any = true;
        const size_t consumed = reader.position();
        if (consumed == 0) break;                 // defensive: no progress
        rest.remove_prefix(std::min(consumed, rest.size()));
    }
    if (!evaluated_any) return Status::invalid("lisp: no forms in input");
    return last;
}

Outcome<ValuePtr> Interp::eval(ValuePtr expr) { return eval_in(std::move(expr), global_, 0); }

Outcome<ValuePtr> Interp::apply(ValuePtr fn, std::vector<ValuePtr> args, int depth) {
    if (!fn) return Status::invalid("lisp: applying nil");
    if (fn->kind == Value::Kind::Builtin) {
        const std::string& name = fn->text;
        try {
            if (name == "+" || name == "*" || name == "-" || name == "/") {
                if (args.empty()) return Value::number(name == "+" || name == "*" ? 0.0 : 0.0);
                double acc = as_num(args[0]);
                for (size_t i = 1; i < args.size(); ++i) {
                    const double b = as_num(args[i]);
                    if (name == "+") acc += b;
                    else if (name == "*") acc *= b;
                    else if (name == "-") acc -= b;
                    else {
                        if (b == 0.0) return Status::invalid("lisp: division by zero");
                        acc /= b;
                    }
                }
                return Value::number(acc);
            }
            if (name == "mod") {
                if (args.size() != 2) return Status::invalid("lisp: mod arity");
                return Value::number(std::fmod(as_num(args[0]), as_num(args[1])));
            }
            if (name == "abs") return Value::number(std::fabs(as_num(args.at(0))));
            if (name == "sqrt") {
                const double a = as_num(args.at(0));
                if (a < 0.0) return Status::invalid("lisp: sqrt of negative");
                return Value::number(std::sqrt(a));
            }
            if (name == "min" || name == "max") {
                double acc = as_num(args.at(0));
                for (size_t i = 1; i < args.size(); ++i)
                    acc = (name == "min") ? std::min(acc, as_num(args[i])) : std::max(acc, as_num(args[i]));
                return Value::number(acc);
            }
            if (name == "=" || name == "<" || name == ">" || name == "<=" || name == ">=") {
                if (args.size() < 2) return Value::boolean_(true);
                for (size_t i = 1; i < args.size(); ++i) {
                    const double a = as_num(args[i - 1]), b = as_num(args[i]);
                    const bool ok = (name == "=") ? a == b : (name == "<") ? a < b
                                   : (name == ">") ? a > b : (name == "<=") ? a <= b : a >= b;
                    if (!ok) return Value::boolean_(false);
                }
                return Value::boolean_(true);
            }
            if (name == "equal?" || name == "eq?") {
                return Value::boolean_(args.size() == 2 && args[0]->to_string() == args[1]->to_string());
            }
            if (name == "not") return Value::boolean_(args.size() == 1 && !args[0]->truthy());
            if (name == "list") return Value::list(args);
            if (name == "car") {
                if (args.at(0)->kind != Value::Kind::List || args[0]->items.empty())
                    return Status::invalid("lisp: car of a non-list");
                return args[0]->items.front();
            }
            if (name == "cdr") {
                if (args.at(0)->kind != Value::Kind::List || args[0]->items.empty())
                    return Status::invalid("lisp: cdr of a non-list");
                return Value::list(std::vector<ValuePtr>(args[0]->items.begin() + 1, args[0]->items.end()));
            }
            if (name == "cons") {
                std::vector<ValuePtr> items{args.at(0)};
                if (args.size() > 1) {
                    if (args[1]->kind == Value::Kind::List)
                        items.insert(items.end(), args[1]->items.begin(), args[1]->items.end());
                    else
                        items.push_back(args[1]);
                }
                return Value::list(std::move(items));
            }
            if (name == "append") {
                std::vector<ValuePtr> items;
                for (const auto& a : args)
                    if (a->kind == Value::Kind::List)
                        items.insert(items.end(), a->items.begin(), a->items.end());
                return Value::list(std::move(items));
            }
            if (name == "length") {
                if (args.at(0)->kind == Value::Kind::List) return Value::number(double(args[0]->items.size()));
                if (args.at(0)->kind == Value::Kind::Str) return Value::number(double(args[0]->text.size()));
                return Status::invalid("lisp: length of a non-sequence");
            }
            if (name == "null?") return Value::boolean_(args.at(0)->kind == Value::Kind::Nil ||
                                                       (args[0]->kind == Value::Kind::List && args[0]->items.empty()));
            if (name == "number?") return Value::boolean_(args.at(0)->kind == Value::Kind::Num);
            if (name == "symbol?") return Value::boolean_(args.at(0)->kind == Value::Kind::Symbol);
            if (name == "list?") return Value::boolean_(args.at(0)->kind == Value::Kind::List);
            if (name == "map") {
                if (args.size() != 2) return Status::invalid("lisp: map arity");
                if (args[1]->kind != Value::Kind::List) return Status::invalid("lisp: map over a non-list");
                std::vector<ValuePtr> out;
                for (const auto& item : args[1]->items) {
                    auto r = apply(args[0], {item}, depth + 1);
                    if (!r) return r.status;
                    out.push_back(*r);
                }
                return Value::list(std::move(out));
            }
            if (name == "apply") {
                if (args.size() != 2 || args[1]->kind != Value::Kind::List)
                    return Status::invalid("lisp: apply expects (fn list)");
                return apply(args[0], args[1]->items, depth + 1);
            }
            if (name == "error") return Status::invalid("lisp: user error: " + args.at(0)->to_string());
            return Status::invalid("lisp: unknown builtin " + name);
        } catch (const std::exception& e) {
            return Status::invalid(e.what());
        }
    }
    if (fn->kind != Value::Kind::Closure) return Status::invalid("lisp: not applicable: " + fn->to_string());
    if (args.size() != fn->params.size())
        return Status::invalid("lisp: arity mismatch calling closure: expected " +
                               std::to_string(fn->params.size()) + ", got " + std::to_string(args.size()));
    auto frame = std::make_shared<Frame>();
    frame->parent = global_;
    for (size_t i = 0; i < args.size(); ++i) frame->vars[fn->params[i]] = args[i];
    ValuePtr result = Value::nil();
    for (const auto& form : fn->items) {
        auto r = eval_in(form, frame, depth + 1);
        if (!r) return r.status;
        result = *r;
    }
    return result;
}

Outcome<ValuePtr> Interp::eval_in(ValuePtr expr, std::shared_ptr<Frame> frame, int depth) {
    if (depth > max_depth_) return Status::invalid("lisp: recursion depth limit exceeded");
    if (!expr) return Value::nil();
    try {
        if (expr->kind == Value::Kind::Symbol) {
            ValuePtr v = frame->find(expr->text);
            if (!v) return Status::invalid("lisp: unbound symbol '" + expr->text + "'");
            return v;
        }
        if (expr->kind != Value::Kind::List) return expr;   // self-evaluating
        if (expr->items.empty()) return expr;               // () evaluates to itself

        const ValuePtr& head = expr->items[0];
        // --- special forms -------------------------------------------------
        if (is_symbol(head, "quote")) {
            if (expr->items.size() != 2) return Status::invalid("lisp: quote arity");
            return expr->items[1];
        }
        if (is_symbol(head, "if")) {
            if (expr->items.size() < 3 || expr->items.size() > 4) return Status::invalid("lisp: if arity");
            auto c = eval_in(expr->items[1], frame, depth + 1);
            if (!c) return c.status;
            if ((*c)->truthy()) return eval_in(expr->items[2], frame, depth + 1);
            return expr->items.size() == 4 ? eval_in(expr->items[3], frame, depth + 1)
                                           : Outcome<ValuePtr>(Value::nil());
        }
        if (is_symbol(head, "cond")) {
            for (size_t i = 1; i < expr->items.size(); ++i) {
                const ValuePtr& clause = expr->items[i];
                if (clause->kind != Value::Kind::List || clause->items.empty())
                    return Status::invalid("lisp: malformed cond clause");
                auto c = eval_in(clause->items[0], frame, depth + 1);
                if (!c) return c.status;
                if ((*c)->truthy()) {
                    ValuePtr last = Value::nil();
                    for (size_t k = 1; k < clause->items.size(); ++k) {
                        auto r = eval_in(clause->items[k], frame, depth + 1);
                        if (!r) return r.status;
                        last = *r;
                    }
                    return last;
                }
            }
            return Value::nil();
        }
        if (is_symbol(head, "lambda")) {
            if (expr->items.size() < 3) return Status::invalid("lisp: lambda arity");
            const ValuePtr& params = expr->items[1];
            if (params->kind != Value::Kind::List) return Status::invalid("lisp: lambda params must be a list");
            auto fn = std::make_shared<Value>();
            fn->kind = Value::Kind::Closure;
            for (const auto& p : params->items) {
                if (p->kind != Value::Kind::Symbol) return Status::invalid("lisp: lambda parameter must be a symbol");
                fn->params.push_back(p->text);
            }
            fn->items.assign(expr->items.begin() + 2, expr->items.end());
            return fn;
        }
        if (is_symbol(head, "define") || is_symbol(head, "defun")) {
            if (expr->items.size() < 3) return Status::invalid("lisp: define arity");
            if (expr->items[1]->kind == Value::Kind::Symbol) {   // (define name value)
                auto v = eval_in(expr->items[2], frame, depth + 1);
                if (!v) return v.status;
                if (!frame->vars.count(expr->items[1]->text) && frame != global_)
                    return Status::invalid("lisp: define must target a global name");
                frame->vars[expr->items[1]->text] = *v;
                return *v;
            }
            // (define (name params...) body...)
            const ValuePtr& sig = expr->items[1];
            if (sig->kind != Value::Kind::List || sig->items.empty() ||
                sig->items[0]->kind != Value::Kind::Symbol)
                return Status::invalid("lisp: malformed function definition");
            auto fn = std::make_shared<Value>();
            fn->kind = Value::Kind::Closure;
            for (size_t i = 1; i < sig->items.size(); ++i) {
                if (sig->items[i]->kind != Value::Kind::Symbol)
                    return Status::invalid("lisp: parameter must be a symbol");
                fn->params.push_back(sig->items[i]->text);
            }
            fn->items.assign(expr->items.begin() + 2, expr->items.end());
            global_->vars[sig->items[0]->text] = fn;
            return fn;
        }
        if (is_symbol(head, "set!")) {
            if (expr->items.size() != 3 || expr->items[1]->kind != Value::Kind::Symbol)
                return Status::invalid("lisp: set! arity");
            auto v = eval_in(expr->items[2], frame, depth + 1);
            if (!v) return v.status;
            for (Frame* f = frame.get(); f; f = f->parent.get()) {
                auto it = f->vars.find(expr->items[1]->text);
                if (it != f->vars.end()) { it->second = *v; return *v; }
            }
            return Status::invalid("lisp: set! of an unbound symbol");
        }
        if (is_symbol(head, "let")) {
            if (expr->items.size() < 3 || expr->items[1]->kind != Value::Kind::List)
                return Status::invalid("lisp: let arity");
            auto inner = std::make_shared<Frame>();
            inner->parent = frame;
            for (const auto& binding : expr->items[1]->items) {
                if (binding->kind != Value::Kind::List || binding->items.size() != 2 ||
                    binding->items[0]->kind != Value::Kind::Symbol)
                    return Status::invalid("lisp: malformed let binding");
                auto v = eval_in(binding->items[1], frame, depth + 1);
                if (!v) return v.status;
                inner->vars[binding->items[0]->text] = *v;
            }
            ValuePtr last = Value::nil();
            for (size_t i = 2; i < expr->items.size(); ++i) {
                auto r = eval_in(expr->items[i], inner, depth + 1);
                if (!r) return r.status;
                last = *r;
            }
            return last;
        }
        if (is_symbol(head, "begin") || is_symbol(head, "progn")) {
            ValuePtr last = Value::nil();
            for (size_t i = 1; i < expr->items.size(); ++i) {
                auto r = eval_in(expr->items[i], frame, depth + 1);
                if (!r) return r.status;
                last = *r;
            }
            return last;
        }
        if (is_symbol(head, "and")) {
            ValuePtr last = Value::boolean_(true);
            for (size_t i = 1; i < expr->items.size(); ++i) {
                auto r = eval_in(expr->items[i], frame, depth + 1);
                if (!r) return r.status;
                last = *r;
                if (!last->truthy()) return last;
            }
            return last;
        }
        if (is_symbol(head, "or")) {
            for (size_t i = 1; i < expr->items.size(); ++i) {
                auto r = eval_in(expr->items[i], frame, depth + 1);
                if (!r) return r.status;
                if ((*r)->truthy()) return *r;
            }
            return Value::boolean_(false);
        }

        // --- application ---------------------------------------------------
        auto fn = eval_in(head, frame, depth + 1);
        if (!fn) return fn.status;
        std::vector<ValuePtr> args;
        args.reserve(expr->items.size() - 1);
        for (size_t i = 1; i < expr->items.size(); ++i) {
            auto a = eval_in(expr->items[i], frame, depth + 1);
            if (!a) return a.status;
            args.push_back(*a);
        }
        return apply(*fn, std::move(args), depth);
    } catch (const std::exception& e) {
        return Status::invalid(std::string("lisp: ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// Live patching
// ---------------------------------------------------------------------------
Outcome<int64_t> Interp::patch(const std::string& name, std::string_view source) {
    // Generation 0 means "before any patch": capture the live definitions the
    // first time a patch is installed, so rollback(0) restores them exactly.
    if (generation_ == 0 && snapshots_.size() <= 1) {
        std::map<std::string, ValuePtr> snap;
        for (const auto& kv : global_->vars)
            if (kv.second && kv.second->kind == Value::Kind::Closure) snap[kv.first] = kv.second;
        snapshots_.assign(1, std::move(snap));
    }
    auto form = read(source);
    if (!form) return form.status;
    auto r = eval_in(*form, global_, 0);
    if (!r) return r.status;
    if (!global_->vars.count(name)) return Status::invalid("lisp: patch did not define '" + name + "'");
    ++generation_;
    Patch p;
    p.generation = generation_;
    p.name = name;
    p.source = std::string(source);
    log_.push_back(p);
    // Snapshot every user definition so rollback is exact.
    std::map<std::string, ValuePtr> snap;
    for (const auto& kv : global_->vars)
        if (kv.second && kv.second->kind == Value::Kind::Closure) snap[kv.first] = kv.second;
    snapshots_.push_back(std::move(snap));
    return generation_;
}

Status Interp::rollback(int64_t target_generation) {
    if (target_generation < 0 || target_generation > generation_)
        return Status::invalid("lisp: rollback target out of range");
    // Restore the exact definition snapshot taken at that generation, then drop
    // the newer history (documented: rollback discards, it does not re-apply).
    std::map<std::string, ValuePtr> restored = snapshots_[size_t(target_generation)];
    for (auto it = global_->vars.begin(); it != global_->vars.end();) {
        if (it->second && it->second->kind == Value::Kind::Closure) it = global_->vars.erase(it);
        else ++it;
    }
    for (const auto& kv : restored) global_->vars[kv.first] = kv.second;
    log_.resize(size_t(target_generation));
    snapshots_.resize(size_t(target_generation) + 1);
    Patch undo;
    undo.generation = target_generation;
    undo.name = "*rollback*";
    undo.source = "rollback to generation " + std::to_string(target_generation);
    undo.rollback = true;
    if (!log_.empty() && log_.back().name == "*rollback*") log_.pop_back();
    log_.push_back(undo);
    generation_ = target_generation;
    return Status::ok();
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class LispModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "lisp.selfmod";
        i.version = "1.0.0";
        i.language = "Scheme/LISP-style (subset)";
        i.role = "recursive self-improvement via reversible live patches";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"lisp.eval", "lisp.patch", "lisp.rollback", "lisp.audit"};
        i.limitations = {
            "no macros/hygiene, no continuations, no tail-call optimisation",
            "numbers are f64; no bignums or exact rationals",
            "patching replaces LISP-level definitions only -- it cannot rewrite machine code",
            "rollback replays the patch log; it restores definitions, not arbitrary side effects",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }
    std::string describe() const override {
        return "Interpreter whose function definitions can be replaced at runtime and "
               "reverted from an ordered patch log (supervised recovery only).";
    }
    Status self_check() override {
        Interp interp;
        auto fact = interp.eval_string(
            "(define (fact n) (if (< n 2) 1 (* n (fact (- n 1)))))");
        if (!fact) return Status::internal(fact.status.message);
        auto f10 = interp.eval_string("(fact 10)");
        if (!f10 || !(*f10) || (*f10)->kind != Value::Kind::Num || (*f10)->num != 3628800.0)
            return Status::internal("lisp: (fact 10) != 3628800");
        auto mapped = interp.eval_string("(map (lambda (x) (* x x)) (list 1 2 3 4))");
        if (!mapped || (*mapped)->to_string() != "(1 4 9 16)")
            return Status::internal("lisp: map/square failed");
        // live patch, then prove the old behaviour is gone and comes back
        auto gen = interp.patch("fact", "(define (fact n) 0)");
        if (!gen) return Status::internal(gen.status.message);
        auto patched = interp.eval_string("(fact 10)");
        if (!patched || (*patched)->num != 0.0) return Status::internal("lisp: patch had no effect");
        Status rb = interp.rollback(0);
        if (!rb) return Status::internal("lisp: rollback failed");
        auto restored = interp.eval_string("(fact 10)");
        if (!restored || (*restored)->num != 3628800.0)
            return Status::internal("lisp: rollback did not restore the definition");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_lisp_module() { return std::make_shared<LispModule>(); }

}  // namespace oct::lisp
