// Octopus Hybrid AI Engine -- Prolog-style logic layer (SLD resolution).
// SPDX-License-Identifier: MIT
#include "octopus/prolog.hpp"

#include "octopus/module.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <cstdio>
#include <sstream>

namespace oct::prolog {

TermPtr Term::atom(std::string n) {
    auto t = std::make_shared<Term>();
    t->kind = Kind::Atom;
    t->name = std::move(n);
    return t;
}
TermPtr Term::var(std::string n) {
    auto t = std::make_shared<Term>();
    t->kind = Kind::Var;
    t->name = std::move(n);
    return t;
}
TermPtr Term::number(double v) {
    auto t = std::make_shared<Term>();
    t->kind = Kind::Num;
    t->num = v;
    return t;
}
TermPtr Term::compound(std::string f, std::vector<TermPtr> a) {
    auto t = std::make_shared<Term>();
    t->kind = Kind::Compound;
    t->name = std::move(f);
    t->args = std::move(a);
    return t;
}
TermPtr Term::list(std::vector<TermPtr> items, TermPtr tail) {
    TermPtr out = tail ? std::move(tail) : atom("[]");
    for (auto it = items.rbegin(); it != items.rend(); ++it)
        out = compound(".", {*it, out});
    return out;
}

bool Term::is_ground() const {
    switch (kind) {
        case Kind::Var: return false;
        case Kind::Atom:
        case Kind::Num: return true;
        case Kind::Compound:
            for (const auto& a : args) if (!a->is_ground()) return false;
            return true;
    }
    return true;
}

std::string Term::to_string() const {
    switch (kind) {
        case Kind::Atom: {
            static const char* ok = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_!=<>+-*/\\^~:.";
            bool plain = !name.empty();
            for (char c : name) if (!std::strchr(ok, c)) plain = false;
            return plain ? name : "'" + name + "'";
        }
        case Kind::Var: return name;
        case Kind::Num: {
            char b[40];
            if (num == std::floor(num) && std::fabs(num) < 1e15) std::snprintf(b, sizeof(b), "%lld", (long long)num);
            else std::snprintf(b, sizeof(b), "%.10g", num);
            return b;
        }
        case Kind::Compound: {
            if (name == "." && args.size() == 2) {   // list sugar
                std::string out = "[";
                const Term* cur = this;
                bool first = true;
                while (cur && cur->kind == Kind::Compound && cur->name == "." && cur->args.size() == 2) {
                    if (!first) out += ",";
                    out += cur->args[0]->to_string();
                    first = false;
                    cur = cur->args[1].get();
                }
                if (!(cur && cur->kind == Kind::Atom && cur->name == "[]"))
                    out += "|" + (cur ? cur->to_string() : std::string("_"));
                out += "]";
                return out;
            }
            std::string out = name + "(";
            for (size_t i = 0; i < args.size(); ++i) {
                if (i) out += ", ";
                out += args[i]->to_string();
            }
            return out + ")";
        }
    }
    return "?";
}

std::string Substitution::to_string() const {
    std::string out;
    for (const auto& kv : bindings) {
        if (!out.empty()) out += ", ";
        out += kv.first + " = " + kv.second->to_string();
    }
    return out.empty() ? "true" : out;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
namespace {

class TermParser {
public:
    explicit TermParser(std::string_view src) : s_(src) {}

    void skip_ws() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) {
            if (s_[i_] == '%') {   // comment to end of line
                while (i_ < s_.size() && s_[i_] != '\n') ++i_;
            } else {
                ++i_;
            }
        }
    }
    bool at_end() { skip_ws(); return i_ >= s_.size(); }
    std::string_view rest() { return s_.substr(i_); }
    bool peek(char c) { skip_ws(); return i_ < s_.size() && s_[i_] == c; }
    void expect(char c) {
        skip_ws();
        if (i_ >= s_.size() || s_[i_] != c) throw OctError(std::string("prolog: expected '") + c + "'");
        ++i_;
    }

    TermPtr parse() { return parse_expression(); }

    // Arithmetic expression with the usual precedence, used to the right of
    // 'is' and the comparison operators (Prolog operands are terms).
    TermPtr parse_arith() { return parse_arith_min(0); }

    // A clause body: goals separated by commas, right-associatively nested.
    TermPtr parse_body() {
        std::vector<TermPtr> goals{parse_expression()};
        while (peek(',')) {
            ++i_;
            goals.push_back(parse_expression());
        }
        TermPtr out = goals.back();
        for (size_t k = goals.size() - 1; k-- > 0;) out = Term::compound(",", {goals[k], out});
        return out;
    }

    // operator level:  :-  is handled by the clause parser
    TermPtr parse_expression() {
        TermPtr lhs = parse_arith();
        for (;;) {
            skip_ws();
            if (i_ + 1 < s_.size() && s_[i_] == '=' && s_[i_ + 1] == '=') { i_ += 2; (void)parse_arith(); continue; }
            if (i_ < s_.size() && (s_[i_] == '<' || s_[i_] == '>' )) {
                size_t j = i_;
                std::string op;
                while (j < s_.size() && (s_[j] == '<' || s_[j] == '>' || s_[j] == '=' )) op += s_[j++];
                if (op == "<" || op == ">" || op == "=<" || op == ">=") {
                    i_ = j;
                    TermPtr rhs = parse_arith();
                    TermPtr opatom = Term::compound(op, {lhs, rhs});
                    lhs = opatom;
                    continue;
                }
                if (op == "\\=" || (i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == '=')) { i_ += 2; TermPtr rhs = parse_arith(); lhs = Term::compound("\\=", {lhs, rhs}); continue; }
            }
            if (i_ + 1 < s_.size() && s_[i_] == '=' && s_[i_ + 1] == '\\' && i_ + 2 < s_.size() && s_[i_ + 2] == '=') {
                i_ += 3;
                TermPtr rhs = parse_arith();
                lhs = Term::compound("=\\=", {lhs, rhs});
                continue;
            }
            if (i_ < s_.size() && s_[i_] == '=' && !(i_ + 1 < s_.size() && s_[i_ + 1] == '=')) {
                i_ += 1;
                TermPtr rhs = parse_arith();
                lhs = Term::compound("=", {lhs, rhs});
                continue;
            }
            if (i_ + 1 < s_.size() && s_[i_] == 'i' && s_[i_ + 1] == 's' &&
                (i_ + 2 >= s_.size() || !std::isalnum(static_cast<unsigned char>(s_[i_ + 2])))) {
                i_ += 2;
                TermPtr rhs = parse_arith();
                lhs = Term::compound("is", {lhs, rhs});
                continue;
            }
            break;
        }
        return lhs;
    }

    TermPtr parse_arithmetic_ops() { return parse_arith_min(0); }

    TermPtr parse_arith_min(int min_prec) {
        TermPtr lhs = parse_unary();
        for (;;) {
            skip_ws();
            int prec = 0;
            std::string op;
            size_t len = 0;
            if (i_ + 1 < s_.size() && (s_.compare(i_, 2, "//") == 0)) { op = "//"; prec = 3; len = 2; }
            else if (i_ < s_.size() && (s_[i_] == '*' || s_[i_] == '/')) { op = std::string(1, s_[i_]); prec = 3; len = 1; }
            else if (i_ + 2 < s_.size() && s_.compare(i_, 3, "mod") == 0 &&
                     (i_ + 3 >= s_.size() || !std::isalnum(static_cast<unsigned char>(s_[i_ + 3])))) { op = "mod"; prec = 3; len = 3; }
            else if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) { op = std::string(1, s_[i_]); prec = 2; len = 1; }
            if (prec < min_prec || len == 0) break;
            i_ += len;
            TermPtr rhs = parse_arith_min(prec + 1);
            lhs = Term::compound(op, {lhs, rhs});
        }
        return lhs;
    }

    TermPtr parse_unary() {
        skip_ws();
        if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) {
            const char c = s_[i_++];
            TermPtr a = parse_unary();
            return c == '-' ? Term::compound("-", {a}) : a;
        }
        if (peek('(')) {
            ++i_;
            TermPtr t = parse_arith_min(0);
            expect(')');
            return t;
        }
        return parse_primary();
    }

    TermPtr parse_primary() {
        skip_ws();
        if (i_ >= s_.size()) throw OctError("prolog: unexpected end of input");
        const char c = s_[i_];
        if (c == '[') return parse_list();
        if (c == '(') {
            ++i_;
            TermPtr t = parse_expression();
            expect(')');
            return t;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || (c == '-' && i_ + 1 < s_.size() &&
                                                           std::isdigit(static_cast<unsigned char>(s_[i_ + 1])))) {
            return Term::number(parse_number());
        }
        if (c == '\'') {
            ++i_;
            std::string name;
            while (i_ < s_.size() && s_[i_] != '\'') name += s_[i_++];
            expect('\'');
            if (i_ < s_.size() && s_[i_] == '(') return parse_args(Term::atom(name));
            return Term::atom(name);
        }
        if (std::isupper(static_cast<unsigned char>(c)) || c == '_') {
            std::string name;
            while (i_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[i_])) || s_[i_] == '_'))
                name += s_[i_++];
            return Term::var(name);
        }
        if (std::islower(static_cast<unsigned char>(c))) {
            const std::string name = parse_name();
            if (i_ < s_.size() && s_[i_] == '(') return parse_args(Term::atom(name));
            return Term::atom(name);
        }
        // operator-ish atoms such as ! or = used as functors
        std::string op;
        while (i_ < s_.size() && !std::isspace(static_cast<unsigned char>(s_[i_])) &&
               s_[i_] != '(' && s_[i_] != ')' && s_[i_] != ',' )
            op += s_[i_++];
        if (op.empty()) throw OctError("prolog: cannot parse token");
        return Term::atom(op);
    }

private:
    std::string parse_name() {
        std::string name;
        while (i_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[i_])) || s_[i_] == '_'))
            name += s_[i_++];
        return name;
    }
    int64_t parse_decimal() {
        int64_t v = 0;
        while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_])))
            v = v * 10 + (s_[i_++] - '0');
        return v;
    }
    double parse_number() {
        const size_t start = i_;
        if (i_ < s_.size() && s_[i_] == '-') ++i_;
        const int64_t integer_part = parse_decimal();
        double frac = 0.0, scale = 1.0;
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) {
                scale /= 10.0;
                frac += (s_[i_++] - '0') * scale;
            }
        }
        double v = double(integer_part) + frac;
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            const bool neg = (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) && (s_[i_++] == '-');
            const int64_t e = parse_decimal();
            v *= std::pow(10.0, neg ? -double(e) : double(e));
        }
        if (i_ == start) throw OctError("prolog: malformed number");
        return v;
    }
    TermPtr parse_args(TermPtr functor_atom) {
        expect('(');
        std::vector<TermPtr> args;
        if (!peek(')')) {
            args.push_back(parse_expression());
            while (peek(',')) { ++i_; args.push_back(parse_expression()); }
        }
        expect(')');
        return Term::compound(functor_atom->name, std::move(args));
    }
    TermPtr parse_list() {
        expect('[');
        std::vector<TermPtr> items;
        TermPtr tail;
        if (!peek(']')) {
            items.push_back(parse_expression());
            while (peek(',')) { ++i_; items.push_back(parse_expression()); }
            if (peek('|')) { ++i_; tail = parse_expression(); }
        }
        expect(']');
        return Term::list(std::move(items), std::move(tail));
    }

    std::string_view s_;
    size_t i_ = 0;
};

bool unify(TermPtr a, TermPtr b, Substitution& s) {
    // Resolve bound variables first.
    while (a->kind == Term::Kind::Var) {
        auto it = s.bindings.find(a->name);
        if (it == s.bindings.end()) break;
        a = it->second;
    }
    while (b->kind == Term::Kind::Var) {
        auto it = s.bindings.find(b->name);
        if (it == s.bindings.end()) break;
        b = it->second;
    }
    if (a->kind == Term::Kind::Var) { s.bindings[a->name] = b; return true; }
    if (b->kind == Term::Kind::Var) { s.bindings[b->name] = a; return true; }
    if (a->kind != b->kind) return false;
    if (a->kind == Term::Kind::Num) return a->num == b->num;
    if (a->kind == Term::Kind::Atom) return a->name == b->name;
    if (a->name != b->name || a->arity() != b->arity()) return false;
    for (size_t i = 0; i < a->args.size(); ++i)
        if (!unify(a->args[i], b->args[i], s)) return false;
    return true;
}

TermPtr apply_subst(TermPtr t, const Substitution& s) {
    if (!t) return t;
    if (t->kind == Term::Kind::Var) {
        auto it = s.bindings.find(t->name);
        return it == s.bindings.end() ? t : apply_subst(it->second, s);
    }
    if (t->args.empty()) return t;
    std::vector<TermPtr> args;
    args.reserve(t->args.size());
    for (const auto& a : t->args) args.push_back(apply_subst(a, s));
    return Term::compound(t->name, std::move(args));
}

bool is_builtin(const std::string& indicator) {
    static const char* k[] = {"=/2", "\\=/2", "is/2", "</2", ">/2", "=</2", ">=/2",
                              "=:=/2", "=\\=/2", "true/0", "fail/0", "false/0", "! /0"};
    for (const char* s : k) if (indicator == s) return true;
    return false;
}

Outcome<double> eval_arith(const TermPtr& t, const Substitution& s) {
    TermPtr r = apply_subst(t, s);
    switch (r->kind) {
        case Term::Kind::Num: return r->num;
        case Term::Kind::Atom:
            if (r->name == "pi") return 3.14159265358979323846;
            if (r->name == "e")  return 2.71828182845904523536;
            return Status::invalid("prolog: not an arithmetic atom: " + r->name);
        default: break;
    }
    if (r->kind != Term::Kind::Compound) return Status::invalid("prolog: not an arithmetic term");
    const std::string& f = r->name;
    if (f == "-" && r->arity() == 1) {
        auto v = eval_arith(r->args[0], s);
        if (!v) return v.status;
        return -*v;
    }
    auto a = eval_arith(r->args[0], s);
    if (!a) return a.status;
    auto b = (r->arity() > 1) ? eval_arith(r->args[1], s) : Outcome<double>(0.0);
    if (r->arity() > 1 && !b) return b.status;
    if (f == "+") return *a + *b;
    if (f == "-") return *a - *b;
    if (f == "*") return *a * *b;
    if (f == "/") return (*b == 0.0) ? Outcome<double>(Status::invalid("prolog: division by zero")) : Outcome<double>(*a / *b);
    if (f == "//") return (*b == 0.0) ? Outcome<double>(Status::invalid("prolog: integer division by zero"))
                                      : Outcome<double>(std::floor(*a / *b));
    if (f == "mod" || f == "rem") return (*b == 0.0) ? Outcome<double>(Status::invalid("prolog: mod by zero"))
                                                     : Outcome<double>(std::fmod(*a, *b));
    if (f == "abs") return std::fabs(*a);
    if (f == "min" && r->arity() == 2) return std::min(*a, *b);
    if (f == "max" && r->arity() == 2) return std::max(*a, *b);
    if (f == "sqrt") return std::sqrt(*a);
    if (f == "round") return std::round(*a);
    return Status::invalid("prolog: unknown arithmetic functor " + f + "/" + std::to_string(r->arity()));
}

}  // namespace

namespace {}  // (parser state is intentionally translation-unit private)

Outcome<std::pair<TermPtr, std::vector<TermPtr>>> parse_clause(std::string_view text) {
    try {
        TermParser p(text);
        TermPtr head = p.parse_expression();
        std::vector<TermPtr> body;
        p.skip_ws();
        std::string_view rest = p.rest();
        size_t pos = rest.find(":-");
        if (pos != std::string_view::npos) {
            TermParser bp(rest.substr(pos + 2));
            TermPtr conj = bp.parse_body();
            // split conjunction on ','/2
            std::function<void(const TermPtr&)> collect = [&](const TermPtr& t) {
                if (t->kind == Term::Kind::Compound && t->name == "," && t->arity() == 2) {
                    collect(t->args[0]);
                    collect(t->args[1]);
                } else {
                    body.push_back(t);
                }
            };
            collect(conj);
        }
        return std::make_pair(head, body);
    } catch (const std::exception& e) {
        return Status::invalid(e.what());
    }
}

Outcome<TermPtr> parse_term(std::string_view text) {
    try {
        TermParser p(text);
        TermPtr t = p.parse_expression();
        if (!p.at_end()) return Status::invalid("prolog: trailing input");
        return t;
    } catch (const std::exception& e) {
        return Status::invalid(e.what());
    }
}

// ---------------------------------------------------------------------------
// KnowledgeBase / solver
// ---------------------------------------------------------------------------
void KnowledgeBase::add_clause(TermPtr head, std::vector<TermPtr> body) {
    Clause c;
    c.head = std::move(head);
    c.body = std::move(body);
    c.order = int64_t(all_.size());
    all_.push_back(c);
    index_[c.head->indicator()].push_back(c);
}

Status KnowledgeBase::add_text(std::string_view clause_text) {
    auto parsed = parse_clause(clause_text);
    if (!parsed) return parsed.status;
    add_clause(parsed->first, parsed->second);
    return Status::ok();
}

size_t KnowledgeBase::clause_count() const { return all_.size(); }
size_t KnowledgeBase::predicate_count() const { return index_.size(); }

std::vector<std::string> KnowledgeBase::predicates() const {
    std::vector<std::string> out;
    out.reserve(index_.size());
    for (const auto& kv : index_) out.push_back(kv.first);
    return out;
}

void KnowledgeBase::declare_complete(std::string_view indicator) {
    complete_.insert(std::string(indicator));
}
bool KnowledgeBase::is_complete(std::string_view indicator) const {
    return complete_.count(std::string(indicator)) != 0;
}

std::string KnowledgeBase::fingerprint() const {
    std::string canon;
    for (const auto& c : all_) {
        canon += c.head->to_string();
        if (!c.body.empty()) {
            canon += " :- ";
            for (size_t i = 0; i < c.body.size(); ++i) {
                if (i) canon += ", ";
                canon += c.body[i]->to_string();
            }
        }
        canon += ".\n";
    }
    return hash::sha256_hex(canon);
}

struct Solver {
    const KnowledgeBase& kb;
    Limits limits;
    int64_t inferences = 0;
    bool hit_limit = false;
    std::vector<std::string> trace;

    bool prove(const std::vector<TermPtr>& goals, Substitution& s, int depth,
               std::vector<Solution>* solutions, const Substitution& answer_template) {
        if (hit_limit) return true;
        if (inferences > limits.max_inferences) { hit_limit = true; return true; }
        if (depth > limits.max_depth) { hit_limit = true; return true; }
        if (goals.empty()) {
            if (solutions) {
                solutions->push_back(Solution{true, s, trace, inferences, false});
            }
            return solutions == nullptr || int64_t(solutions->size()) >= limits.max_solutions;
        }
        const TermPtr goal = goals.front();
        std::vector<TermPtr> rest(goals.begin() + 1, goals.end());

        // --- builtins -----------------------------------------------------
        const std::string ind = goal->indicator();
        if (ind == "true/0") return prove(rest, s, depth, solutions, answer_template);
        if (ind == "fail/0" || ind == "false/0") return false;
        if (ind == "=/2") {
            Substitution s2 = s;
            if (!unify(goal->args[0], goal->args[1], s2)) return false;
            ++inferences;
            return prove(rest, s2, depth, solutions, answer_template);
        }
        if (ind == "\\=/2") {
            Substitution s2 = s;
            if (unify(goal->args[0], goal->args[1], s2)) return false;
            ++inferences;
            return prove(rest, s, depth, solutions, answer_template);
        }
        if (ind == "is/2") {
            auto v = eval_arith(goal->args[1], s);
            if (!v) return false;
            Substitution s2 = s;
            if (!unify(goal->args[0], Term::number(*v), s2)) return false;
            ++inferences;
            return prove(rest, s2, depth, solutions, answer_template);
        }
        if (ind == "</2" || ind == ">/2" || ind == "=</2" || ind == ">=/2" ||
            ind == "=:=/2" || ind == "=\\=/2") {
            auto a = eval_arith(goal->args[0], s);
            auto b = eval_arith(goal->args[1], s);
            if (!a || !b) return false;
            bool ok = false;
            if (ind == "</2") ok = *a < *b;
            else if (ind == ">/2") ok = *a > *b;
            else if (ind == "=</2") ok = *a <= *b;
            else if (ind == ">=/2") ok = *a >= *b;
            else if (ind == "=:=/2") ok = *a == *b;
            else ok = *a != *b;
            ++inferences;
            return ok && prove(rest, s, depth, solutions, answer_template);
        }

        // --- user predicates ---------------------------------------------
        auto it = kb.index_.find(ind);
        if (it == kb.index_.end()) {
            ++inferences;
            trace.push_back("no clauses for " + ind);
            return false;   // caller decides Unknown vs Refuted
        }
        for (const auto& clause : it->second) {
            Substitution s2 = s;
            // Rename clause variables per invocation to keep derivations sound.
            static int64_t call_counter = 0;
            const std::string suffix = "_" + std::to_string(++call_counter);
            std::function<TermPtr(const TermPtr&)> rename = [&](const TermPtr& t) -> TermPtr {
                if (t->kind == Term::Kind::Var) {
                    if (t->name == "_") return Term::var("_G" + suffix);
                    return Term::var(t->name + suffix);
                }
                if (t->args.empty()) return t;
                std::vector<TermPtr> args;
                for (const auto& a : t->args) args.push_back(rename(a));
                return Term::compound(t->name, std::move(args));
            };
            TermPtr head = rename(clause.head);
            if (!unify(goal, head, s2)) continue;
            ++inferences;
            std::vector<TermPtr> body;
            body.reserve(clause.body.size() + rest.size());
            for (const auto& g : clause.body) body.push_back(rename(g));
            for (const auto& g : rest) body.push_back(g);
            trace.push_back("resolve " + goal->to_string() + " with " + clause.head->to_string());
            const size_t trace_mark = trace.size();
            if (prove(body, s2, depth + 1, solutions, answer_template)) return true;
            trace.resize(trace_mark);
        }
        return false;
    }
};

Solution KnowledgeBase::solve(const TermPtr& goal, const Limits& limits) const {
    Solver solver{*this, limits, 0, false, {}};
    Substitution s;
    std::vector<Solution> sols;
    bool stop = solver.prove({goal}, s, 0, &sols, {});
    (void)stop;
    if (!sols.empty()) {
        Solution out = sols.front();
        out.inferences = solver.inferences;
        out.hit_limit = solver.hit_limit;
        return out;
    }
    Solution out;
    out.proved = false;
    out.inferences = solver.inferences;
    out.hit_limit = solver.hit_limit;
    return out;
}

std::vector<Solution> KnowledgeBase::solve_all(const TermPtr& goal, const Limits& limits) const {
    Solver solver{*this, limits, 0, false, {}};
    Substitution s;
    std::vector<Solution> sols;
    solver.prove({goal}, s, 0, &sols, {});
    for (auto& sol : sols) sol.hit_limit = solver.hit_limit;
    return sols;
}

// ---------------------------------------------------------------------------
// Verification -- the anti-hallucination entry point
// ---------------------------------------------------------------------------
Json Verification::to_json() const {
    Json j;
    j.begin_object();
    j.field("claim", claim);
    switch (verdict) {
        case Verdict::Grounded: j.field("verdict", "grounded"); break;
        case Verdict::Refuted:  j.field("verdict", "refuted"); break;
        case Verdict::Unknown:  j.field("verdict", "unknown"); break;
    }
    j.field("reason", reason);
    j.field("inferences", inferences);
    j.field("truncated", hit_limit);
    j.key("evidence");
    j.begin_array();
    for (const auto& e : evidence) j.value(e);
    j.end_array();
    j.end_object();
    return j;
}

Verification verify(const KnowledgeBase& kb, const TermPtr& claim, const Limits& limits) {
    Verification v;
    v.claim = claim->to_string();
    // A claim with variables is a query, not an assertion: it can only be
    // grounded under a substitution, and never refuted wholesale.
    if (!claim->is_ground()) {
        Limits l = limits;
        l.max_solutions = 1;
        const Solution s = kb.solve(claim, l);
        v.inferences = s.inferences;
        v.hit_limit = s.hit_limit;
        if (s.proved) {
            v.verdict = Verdict::Grounded;
            v.reason = "query has a solution under " + s.answer.to_string();
            v.evidence = s.proof;
        } else {
            v.verdict = Verdict::Unknown;
            v.reason = s.hit_limit ? "search truncated by limits"
                                   : "no derivation found for a non-ground query";
        }
        return v;
    }
    const Solution s = kb.solve(claim, limits);
    v.inferences = s.inferences;
    v.hit_limit = s.hit_limit;
    if (s.proved) {
        v.verdict = Verdict::Grounded;
        v.reason = "derivation found";
        v.evidence = s.proof;
        return v;
    }
    if (s.hit_limit) {
        v.verdict = Verdict::Unknown;
        v.reason = "search truncated by limits; refusing to claim false";
        return v;
    }
    if (kb.is_complete(claim->indicator())) {
        v.verdict = Verdict::Refuted;
        v.reason = "predicate " + claim->indicator() +
                   " is declared complete and no derivation exists";
        return v;
    }
    v.verdict = Verdict::Unknown;
    v.reason = "open world: predicate " + claim->indicator() +
               " is not declared complete, so failure is not falsity";
    return v;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class PrologModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "prolog.logic";
        i.version = "1.0.0";
        i.language = "Prolog-style (Horn clauses, SLD resolution)";
        i.role = "claim verification and hallucination rejection";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"logic.assert", "logic.query", "logic.verify", "logic.prove"};
        i.limitations = {
            "no cut/1, no assert/retract, no tabling; pure Horn clauses only",
            "unification omits the occurs check",
            "predicates are open-world by default: unprovable != false; declare_complete is required to refute",
            "arithmetic is f64 only; no unbounded integers",
            "search is depth-first with an inference budget; truncation reports Unknown",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }
    std::string describe() const override {
        return "SLD-resolution engine used to ground or refuse claims; it cannot "
               "report a claim as true without an actual derivation.";
    }
    Status self_check() override {
        KnowledgeBase kb;
        if (!kb.add_text("parent(homer, bart).")) return Status::internal("prolog: fact 1");
        if (!kb.add_text("parent(homer, lisa).")) return Status::internal("prolog: fact 2");
        if (!kb.add_text("parent(abe, homer).")) return Status::internal("prolog: fact 3");
        kb.declare_complete("parent/2");
        if (!kb.add_text("ancestor(X, Y) :- parent(X, Y).")) return Status::internal("prolog: rule 1");
        if (!kb.add_text("ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).")) return Status::internal("prolog: rule 2");
        Limits lim;
        lim.max_solutions = 8;
        auto bart = parse_term("ancestor(abe, bart)");
        if (!bart) return Status::internal(bart.status.message);
        auto ground = verify(kb, *bart, lim);
        if (ground.verdict != Verdict::Grounded) return Status::internal("prolog: abe->bart not grounded");
        auto nope = parse_term("parent(abe, bart)");
        if (!nope) return Status::internal(nope.status.message);
        auto refuted = verify(kb, *nope, lim);
        if (refuted.verdict != Verdict::Refuted) return Status::internal("prolog: complete-predicate refutation failed");
        auto unknown_pred = parse_term("wears(bart, hat)");
        if (!unknown_pred) return Status::internal(unknown_pred.status.message);
        auto unknown = verify(kb, *unknown_pred, lim);
        if (unknown.verdict != Verdict::Unknown) return Status::internal("prolog: open world leaked a verdict");
        auto arith = parse_term("X is 2 + 3 * 4");
        if (!arith) return Status::internal(arith.status.message);
        auto sol = kb.solve(*arith, lim);
        // 'is/2' with an unbound left side should bind X to 14
        if (!sol.proved || !sol.answer.bindings.count("X") ||
            sol.answer.bindings["X"]->num != 14.0)
            return Status::internal("prolog: arithmetic evaluation failed");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_prolog_module() { return std::make_shared<PrologModule>(); }

}  // namespace oct::prolog
