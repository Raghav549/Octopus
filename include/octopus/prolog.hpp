// Octopus Hybrid AI Engine -- Prolog-style logic layer.
//
// Role in the engine (docs/DESIGN.md): the "no guessing" half of the
// correctness model. Numeric kernels and the model router publish *facts*
// (measured residuals, backend identity, hardware limits); claims are only
// reported as grounded when a Prolog-style derivation actually succeeds.
//
// Honest scope:
//   * SLD resolution, left-to-right selection, chronological backtracking,
//     first-solution and bounded all-solutions search. No cuts, no
//     assert/retract, no tabling.
//   * Unification is without the occurs check (documented limitation).
//   * The world is open by default: an unprovable goal is UNKNOWN, never
//     "false". A predicate must be explicitly declared complete before its
//     failure is allowed to mean REFUTED. This is the anti-hallucination rule:
//     the engine may not convert ignorance into a fact.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <map>
#include <memory>
#include <set>

namespace oct { class Module; }

namespace oct::prolog {

struct Term;
using TermPtr = std::shared_ptr<Term>;

struct Term {
    enum class Kind : uint8_t { Atom, Var, Num, Compound };
    Kind                 kind = Kind::Atom;
    std::string          name;      // atom / variable name / functor
    double               num = 0.0;
    std::vector<TermPtr> args;

    static TermPtr atom(std::string n);
    static TermPtr var(std::string n);
    static TermPtr number(double v);
    static TermPtr compound(std::string f, std::vector<TermPtr> a);
    // [h|t] list construction sugar (right-nested '.'/2 compounds).
    static TermPtr list(std::vector<TermPtr> items, TermPtr tail = nullptr);

    int         arity() const { return static_cast<int>(args.size()); }
    bool        is_ground() const;
    std::string to_string() const;
    std::string indicator() const { return name + "/" + std::to_string(arity()); }
};

// Parse a Prolog clause or goal, e.g. "ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y)."
// Returns the parsed term plus whether the clause had a body.
Outcome<std::pair<TermPtr, std::vector<TermPtr>>> parse_clause(std::string_view text);
Outcome<TermPtr> parse_term(std::string_view text);

struct Substitution {
    std::map<std::string, TermPtr> bindings;
    bool empty() const { return bindings.empty(); }
    std::string to_string() const;
};

struct Solution {
    bool         proved = false;
    Substitution answer;
    std::vector<std::string> proof;   // one line per resolution step
    int64_t      inferences = 0;
    bool         hit_limit = false;   // true => the search was truncated
};

struct Limits {
    int64_t max_inferences = 200000;
    int     max_depth = 256;
    int64_t max_solutions = 1;
};

class KnowledgeBase {
public:
    // Clauses are stored in insertion order; resolution is deterministic.
    void add_clause(TermPtr head, std::vector<TermPtr> body = {});
    Status add_text(std::string_view clause_text);
    size_t clause_count() const;
    size_t predicate_count() const;
    std::vector<std::string> predicates() const;

    // Predicates declared complete: failure to prove them means REFUTED.
    void declare_complete(std::string_view indicator);
    bool is_complete(std::string_view indicator) const;

    Solution solve(const TermPtr& goal, const Limits& limits = {}) const;
    std::vector<Solution> solve_all(const TermPtr& goal, const Limits& limits = {}) const;

    std::string fingerprint() const;   // SHA-256 over the canonical clause set

private:
    struct Clause { TermPtr head; std::vector<TermPtr> body; int64_t order = 0; };
    std::map<std::string, std::vector<Clause>> index_;   // key = name/arity
    std::vector<Clause> all_;
    std::set<std::string> complete_;
    friend struct Solver;
};

enum class Verdict : uint8_t { Grounded, Refuted, Unknown };

struct Verification {
    Verdict     verdict = Verdict::Unknown;
    std::string claim;
    std::string reason;
    std::vector<std::string> evidence;      // proof lines for Grounded
    int64_t     inferences = 0;
    bool        hit_limit = false;
    Json        to_json() const;
};

// The single entry point used by the engine: never returns Grounded without a
// derivation, never returns Refuted without a complete predicate.
Verification verify(const KnowledgeBase& kb, const TermPtr& claim, const Limits& limits = {});

std::shared_ptr<oct::Module> make_prolog_module();

}  // namespace oct::prolog
