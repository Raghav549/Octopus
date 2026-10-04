// Octopus Hybrid AI Engine -- supervised recovery.
//
// The architecture spec is explicit about what "self-healing" may mean:
// health checks, watchdogs, invariants, restart, rollback -- never in-place
// machine-code rewriting during inference. This module implements exactly that
// set, deterministically and with an audit trail:
//
//   * checks       : named functions returning Status (healthy / reason)
//   * watchdogs    : a check that exceeds its deadline is treated as failed
//   * invariants   : Prolog goals that must stay provable (a violated invariant
//                    is a policy violation, not a crash)
//   * actions      : restart callback, rollback callback, degrade, or fail closed
//   * history      : every check result and action is appended to an audit chain
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"
#include "octopus/intercal.hpp"
#include "octopus/lisp.hpp"
#include "octopus/prolog.hpp"
#include "octopus/smalltalk.hpp"

#include <functional>

namespace oct { class Module; }

namespace oct::supervisor {

enum class Severity : uint8_t { Info, Warn, Critical };
const char* severity_name(Severity s) noexcept;

enum class Action : uint8_t { None, Restart, Rollback, Degrade, FailClosed };
const char* action_name(Action a) noexcept;

struct Check {
    std::string                  name;
    std::function<Status()>      run;             // Status::ok() == healthy
    double                       timeout_seconds = 0.0;   // 0 = no watchdog
    Severity                     severity = Severity::Warn;
};

struct InvariantCheck {
    std::string        name;
    prolog::TermPtr    goal;      // must stay provable
    prolog::Limits     limits{};
    Severity           severity = Severity::Critical;
};

struct OutcomeRecord {
    int64_t     seq = 0;
    std::string check;
    bool        healthy = false;
    std::string detail;
    double      seconds = 0.0;
    bool        timed_out = false;
    Action      action = Action::None;
    std::string when;
};

class Supervisor {
public:
    void add_check(Check c);
    void add_invariant(const prolog::KnowledgeBase& kb, InvariantCheck inv);
    // Recovery hooks, invoked when a check fails according to the policy.
    void set_restart(std::function<Status()> fn) { restart_ = std::move(fn); }
    void set_rollback(std::function<Status(int64_t)> fn) { rollback_ = std::move(fn); }

    // Run every check once. `max_failures_before_action` allows a transient
    // blip to be tolerated before recovery is triggered.
    std::vector<OutcomeRecord> tick(int max_failures_before_action = 1);

    bool        healthy() const { return healthy_; }
    int64_t     restarts() const { return restarts_; }
    Json        report() const;
    const intercal::AuditChain& audit() const { return audit_; }

    // Autonomous Agent Supervisor: orchestrates the Smalltalk live-cell
    // messaging watchdog with automated LISP code rewrites and Prolog SLD
    // invariant sweeps to heal internal logic exceptions live without stopping
    // the primary runtime engine loop.
    smalltalk::ActorSystem::LiveHealReport heal_actor_exception(
        smalltalk::ActorSystem& actors,
        lisp::Interp& interp,
        prolog::KnowledgeBase& kb,
        const std::string& actor_name,
        const std::string& selector,
        const smalltalk::Args& args,
        const std::string& lisp_symbol,
        const std::string& lisp_patch_expr,
        const std::string& prolog_invariant);

private:
    struct InvariantBinding { const prolog::KnowledgeBase* kb; InvariantCheck inv; };
    std::vector<Check>              checks_;
    std::vector<InvariantBinding>   invariants_;
    std::function<Status()>         restart_;
    std::function<Status(int64_t)>  rollback_;
    std::vector<OutcomeRecord>      history_;
    std::map<std::string, int>      failures_;
    intercal::AuditChain            audit_;
    bool                            healthy_ = true;
    int64_t                         restarts_ = 0;
    int64_t                         seq_ = 0;
    int64_t                         generation_ = 0;
};

std::shared_ptr<oct::Module> make_supervisor_module();

}  // namespace oct::supervisor
