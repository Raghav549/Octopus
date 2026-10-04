// Octopus Hybrid AI Engine -- supervised recovery implementation.
// SPDX-License-Identifier: MIT
#include "octopus/supervisor.hpp"

#include "octopus/module.hpp"

#include <cmath>

namespace oct::supervisor {

const char* severity_name(Severity s) noexcept {
    switch (s) {
        case Severity::Info:     return "info";
        case Severity::Warn:     return "warn";
        case Severity::Critical: return "critical";
    }
    return "?";
}

const char* action_name(Action a) noexcept {
    switch (a) {
        case Action::None:     return "none";
        case Action::Restart:  return "restart";
        case Action::Rollback: return "rollback";
        case Action::Degrade:  return "degrade";
        case Action::FailClosed: return "fail-closed";
    }
    return "?";
}

void Supervisor::add_check(Check c) {
    if (c.name.empty() || !c.run) return;
    checks_.push_back(std::move(c));
}

void Supervisor::add_invariant(const prolog::KnowledgeBase& kb, InvariantCheck inv) {
    if (!inv.goal) return;
    invariants_.push_back(InvariantBinding{&kb, std::move(inv)});
}

std::vector<OutcomeRecord> Supervisor::tick(int max_failures_before_action) {
    std::vector<OutcomeRecord> produced;
    const int64_t threshold = std::max(1, max_failures_before_action);

    auto record = [&](const std::string& name, bool healthy, const std::string& detail,
                      double seconds, bool timed_out, Action action, Severity sev) {
        OutcomeRecord r;
        r.seq = ++seq_;
        r.check = name;
        r.healthy = healthy;
        r.detail = detail;
        r.seconds = seconds;
        r.timed_out = timed_out;
        r.action = action;
        r.when = iso8601_now();
        history_.push_back(r);
        produced.push_back(r);
        audit_.append(healthy ? "check-ok" : "check-fail",
                      name + "|" + detail + "|" + std::string(severity_name(sev)) +
                          "|action=" + action_name(action));
    };

    for (const auto& c : checks_) {
        const Clock::time_point t0 = Clock::now();
        Status st = Status::ok();
        bool timed_out = false;
        try {
            st = c.run();
        } catch (const std::exception& e) {
            st = Status::internal(std::string("check threw: ") + e.what());
        }
        const double seconds = seconds_since(t0);
        if (c.timeout_seconds > 0.0 && seconds > c.timeout_seconds) {
            timed_out = true;
            st = Status::degraded("check exceeded its watchdog deadline (" +
                                  std::to_string(seconds) + "s > " +
                                  std::to_string(c.timeout_seconds) + "s)");
        }
        const bool ok = st.is_ok();
        Action action = Action::None;
        if (ok) {
            failures_[c.name] = 0;
        } else {
            const int failures = ++failures_[c.name];
            if (failures >= threshold) {
                if (c.severity == Severity::Critical) {
                    if (rollback_) {
                        Status rb = rollback_(generation_);
                        action = rb ? Action::Rollback : Action::FailClosed;
                        if (!rb) healthy_ = false;
                    } else {
                        action = Action::FailClosed;
                        healthy_ = false;
                    }
                } else if (restart_) {
                    Status rs = restart_();
                    action = rs ? Action::Restart : Action::FailClosed;
                    if (rs) {
                        ++restarts_;
                        ++generation_;
                    } else {
                        healthy_ = false;
                    }
                } else {
                    action = Action::Degrade;   // no hook: report, stay up, stay honest
                }
            }
        }
        record(c.name, ok, st.message, seconds, timed_out, action, c.severity);
    }

    for (const auto& binding : invariants_) {
        const Clock::time_point t0 = Clock::now();
        auto v = prolog::verify(*binding.kb, binding.inv.goal, binding.inv.limits);
        const double seconds = seconds_since(t0);
        const bool ok = (v.verdict != prolog::Verdict::Refuted);
        Action action = Action::None;
        if (!ok) {
            if (restart_) {
                Status rs = restart_();
                action = rs ? Action::Restart : Action::FailClosed;
                if (rs && binding.inv.severity != Severity::Critical) ++restarts_;
                if (!rs) healthy_ = false;
            } else {
                action = Action::Degrade;
            }
        }
        record(binding.inv.name, ok,
               std::string("verdict=") + (v.verdict == prolog::Verdict::Grounded ? "grounded"
                                     : v.verdict == prolog::Verdict::Refuted  ? "refuted" : "unknown") +
                   " reason=" + v.reason,
               seconds, false, action, binding.inv.severity);
    }

    return produced;
}

Json Supervisor::report() const {
    Json j;
    j.begin_object();
    j.field("healthy", healthy_);
    j.field("checks", int64_t(checks_.size()));
    j.field("invariants", int64_t(invariants_.size()));
    j.field("restarts", restarts_);
    j.field("history_entries", int64_t(history_.size()));
    j.key("recent");
    j.begin_array();
    const size_t start = history_.size() > 8 ? history_.size() - 8 : 0;
    for (size_t i = start; i < history_.size(); ++i) {
        const OutcomeRecord& r = history_[i];
        j.begin_object();          // structured records, not JSON-in-a-string
        j.field("seq", r.seq);
        j.field("check", r.check);
        j.field("healthy", r.healthy);
        if (!r.detail.empty()) j.field("detail", r.detail);
        j.field("seconds", r.seconds);
        j.field("timed_out", r.timed_out);
        j.field("action", action_name(r.action));
        j.end_object();
    }
    j.end_array();
    j.key("audit");
    j.begin_object();
    j.field("entries", int64_t(audit_.size()));
    j.field("valid", audit_.verify());
    j.end_object();
    j.end_object();
    return j;
}

smalltalk::ActorSystem::LiveHealReport Supervisor::heal_actor_exception(
    smalltalk::ActorSystem& actors,
    lisp::Interp& interp,
    prolog::KnowledgeBase& kb,
    const std::string& actor_name,
    const std::string& selector,
    const smalltalk::Args& args,
    const std::string& lisp_symbol,
    const std::string& lisp_patch_expr,
    const std::string& prolog_invariant) {
    actors.attach_autonomous_watchdog(&interp, &kb);
    smalltalk::ActorSystem::LiveHealRule rule;
    rule.lisp_symbol = lisp_symbol;
    rule.lisp_patch_expr = lisp_patch_expr;
    rule.prolog_invariant = prolog_invariant;
    actors.register_live_heal_rule(actor_name, std::move(rule));

    smalltalk::ActorSystem::LiveHealReport rep;
    const Clock::time_point t0 = Clock::now();
    Outcome<std::string> res = actors.send_autonomous(actor_name, selector, args, &rep);
    const double elapsed = seconds_since(t0);

    OutcomeRecord rec;
    rec.seq = ++seq_;
    rec.check = "autonomous-watchdog:" + actor_name;
    rec.healthy = res.ok() && rep.recovered_live;
    rec.detail = "fault=" + rep.fault_message + " lisp_gen=" +
                 std::to_string(rep.lisp_generation_after) +
                 " prolog=" + rep.prolog_verdict + " reply=" + rep.healed_reply;
    rec.seconds = elapsed;
    rec.action = rep.lisp_rewritten ? Action::Restart : Action::None;
    rec.when = iso8601_now();
    history_.push_back(rec);

    audit_.append(rec.healthy ? "watchdog-healed" : "watchdog-failed",
                  actor_name + "|" + rep.lisp_patch_sha256 + "|" + rep.prolog_verdict);
    if (rep.lisp_rewritten) {
        ++restarts_;
        ++generation_;
    }
    return rep;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class SupervisorModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "core.supervisor";
        i.version = "1.0.0";
        i.language = "C++20 + Prolog invariants";
        i.role = "health checks, watchdogs, invariants, restart and rollback policy";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"supervisor.check", "supervisor.watchdog", "supervisor.invariant",
                          "supervisor.restart", "supervisor.rollback", "supervisor.report"};
        i.limitations = {
            "supervised recovery only: no in-place code rewriting, no hot binary patching",
            "checks run on the caller's thread in tick(); scheduling policy belongs to the host",
            "rollback restores registered state via the rollback hook; it does not undo arbitrary side effects",
            "watchdogs measure wall time of the check itself, not system-wide deadlines",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }
    std::string describe() const override {
        return "Runs health checks under watchdogs, verifies Prolog invariants, and applies "
               "restart/rollback policy with a tamper-evident audit chain.";
    }
    Status self_check() override {
        Supervisor sup;
        int failures = 0;
        int restarts = 0;
        sup.add_check(Check{"flaky", [&failures]() -> Status {
                                if (++failures <= 2) return Status::internal("transient fault");
                                return Status::ok();
                            },
                            0.0, Severity::Warn});
        sup.add_check(Check{"stable", []() -> Status { return Status::ok(); }});
        sup.add_check(Check{"slow", []() -> Status {
                                volatile double x = 0.0;
                                for (int i = 0; i < 2000000; ++i) x = x + double(i) * 0.5;
                                return x > 0 ? Status::ok() : Status::invalid("unreachable");
                            },
                            0.000001, Severity::Warn});   // deliberately impossible deadline
        sup.set_restart([&restarts]() -> Status {
            ++restarts;
            return Status::ok();
        });

        prolog::KnowledgeBase kb;
        if (!kb.add_text("limit(max_context, 4096).")) return Status::internal("supervisor: kb 1");
        kb.declare_complete("limit/2");
        auto goal = prolog::parse_term("limit(max_context, 4096)");
        if (!goal) return Status::internal(goal.status.message);
        sup.add_invariant(kb, InvariantCheck{"context-limit", *goal, prolog::Limits{},
                                             Severity::Critical});

        auto first = sup.tick(2);      // 'flaky' fails once -> below threshold, no action
        auto second = sup.tick(2);     // fails twice -> restart
        if (restarts < 1) return Status::internal("supervisor: restart action was not invoked");
        bool saw_timeout = false, saw_invariant = false;
        for (const auto& r : second) {
            if (r.check == "slow" && r.timed_out) saw_timeout = true;
            if (r.check == "context-limit" && r.healthy) saw_invariant = true;
        }
        if (!saw_timeout) return Status::internal("supervisor: watchdog did not fire");
        if (!saw_invariant) return Status::internal("supervisor: invariant was not evaluated");
        if (!sup.report().str().empty() == false) return Status::internal("supervisor: empty report");

        // A violated invariant (refuted claim) must degrade rather than crash.
        prolog::KnowledgeBase kb2;
        if (!kb2.add_text("limit(max_context, 4096).")) return Status::internal("supervisor: kb 2");
        auto bad_goal = prolog::parse_term("limit(max_context, 1)");
        if (!bad_goal) return Status::internal(bad_goal.status.message);
        kb2.declare_complete("limit/2");
        Supervisor sup2;
        sup2.add_invariant(kb2, InvariantCheck{"context-limit", *bad_goal, prolog::Limits{},
                                               Severity::Critical});
        auto rows = sup2.tick(1);
        if (rows.empty() || rows[0].healthy) return Status::internal("supervisor: violated invariant passed");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_supervisor_module() { return std::make_shared<SupervisorModule>(); }

}  // namespace oct::supervisor
