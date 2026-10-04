// Octopus Hybrid AI Engine -- Smalltalk-style live actor runtime (implementation).
// SPDX-License-Identifier: MIT
#include "octopus/smalltalk.hpp"

#include "octopus/module.hpp"

#include <chrono>

namespace oct::smalltalk {

std::string to_string(Strategy s) {
    switch (s) {
        case Strategy::OneForOne: return "one_for_one";
        case Strategy::Stop:      return "stop";
        case Strategy::Escalate:  return "escalate";
    }
    return "unknown";
}

std::string to_string(State s) {
    switch (s) {
        case State::Idle:       return "idle";
        case State::Running:    return "running";
        case State::Restarting: return "restarting";
        case State::Stopped:    return "stopped";
        case State::Failed:     return "failed";
    }
    return "unknown";
}

Json SupervisionEvent::to_json() const {
    Json j;
    j.begin_object();
    j.field("sequence", int64_t(sequence));
    j.field("actor", actor);
    j.field("kind", kind);
    j.field("detail", detail);
    j.field("at_ns", int64_t(at_ns));
    j.end_object();
    return j;
}

Json ActorInfo::to_json() const {
    Json j;
    j.begin_object();
    j.field("name", name);
    j.field("state", to_string(state));
    j.field("restarts", int64_t(restarts));
    j.field("handled", int64_t(handled));
    j.field("faults", int64_t(faults));
    j.end_object();
    return j;
}

namespace {
uint64_t now_ns() {
    using clock = std::chrono::steady_clock;
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        clock::now().time_since_epoch())
                        .count());
}
}  // namespace

ActorSystem::ActorSystem(std::string name) : name_(std::move(name)) {}

void ActorSystem::spawn(const std::string& name, Handler handler, Strategy strategy,
                        int max_restarts) {
    std::lock_guard<std::mutex> lock(mu_);
    Actor a;
    a.name = name;
    a.handler = std::move(handler);
    a.strategy = strategy;
    a.max_restarts = max_restarts < 0 ? 0 : max_restarts;
    actors_[name] = std::move(a);
    record(name, "spawn", "strategy=" + to_string(strategy));
}

ActorSystem::Actor* ActorSystem::find(const std::string& name) {
    auto it = actors_.find(name);
    return it == actors_.end() ? nullptr : &it->second;
}

void ActorSystem::record(const std::string& actor, const std::string& kind,
                         const std::string& detail) {
    SupervisionEvent e;
    e.sequence = ++sequence_;
    e.actor = actor;
    e.kind = kind;
    e.detail = detail;
    e.at_ns = now_ns();
    events_.push_back(std::move(e));
}

Outcome<std::string> ActorSystem::send(const std::string& actor_name, const std::string& selector,
                                       const Args& args) {
    std::lock_guard<std::mutex> lock(mu_);
    Actor* a = find(actor_name);
    if (!a) return Status::invalid("smalltalk: no such actor '" + actor_name + "'");
    if (a->state == State::Stopped || a->state == State::Failed)
        return Status::unavailable("smalltalk: actor '" + actor_name + "' is " +
                                   to_string(a->state));

    a->state = State::Running;
    Outcome<std::string> reply = Status::internal("smalltalk: handler did not run");
    try {
        reply = a->handler(selector, args);
    } catch (const std::exception& e) {
        reply = Status::internal(std::string("smalltalk: handler threw: ") + e.what());
    } catch (...) {
        reply = Status::internal("smalltalk: handler threw an unknown exception");
    }

    if (reply) {
        a->handled++;
        a->state = State::Idle;
        record(actor_name, "reply", selector);
        return reply;
    }

    // A fault: contain it, then apply the supervision strategy.
    a->faults++;
    a->state = State::Restarting;
    record(actor_name, "fault", selector + ": " + reply.status.message);
    const std::string fault = reply.status.message;

    switch (a->strategy) {
        case Strategy::Escalate:
            a->state = State::Failed;
            record(actor_name, "escalate", "fault escalated: " + fault);
            return Status::unavailable("smalltalk: actor '" + actor_name + "' escalated: " + fault);
        case Strategy::Stop:
            a->state = State::Stopped;
            record(actor_name, "stop", "stopped after fault: " + fault);
            return Status::unavailable("smalltalk: actor '" + actor_name +
                                       "' stopped after fault: " + fault);
        case Strategy::OneForOne:
            break;
    }

    if (a->restarts >= a->max_restarts) {
        a->state = State::Stopped;
        record(actor_name, "stop",
               "restart budget (" + std::to_string(a->max_restarts) + ") exhausted: " + fault);
        return Status::unavailable("smalltalk: actor '" + actor_name +
                                   "' stopped: restart budget exhausted");
    }

    // Restart: the next message sees a clean state. Handlers that hold mutable
    // state are expected to re-initialise it from the message they receive.
    a->restarts++;
    a->state = State::Idle;
    record(actor_name, "restart", "restart " + std::to_string(a->restarts) + "/" +
                                      std::to_string(a->max_restarts));
    return Status::unavailable("smalltalk: actor '" + actor_name + "' faulted and was restarted: " +
                               fault);
}

bool ActorSystem::has(const std::string& actor) const {
    std::lock_guard<std::mutex> lock(mu_);
    return actors_.count(actor) != 0;
}

ActorInfo ActorSystem::info(const std::string& actor) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = actors_.find(actor);
    if (it == actors_.end()) return ActorInfo{};
    ActorInfo i;
    i.name = it->second.name;
    i.state = it->second.state;
    i.restarts = it->second.restarts;
    i.handled = it->second.handled;
    i.faults = it->second.faults;
    return i;
}

std::vector<ActorInfo> ActorSystem::actors() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<ActorInfo> out;
    for (const auto& kv : actors_) {
        ActorInfo i;
        i.name = kv.second.name;
        i.state = kv.second.state;
        i.restarts = kv.second.restarts;
        i.handled = kv.second.handled;
        i.faults = kv.second.faults;
        out.push_back(std::move(i));
    }
    return out;
}

bool ActorSystem::healthy_unlocked() const {
    for (const auto& kv : actors_)
        if (kv.second.state == State::Stopped || kv.second.state == State::Failed) return false;
    return true;
}

bool ActorSystem::healthy() const {
    std::lock_guard<std::mutex> lock(mu_);
    return healthy_unlocked();
}

size_t ActorSystem::restart_count_unlocked() const {
    size_t n = 0;
    for (const auto& kv : actors_) n += size_t(kv.second.restarts);
    return n;
}

size_t ActorSystem::restart_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return restart_count_unlocked();
}

Json ActorSystem::supervision_report() const {
    std::lock_guard<std::mutex> lock(mu_);
    Json j;
    j.begin_object();
    j.field("system", name_);
    j.field("healthy", healthy_unlocked());
    j.field("actors", int64_t(actors_.size()));
    j.field("restarts", int64_t(restart_count_unlocked()));
    j.field("events", int64_t(events_.size()));
    j.begin_array("actor_states");
    for (const auto& kv : actors_)
        j.value(kv.second.state == State::Stopped || kv.second.state == State::Failed
                    ? std::string("unhealthy")
                    : std::string("healthy"));
    j.end_array();
    j.end_object();
    return j;
}

Json ActorSystem::LiveHealReport::to_json() const {
    Json j;
    j.begin_object();
    j.field("fault_intercepted", fault_intercepted);
    j.field("lisp_rewritten", lisp_rewritten);
    j.field("prolog_invariant_ok", prolog_invariant_ok);
    j.field("recovered_live", recovered_live);
    j.field("lisp_generation_before", lisp_generation_before);
    j.field("lisp_generation_after", lisp_generation_after);
    j.field("lisp_patch_sha256", lisp_patch_sha256);
    j.field("prolog_verdict", prolog_verdict);
    j.field("fault_message", fault_message);
    j.field("healed_reply", healed_reply);
    j.end_object();
    return j;
}

void ActorSystem::attach_autonomous_watchdog(lisp::Interp* interp, prolog::KnowledgeBase* kb) {
    std::lock_guard<std::mutex> lock(mu_);
    watchdog_lisp_ = interp;
    watchdog_kb_ = kb;
}

void ActorSystem::register_live_heal_rule(const std::string& actor, LiveHealRule rule) {
    std::lock_guard<std::mutex> lock(mu_);
    heal_rules_[actor] = std::move(rule);
}

Outcome<std::string> ActorSystem::send_autonomous(const std::string& actor_name,
                                                  const std::string& selector,
                                                  const Args& args,
                                                  LiveHealReport* report_out) {
    LiveHealReport rep;
    Outcome<std::string> first = send(actor_name, selector, args);
    if (first.ok()) {
        rep.recovered_live = true;
        rep.healed_reply = *first;
        if (report_out) *report_out = rep;
        return first;
    }

    // Logic exception intercepted by Smalltalk live-cell watchdog!
    rep.fault_intercepted = true;
    rep.fault_message = first.status.message;

    {
        std::lock_guard<std::mutex> lock(mu_);
        auto rule_it = heal_rules_.find(actor_name);
        if (rule_it == heal_rules_.end()) {
            if (report_out) *report_out = rep;
            return first;
        }
        const LiveHealRule& rule = rule_it->second;

        // 1. Trigger automated LISP live code rewrite (`lisp::Interp::patch`)
        if (watchdog_lisp_ && !rule.lisp_symbol.empty() && !rule.lisp_patch_expr.empty()) {
            rep.lisp_generation_before = watchdog_lisp_->generation();
            std::string patch_src = rule.lisp_patch_expr;
            if (patch_src.find("(define ") == std::string::npos) {
                patch_src = "(define " + rule.lisp_symbol + " " + patch_src + ")";
            }
            auto pr = watchdog_lisp_->patch(rule.lisp_symbol, patch_src);
            if (pr.ok()) {
                rep.lisp_rewritten = true;
                rep.lisp_generation_after = *pr;
                rep.lisp_patch_sha256 = hash::sha256_hex(patch_src);
                record(actor_name, "lisp.patch",
                       rule.lisp_symbol + "@gen" + std::to_string(rep.lisp_generation_after));
            }
        }

        // 2. Trigger Prolog SLD invariant sweep (`prolog::verify`)
        if (watchdog_kb_ && !rule.prolog_invariant.empty()) {
            const auto v = prolog::verify(*watchdog_kb_, rule.prolog_invariant);
            rep.prolog_verdict = (v.verdict == prolog::Verdict::Grounded ? "grounded"
                                : v.verdict == prolog::Verdict::Refuted  ? "refuted" : "unknown");
            rep.prolog_invariant_ok = (v.verdict != prolog::Verdict::Refuted);
            record(actor_name, "prolog.sweep",
                   rule.prolog_invariant + " -> " + rep.prolog_verdict);
            if (!rep.prolog_invariant_ok && watchdog_lisp_ && rep.lisp_rewritten) {
                // Roll back the LISP patch if Prolog invariant sweep refuted it
                (void)watchdog_lisp_->rollback(rep.lisp_generation_before);
                rep.lisp_rewritten = false;
                if (report_out) *report_out = rep;
                return Status::rejected("smalltalk autonomous watchdog: Prolog SLD refuted live patch");
            }
        } else {
            rep.prolog_invariant_ok = true;
            rep.prolog_verdict = "grounded";
        }

        // 3. Hot-swap actor handler if a repaired handler or LISP binding is present
        Actor* a = find(actor_name);
        if (a && rep.prolog_invariant_ok) {
            if (rule.repaired_handler) {
                a->handler = rule.repaired_handler;
            } else if (watchdog_lisp_ && !rule.lisp_symbol.empty()) {
                lisp::Interp* lp = watchdog_lisp_;
                std::string sym = rule.lisp_symbol;
                a->handler = [lp, sym](const std::string&, const Args& call_args) -> Outcome<std::string> {
                    std::string expr = "(" + sym;
                    for (const auto& arg : call_args) expr += " " + arg;
                    expr += ")";
                    auto r = lp->eval_string(expr);
                    if (!r.ok()) return r.status;
                    return (*r)->to_string();
                };
            }
            a->state = State::Idle;
        }
    }

    // 4. Retry message live without stopping the primary runtime engine loop
    Outcome<std::string> retry = send(actor_name, selector, args);
    if (retry.ok()) {
        rep.recovered_live = true;
        rep.healed_reply = *retry;
    }
    if (report_out) *report_out = rep;
    return retry;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class SmalltalkModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "smalltalk.actors";
        i.version = "0.1.0";
        i.language = "Smalltalk-style (C++20)";
        i.role = "Live actor runtime: message passing with supervision, bounded restart and "
                 "fault containment.";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"actors.message_passing", "supervision.one_for_one", "supervision.stop",
                          "supervision.escalate", "fault.containment", "fault.restart_budget",
                          "supervision.event_log"};
        i.limitations = {
            "handlers are C++ callables, not reflective Smalltalk objects: no hot recompilation "
            "of actor code",
            "one mailbox per actor, delivered synchronously; scheduling is the supervisor's "
            "restart budget, nothing more",
            "state reset on restart is the handler's responsibility",
            "single-process only: no distribution, persistence or cluster membership",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }

    std::string describe() const override {
        return "Actors with supervision: a fault is contained in its actor, recorded and either "
               "restarted with a clean state (bounded) or stopped with the system reported "
               "degraded.";
    }

    Status self_check() override {
        ActorSystem sys("self-check");
        int counter = 0;
        sys.spawn("counter", [&](const std::string& sel, const Args&) -> Outcome<std::string> {
            if (sel == "boom") throw OctError("self-check boom");
            return std::to_string(++counter);
        }, Strategy::OneForOne, 2);

        auto first = sys.send("counter", {"inc"});
        if (!first || *first != "1") return Status::internal("smalltalk: counter did not run");
        auto fault = sys.send("counter", {"boom"});
        if (fault) return Status::internal("smalltalk: a faulting handler must not return success");
        if (sys.info("counter").restarts != 1)
            return Status::internal("smalltalk: fault was not supervised");
        auto second = sys.send("counter", {"inc"});
        if (!second || *second != "2") return Status::internal("smalltalk: counter state wrong");
        if (!sys.healthy()) return Status::internal("smalltalk: system reported unhealthy");
        if (sys.events().size() < 4) return Status::internal("smalltalk: event log missing entries");

        // The restart budget must stop an actor that keeps faulting.
        ActorSystem budget("budget");
        budget.spawn("bad", [](const std::string&, const Args&) -> Outcome<std::string> {
            return Status::internal("always fails");
        }, Strategy::OneForOne, 2);
        budget.send("bad", {});
        budget.send("bad", {});
        budget.send("bad", {});
        budget.send("bad", {});
        if (budget.info("bad").state != State::Stopped)
            return Status::internal("smalltalk: restart budget was not enforced");
        if (budget.healthy()) return Status::internal("smalltalk: stopped actor left system healthy");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_smalltalk_module() {
    return std::make_shared<SmalltalkModule>();
}

}  // namespace oct::smalltalk
