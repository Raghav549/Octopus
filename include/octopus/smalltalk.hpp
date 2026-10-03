// Octopus Hybrid AI Engine -- Smalltalk-style live actor runtime.
//
// Role in the engine (ARCHITECTURE_SPEC.md, capability "immortal resilience"):
// message-passing actors with supervision. A handler fault is contained by the
// actor it happened in; the supervisor records it, restarts the actor with a
// clean state (bounded restart budget) and, when the budget is exhausted,
// stops the actor and reports the system as degraded -- it never takes the
// process down and never silently continues as if nothing happened.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace oct { class Module; }

namespace oct::smalltalk {

using Args     = std::vector<std::string>;
// Handler receives the selector and the arguments of one message.
using Handler  = std::function<Outcome<std::string>(const std::string& selector, const Args&)>;

enum class Strategy : uint8_t { OneForOne, Stop, Escalate };
enum class State    : uint8_t { Idle, Running, Restarting, Stopped, Failed };

std::string to_string(Strategy s);
std::string to_string(State s);

struct SupervisionEvent {
    uint64_t    sequence = 0;
    std::string actor;
    std::string kind;        // "spawn" | "fault" | "restart" | "stop" | "escalate" | "reply"
    std::string detail;
    uint64_t    at_ns = 0;
    Json to_json() const;
};

struct ActorInfo {
    std::string name;
    State       state = State::Idle;
    int         restarts = 0;
    uint64_t    handled = 0;
    uint64_t    faults = 0;
    Json to_json() const;
};

class ActorSystem {
public:
    explicit ActorSystem(std::string name = "smalltalk");

    // Spawn an actor. The handler is invoked for every delivered message; it may
    // throw or return an error Status, both are treated as faults and contained.
    void spawn(const std::string& name, Handler handler,
               Strategy strategy = Strategy::OneForOne, int max_restarts = 3);

    // Deliver one message synchronously. A fault never propagates to the caller:
    // it returns Status::unavailable with the fault recorded in the event log.
    Outcome<std::string> send(const std::string& actor, const std::string& selector,
                              const Args& args = {});

    bool   has(const std::string& actor) const;
    ActorInfo info(const std::string& actor) const;
    std::vector<ActorInfo> actors() const;
    const std::vector<SupervisionEvent>& events() const { return events_; }
    const std::string& name() const { return name_; }
    bool   healthy() const;                  // no actor stopped or failed
    size_t restart_count() const;
    Json   supervision_report() const;

private:
    struct Actor {
        std::string  name;
        Handler      handler;
        Strategy     strategy = Strategy::OneForOne;
        int          max_restarts = 3;
        int          restarts = 0;
        uint64_t     handled = 0;
        uint64_t     faults = 0;
        State        state = State::Idle;
    };
    void record(const std::string& actor, const std::string& kind, const std::string& detail);
    Actor* find(const std::string& name);

    std::string                  name_;
    std::map<std::string, Actor> actors_;
    std::vector<SupervisionEvent> events_;
    mutable std::mutex           mu_;
    uint64_t                     sequence_ = 0;
};

std::shared_ptr<oct::Module> make_smalltalk_module();

}  // namespace oct::smalltalk
