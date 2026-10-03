// Octopus Hybrid AI Engine -- module contract and capability registry.
//
// Every "language DNA" subsystem is a Module with an explicit, versioned
// contract. Modules communicate only through this interface + the engine's
// typed request bus; they cannot patch each other's machine code.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <map>
#include <set>
#include <shared_mutex>

namespace oct {

// Capability strings are dotted and stable: "numeric.kernel.sod1d".
using Capability = std::string;

enum class Trust : uint8_t {
    Core,        // shipped with the engine, compiled in
    Verified,    // external, manifest hash checked at load
    Untrusted,   // experimental; fail-closed, no side effects
};

const char* trust_name(Trust t) noexcept;

struct HealthReport {
    bool        healthy = true;
    std::string detail;
};

struct ModuleInfo {
    std::string name;         // "fortran.kernels"
    std::string version;      // semantic-ish: "1.2.0"
    std::string language;     // "Fortran 2018/2023", "C++20", "Prolog-style", ...
    std::string role;         // one-line human description
    Trust       trust = Trust::Core;
    bool        compiled_in = true;
    std::vector<Capability> capabilities;
    std::vector<std::string> limitations;   // honest, machine-readable caveats
    std::string build_id;                   // compiler/flag fingerprint
};

class Module {
public:
    virtual ~Module() = default;
    virtual ModuleInfo           info() const = 0;
    virtual HealthReport         health() const { return {true, "ok"}; }
    virtual std::string          describe() const { return info().role; }
    // Self-check executed by `octopus selftest`; must actually exercise code.
    virtual Status               self_check() = 0;
};

// ---------------------------------------------------------------------------
// Registry: thread-safe routing table, exact and prefix capability lookup.
// ---------------------------------------------------------------------------
class Registry {
public:
    void add(std::shared_ptr<Module> m);

    // Exact capability lookup ("numeric.kernel.heat2d").
    std::shared_ptr<Module> resolve(const Capability& cap) const;
    // Prefix lookup ("numeric.kernel") -> all matching modules, name-ordered.
    std::vector<std::shared_ptr<Module>> resolve_prefix(const std::string& prefix) const;
    // Semantic dispatch helper: try exact, then progressively shorter prefixes.
    std::shared_ptr<Module> dispatch(const Capability& cap, Capability* matched = nullptr) const;

    std::vector<ModuleInfo>    inventory() const;
    std::vector<std::string>   all_capabilities() const;
    bool                       empty() const;
    size_t                     size() const;

    // Every module's self_check(), reported per module.
    struct SelfCheckResult { std::string module; Status status; };
    std::vector<SelfCheckResult> self_check_all() const;

private:
    mutable std::shared_mutex                       mu_;
    std::vector<std::shared_ptr<Module>>            modules_;   // registration order
    std::map<Capability, std::shared_ptr<Module>>   exact_;
};

// Helper: a module that owns a fixed capability set.
class SimpleModule : public Module {
public:
    explicit SimpleModule(ModuleInfo mi) : mi_(std::move(mi)) {}
    ModuleInfo info() const override { return mi_; }
protected:
    ModuleInfo mi_;
};

std::string build_fingerprint();   // compiler + flags + version, for manifests

}  // namespace oct
