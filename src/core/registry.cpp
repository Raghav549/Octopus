// Octopus Hybrid AI Engine -- module registry.
// SPDX-License-Identifier: MIT
#include "octopus/module.hpp"

#include <algorithm>
#include <sstream>

namespace oct {

const char* trust_name(Trust t) noexcept {
    switch (t) {
        case Trust::Core:      return "core";
        case Trust::Verified:  return "verified";
        case Trust::Untrusted: return "untrusted";
    }
    return "?";
}

void Registry::add(std::shared_ptr<Module> m) {
    if (!m) return;
    ModuleInfo mi = m->info();
    std::unique_lock lock(mu_);
    modules_.push_back(m);
    for (const auto& cap : mi.capabilities) exact_[cap] = m;
}

std::shared_ptr<Module> Registry::resolve(const Capability& cap) const {
    std::shared_lock lock(mu_);
    auto it = exact_.find(cap);
    return it == exact_.end() ? nullptr : it->second;
}

std::vector<std::shared_ptr<Module>> Registry::resolve_prefix(const std::string& prefix) const {
    std::shared_lock lock(mu_);
    std::vector<std::shared_ptr<Module>> out;
    std::set<std::string> seen;
    for (const auto& kv : exact_) {
        if (kv.first == prefix || kv.first.rfind(prefix + ".", 0) == 0) {
            if (seen.insert(kv.first).second) out.push_back(kv.second);
        }
    }
    return out;
}

std::shared_ptr<Module> Registry::dispatch(const Capability& cap, Capability* matched) const {
    // Exact match first, then progressively shorter dotted prefixes so that a
    // caller asking for "numeric.kernel.foo" still reaches a "numeric.kernel"
    // provider -- but never silently crosses a top-level namespace.
    std::string probe = cap;
    for (;;) {
        if (auto m = resolve(probe)) {
            if (matched) *matched = probe;
            return m;
        }
        auto dot = probe.rfind('.');
        if (dot == std::string::npos) return nullptr;
        probe = probe.substr(0, dot);
    }
}

std::vector<ModuleInfo> Registry::inventory() const {
    std::shared_lock lock(mu_);
    std::vector<ModuleInfo> out;
    out.reserve(modules_.size());
    for (const auto& m : modules_) out.push_back(m->info());
    return out;
}

std::vector<std::string> Registry::all_capabilities() const {
    std::shared_lock lock(mu_);
    std::vector<std::string> out;
    out.reserve(exact_.size());
    for (const auto& kv : exact_) out.push_back(kv.first);
    return out;
}

bool Registry::empty() const {
    std::shared_lock lock(mu_);
    return modules_.empty();
}

size_t Registry::size() const {
    std::shared_lock lock(mu_);
    return modules_.size();
}

std::vector<Registry::SelfCheckResult> Registry::self_check_all() const {
    std::vector<std::shared_ptr<Module>> mods;
    {
        std::shared_lock lock(mu_);
        mods = modules_;
    }
    std::vector<SelfCheckResult> out;
    out.reserve(mods.size());
    for (const auto& m : mods) {
        SelfCheckResult r;
        r.module = m->info().name;
        try {
            r.status = m->self_check();
        } catch (const std::exception& e) {
            r.status = Status::internal(std::string("self_check threw: ") + e.what());
        }
        out.push_back(std::move(r));
    }
    return out;
}

std::string build_fingerprint() {
    std::ostringstream os;
    os << kEngineName << "/" << kVersionString << " "
       << host_info().arch << " "
#if defined(__clang__)
       << "clang/" << __clang_major__ << "." << __clang_minor__ << "."
#elif defined(__GNUC__)
       << "gcc/" << __GNUC__ << "." << __GNUC_MINOR__ << "."
#else
       << "cc/unknown "
#endif
       << (host_info().has_omp ? "omp " : "")
       << (host_info().has_tsan ? "tsan " : "")
       << cpu_features_string();
    return os.str();
}

}  // namespace oct
