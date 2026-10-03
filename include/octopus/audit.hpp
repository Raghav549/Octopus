// Octopus Hybrid AI Engine -- hash-chained local audit log.
//
// Design note (see docs/SECURITY.md): an append-only hash chain gives *tamper
// evidence*, not tamper immunity. Entries are optionally obfuscated by the
// INTERCAL-style codec at rest (obfuscation is explicitly not cryptography).
// The log never leaves the machine: no telemetry, no sockets.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <span>

namespace oct {

struct AuditEntry {
    uint64_t    seq = 0;
    uint64_t    unix_time = 0;
    std::string actor;     // module that performed the action
    std::string action;    // verb
    std::string subject;   // object / detail (redacted by policy)
    std::string payload;   // canonical JSON detail
    hash::Sha256Digest prev{};
    hash::Sha256Digest hash{};
    std::string encode() const;                       // canonical line (pre-obfuscation)
    static hash::Sha256Digest digest_of(const AuditEntry& e);
};

class AuditLog {
public:
    explicit AuditLog(std::string path = {}, uint64_t obfuscation_key = 0);

    void append(std::string actor, std::string action, std::string subject,
                std::string payload_json = "{}");

    const std::vector<AuditEntry>& entries() const { return entries_; }
    size_t size() const { return entries_.size(); }

    // Recomputes every link; returns index of the first broken entry, or -1.
    // Also detects truncated/payload-mutated logs written by flush().
    int64_t verify_chain() const;

    // Persisted form: obfuscated, hash-chained lines. Verifiable after reload.
    Status flush();
    Status load();          // replaces in-memory entries from the persisted file
    Status verify_file() const;

    // Deliberately-mutating test hook (used by the tamper-detection test).
    void corrupt_entry_for_test(size_t idx);

    std::string to_json() const;

private:
    std::string                path_;
    uint64_t                   key_ = 0;
    hash::Sha256Digest         head_{};
    std::vector<AuditEntry>    entries_;
};

// Obfuscation codec (INTERCAL-inspired layer). Deterministic stream built from
// SHA-256(keystream) -- *not* encryption, and documented as such: it defeats
// casual `strings`/hexdump inspection and nothing more.
namespace obfuscate {
std::vector<uint8_t> encode(std::span<const uint8_t> plain, uint64_t key);
std::vector<uint8_t> decode(std::span<const uint8_t> cipher, uint64_t key);
}  // namespace obfuscate

}  // namespace oct
