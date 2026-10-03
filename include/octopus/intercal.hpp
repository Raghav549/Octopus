// Octopus Hybrid AI Engine -- INTERCAL-style guardrail layer.
//
// Role in the engine (ARCHITECTURE_SPEC.md, "hardware-level security shield"):
// data at rest is kept in an obfuscated, tamper-evident envelope so that casual
// inspection of the binary or of a snapshot does not reveal plaintext, and an
// edited envelope can never be silently accepted. It also provides the
// append-only audit chain.
//
// HONESTY: this is obfuscation + tamper evidence, NOT a cipher. The keystream is
// a SHA-256 counter stream over a public construction; there is no claim of
// cryptographic strength, no key management, and INTERCAL is used here only as
// the deliberately cryptic *encoding* of the guardrail, never as a security
// proof. See docs/LIMITATIONS.md.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <string>
#include <vector>

namespace oct { class Module; }

namespace oct::intercal {

// Envelope layout: magic "OCT1" | nonce(16) | ciphertext(..) | tag(32).
// The tag is SHA-256 over magic||nonce||ciphertext||key.
inline constexpr char kMagic[4] = {'O', 'C', 'T', '1'};
inline constexpr size_t kNonceBytes = 16;
inline constexpr size_t kTagBytes   = 32;

std::string encode(const std::string& plaintext, const std::string& key);
// Returns Status::invalid (never a partial plaintext) when the tag or magic is
// wrong, or when the envelope is truncated.
Outcome<std::string> decode(const std::string& envelope, const std::string& key);

// Flip `iterations` bits in random positions of `blob` and count how many
// mutations decode() accepts. Any accepted mutation is a silent failure.
struct FuzzReport {
    size_t iterations = 0;
    size_t accepted = 0;      // must be 0
    size_t rejected = 0;
    Json to_json() const;
};
FuzzReport fuzz_guard(std::string blob, const std::string& key, size_t iterations, uint64_t seed = 1);

// ---------------------------------------------------------------------------
// Audit chain: append-only, hash-linked records.
// ---------------------------------------------------------------------------
struct AuditEntry {
    uint64_t    sequence = 0;
    std::string kind;
    std::string payload_hash;   // SHA-256 of the payload
    std::string prev_hash;      // hash of the previous entry ("genesis" for the first)
    std::string hash;           // SHA-256 over sequence|kind|payload_hash|prev_hash
    Json to_json() const;
};

class AuditChain {
public:
    AuditChain() = default;
    AuditEntry append(const std::string& kind, const std::string& payload);
    // Re-insert a persisted entry verbatim (log replay). verify() then checks
    // whether the replayed log is intact -- an edited entry will not match.
    void append_raw(const AuditEntry& e) { entries_.push_back(e); }
    // True only if every link recomputes; returns the first broken sequence.
    bool verify(size_t* first_broken = nullptr) const;
    const std::vector<AuditEntry>& entries() const { return entries_; }
    size_t size() const { return entries_.size(); }
    Json to_json() const;

private:
    std::vector<AuditEntry> entries_;
};

std::shared_ptr<oct::Module> make_intercal_module();

}  // namespace oct::intercal
