// Octopus Hybrid AI Engine -- INTERCAL-style guardrail (implementation).
// SPDX-License-Identifier: MIT
#include "octopus/intercal.hpp"

#include "octopus/module.hpp"

#include <algorithm>
#include <cstring>

namespace oct::intercal {

namespace {

// SHA-256 digest as 32 raw bytes (the public hash API returns hex).
std::string sha256_bytes(const std::string& data) {
    const std::string hex = hash::sha256_hex(data);
    std::string out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        auto nib = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return 0;
        };
        out.push_back(char((nib(hex[i]) << 4) | nib(hex[i + 1])));
    }
    return out;
}

std::string keystream_block(const std::string& key, const std::string& nonce, uint64_t counter) {
    std::string material = "OCT1-stream";
    material += key;
    material += nonce;
    material.push_back(char(counter & 0xFF));
    material.push_back(char((counter >> 8) & 0xFF));
    material.push_back(char((counter >> 16) & 0xFF));
    material.push_back(char((counter >> 24) & 0xFF));
    material.push_back(char((counter >> 32) & 0xFF));
    material.push_back(char((counter >> 40) & 0xFF));
    material.push_back(char((counter >> 48) & 0xFF));
    material.push_back(char((counter >> 56) & 0xFF));
    return sha256_bytes(material);
}

std::string tag_of(const std::string& magic, const std::string& nonce, const std::string& body,
                   const std::string& key) {
    return sha256_bytes(magic + nonce + body + key);
}

}  // namespace

std::string encode(const std::string& plaintext, const std::string& key) {
    // Deterministic nonce: the guardrail must be reproducible for the audit
    // chain and for tests; this is obfuscation, not a nonce-based cipher.
    std::string nonce = sha256_bytes("nonce|" + key).substr(0, kNonceBytes);
    std::string body(plaintext.size(), '\0');
    for (size_t i = 0; i < plaintext.size(); ++i) {
        const std::string block = keystream_block(key, nonce, i / 32);
        body[i] = char(uint8_t(plaintext[i]) ^ uint8_t(block[i % 32]));
    }
    const std::string magic(kMagic, sizeof(kMagic));
    return magic + nonce + body + tag_of(magic, nonce, body, key);
}

Outcome<std::string> decode(const std::string& envelope, const std::string& key) {
    if (envelope.size() < 4 + kNonceBytes + kTagBytes)
        return Status::invalid("intercal: envelope truncated");
    const std::string magic = envelope.substr(0, 4);
    if (magic != std::string(kMagic, sizeof(kMagic)))
        return Status::invalid("intercal: bad magic");
    const std::string nonce = envelope.substr(4, kNonceBytes);
    const size_t body_len = envelope.size() - 4 - kNonceBytes - kTagBytes;
    const std::string body = envelope.substr(4 + kNonceBytes, body_len);
    const std::string tag = envelope.substr(envelope.size() - kTagBytes);
    if (tag_of(magic, nonce, body, key) != tag)
        return Status::invalid("intercal: authentication tag mismatch");
    std::string out(body_len, '\0');
    for (size_t i = 0; i < body_len; ++i) {
        const std::string block = keystream_block(key, nonce, i / 32);
        out[i] = char(uint8_t(body[i]) ^ uint8_t(block[i % 32]));
    }
    return out;
}

Json FuzzReport::to_json() const {
    Json j;
    j.begin_object();
    j.field("iterations", int64_t(iterations));
    j.field("accepted", int64_t(accepted));
    j.field("rejected", int64_t(rejected));
    j.end_object();
    return j;
}

FuzzReport fuzz_guard(std::string blob, const std::string& key, size_t iterations, uint64_t seed) {
    FuzzReport r;
    r.iterations = iterations;
    if (blob.empty()) return r;
    Rng rng(seed);
    for (size_t i = 0; i < iterations; ++i) {
        const size_t pos = size_t(rng.next_u64() % blob.size());
        const uint8_t bit = uint8_t(1u << (rng.next_u64() % 8));
        std::string mutated = blob;
        mutated[pos] = char(uint8_t(mutated[pos]) ^ bit);
        auto out = decode(mutated, key);
        if (out) ++r.accepted; else ++r.rejected;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Audit chain
// ---------------------------------------------------------------------------
Json AuditEntry::to_json() const {
    Json j;
    j.begin_object();
    j.field("sequence", int64_t(sequence));
    j.field("kind", kind);
    j.field("payload_hash", payload_hash);
    j.field("prev_hash", prev_hash);
    j.field("hash", hash);
    j.end_object();
    return j;
}

AuditEntry AuditChain::append(const std::string& kind, const std::string& payload) {
    AuditEntry e;
    e.sequence = uint64_t(entries_.size()) + 1;
    e.kind = kind;
    e.payload_hash = hash::sha256_hex(payload);
    e.prev_hash = entries_.empty() ? std::string("genesis") : entries_.back().hash;
    e.hash = hash::sha256_hex(std::to_string(e.sequence) + "|" + e.kind + "|" + e.payload_hash +
                              "|" + e.prev_hash);
    entries_.push_back(e);
    return e;
}

bool AuditChain::verify(size_t* first_broken) const {
    for (size_t i = 0; i < entries_.size(); ++i) {
        const AuditEntry& e = entries_[i];
        const std::string expect_prev = i == 0 ? std::string("genesis") : entries_[i - 1].hash;
        const std::string expect_hash = hash::sha256_hex(std::to_string(e.sequence) + "|" +
                                                         e.kind + "|" + e.payload_hash + "|" +
                                                         e.prev_hash);
        if (e.prev_hash != expect_prev || e.hash != expect_hash) {
            if (first_broken) *first_broken = i;
            return false;
        }
    }
    return true;
}

Json AuditChain::to_json() const {
    Json j;
    j.begin_object();
    j.field("entries", int64_t(entries_.size()));
    j.field("valid", verify());
    j.begin_array("chain");
    for (const auto& e : entries_) {
        j.begin_object();
        j.field("sequence", int64_t(e.sequence));
        j.field("kind", e.kind);
        j.field("hash", e.hash.substr(0, 16));
        j.end_object();
    }
    j.end_array();
    j.end_object();
    return j;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class IntercalModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "intercal.guard";
        i.version = "0.1.0";
        i.language = "INTERCAL-style guardrail (C++20)";
        i.role = "Obfuscated, tamper-evident envelope for data at rest plus the append-only "
                 "audit chain.";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"guard.encode", "guard.decode", "guard.tamper_evidence",
                          "audit.append", "audit.verify", "guard.fuzz"};
        i.limitations = {
            "obfuscation and tamper evidence only: this is NOT a cipher and must not be used to "
            "protect secrets from a determined attacker",
            "the keystream construction is public and has no formal security proof; no key "
            "derivation, rotation or secure storage",
            "deterministic nonce: identical plaintext and key produce identical envelopes, which "
            "is required for reproducibility and is a known weakness for confidentiality",
            "INTERCAL is used as a deliberately cryptic encoding style, not as a security "
            "argument; no claim of immunity to decompilation"};
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }

    std::string describe() const override {
        return "Wraps payloads in an obfuscated, tagged envelope (reproducible keystream, "
               "SHA-256 tag) and maintains a hash-linked audit log of engine events.";
    }

    Status self_check() override {
        const std::string key = "self-check-key";
        const std::string msg = "octopus guardrail payload 0123456789";
        const std::string env = encode(msg, key);
        auto back = decode(env, key);
        if (!back || *back != msg) return Status::internal("intercal: round-trip failed");
        if (decode(env, "wrong-key")) return Status::internal("intercal: wrong key accepted");
        if (back->find(msg) == std::string::npos)
            return Status::internal("intercal: plaintext did not survive");

        FuzzReport f = fuzz_guard(env, key, 512, 7);
        if (f.accepted != 0)
            return Status::internal("intercal: " + std::to_string(f.accepted) +
                                    " tampered envelopes were accepted");

        AuditChain chain;
        chain.append("module.start", "intercal.guard");
        chain.append("guard.encode", "len=" + std::to_string(msg.size()));
        chain.append("module.ok", "self-check");
        if (!chain.verify()) return Status::internal("intercal: audit chain did not verify");

        // Prove that verification catches an edited log: replay the entries into
        // a new chain with one kind changed.
        AuditChain tampered;
        size_t i = 0;
        for (const AuditEntry& e : chain.entries()) {
            AuditEntry copy = e;
            if (i == 1) copy.kind += ".tampered";
            tampered.append_raw(copy);
            ++i;
        }
        size_t broken = 0;
        if (tampered.verify(&broken)) return Status::internal("intercal: edited log verified");
        if (broken != 1) return Status::internal("intercal: wrong entry flagged as broken");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_intercal_module() {
    return std::make_shared<IntercalModule>();
}

}  // namespace oct::intercal
