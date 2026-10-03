// Octopus Hybrid AI Engine -- hash-chained audit log + obfuscation codec.
// SPDX-License-Identifier: MIT
#include "octopus/audit.hpp"

#include <fstream>
#include <sstream>

namespace oct {

namespace obfuscate {

// Deterministic keystream: block i = SHA-256(key || i) (32-byte blocks).
// This is an obfuscation layer, NOT a cipher with a security claim. Its only
// purpose is to stop casual inspection from reading audit lines / constants
// directly out of a binary image with `strings`.
static void keystream(uint64_t key, size_t n, std::vector<uint8_t>& out) {
    out.clear();
    out.reserve(n + 32);
    for (uint64_t i = 0; out.size() < n; ++i) {
        hash::Sha256 h;
        h.update(&key, sizeof(key));
        h.update(&i, sizeof(i));
        auto d = h.finalize();
        out.insert(out.end(), d.bytes, d.bytes + 32);
    }
    out.resize(n);
}

std::vector<uint8_t> encode(std::span<const uint8_t> plain, uint64_t key) {
    std::vector<uint8_t> ks;
    keystream(key, plain.size(), ks);
    std::vector<uint8_t> out(plain.size());
    for (size_t i = 0; i < plain.size(); ++i) out[i] = uint8_t(plain[i] ^ ks[i]);
    return out;
}

std::vector<uint8_t> decode(std::span<const uint8_t> cipher, uint64_t key) {
    return encode(cipher, key);   // XOR is an involution
}

}  // namespace obfuscate

// ---------------------------------------------------------------------------
std::string AuditEntry::encode() const {
    // Canonical, unambiguous, unit-separated line.
    std::ostringstream os;
    os << seq << '\x1f' << unix_time << '\x1f' << actor << '\x1f' << action << '\x1f'
       << subject << '\x1f' << payload << '\x1f' << prev.hex();
    return os.str();
}

hash::Sha256Digest AuditEntry::digest_of(const AuditEntry& e) {
    return hash::sha256(e.encode());
}

AuditLog::AuditLog(std::string path, uint64_t obfuscation_key)
    : path_(std::move(path)), key_(obfuscation_key) {}

void AuditLog::append(std::string actor, std::string action, std::string subject,
                      std::string payload_json) {
    AuditEntry e;
    e.seq = uint64_t(entries_.size()) + 1;
    e.unix_time = uint64_t(unix_time_seconds());
    e.actor = std::move(actor);
    e.action = std::move(action);
    e.subject = std::move(subject);
    e.payload = std::move(payload_json);
    e.prev = head_;
    // Hash covers the entry body and the previous head -> append-only chain.
    e.hash = AuditEntry::digest_of(e);
    head_ = e.hash;
    entries_.push_back(std::move(e));
}

int64_t AuditLog::verify_chain() const {
    hash::Sha256Digest prev{};
    for (size_t i = 0; i < entries_.size(); ++i) {
        const AuditEntry& e = entries_[i];
        if (e.seq != uint64_t(i) + 1) return int64_t(i);
        if (e.prev != prev) return int64_t(i);
        hash::Sha256Digest expect = AuditEntry::digest_of(e);
        if (expect != e.hash) return int64_t(i);
        prev = e.hash;
    }
    return -1;
}

Status AuditLog::flush() {
    if (path_.empty()) return Status::ok("no audit path configured");
    std::ofstream f(path_, std::ios::binary | std::ios::trunc);
    if (!f) return Status::internal("cannot open audit log for write: " + path_);
    // Header line: magic + key id + entry count. Obfuscated by default.
    std::ostringstream hdr;
    hdr << "OCTAUDIT1 " << entries_.size() << " " << head_.hex() << "\n";
    std::string hdr_s = hdr.str();
    auto enc_hdr = obfuscate::encode(std::span<const uint8_t>(
                          reinterpret_cast<const uint8_t*>(hdr_s.data()), hdr_s.size()), key_);
    f.write(reinterpret_cast<const char*>(enc_hdr.data()), std::streamsize(enc_hdr.size()));
    f.put('\n');
    for (const auto& e : entries_) {
        std::string line = e.encode() + '\x1f' + e.hash.hex() + '\n';
        auto enc = obfuscate::encode(std::span<const uint8_t>(
                         reinterpret_cast<const uint8_t*>(line.data()), line.size()), key_);
        f.write(reinterpret_cast<const char*>(enc.data()), std::streamsize(enc.size()));
    }
    if (!f) return Status::internal("audit log write failed: " + path_);
    return Status::ok(std::to_string(entries_.size()) + " entries");
}

Status AuditLog::load() {
    if (path_.empty()) return Status::ok("no audit path configured");
    std::ifstream f(path_, std::ios::binary);
    if (!f) return Status::ok("no persisted audit log yet");
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t nl = all.find('\n');
    if (nl == std::string::npos) return Status::invalid("audit log header missing");
    auto hdr_plain = obfuscate::decode(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(all.data()), nl), key_);
    std::string hdr(reinterpret_cast<const char*>(hdr_plain.data()), hdr_plain.size());
    if (hdr.rfind("OCTAUDIT1 ", 0) != 0)
        return Status::invalid("audit log header invalid (wrong key or corrupt log)");

    entries_.clear();
    head_ = hash::Sha256Digest{};
    size_t pos = nl + 1;
    while (pos < all.size()) {
        size_t eol = all.find('\n', pos);
        if (eol == std::string::npos) eol = all.size();
        std::string chunk = all.substr(pos, eol - pos);
        pos = eol + 1;
        if (chunk.empty()) continue;
        auto plain = obfuscate::decode(std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(chunk.data()), chunk.size()), key_);
        std::string line(reinterpret_cast<const char*>(plain.data()), plain.size());

        // Fields: seq US time US actor US action US subject US payload US prev US hash
        std::vector<std::string> f;
        size_t start = 0;
        for (;;) {
            size_t us = line.find('\x1f', start);
            if (us == std::string::npos) { f.push_back(line.substr(start)); break; }
            f.push_back(line.substr(start, us - start));
            start = us + 1;
        }
        if (f.size() != 8) return Status::invalid("audit entry field count wrong");

        AuditEntry e;
        e.seq = std::stoull(f[0]);
        e.unix_time = std::stoull(f[1]);
        e.actor = f[2];
        e.action = f[3];
        e.subject = f[4];
        e.payload = f[5];
        auto hex_to_bytes = [](const std::string& h, uint8_t* out) {
            if (h.size() != 64) throw OctError("bad digest length");
            for (size_t i = 0; i < 32; ++i)
                out[i] = uint8_t(std::stoi(h.substr(i * 2, 2), nullptr, 16));
        };
        try {
            hex_to_bytes(f[6], e.prev.bytes);
            hex_to_bytes(f[7], e.hash.bytes);
        } catch (const std::exception& ex) {
            return Status::invalid(std::string("audit digest parse: ") + ex.what());
        }
        head_ = e.hash;
        entries_.push_back(std::move(e));
    }
    return Status::ok(std::to_string(entries_.size()) + " entries loaded");
}

Status AuditLog::verify_file() const {
    std::ifstream f(path_, std::ios::binary);
    if (!f) return Status::invalid("cannot open audit log: " + path_);
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t nl = all.find('\n');
    if (nl == std::string::npos) return Status::invalid("audit log header missing");
    auto hdr_plain = obfuscate::decode(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(all.data()), nl), key_);
    std::string hdr(reinterpret_cast<const char*>(hdr_plain.data()), hdr_plain.size());
    if (hdr.rfind("OCTAUDIT1 ", 0) != 0)
        return Status::rejected("audit header not decodable with this key");

    std::istringstream hs(hdr.substr(10));
    uint64_t count = 0;
    std::string expect_head;
    hs >> count >> expect_head;

    std::vector<AuditEntry> parsed;
    size_t pos = nl + 1, n_entries = 0;
    hash::Sha256Digest prev{};
    while (pos < all.size()) {
        size_t eol = all.find('\n', pos);
        if (eol == std::string::npos) eol = all.size();
        std::string chunk = all.substr(pos, eol - pos);
        pos = eol + 1;
        if (chunk.empty()) continue;
        n_entries++;
        auto plain = obfuscate::decode(std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(chunk.data()), chunk.size()), key_);
        std::string line(reinterpret_cast<const char*>(plain.data()), plain.size());
        std::vector<std::string> f;
        size_t start = 0;
        for (;;) {
            size_t us = line.find('\x1f', start);
            if (us == std::string::npos) { f.push_back(line.substr(start)); break; }
            f.push_back(line.substr(start, us - start));
            start = us + 1;
        }
        if (f.size() != 8) return Status::rejected("entry " + std::to_string(n_entries) + ": field count");
        AuditEntry e;
        e.seq = std::stoull(f[0]);
        e.unix_time = std::stoull(f[1]);
        e.actor = f[2]; e.action = f[3]; e.subject = f[4]; e.payload = f[5];
        auto parse_hex = [](const std::string& h, uint8_t* out, size_t n) {
            if (h.size() != n * 2) throw OctError("bad digest length");
            for (size_t i = 0; i < n; ++i) out[i] = uint8_t(std::stoi(h.substr(i * 2, 2), nullptr, 16));
        };
        parse_hex(f[6], e.prev.bytes, 32);
        parse_hex(f[7], e.hash.bytes, 32);
        if (e.prev != prev) return Status::rejected("chain break at entry " + std::to_string(n_entries));
        if (AuditEntry::digest_of(e) != e.hash)
            return Status::rejected("payload tampered at entry " + std::to_string(n_entries));
        prev = e.hash;
        parsed.push_back(std::move(e));
    }
    if (n_entries != count)
        return Status::rejected("entry count mismatch: header=" + std::to_string(count) +
                                " actual=" + std::to_string(n_entries) +
                                " (log truncated or extended)");
    if (prev.hex() != expect_head)
        return Status::rejected("head mismatch: header=" + expect_head + " computed=" + prev.hex());
    return Status::ok(std::to_string(n_entries) + " entries verified");
}

void AuditLog::corrupt_entry_for_test(size_t idx) {
    if (idx >= entries_.size()) return;
    entries_[idx].payload += "!";
}

std::string AuditLog::to_json() const {
    Json j;
    j.begin_object();
    j.field("entries", int64_t(entries_.size()));
    j.field("head", head_.hex());
    j.field("chain_ok", verify_chain() < 0);
    j.key("items");
    j.begin_array();
    for (const auto& e : entries_) {
        j.begin_object();
        j.field("seq", int64_t(e.seq));
        j.field("t", int64_t(e.unix_time));
        j.field("actor", e.actor);
        j.field("action", e.action);
        j.field("subject", e.subject);
        j.field("hash", e.hash.hex());
        j.end_object();
    }
    j.end_array();
    j.end_object();
    return j.take();
}

}  // namespace oct
