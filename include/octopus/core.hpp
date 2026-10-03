// Octopus Hybrid AI Engine -- core primitives.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <optional>
#include <stdexcept>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <cmath>

namespace oct {

inline constexpr int    kVersionMajor = 0;
inline constexpr int    kVersionMinor = 2;
inline constexpr const char* kVersionString = "0.2.0";
inline constexpr const char* kEngineName    = "Octopus";

// ---------------------------------------------------------------------------
// Status / Outcome
// ---------------------------------------------------------------------------
enum class Code : uint8_t {
    Ok = 0,
    Rejected,       // policy / verification refused the request
    Degraded,       // ran, but without the preferred backend
    Invalid,        // malformed input
    Unavailable,    // capability not compiled in / not detected
    Timeout,
    Internal,
};

const char* to_string(Code c) noexcept;

struct Status {
    Code        code = Code::Ok;
    std::string message;

    Status() = default;
    Status(Code c, std::string msg) : code(c), message(std::move(msg)) {}

    static Status ok(std::string msg = {}) { return {Code::Ok, std::move(msg)}; }
    static Status rejected(std::string msg) { return {Code::Rejected, std::move(msg)}; }
    static Status degraded(std::string msg) { return {Code::Degraded, std::move(msg)}; }
    static Status invalid(std::string msg) { return {Code::Invalid, std::move(msg)}; }
    static Status unavailable(std::string msg) { return {Code::Unavailable, std::move(msg)}; }
    static Status internal(std::string msg) { return {Code::Internal, std::move(msg)}; }

    bool is_ok() const noexcept { return code == Code::Ok || code == Code::Degraded; }
    explicit operator bool() const noexcept { return is_ok(); }
};

template <class T>
struct Outcome {
    Status                          status;
    std::optional<T>                value;

    Outcome() = default;
    Outcome(T v) : status(Status::ok()), value(std::move(v)) {}
    Outcome(Status s) : status(std::move(s)) {}
    Outcome(Status s, T v) : status(std::move(s)), value(std::move(v)) {}

    bool is_ok() const noexcept { return status.is_ok() && value.has_value(); }
    explicit operator bool() const noexcept { return is_ok(); }
    const T& operator*() const { return *value; }
    T& operator*() { return *value; }
    const T* operator->() const { return &*value; }
    T* operator->() { return &*value; }
};

// ---------------------------------------------------------------------------
// Diagnostics: local-only, opt-in, never networked (see docs/SECURITY.md).
// ---------------------------------------------------------------------------
enum class LogLevel : uint8_t { Trace = 0, Debug, Info, Warn, Error, Off };

struct LogSink {
    LogLevel    level = LogLevel::Info;
    std::mutex  mu;
    std::function<void(LogLevel, const std::string&)> sink;  // null => stderr
};

LogSink&  log_sink();
void      set_log_level(LogLevel l);
LogLevel  log_level();
void      log_message(LogLevel l, std::string_view msg);

inline void log_trace(std::string_view m) { log_message(LogLevel::Trace, m); }
inline void log_debug(std::string_view m) { log_message(LogLevel::Debug, m); }
inline void log_info (std::string_view m) { log_message(LogLevel::Info,  m); }
inline void log_warn (std::string_view m) { log_message(LogLevel::Warn,  m); }
inline void log_error(std::string_view m) { log_message(LogLevel::Error, m); }

// ---------------------------------------------------------------------------
// Hashing / encoding
// ---------------------------------------------------------------------------
namespace hash {
// FIPS 180-4 SHA-256. Used for audit chains, model manifests, provenance.
struct Sha256Digest {
    uint8_t bytes[32]{};
    std::string hex() const;
    bool operator==(const Sha256Digest& o) const noexcept {
        return std::memcmp(bytes, o.bytes, 32) == 0;
    }
    bool operator!=(const Sha256Digest& o) const noexcept { return !(*this == o); }
};

class Sha256 {
public:
    Sha256();
    void update(const void* data, size_t len);
    void update(std::string_view s) { update(s.data(), s.size()); }
    Sha256Digest finalize();
private:
    void compress(const uint8_t block[64]);
    uint32_t h_[8];
    uint8_t  buf_[64];
    uint64_t total_ = 0;
    size_t   buflen_ = 0;
};

Sha256Digest sha256(std::string_view data);
Sha256Digest sha256(const void* data, size_t len);
std::string  sha256_hex(std::string_view data);

// Non-cryptographic, stable 64-bit hash for cache keys and bucket routing.
uint64_t fnv1a64(std::string_view s);
}  // namespace hash

std::string hex_encode(const void* data, size_t len);
std::string to_lower(std::string s);

// ---------------------------------------------------------------------------
// Deterministic RNG (no <random> implementation variance across libstdc++/MSVC)
// ---------------------------------------------------------------------------
class Rng {
public:
    explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ULL) : s_(seed ? seed : 1) {}
    uint64_t next_u64();
    double   next_unit();                                     // [0,1)
    double   next_normal();                                   // Box-Muller
    int64_t  range(int64_t lo, int64_t hi_inclusive);
private:
    uint64_t s_;
};

// ---------------------------------------------------------------------------
// Minimal JSON emitter (structured, reproducible CLI output; no pretty print)
// ---------------------------------------------------------------------------
class Json {
public:
    Json& begin_object();
    Json& end_object();
    Json& begin_array(std::string_view key = {});
    Json& end_array();
    Json& key(std::string_view k);
    Json& value(std::string_view s);
    Json& value(const char* s) { return value(std::string_view(s)); }
    Json& value(bool b);
    Json& value(int64_t v);
    Json& value(uint64_t v);
    Json& value(double v, int precision = 10);
    Json& field(std::string_view k, std::string_view v) { return key(k).value(v); }
    // Without this overload a `const char*` argument binds to the bool overload
    // (pointer-to-bool conversion) and the field silently prints true/false.
    Json& field(std::string_view k, const char* v) { return key(k).value(std::string_view(v)); }
    Json& field(std::string_view k, bool v) { return key(k).value(v); }
    Json& field(std::string_view k, int64_t v) { return key(k).value(v); }
    Json& field(std::string_view k, uint64_t v) { return key(k).value(v); }
    Json& field(std::string_view k, int v) { return key(k).value(static_cast<int64_t>(v)); }
    Json& field(std::string_view k, double v, int p = 10) { return key(k).value(v, p); }
    const std::string& str() const { return out_; }
    std::string take() { return std::move(out_); }
private:
    void comma();
    std::string out_;
    std::vector<bool> first_;   // per nesting level: is next element the first?
    bool pending_key_ = false;
};

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0);
int64_t unix_time_seconds();
std::string iso8601_now();

// ---------------------------------------------------------------------------
// Hardware / environment detection (used for honest capability reporting)
// ---------------------------------------------------------------------------
struct HostInfo {
    unsigned hw_threads = 0;
    std::string arch;
    std::string os;
    bool has_avx2 = false;
    bool has_avx512 = false;
    bool has_neon = false;
    bool has_omp = false;
    bool has_tsan = false;
    uint64_t total_ram_bytes = 0;
};
const HostInfo& host_info();
std::string cpu_features_string();

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
class OctError : public std::runtime_error {
public:
    explicit OctError(const std::string& what) : std::runtime_error(what) {}
};

inline void require(bool cond, const std::string& msg) {
    if (!cond) throw OctError(msg);
}

}  // namespace oct
