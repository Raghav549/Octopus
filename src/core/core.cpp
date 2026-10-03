// Octopus Hybrid AI Engine -- core primitives implementation.
// SPDX-License-Identifier: MIT
#include "octopus/core.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <thread>
#include <ctime>
#include <sys/utsname.h>

namespace oct {

// ---------------------------------------------------------------------------
const char* to_string(Code c) noexcept {
    switch (c) {
        case Code::Ok:          return "ok";
        case Code::Rejected:    return "rejected";
        case Code::Degraded:    return "degraded";
        case Code::Invalid:     return "invalid";
        case Code::Unavailable: return "unavailable";
        case Code::Timeout:     return "timeout";
        case Code::Internal:    return "internal";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
static const char* level_name(LogLevel l) {
    switch (l) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Off:   return "OFF  ";
    }
    return "?????";
}

LogSink& log_sink() {
    static LogSink s;
    return s;
}

void set_log_level(LogLevel l) { log_sink().level = l; }
LogLevel log_level() { return log_sink().level; }

void log_message(LogLevel l, std::string_view msg) {
    LogSink& s = log_sink();
    if (static_cast<int>(l) < static_cast<int>(s.level)) return;
    std::string line = std::string("[") + level_name(l) + "] " + std::string(msg) + "\n";
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.sink) {
        s.sink(l, line);
    } else {
        std::fputs(line.c_str(), stderr);
    }
}

// ---------------------------------------------------------------------------
namespace hash {

// Verified round constants: K[i] = floor(frac(cbrt(prime_i)) * 2^32).
static constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

Sha256::Sha256() {
    h_[0] = 0x6a09e667; h_[1] = 0xbb67ae85; h_[2] = 0x3c6ef372; h_[3] = 0xa54ff53a;
    h_[4] = 0x510e527f; h_[5] = 0x9b05688c; h_[6] = 0x1f83d9ab; h_[7] = 0x5be0cd19;
    std::memset(buf_, 0, sizeof(buf_));
}

void Sha256::compress(const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
               (uint32_t(block[i * 4 + 2]) << 8) | uint32_t(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + kK[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

void Sha256::update(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    total_ += len;
    while (len > 0) {
        size_t take = std::min(len, size_t(64) - buflen_);
        std::memcpy(buf_ + buflen_, p, take);
        buflen_ += take; p += take; len -= take;
        if (buflen_ == 64) { compress(buf_); buflen_ = 0; }
    }
}

Sha256Digest Sha256::finalize() {
    uint64_t bits = total_ * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t zero = 0;
    while (buflen_ != 56) update(&zero, 1);
    uint8_t lenbuf[8];
    for (int i = 0; i < 8; ++i) lenbuf[i] = uint8_t((bits >> (56 - 8 * i)) & 0xFF);
    // Append length without counting it in total_.
    uint64_t saved = total_;
    update(lenbuf, 8);
    total_ = saved;
    Sha256Digest d;
    for (int i = 0; i < 8; ++i) {
        d.bytes[i * 4 + 0] = uint8_t((h_[i] >> 24) & 0xFF);
        d.bytes[i * 4 + 1] = uint8_t((h_[i] >> 16) & 0xFF);
        d.bytes[i * 4 + 2] = uint8_t((h_[i] >> 8) & 0xFF);
        d.bytes[i * 4 + 3] = uint8_t(h_[i] & 0xFF);
    }
    return d;
}

std::string Sha256Digest::hex() const { return hex_encode(bytes, 32); }

Sha256Digest sha256(const void* data, size_t len) {
    Sha256 h; h.update(data, len); return h.finalize();
}
Sha256Digest sha256(std::string_view data) { return sha256(data.data(), data.size()); }
std::string  sha256_hex(std::string_view data) { return sha256(data).hex(); }

uint64_t fnv1a64(std::string_view s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) { h ^= c; h *= 0x100000001b3ULL; }
    return h;
}

}  // namespace hash

std::string hex_encode(const void* data, size_t len) {
    static const char* d = "0123456789abcdef";
    const uint8_t* p = static_cast<const uint8_t*>(data);
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(d[p[i] >> 4]);
        out.push_back(d[p[i] & 0xF]);
    }
    return out;
}

std::string to_lower(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// ---------------------------------------------------------------------------
uint64_t Rng::next_u64() {
    // SplitMix64: identical results on every platform/libstdc++.
    uint64_t z = (s_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double Rng::next_unit() { return double(next_u64() >> 11) * (1.0 / 9007199254740992.0); }

double Rng::next_normal() {
    double u1 = std::max(next_unit(), 1e-15);
    double u2 = next_unit();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}

int64_t Rng::range(int64_t lo, int64_t hi_inclusive) {
    if (hi_inclusive <= lo) return lo;
    uint64_t span = uint64_t(hi_inclusive - lo) + 1;
    return lo + int64_t(next_u64() % span);
}

// ---------------------------------------------------------------------------
static std::string json_escape(std::string_view s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o.push_back(char(c));
                }
        }
    }
    return o;
}

void Json::comma() {
    if (pending_key_) { pending_key_ = false; return; }
    if (!first_.empty() && !first_.back()) out_ += ",";
    if (!first_.empty()) first_.back() = false;
}

Json& Json::begin_object() { comma(); out_ += "{"; first_.push_back(true); return *this; }
Json& Json::end_object()   { out_ += "}"; if (!first_.empty()) first_.pop_back(); pending_key_ = false; return *this; }

Json& Json::begin_array(std::string_view k) {
    if (!k.empty()) this->key(k);
    comma(); out_ += "["; first_.push_back(true); return *this;
}
Json& Json::end_array() { out_ += "]"; if (!first_.empty()) first_.pop_back(); pending_key_ = false; return *this; }

Json& Json::key(std::string_view k) {
    comma();
    out_ += "\"" + json_escape(k) + "\":";
    pending_key_ = true;
    return *this;
}

Json& Json::value(std::string_view s) { comma(); out_ += "\"" + json_escape(s) + "\""; return *this; }
Json& Json::value(bool b)             { comma(); out_ += b ? "true" : "false"; return *this; }
Json& Json::value(int64_t v)          { comma(); out_ += std::to_string(v); return *this; }
Json& Json::value(uint64_t v)         { comma(); out_ += std::to_string(v); return *this; }

Json& Json::value(double v, int precision) {
    comma();
    if (!std::isfinite(v)) { out_ += "null"; return *this; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    out_ += buf;
    return *this;
}

// ---------------------------------------------------------------------------
double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int64_t unix_time_seconds() { return int64_t(std::time(nullptr)); }

std::string iso8601_now() {
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    gmtime_s(&tmv, &t);
#else
    gmtime_r(&t, &tmv);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return buf;
}

// ---------------------------------------------------------------------------
static HostInfo detect_host() {
    HostInfo h;
    h.hw_threads = std::max(1u, std::thread::hardware_concurrency());
#if defined(__x86_64__) || defined(_M_X64)
    h.arch = "x86_64";
    __builtin_cpu_init();
    h.has_avx2   = __builtin_cpu_supports("avx2");
    h.has_avx512 = __builtin_cpu_supports("avx512f");
#elif defined(__aarch64__) || defined(_M_ARM64)
    h.arch = "aarch64";
    h.has_neon = true;
#else
    h.arch = "unknown";
#endif
    struct utsname u{};
    if (uname(&u) == 0) h.os = std::string(u.sysname) + " " + u.release;
    else h.os = "unknown";

    std::ifstream mi("/proc/meminfo");
    std::string line;
    while (std::getline(mi, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            unsigned long kb = 0;
            std::sscanf(line.c_str(), "MemTotal: %lu kB", &kb);
            h.total_ram_bytes = uint64_t(kb) * 1024;
            break;
        }
    }
#ifdef _OPENMP
    h.has_omp = true;
#endif
#if defined(__SANITIZE_THREAD__)
    h.has_tsan = true;
#endif
    return h;
}

const HostInfo& host_info() {
    static HostInfo h = detect_host();
    return h;
}

std::string cpu_features_string() {
    std::string s;
    const HostInfo& h = host_info();
    if (h.has_avx2)   s += "avx2 ";
    if (h.has_avx512) s += "avx512f ";
    if (h.has_neon)   s += "neon ";
    if (h.has_omp)    s += "openmp ";
    if (h.has_tsan)   s += "tsan ";
    if (s.empty())    s = "baseline ";
    s += "(" + h.arch + ")";
    return s;
}

}  // namespace oct
