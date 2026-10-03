// Octopus Hybrid AI Engine -- GGUF reader/writer.
// SPDX-License-Identifier: MIT
#include "octopus/gguf.hpp"

#include <fstream>
#include <cstring>
#include <sstream>

namespace oct::gguf {

const char* value_type_name(ValueType t) noexcept {
    switch (t) {
        case ValueType::UINT8:   return "u8";
        case ValueType::INT8:    return "i8";
        case ValueType::UINT16:  return "u16";
        case ValueType::INT16:   return "i16";
        case ValueType::UINT32:  return "u32";
        case ValueType::INT32:   return "i32";
        case ValueType::FLOAT32: return "f32";
        case ValueType::BOOL:    return "bool";
        case ValueType::STRING:  return "string";
        case ValueType::ARRAY:   return "array";
        case ValueType::UINT64:  return "u64";
        case ValueType::INT64:   return "i64";
        case ValueType::FLOAT64: return "f64";
    }
    return "?";
}

size_t value_type_size(ValueType t) noexcept {
    switch (t) {
        case ValueType::UINT8:
        case ValueType::INT8:
        case ValueType::BOOL:    return 1;
        case ValueType::UINT16:
        case ValueType::INT16:   return 2;
        case ValueType::UINT32:
        case ValueType::INT32:
        case ValueType::FLOAT32: return 4;
        case ValueType::UINT64:
        case ValueType::INT64:
        case ValueType::FLOAT64: return 8;
        default: return 0;   // string/array: variable
    }
}

uint64_t Value::as_u64() const {
    if (auto* p = std::get_if<uint64_t>(&data)) return *p;
    if (auto* p = std::get_if<int64_t>(&data)) return uint64_t(*p);
    if (auto* p = std::get_if<double>(&data)) return uint64_t(*p);
    if (auto* p = std::get_if<bool>(&data)) return *p ? 1 : 0;
    throw OctError("GGUF value is not numeric");
}
int64_t Value::as_i64() const {
    if (auto* p = std::get_if<int64_t>(&data)) return *p;
    if (auto* p = std::get_if<uint64_t>(&data)) return int64_t(*p);
    if (auto* p = std::get_if<double>(&data)) return int64_t(*p);
    throw OctError("GGUF value is not integer");
}
double Value::as_f64() const {
    if (auto* p = std::get_if<double>(&data)) return *p;
    if (auto* p = std::get_if<uint64_t>(&data)) return double(*p);
    if (auto* p = std::get_if<int64_t>(&data)) return double(*p);
    throw OctError("GGUF value is not floating point");
}
bool Value::as_bool() const {
    if (auto* p = std::get_if<bool>(&data)) return *p;
    return as_u64() != 0;
}
std::string Value::as_string() const {
    if (auto* p = std::get_if<std::string>(&data)) return *p;
    throw OctError("GGUF value is not a string");
}
const ArrayValue& Value::as_array() const {
    if (auto* p = std::get_if<ArrayValue>(&data)) return *p;
    throw OctError("GGUF value is not an array");
}

std::string Value::to_display() const {
    switch (type) {
        case ValueType::BOOL:   return as_bool() ? "true" : "false";
        case ValueType::STRING: return "\"" + as_string() + "\"";
        case ValueType::FLOAT32:
        case ValueType::FLOAT64: {
            char b[64]; std::snprintf(b, sizeof(b), "%.6g", as_f64()); return b;
        }
        case ValueType::ARRAY: {
            const auto& a = as_array();
            std::ostringstream os;
            os << "[" << a.size() << " x " << (a.empty() ? "?" : value_type_name(a[0].type)) << "]";
            return os.str();
        }
        default: return std::to_string(as_i64());
    }
}

// ---------------------------------------------------------------------------
const char* ggml_type_name(GgmlType t) noexcept {
    switch (t) {
        case GgmlType::F32: return "f32";
        case GgmlType::F16: return "f16";
        case GgmlType::Q4_0: return "q4_0";
        case GgmlType::Q4_1: return "q4_1";
        case GgmlType::Q5_0: return "q5_0";
        case GgmlType::Q5_1: return "q5_1";
        case GgmlType::Q8_0: return "q8_0";
        case GgmlType::Q8_1: return "q8_1";
        case GgmlType::Q2_K: return "q2_k";
        case GgmlType::Q3_K: return "q3_k";
        case GgmlType::Q4_K: return "q4_k";
        case GgmlType::Q5_K: return "q5_k";
        case GgmlType::Q6_K: return "q6_k";
        case GgmlType::Q8_K: return "q8_k";
        case GgmlType::IQ2_XXS: return "iq2_xxs";
        case GgmlType::IQ2_XS: return "iq2_xs";
        case GgmlType::IQ3_XXS: return "iq3_xxs";
        case GgmlType::IQ1_S: return "iq1_s";
        case GgmlType::IQ4_NL: return "iq4_nl";
        case GgmlType::IQ3_S: return "iq3_s";
        case GgmlType::IQ2_S: return "iq2_s";
        case GgmlType::IQ4_XS: return "iq4_xs";
        case GgmlType::I8: return "i8";
        case GgmlType::I16: return "i16";
        case GgmlType::I32: return "i32";
        case GgmlType::I64: return "i64";
        case GgmlType::F64: return "f64";
        case GgmlType::BF16: return "bf16";
        case GgmlType::TQ1_0: return "tq1_0";
        case GgmlType::TQ2_0: return "tq2_0";
        default: return "unknown";
    }
}

bool ggml_type_is_quantized(GgmlType t) noexcept {
    switch (t) {
        case GgmlType::F32: case GgmlType::F16: case GgmlType::BF16:
        case GgmlType::F64: case GgmlType::I8:  case GgmlType::I16:
        case GgmlType::I32: case GgmlType::I64:
            return false;
        default: return true;
    }
}

double ggml_type_bits(GgmlType t) noexcept {
    // Stored bits per element, including per-block scale overhead, derived from
    // block sizes (Q*_0: 32-element blocks; Q*_K: 256-element superblocks).
    switch (t) {
        case GgmlType::F32: return 32.0;
        case GgmlType::F16: return 16.0;
        case GgmlType::BF16: return 16.0;
        case GgmlType::F64: return 64.0;
        case GgmlType::I8: return 8.0;
        case GgmlType::I16: return 16.0;
        case GgmlType::I32: return 32.0;
        case GgmlType::I64: return 64.0;
        case GgmlType::Q4_0: return 32.0 * 18 / 32;             // 18 bytes / 32 weights
        case GgmlType::Q4_1: return 32.0 * 20 / 32;
        case GgmlType::Q5_0: return 32.0 * 22 / 32;
        case GgmlType::Q5_1: return 32.0 * 24 / 32;
        case GgmlType::Q8_0: return 32.0 * 34 / 32;
        case GgmlType::Q8_1: return 32.0 * 36 / 32;
        case GgmlType::Q2_K: return 256.0 * 84 / 256;
        case GgmlType::Q3_K: return 256.0 * 110 / 256;
        case GgmlType::Q4_K: return 256.0 * 144 / 256;
        case GgmlType::Q5_K: return 256.0 * 176 / 256;
        case GgmlType::Q6_K: return 256.0 * 210 / 256;
        case GgmlType::Q8_K: return 256.0 * 292 / 256;
        case GgmlType::TQ1_0: return 256.0 * 54 / 256;
        case GgmlType::TQ2_0: return 256.0 * 66 / 256;
        default: return 4.5;   // IQ* families: approximate (documented)
    }
}

// ---------------------------------------------------------------------------
namespace {

class Reader {
public:
    Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
    size_t pos() const { return pos_; }
    size_t remaining() const { return n_ - pos_; }

    void need(size_t k) const {
        if (k > n_ - pos_) throw OctError("GGUF: unexpected end of data");
    }
    uint8_t u8()  { need(1); return p_[pos_++]; }
    int8_t  i8()  { return int8_t(u8()); }
    uint16_t u16() { need(2); uint16_t v = uint16_t(p_[pos_]) | uint16_t(p_[pos_ + 1]) << 8; pos_ += 2; return v; }
    int16_t  i16() { return int16_t(u16()); }
    uint32_t u32() {
        need(4);
        uint32_t v = uint32_t(p_[pos_]) | uint32_t(p_[pos_ + 1]) << 8 |
                     uint32_t(p_[pos_ + 2]) << 16 | uint32_t(p_[pos_ + 3]) << 24;
        pos_ += 4; return v;
    }
    int32_t i32() { return int32_t(u32()); }
    uint64_t u64() {
        need(8);
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) v = (v << 8) | p_[pos_ + size_t(i)];
        pos_ += 8; return v;
    }
    int64_t i64() { return int64_t(u64()); }
    float f32() { uint32_t b = u32(); float f; std::memcpy(&f, &b, 4); return f; }
    double f64() { uint64_t b = u64(); double d; std::memcpy(&d, &b, 8); return d; }
    std::string str() {
        uint64_t len = u64();
        need(size_t(len));
        std::string s(reinterpret_cast<const char*>(p_ + pos_), size_t(len));
        pos_ += size_t(len);
        return s;
    }

private:
    const uint8_t* p_;
    size_t n_;
    size_t pos_ = 0;
};

Value read_value(Reader& r, ValueType t, int depth = 0) {
    Value v;
    v.type = t;
    switch (t) {
        case ValueType::UINT8:   v.data = uint64_t(r.u8()); break;
        case ValueType::INT8:    v.data = int64_t(r.i8()); break;
        case ValueType::UINT16:  v.data = uint64_t(r.u16()); break;
        case ValueType::INT16:   v.data = int64_t(r.i16()); break;
        case ValueType::UINT32:  v.data = uint64_t(r.u32()); break;
        case ValueType::INT32:   v.data = int64_t(r.i32()); break;
        case ValueType::FLOAT32: v.data = double(r.f32()); break;
        case ValueType::BOOL:    v.data = (r.u8() != 0); break;
        case ValueType::STRING:  v.data = r.str(); break;
        case ValueType::UINT64:  v.data = r.u64(); break;
        case ValueType::INT64:   v.data = r.i64(); break;
        case ValueType::FLOAT64: v.data = r.f64(); break;
        case ValueType::ARRAY: {
            if (depth > 1) throw OctError("GGUF: nested arrays are not permitted");
            uint32_t et = r.u32();
            if (et > uint32_t(ValueType::FLOAT64)) throw OctError("GGUF: invalid array element type");
            uint64_t count = r.u64();
            // Guard against absurd allocations from corrupt headers.
            if (count > (1ull << 28)) throw OctError("GGUF: implausible array length");
            ArrayValue arr;
            arr.reserve(size_t(count));
            for (uint64_t i = 0; i < count; ++i)
                arr.push_back(read_value(r, ValueType(et), depth + 1));
            v.data = std::move(arr);
            break;
        }
        default:
            throw OctError("GGUF: unknown value type " + std::to_string(uint32_t(t)));
    }
    return v;
}

uint64_t tensor_payload_bytes(GgmlType t, uint64_t n_elements) {
    // Block-quantised types store whole blocks of 32 (Q*_0/Q*_1) or 256 (…_K).
    auto blocks = [&](uint64_t block, uint64_t bytes_per_block) -> uint64_t {
        if (n_elements % block != 0)
            throw OctError("GGUF: tensor element count not divisible by block size");
        return (n_elements / block) * bytes_per_block;
    };
    switch (t) {
        case GgmlType::F32:  return n_elements * 4;
        case GgmlType::F16:  return n_elements * 2;
        case GgmlType::BF16: return n_elements * 2;
        case GgmlType::F64:  return n_elements * 8;
        case GgmlType::I8:   return n_elements;
        case GgmlType::I16:  return n_elements * 2;
        case GgmlType::I32:  return n_elements * 4;
        case GgmlType::I64:  return n_elements * 8;
        case GgmlType::Q4_0: return blocks(32, 18);
        case GgmlType::Q4_1: return blocks(32, 20);
        case GgmlType::Q5_0: return blocks(32, 22);
        case GgmlType::Q5_1: return blocks(32, 24);
        case GgmlType::Q8_0: return blocks(32, 34);
        case GgmlType::Q8_1: return blocks(32, 36);
        case GgmlType::Q2_K: return blocks(256, 84);
        case GgmlType::Q3_K: return blocks(256, 110);
        case GgmlType::Q4_K: return blocks(256, 144);
        case GgmlType::Q5_K: return blocks(256, 176);
        case GgmlType::Q6_K: return blocks(256, 210);
        case GgmlType::Q8_K: return blocks(256, 292);
        case GgmlType::TQ1_0: return blocks(256, 54);
        case GgmlType::TQ2_0: return blocks(256, 66);
        default:
            throw OctError(std::string("GGUF: unsupported tensor type in payload-size computation: ") +
                           ggml_type_name(t));
    }
}

}  // namespace

Outcome<ModelInfo> read_model_info_from_buffer(std::span<const uint8_t> buf) {
    try {
        Reader r(buf.data(), buf.size());
        uint32_t magic = r.u32();
        if (magic != 0x46554747u)   // "GGUF" little-endian
            return Status::invalid("not a GGUF file (bad magic)");
        ModelInfo info;
        info.version = r.u32();
        if (info.version < 2 || info.version > 3)
            return Status::invalid("unsupported GGUF version " + std::to_string(info.version));
        info.n_tensors = r.u64();
        info.n_kv = r.u64();
        if (info.n_kv > 4096) return Status::invalid("implausible metadata count");
        if (info.n_tensors > 65536) return Status::invalid("implausible tensor count");

        for (uint64_t i = 0; i < info.n_kv; ++i) {
            std::string key = r.str();
            uint32_t vt = r.u32();
            if (vt > uint32_t(ValueType::FLOAT64))
                return Status::invalid("invalid metadata value type for key " + key);
            info.metadata[key] = read_value(r, ValueType(vt));
        }

        for (uint64_t i = 0; i < info.n_tensors; ++i) {
            TensorInfo t;
            t.name = r.str();
            uint32_t nd = r.u32();
            if (nd == 0 || nd > 4) return Status::invalid("invalid tensor rank for " + t.name);
            t.dimensions.resize(nd);
            t.n_elements = 1;
            for (uint32_t d = 0; d < nd; ++d) {
                t.dimensions[d] = r.u64();
                t.n_elements *= t.dimensions[d];
            }
            uint32_t tt = r.u32();
            t.type = GgmlType(tt);
            t.offset = r.u64();
            t.n_bytes = tensor_payload_bytes(t.type, t.n_elements);
            info.tensors.push_back(std::move(t));
        }

        // Data section is aligned to `general.alignment` (default 32).
        if (auto al = info.get_u64("general.alignment"); al && *al >= 8 && (*al & (*al - 1)) == 0) {
            info.alignment = *al;
        }
        // Trailing padding before the data section (Section 5.1: header is
        // padded so that the tensor data starts aligned).
        // Some writers (older versions) used a raw data_section_offset key.
        if (auto off = info.get_u64("data_section_offset"); off) {
            info.data_section_offset = *off;
        } else {
            size_t cur = r.pos();
            size_t pad = (info.alignment - (cur % info.alignment)) % info.alignment;
            info.data_section_offset = cur + pad;
        }
        return info;
    } catch (const std::exception& e) {
        return Status::invalid(std::string("GGUF parse error: ") + e.what());
    }
}

Outcome<ModelInfo> read_model_info(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return Status::invalid("cannot open model file: " + path);
    // Read the whole file only if small; otherwise read incrementally up to 64 MiB
    // which is far beyond any realistic GGUF header (metadata + tensor table).
    constexpr size_t kMaxHeader = 64u << 20;
    std::vector<uint8_t> buf;
    buf.resize(1u << 16);
    size_t filled = 0;
    ModelInfo info;
    for (;;) {
        f.read(reinterpret_cast<char*>(buf.data() + filled), std::streamsize(buf.size() - filled));
        filled += size_t(f.gcount());
        if (f.gcount() == 0 && filled != buf.size()) break;
        auto out = read_model_info_from_buffer(std::span<const uint8_t>(buf.data(), filled));
        if (out.is_ok()) return out;
        if (out.status.code != Code::Invalid) return out;
        // "unexpected end of data" means we need more bytes; anything else is fatal.
        if (out.status.message.find("end of data") == std::string::npos) return out;
        if (buf.size() >= kMaxHeader)
            return Status::invalid("GGUF header exceeds " + std::to_string(kMaxHeader) + " bytes");
        buf.resize(std::min(buf.size() * 2, kMaxHeader));
    }
    return Status::invalid("cannot parse GGUF header: " + path);
}

// ---------------------------------------------------------------------------
std::optional<uint64_t> ModelInfo::get_u64(std::string_view key) const {
    auto it = metadata.find(std::string(key));
    if (it == metadata.end()) return std::nullopt;
    try { return it->second.as_u64(); } catch (...) { return std::nullopt; }
}
std::optional<double> ModelInfo::get_f64(std::string_view key) const {
    auto it = metadata.find(std::string(key));
    if (it == metadata.end()) return std::nullopt;
    try { return it->second.as_f64(); } catch (...) { return std::nullopt; }
}
std::optional<std::string> ModelInfo::get_str(std::string_view key) const {
    auto it = metadata.find(std::string(key));
    if (it == metadata.end()) return std::nullopt;
    try { return it->second.as_string(); } catch (...) { return std::nullopt; }
}
std::optional<std::vector<std::string>> ModelInfo::get_str_array(std::string_view key) const {
    auto it = metadata.find(std::string(key));
    if (it == metadata.end()) return std::nullopt;
    try {
        std::vector<std::string> out;
        for (const auto& v : it->second.as_array()) out.push_back(v.as_string());
        return out;
    } catch (...) { return std::nullopt; }
}

ModelInfo::Facts ModelInfo::facts() const {
    Facts f;
    f.architecture = get_str("general.architecture").value_or("");
    f.name         = get_str("general.name").value_or("");
    if (!f.architecture.empty()) {
        const std::string p = f.architecture + ".";
        auto g = [&](const char* k) -> int64_t {
            return int64_t(get_u64(p + k).value_or(0));
        };
        f.n_layer     = g("block_count");
        f.n_embd      = g("embedding_length");
        f.n_head      = g("attention.head_count");
        f.n_head_kv   = g("attention.head_count_kv");
        f.n_ff        = g("feed_forward_length");
        f.n_ctx_train = g("context_length");
        f.n_ctx_max   = g("context_length_max");
    }
    // Vocabulary size: the authoritative source is the token array itself (its
    // length is the vocabulary size). Fall back to the declared vocab_size, then
    // to the token-embedding tensor shape -- never to a hard-coded constant.
    if (auto arr = get_str_array("tokenizer.ggml.tokens"); arr && !arr->empty()) {
        f.n_vocab = int64_t(arr->size());
        f.has_tokenizer = true;
    } else {
        const std::string prefix = f.architecture.empty() ? std::string() : f.architecture + ".";
        if (auto nv = get_u64(prefix + "vocab_size"); nv) {
            f.n_vocab = int64_t(*nv);
        } else {
            for (const auto& t : tensors) {
                if (t.name != "token_embd.weight" || t.dimensions.size() != 2) continue;
                // GGUF dimensions are stored fastest-varying last; the embedding
                // matrix is [n_embd, n_vocab] for LLaMA-style models.
                f.n_vocab = int64_t(std::max(t.dimensions[0], t.dimensions[1]));
                break;
            }
        }
    }
    uint64_t bytes = 0;
    for (const auto& t : tensors) {
        f.param_count += int64_t(t.n_elements);
        bytes += t.n_bytes;
    }
    f.weight_bytes = bytes;
    f.bpw = f.param_count > 0 ? double(bytes) * 8.0 / double(f.param_count) : 0.0;

    // A complete LLaMA-style decoder block needs at least these tensors.
    if (!f.architecture.empty() && f.n_layer > 0) {
        int have = 0;
        auto has = [&](const std::string& n) {
            for (const auto& t : tensors) if (t.name == n) return true;
            return false;
        };
        if (has("token_embd.weight")) have++;
        if (has("output_norm.weight")) have++;
        if (has("blk.0.attn_q.weight")) have++;
        if (has("blk.0.attn_k.weight")) have++;
        if (has("blk.0.attn_v.weight")) have++;
        if (has("blk.0.attn_output.weight")) have++;
        if (has("blk.0.ffn_gate.weight")) have++;
        if (has("blk.0.ffn_down.weight")) have++;
        if (has("blk.0.ffn_up.weight")) have++;
        f.complete_llama_block = (have == 9);
    }
    return f;
}

std::string ModelInfo::summary() const {
    Facts f = facts();
    std::ostringstream os;
    os << "GGUF v" << version << " | arch=" << (f.architecture.empty() ? "?" : f.architecture)
       << " layers=" << f.n_layer << " embd=" << f.n_embd << " heads=" << f.n_head
       << " vocab=" << f.n_vocab << " tensors=" << n_tensors;
    os << " | params=" << f.param_count << " weight_bytes=" << f.weight_bytes
       << " bpw=" << std::fixed;
    os.precision(3);
    os << f.bpw;
    return os.str();
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------
void Writer::set_metadata(std::string key, Value v) {
    for (auto& kv : meta_) {
        if (kv.first == key) { kv.second = std::move(v); return; }
    }
    meta_.emplace_back(std::move(key), std::move(v));
}
void Writer::set_u32(std::string key, uint32_t v) { Value x; x.type = ValueType::UINT32; x.data = uint64_t(v); set_metadata(std::move(key), std::move(x)); }
void Writer::set_u64(std::string key, uint64_t v) { Value x; x.type = ValueType::UINT64; x.data = v; set_metadata(std::move(key), std::move(x)); }
void Writer::set_i32(std::string key, int32_t v) { Value x; x.type = ValueType::INT32; x.data = int64_t(v); set_metadata(std::move(key), std::move(x)); }
void Writer::set_f32(std::string key, float v) { Value x; x.type = ValueType::FLOAT32; x.data = double(v); set_metadata(std::move(key), std::move(x)); }
void Writer::set_str(std::string key, std::string v) { Value x; x.type = ValueType::STRING; x.data = std::move(v); set_metadata(std::move(key), std::move(x)); }
void Writer::set_bool(std::string key, bool v) { Value x; x.type = ValueType::BOOL; x.data = v; set_metadata(std::move(key), std::move(x)); }

void Writer::set_str_array(std::string key, const std::vector<std::string>& v) {
    Value x; x.type = ValueType::ARRAY; x.array_element_type = ValueType::STRING;
    ArrayValue arr;
    arr.reserve(v.size());
    for (const auto& s : v) { Value e; e.type = ValueType::STRING; e.data = s; arr.push_back(std::move(e)); }
    x.data = std::move(arr);
    set_metadata(std::move(key), std::move(x));
}
void Writer::set_f32_array(std::string key, const std::vector<float>& v) {
    Value x; x.type = ValueType::ARRAY; x.array_element_type = ValueType::FLOAT32;
    ArrayValue arr;
    arr.reserve(v.size());
    for (float s : v) { Value e; e.type = ValueType::FLOAT32; e.data = double(s); arr.push_back(std::move(e)); }
    x.data = std::move(arr);
    set_metadata(std::move(key), std::move(x));
}
void Writer::set_i32_array(std::string key, const std::vector<int32_t>& v) {
    Value x; x.type = ValueType::ARRAY; x.array_element_type = ValueType::INT32;
    ArrayValue arr;
    arr.reserve(v.size());
    for (int32_t s : v) { Value e; e.type = ValueType::INT32; e.data = int64_t(s); arr.push_back(std::move(e)); }
    x.data = std::move(arr);
    set_metadata(std::move(key), std::move(x));
}

void Writer::add_tensor(std::string name, std::vector<uint64_t> dims, GgmlType type,
                        std::vector<uint8_t> payload) {
    Pending p;
    p.name = std::move(name);
    p.dims = std::move(dims);
    p.type = type;
    p.payload = std::move(payload);
    tensors_.push_back(std::move(p));
}

static void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t((v >> (8 * i)) & 0xFF));
}
static void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(uint8_t((v >> (8 * i)) & 0xFF));
}
static void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

static void write_value(std::vector<uint8_t>& b, const Value& v) {
    put_u32(b, uint32_t(v.type));
    switch (v.type) {
        case ValueType::UINT8:   b.push_back(uint8_t(v.as_u64())); break;
        case ValueType::INT8:    b.push_back(uint8_t(uint8_t(v.as_i64()))); break;
        case ValueType::UINT16:  { uint16_t x = uint16_t(v.as_u64()); b.push_back(uint8_t(x & 0xFF)); b.push_back(uint8_t(x >> 8)); break; }
        case ValueType::INT16:   { int16_t x = int16_t(v.as_i64()); uint16_t u = uint16_t(x); b.push_back(uint8_t(u & 0xFF)); b.push_back(uint8_t(u >> 8)); break; }
        case ValueType::UINT32:  put_u32(b, uint32_t(v.as_u64())); break;
        case ValueType::INT32:   put_u32(b, uint32_t(int32_t(v.as_i64()))); break;
        case ValueType::FLOAT32: { float f = float(v.as_f64()); uint32_t u; std::memcpy(&u, &f, 4); put_u32(b, u); break; }
        case ValueType::BOOL:    b.push_back(v.as_bool() ? 1 : 0); break;
        case ValueType::STRING:  put_str(b, v.as_string()); break;
        case ValueType::UINT64:  put_u64(b, v.as_u64()); break;
        case ValueType::INT64:   put_u64(b, uint64_t(v.as_i64())); break;
        case ValueType::FLOAT64: { double d = v.as_f64(); uint64_t u; std::memcpy(&u, &d, 8); put_u64(b, u); break; }
        case ValueType::ARRAY: {
            const auto& a = v.as_array();
            uint32_t et = a.empty() ? uint32_t(v.array_element_type) : uint32_t(a[0].type);
            put_u32(b, et);
            put_u64(b, a.size());
            for (const auto& e : a) {
                // element type is implied by the array header: write payload only
                Value tmp = e;
                switch (ValueType(et)) {
                    case ValueType::STRING: put_str(b, e.as_string()); break;
                    case ValueType::FLOAT32: { float f = float(e.as_f64()); uint32_t u; std::memcpy(&u, &f, 4); put_u32(b, u); break; }
                    case ValueType::INT32:  put_u32(b, uint32_t(int32_t(e.as_i64()))); break;
                    case ValueType::UINT32: put_u32(b, uint32_t(e.as_u64())); break;
                    case ValueType::UINT64: put_u64(b, e.as_u64()); break;
                    case ValueType::INT64:  put_u64(b, uint64_t(e.as_i64())); break;
                    case ValueType::FLOAT64: { double d = e.as_f64(); uint64_t u; std::memcpy(&u, &d, 8); put_u64(b, u); break; }
                    case ValueType::BOOL:   b.push_back(e.as_bool() ? 1 : 0); break;
                    default: throw OctError("Writer: unsupported array element type");
                }
                (void)tmp;
            }
            break;
        }
        default: throw OctError("Writer: unsupported metadata type");
    }
}

std::vector<uint8_t> Writer::encode() const {
    std::vector<uint8_t> b;
    put_u32(b, 0x46554747u);   // "GGUF"
    put_u32(b, 3);             // version
    put_u64(b, tensors_.size());
    put_u64(b, meta_.size());
    for (const auto& kv : meta_) {
        put_str(b, kv.first);
        write_value(b, kv.second);
    }
    // Tensor directory (offsets are placeholders, fixed below).
    uint64_t running = 0;
    std::vector<uint64_t> offsets(tensors_.size(), 0);
    for (size_t i = 0; i < tensors_.size(); ++i) {
        const auto& t = tensors_[i];
        offsets[i] = running;
        put_str(b, t.name);
        put_u32(b, uint32_t(t.dims.size()));
        for (uint64_t d : t.dims) put_u64(b, d);
        put_u32(b, uint32_t(t.type));
        put_u64(b, running);
        uint64_t elems = 1;
        for (uint64_t d : t.dims) elems *= d;
        uint64_t nbytes = tensor_payload_bytes(t.type, elems);
        if (nbytes != t.payload.size()) {
            throw OctError("Writer: payload size mismatch for tensor " + t.name + " (expected " +
                           std::to_string(nbytes) + ", got " + std::to_string(t.payload.size()) + ")");
        }
        uint64_t aligned = (nbytes + alignment_ - 1) / alignment_ * alignment_;
        running += aligned;
    }
    // Pad header so the data section starts on an alignment boundary.
    while (b.size() % alignment_ != 0) b.push_back(0);
    data_start_ = b.size();
    for (const auto& t : tensors_) {
        b.insert(b.end(), t.payload.begin(), t.payload.end());
        while (b.size() % alignment_ != 0) b.push_back(0);
    }
    return b;
}

Status Writer::write_file(const std::string& path) const {
    try {
        auto bytes = encode();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return Status::internal("cannot write " + path);
        f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        if (!f) return Status::internal("write failed for " + path);
        return Status::ok(std::to_string(bytes.size()) + " bytes");
    } catch (const std::exception& e) {
        return Status::internal(e.what());
    }
}

}  // namespace oct::gguf
