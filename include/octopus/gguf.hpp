// Octopus Hybrid AI Engine -- GGUF container reader/writer.
//
// Real, dependency-free implementation of the GGUF v3 container used by
// llama.cpp. The engine reads *all* model dimensions from the file (never from
// hard-coded constants) and can also write minimal valid containers, which is
// how the offline CI fixture model is produced (tools/ggufgen).
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <map>
#include <span>
#include <variant>

namespace oct::gguf {

enum class ValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12,
};

const char* value_type_name(ValueType t) noexcept;
size_t      value_type_size(ValueType t) noexcept;

struct Value;
using ArrayValue = std::vector<Value>;

struct Value {
    ValueType type = ValueType::UINT32;
    std::variant<uint64_t, int64_t, double, bool, std::string, ArrayValue> data;

    bool        is_array() const { return type == ValueType::ARRAY; }
    uint64_t    as_u64()   const;
    int64_t     as_i64()   const;
    double      as_f64()   const;
    bool        as_bool()  const;
    std::string as_string() const;
    const ArrayValue& as_array() const;
    std::string to_display() const;
};

enum class GgmlType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8,
    Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20,
    IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27,
    F64 = 28, BF16 = 30, TQ1_0 = 34, TQ2_0 = 35, UNKNOWN = 0xFFFFFFFFu,
};

const char* ggml_type_name(GgmlType t) noexcept;
double      ggml_type_bits(GgmlType t) noexcept;   // bits per stored element
bool        ggml_type_is_quantized(GgmlType t) noexcept;

struct TensorInfo {
    std::string      name;
    std::vector<uint64_t> dimensions;   // GGUF order: fastest-varying last
    GgmlType         type = GgmlType::F32;
    uint64_t         offset = 0;        // relative to data section start
    uint64_t         n_elements = 0;    // product of dimensions
    uint64_t         n_bytes = 0;       // computed from type + element count
};

struct ModelInfo {
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    uint64_t alignment = 32;
    uint64_t data_section_offset = 0;
    std::map<std::string, Value> metadata;
    std::vector<TensorInfo> tensors;

    // --- typed metadata accessors (return nullopt when absent / wrong type) ---
    std::optional<uint64_t>    get_u64(std::string_view key) const;
    std::optional<double>      get_f64(std::string_view key) const;
    std::optional<std::string> get_str(std::string_view key) const;
    std::optional<std::vector<std::string>> get_str_array(std::string_view key) const;

    // --- derived model facts (all from the file; no hard-coded shapes) ------
    struct Facts {
        std::string architecture;
        std::string name;
        int64_t  n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0;
        int64_t  n_ff = 0, n_vocab = 0, n_ctx_train = 0, n_ctx_max = 0;
        int64_t  param_count = 0;      // element count summed over all tensors
        uint64_t weight_bytes = 0;     // bytes of tensor data in the file
        double   bpw = 0.0;            // bits per weight actually stored
        bool     has_tokenizer = false;
        bool     complete_llama_block = false;
    };
    Facts facts() const;
    std::string summary() const;
};

// Reads only header/metadata/tensor-table (no tensor payload). Cheap enough
// to be used for model metadata tests and for capability negotiation.
Outcome<ModelInfo> read_model_info(const std::string& path);
// Same, but from a memory buffer (used by tests with synthetic containers).
Outcome<ModelInfo> read_model_info_from_buffer(std::span<const uint8_t> buf);

// ---------------------------------------------------------------------------
// Writer (used to synthesise the tiny CI fixture model and for manifests)
// ---------------------------------------------------------------------------
class Writer {
public:
    explicit Writer(uint64_t alignment = 32) : alignment_(alignment) {}

    void set_metadata(std::string key, Value v);
    void set_u32(std::string key, uint32_t v);
    void set_u64(std::string key, uint64_t v);
    void set_i32(std::string key, int32_t v);
    void set_f32(std::string key, float v);
    void set_str(std::string key, std::string v);
    void set_bool(std::string key, bool v);
    void set_str_array(std::string key, const std::vector<std::string>& v);
    void set_f32_array(std::string key, const std::vector<float>& v);
    void set_i32_array(std::string key, const std::vector<int32_t>& v);

    // Tensor payloads are raw little-endian bytes in the declared ggml type.
    void add_tensor(std::string name, std::vector<uint64_t> dims, GgmlType type,
                    std::vector<uint8_t> payload);

    std::vector<uint8_t> encode() const;
    Status               write_file(const std::string& path) const;

private:
    uint64_t alignment_;
    std::vector<std::pair<std::string, Value>> meta_;   // insertion order
    struct Pending { std::string name; std::vector<uint64_t> dims; GgmlType type; std::vector<uint8_t> payload; };
    std::vector<Pending> tensors_;
    mutable uint64_t     data_start_ = 0;
};

}  // namespace oct::gguf
