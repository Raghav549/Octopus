// Octopus Hybrid AI Engine -- dense N-d tensor / array primitives.
// Shared by the APL-style array IR, the numerics kernels and the Piet canvas.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <array>
#include <span>
#include <variant>

namespace oct {

enum class DType : uint8_t { F32 = 0, F64, I32, I64, U8, BOOL };

const char* dtype_name(DType t) noexcept;
size_t      dtype_size(DType t) noexcept;

using Index = std::vector<int64_t>;

struct ShapeSpec {
    Index dims;
    size_t element_count() const;
    std::string to_string() const;
    bool operator==(const ShapeSpec& o) const { return dims == o.dims; }
};

// A dense array with an explicit dtype. Storage is one of the variants; all
// transformations must preserve or explicitly declare shape + dtype semantics
// (see docs/DESIGN.md, "array contract").
class Array {
public:
    using Storage = std::variant<std::vector<float>, std::vector<double>,
                                 std::vector<int32_t>, std::vector<int64_t>,
                                 std::vector<uint8_t>>;

    Array() = default;
    static Array zeros(DType t, Index shape);
    static Array from_f64(std::vector<double> v, Index shape = {});
    static Array from_f32(std::vector<float> v, Index shape = {});
    static Array from_i64(std::vector<int64_t> v, Index shape = {});
    static Array from_u8(std::vector<uint8_t> v, Index shape = {});

    DType dtype() const noexcept { return dtype_; }
    const Index& shape() const noexcept { return shape_; }
    size_t size() const noexcept { return n_; }
    size_t bytes() const noexcept { return n_ * dtype_size(dtype_); }
    bool   empty() const noexcept { return n_ == 0; }

    // Strictly-typed, checked accessors (throw OctError on dtype mismatch).
    const std::vector<double>&  as_f64() const;
    const std::vector<float>&   as_f32() const;
    const std::vector<int64_t>& as_i64() const;
    const std::vector<uint8_t>& as_u8()  const;
    std::vector<double>&        mut_f64();

    double  scalar_f64(size_t i = 0) const;
    int64_t scalar_i64(size_t i = 0) const;

    // Conversions (explicit; never implicit in the IR).
    Array cast(DType t) const;
    Array reshape(Index new_shape) const;   // element count must match

    std::string fingerprint() const;        // SHA-256 over dtype+shape+payload
    std::string describe() const;

private:
    Storage storage_;
    DType   dtype_ = DType::F64;
    Index   shape_{};
    size_t  n_ = 0;
};

// Basic vector reduction helpers shared by kernels (deterministic order).
namespace reduce {
double sum(const std::vector<double>& v);
double mean(const std::vector<double>& v);
double min(const std::vector<double>& v);
double max(const std::vector<double>& v);
double l2_norm(const std::vector<double>& v);
// Maximum absolute difference (sup-norm) -- the workhorse of kernel validation.
double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b);
// Relative L2 error, normalised by the reference norm (0 when ref is all-zero).
double rel_l2(const std::vector<double>& a, const std::vector<double>& b);
bool   all_finite(const std::vector<double>& v);
}  // namespace reduce

}  // namespace oct
