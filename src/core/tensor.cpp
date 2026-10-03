// Octopus Hybrid AI Engine -- tensor implementation.
// SPDX-License-Identifier: MIT
#include "octopus/tensor.hpp"

#include <numeric>
#include <algorithm>
#include <sstream>

namespace oct {

const char* dtype_name(DType t) noexcept {
    switch (t) {
        case DType::F32:  return "f32";
        case DType::F64:  return "f64";
        case DType::I32:  return "i32";
        case DType::I64:  return "i64";
        case DType::U8:   return "u8";
        case DType::BOOL: return "bool";
    }
    return "?";
}

size_t dtype_size(DType t) noexcept {
    switch (t) {
        case DType::F32:  return 4;
        case DType::F64:  return 8;
        case DType::I32:  return 4;
        case DType::I64:  return 8;
        case DType::U8:   return 1;
        case DType::BOOL: return 1;
    }
    return 0;
}

size_t ShapeSpec::element_count() const {
    size_t n = 1;
    for (int64_t d : dims) {
        if (d < 0) throw OctError("ShapeSpec: negative dimension");
        n *= size_t(d);
    }
    return n;
}

std::string ShapeSpec::to_string() const {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(dims[i]);
    }
    return s + "]";
}

static Index normalize_shape(const Index& shape, size_t n) {
    // A single unknown dimension (-1) is allowed and inferred.
    Index out = shape;
    if (out.empty()) return {};  // scalar / vector with implicit shape
    int64_t unknown = -1;
    size_t known = 1;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] < 0) {
            if (unknown >= 0) throw OctError("Array: more than one unknown dimension");
            unknown = int64_t(i);
        } else {
            known *= size_t(out[i]);
        }
    }
    if (unknown >= 0) {
        if (known == 0 || n % known != 0) throw OctError("Array: cannot infer shape");
        out[size_t(unknown)] = int64_t(n / known);
    }
    size_t total = 1;
    for (int64_t d : out) total *= size_t(d);
    if (total != n) {
        throw OctError("Array: shape does not match element count (" +
                       std::to_string(total) + " vs " + std::to_string(n) + ")");
    }
    return out;
}

Array Array::zeros(DType t, Index shape) {
    ShapeSpec spec{shape};
    size_t n = spec.element_count();
    Array a;
    a.dtype_ = t;
    a.n_ = n;
    switch (t) {
        case DType::F32:  a.storage_ = std::vector<float>(n, 0.0f); break;
        case DType::F64:  a.storage_ = std::vector<double>(n, 0.0); break;
        case DType::I32:  a.storage_ = std::vector<int32_t>(n, 0); break;
        case DType::I64:  a.storage_ = std::vector<int64_t>(n, 0); break;
        case DType::U8:
        case DType::BOOL: a.storage_ = std::vector<uint8_t>(n, 0); break;
    }
    a.shape_ = normalize_shape(shape, n);
    return a;
}

#define OCT_ARRAY_FACTORY(fn, ctype, vec, dt)                                   \
    Array Array::fn(std::vector<ctype> v, Index shape) {                         \
        Array a;                                                                 \
        size_t n = v.size();                                                     \
        a.dtype_ = dt;                                                           \
        a.n_ = n;                                                                \
        a.shape_ = normalize_shape(shape, n);                                    \
        a.storage_ = std::move(v);                                               \
        (void)sizeof(vec);                                                       \
        return a;                                                                \
    }

OCT_ARRAY_FACTORY(from_f64, double,  std::vector<double>,  DType::F64)
OCT_ARRAY_FACTORY(from_f32, float,   std::vector<float>,   DType::F32)
OCT_ARRAY_FACTORY(from_i64, int64_t, std::vector<int64_t>, DType::I64)
OCT_ARRAY_FACTORY(from_u8,  uint8_t, std::vector<uint8_t>, DType::U8)
#undef OCT_ARRAY_FACTORY

const std::vector<double>& Array::as_f64() const {
    if (dtype_ != DType::F64) throw OctError("Array::as_f64: dtype is " + std::string(dtype_name(dtype_)));
    return std::get<std::vector<double>>(storage_);
}
const std::vector<float>& Array::as_f32() const {
    if (dtype_ != DType::F32) throw OctError("Array::as_f32: dtype is " + std::string(dtype_name(dtype_)));
    return std::get<std::vector<float>>(storage_);
}
const std::vector<int64_t>& Array::as_i64() const {
    if (dtype_ != DType::I64) throw OctError("Array::as_i64: dtype is " + std::string(dtype_name(dtype_)));
    return std::get<std::vector<int64_t>>(storage_);
}
const std::vector<uint8_t>& Array::as_u8() const {
    if (dtype_ != DType::U8 && dtype_ != DType::BOOL)
        throw OctError("Array::as_u8: dtype is " + std::string(dtype_name(dtype_)));
    return std::get<std::vector<uint8_t>>(storage_);
}
std::vector<double>& Array::mut_f64() {
    if (dtype_ != DType::F64) throw OctError("Array::mut_f64: dtype is " + std::string(dtype_name(dtype_)));
    return std::get<std::vector<double>>(storage_);
}

double Array::scalar_f64(size_t i) const {
    if (i >= n_) throw OctError("Array::scalar_f64: index out of range");
    switch (dtype_) {
        case DType::F64: return std::get<std::vector<double>>(storage_)[i];
        case DType::F32: return double(std::get<std::vector<float>>(storage_)[i]);
        case DType::I32: return double(std::get<std::vector<int32_t>>(storage_)[i]);
        case DType::I64: return double(std::get<std::vector<int64_t>>(storage_)[i]);
        case DType::U8:
        case DType::BOOL: return double(std::get<std::vector<uint8_t>>(storage_)[i]);
    }
    return 0.0;
}

int64_t Array::scalar_i64(size_t i) const {
    if (i >= n_) throw OctError("Array::scalar_i64: index out of range");
    switch (dtype_) {
        case DType::F64: return int64_t(std::get<std::vector<double>>(storage_)[i]);
        case DType::F32: return int64_t(std::get<std::vector<float>>(storage_)[i]);
        case DType::I32: return int64_t(std::get<std::vector<int32_t>>(storage_)[i]);
        case DType::I64: return std::get<std::vector<int64_t>>(storage_)[i];
        case DType::U8:
        case DType::BOOL: return int64_t(std::get<std::vector<uint8_t>>(storage_)[i]);
    }
    return 0;
}

Array Array::cast(DType t) const {
    if (t == dtype_) return *this;
    Array out;
    out.dtype_ = t;
    out.n_ = n_;
    out.shape_ = shape_;
    switch (t) {
        case DType::F64: {
            std::vector<double> v(n_);
            for (size_t i = 0; i < n_; ++i) v[i] = scalar_f64(i);
            out.storage_ = std::move(v);
            break;
        }
        case DType::F32: {
            std::vector<float> v(n_);
            for (size_t i = 0; i < n_; ++i) v[i] = float(scalar_f64(i));
            out.storage_ = std::move(v);
            break;
        }
        case DType::I64: {
            std::vector<int64_t> v(n_);
            for (size_t i = 0; i < n_; ++i) v[i] = scalar_i64(i);
            out.storage_ = std::move(v);
            break;
        }
        case DType::I32: {
            std::vector<int32_t> v(n_);
            for (size_t i = 0; i < n_; ++i) v[i] = int32_t(scalar_i64(i));
            out.storage_ = std::move(v);
            break;
        }
        case DType::U8:
        case DType::BOOL: {
            std::vector<uint8_t> v(n_);
            for (size_t i = 0; i < n_; ++i) v[i] = uint8_t(scalar_i64(i));
            out.storage_ = std::move(v);
            break;
        }
    }
    return out;
}

Array Array::reshape(Index new_shape) const {
    Array a = *this;
    a.shape_ = normalize_shape(new_shape, n_);
    return a;
}

std::string Array::fingerprint() const {
    hash::Sha256 h;
    h.update(dtype_name(dtype_), std::strlen(dtype_name(dtype_)));
    for (int64_t d : shape_) h.update(&d, sizeof(d));
    if (auto* p = std::get_if<std::vector<double>>(&storage_))  h.update(p->data(), p->size() * sizeof(double));
    if (auto* p = std::get_if<std::vector<float>>(&storage_))   h.update(p->data(), p->size() * sizeof(float));
    if (auto* p = std::get_if<std::vector<int32_t>>(&storage_)) h.update(p->data(), p->size() * sizeof(int32_t));
    if (auto* p = std::get_if<std::vector<int64_t>>(&storage_)) h.update(p->data(), p->size() * sizeof(int64_t));
    if (auto* p = std::get_if<std::vector<uint8_t>>(&storage_)) h.update(p->data(), p->size());
    return h.finalize().hex();
}

std::string Array::describe() const {
    std::ostringstream os;
    os << dtype_name(dtype_) << ShapeSpec{shape_}.to_string() << " (" << n_ << " elems, " << bytes() << " B)";
    return os.str();
}

namespace reduce {

double sum(const std::vector<double>& v) {
    // Kahan-Neumaier compensated summation: order-stable, no drift.
    double s = 0.0, c = 0.0;
    for (double x : v) {
        double t = s + x;
        c += (std::abs(s) >= std::abs(x)) ? ((s - t) + x) : ((x - t) + s);
        s = t;
    }
    return s + c;
}

double mean(const std::vector<double>& v) { return v.empty() ? 0.0 : sum(v) / double(v.size()); }

double min(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double m = v[0];
    for (double x : v) m = std::min(m, x);
    return m;
}

double max(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double m = v[0];
    for (double x : v) m = std::max(m, x);
    return m;
}

double l2_norm(const std::vector<double>& v) {
    double s = 0.0, c = 0.0;
    for (double x : v) {
        double sq = x * x;
        double t = s + sq;
        c += (std::abs(s) >= std::abs(sq)) ? ((s - t) + sq) : ((sq - t) + s);
        s = t;
    }
    return std::sqrt(s + c);
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw OctError("max_abs_diff: size mismatch");
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

double rel_l2(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw OctError("rel_l2: size mismatch");
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    if (den == 0.0) return std::sqrt(num);
    return std::sqrt(num / den);
}

bool all_finite(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace reduce

}  // namespace oct
