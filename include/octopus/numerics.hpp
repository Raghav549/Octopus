// Octopus Hybrid AI Engine -- Fortran 2023 Absolute Physics Core interface.
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// Every numerical kernel (fluid dynamics, Kepler/N-body gravity, elliptic &
// parabolic PDEs, thermodynamics, linear systems, covariant tensor matrix
// fields) is bound strictly to the Fortran 2023 ISO_C_BINDING execution core
// (src/fortran/octopus_kernels.f90). C++ reference logic replacements are
// strictly banned. When a Fortran compiler (gfortran/lfortran) is absent on
// the compilation host, live execution blocks report UNSUPPORTED_HARDWARE_SKIP
// rather than downscaling to standard C++ routines.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"
#include "octopus/tensor.hpp"

#include <map>
#include <memory>

namespace oct::numerics {

// Only the native Fortran 2023 backend is permitted; C++ fallback is banned.
enum class Backend : uint8_t { Fortran = 1 };

const char* backend_name(Backend b) noexcept;

struct ProblemSpec {
    std::string                         kernel;     // "sod1d", "heat2d", ...
    std::map<std::string, double>       params;
    double                              tolerance = 1e-9;
    double                              t_end = 1.0;
    int64_t                             steps = 0;      // 0 => solver default
    std::string                         method;         // "" => solver default
    Backend                             backend = Backend::Fortran;
    bool                                collect_history = false;

    double get(std::string_view k, double dflt) const {
        auto it = params.find(std::string(k));
        return it == params.end() ? dflt : it->second;
    }
    int64_t get_i(std::string_view k, int64_t dflt) const {
        auto it = params.find(std::string(k));
        return it == params.end() ? dflt : int64_t(it->second);
    }
};

struct Diagnostics {
    double residual = 0.0;             // max |discrete residual|
    double conservation_error = 0.0;   // max relative drift of invariants
    double measured_order = 0.0;       // spatial/temporal order vs refined grid
    int64_t iterations = 0;
    double  wall_seconds = 0.0;
    double  cells_per_second = 0.0;
    std::vector<std::pair<std::string, double>> metrics;   // named extra diagnostics

    void set(std::string name, double v) { metrics.emplace_back(std::move(name), v); }
    double get(std::string_view n, double dflt = 0.0) const;
};

struct Result {
    std::string              kernel;
    std::string              method;
    std::string              backend;        // human-readable, honest
    std::string              units;
    std::vector<int64_t>     shape;
    std::vector<double>      data;           // flat, last index fastest
    std::vector<double>      x, y;           // optional physical grids
    std::vector<double>      history;        // optional: state snapshots (flat)
    int64_t                  history_stride = 0;
    Diagnostics              diag;
    ProblemSpec              spec;
    std::string              fingerprint;    // SHA-256 over spec+method+backend
    bool                     hardware_skipped = false;
    Status                   status = Status::ok();

    Json   to_json(bool include_data = false) const;
    Array  as_array() const;
};

struct Validation {
    bool        reference_ok = false;
    bool        tol_ok = false;
    double      max_abs_error = 0.0;
    double      rel_l2_error = 0.0;
    double      tolerance = 0.0;
    std::string reference;
    bool        conservation_ok = false;
    double      conservation_error = 0.0;
    bool        order_ok = false;
    double      measured_order = 0.0;
    double      expected_order = 0.0;
    bool        divergence_ok = true;
    double      max_divergence = 0.0;
    bool        hardware_skipped = false;
    std::string notes;
    std::vector<std::pair<std::string, double>> metrics;

    void set_metric(std::string name, double v) { metrics.emplace_back(std::move(name), v); }
    double metric(std::string_view name, double dflt = 0.0) const {
        for (const auto& kv : metrics) if (kv.first == name) return kv.second;
        return dflt;
    }

    bool accepted() const { return !hardware_skipped && reference_ok && tol_ok && conservation_ok; }
    Json to_json() const;
};

class Kernel {
public:
    virtual ~Kernel() = default;
    virtual std::string name() const = 0;
    virtual std::string method() const = 0;
    virtual std::string units() const = 0;
    virtual std::vector<std::pair<std::string, std::string>> parameters() const = 0;
    virtual Result run(const ProblemSpec& spec) const = 0;
    virtual Validation validate(const ProblemSpec& spec) const = 0;
    virtual ProblemSpec validation_spec() const {
        ProblemSpec s;
        s.kernel = name();
        return s;
    }
};

class KernelLibrary {
public:
    static const KernelLibrary& instance();
    std::shared_ptr<const Kernel> find(std::string_view name) const;
    std::vector<std::string> names() const;
    std::vector<std::string> describe_all() const;

    bool        fortran_available() const;
    std::string fortran_compiler() const;
    std::string fortran_backend_id() const;
private:
    KernelLibrary();
    std::vector<std::shared_ptr<const Kernel>> kernels_;
};

// ---------------------------------------------------------------------------
// Fortran 2023 C ABI bridge (linked from octopus_kernels.f90 or fortran_hw_skip.cpp)
// ---------------------------------------------------------------------------
namespace fortran_bridge {
inline constexpr int kErrUnsupportedHardwareSkip = -99;
bool available();
const char* compiler_id();
Status skip_status(std::string_view kernel_name);
}  // namespace fortran_bridge

// ---------------------------------------------------------------------------
// Linear algebra utilities (used by APL ⌹ domino matrix division).
// ---------------------------------------------------------------------------
namespace linalg {
Outcome<std::vector<double>> solve(std::vector<double> A, std::vector<double> b, int64_t n,
                                   double* residual_out = nullptr);
Outcome<std::vector<double>> cg(const std::vector<double>& A, const std::vector<double>& b,
                                int64_t n, double tol, int64_t max_iter,
                                int64_t* iters_out, double* residual_out);
std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B,
                           int64_t n, int64_t k, int64_t m);
double inf_norm(const std::vector<double>& v);
double condition_1norm(const std::vector<double>& A, int64_t n);
double condition_estimate(const std::vector<double>& A, int64_t n);
}  // namespace linalg

}  // namespace oct::numerics
