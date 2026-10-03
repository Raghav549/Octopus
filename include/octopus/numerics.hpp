// Octopus Hybrid AI Engine -- numerical kernel layer.
//
// Contract (docs/NUMERICS.md): every kernel declares its method, units,
// parameters, tolerances and validation reference; every run returns
// diagnostics (residual / conservation error) alongside the data. Kernels are
// pure: same spec + same backend => bitwise-stable results on a given machine.
//
// The C++ reference implementation is always present. The Fortran 2018 backend
// (src/fortran/octopus_kernels.f90) is used when a compiler was available at
// configure time; the CLI reports which backend answered, and never pretends
// Fortran ran when it did not.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"
#include "octopus/tensor.hpp"

#include <map>
#include <memory>

namespace oct::numerics {

enum class Backend : uint8_t { CxxLd = 0, Fortran = 1 };

const char* backend_name(Backend b) noexcept;

struct ProblemSpec {
    std::string                         kernel;     // "sod1d", "heat2d", ...
    std::map<std::string, double>       params;
    double                              tolerance = 1e-9;
    double                              t_end = 1.0;
    int64_t                             steps = 0;      // 0 => solver default
    std::string                         method;         // "" => solver default
    Backend                             backend = Backend::CxxLd;
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

    Json   to_json(bool include_data = false) const;
    Array  as_array() const;
};

// Validation is the only thing allowed to say "correct": it compares against an
// independently-derived reference (analytic solution, manufactured solution, or
// a cross-backend run) and against the declared tolerance.
struct Validation {
    bool        reference_ok = false;
    bool        tol_ok = false;
    double      max_abs_error = 0.0;
    double      rel_l2_error = 0.0;
    double      tolerance = 0.0;
    std::string reference;                  // "analytic", "exact-riemann", ...
    bool        conservation_ok = false;
    double      conservation_error = 0.0;
    bool        order_ok = false;
    double      measured_order = 0.0;
    double      expected_order = 0.0;
    bool        divergence_ok = true;       // fluid kernels only
    double      max_divergence = 0.0;
    std::string notes;
    std::vector<std::pair<std::string, double>> metrics;   // extra evidence

    void set_metric(std::string name, double v) { metrics.emplace_back(std::move(name), v); }
    double metric(std::string_view name, double dflt = 0.0) const {
        for (const auto& kv : metrics) if (kv.first == name) return kv.second;
        return dflt;
    }

    bool accepted() const { return reference_ok && tol_ok && conservation_ok; }
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
    // Canonical parameters (including the tolerance) under which this kernel's
    // validation case is meaningful. An empty spec means "kernel defaults";
    // callers that need a self-contained check use this instead of guessing.
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
// Fortran bridge (raw C ABI exported by the Fortran module; see the .f90 file).
// Returns false when the Fortran backend is not linked in.
// ---------------------------------------------------------------------------
namespace fortran_bridge {
bool available();
const char* compiler_id();
extern "C" {
// Each returns 0 on success, non-zero on Fortran-side error.
}
}  // namespace fortran_bridge

// ---------------------------------------------------------------------------
// Linear algebra helpers (used by APL ⌹, Least squares, covariance, PCA).
// ---------------------------------------------------------------------------
namespace linalg {
// Solve A x = b for square A (n x n, row-major) via LU with partial pivoting.
// Returns residual ||Ax-b||_inf / (||A||_inf ||x||_inf + tiny).
Outcome<std::vector<double>> solve(std::vector<double> A, std::vector<double> b, int64_t n,
                                   double* residual_out = nullptr);
// Conjugate gradient for SPD systems; reports iterations and true residual.
Outcome<std::vector<double>> cg(const std::vector<double>& A, const std::vector<double>& b,
                                int64_t n, double tol, int64_t max_iter,
                                int64_t* iters_out, double* residual_out);
std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B,
                           int64_t n, int64_t k, int64_t m);
double inf_norm(const std::vector<double>& v);
// Exact 1-norm condition number via n LU back-substitutions (O(n^3)).
double condition_1norm(const std::vector<double>& A, int64_t n);
// Cheap 1-norm condition *estimate* (Hager/Higham style); prefer the exact one
// whenever the matrix is small, and always report which was used.
double condition_estimate(const std::vector<double>& A, int64_t n);
}  // namespace linalg

}  // namespace oct::numerics
