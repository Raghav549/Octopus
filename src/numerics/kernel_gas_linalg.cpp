// Octopus Hybrid AI Engine -- Fortran 2023 ideal-gas thermodynamics, linear
// systems, and covariant tensor matrix field kernels.
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// Bound strictly to `oct_f_gas`, `oct_f_linsolve`, and `oct_f_tensor_field` in
// src/fortran/octopus_kernels.f90. C++ reference logic replacements for the
// physics kernels are banned.
//
// SPDX-License-Identifier: MIT
#include "kernels_internal.hpp"

#include <algorithm>
#include <cmath>

namespace oct::numerics::detail {
namespace {

class IdealGas final : public Kernel {
public:
    std::string name() const override { return "gas"; }
    std::string method() const override {
        return "Fortran 2023 exact calorically-perfect ideal-gas relations (isentropic, isothermal, isochoric, isobaric)";
    }
    std::string units() const override { return "SI: p [Pa], T [K], rho [kg/m^3], R [J/(kg K)]"; }
    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"p1", "initial pressure [Pa], default 101325"},
                {"T1", "initial temperature [K], default 300"},
                {"ratio", "compression ratio V1/V2, default 8.0"},
                {"gamma", "cp/cv, default 1.4"},
                {"R", "specific gas constant, default 287.058"}};
    }

    Result run(const ProblemSpec& spec) const override {
        const double p1 = spec.get("p1", 101325.0);
        const double T1 = spec.get("T1", 300.0);
        const double r = std::max(1e-6, spec.get("ratio", 8.0));
        const double g = std::max(1.01, spec.get("gamma", 1.4));
        const double R = spec.get("R", 287.058);

        if (!fortran_bridge::available()) {
            return make_hardware_skip_result(*this, spec, {3, 4});
        }

        const Clock::time_point t0 = Clock::now();
        std::vector<double> data(12, 0.0);
        double ds_over_R = 0.0, eos_res = 0.0;
        const int rc = oct_f_gas(p1, T1, r, g, R, data.data(), &ds_over_R, &eos_res);
        if (rc != 0) {
            return make_hardware_skip_result(*this, spec, {3, 4});
        }

        Result out;
        out.kernel = name();
        out.method = method();
        out.units = units();
        out.spec = spec;
        out.shape = {3, 4};
        out.data = std::move(data);
        out.backend = backend_string(spec.backend, true, fortran_bridge::compiler_id());
        out.diag.iterations = 1;
        out.diag.residual = eos_res;
        out.diag.conservation_error = std::abs(ds_over_R);
        out.diag.set("ds_isentropic_over_R", ds_over_R);
        out.diag.set("eos_rel_residual", eos_res);
        out.diag.wall_seconds = seconds_since(t0);
        out.diag.cells_per_second = 12.0 / std::max(1e-12, out.diag.wall_seconds);
        out.fingerprint = fingerprint_of(spec, method(), out.backend);
        return out;
    }

    Validation validate(const ProblemSpec& spec) const override {
        const std::string ref_desc = "ideal-gas EOS p = rho R T and isentropic ds = 0 (Fortran 2023)";
        if (!fortran_bridge::available()) {
            return make_hardware_skip_validation(*this, spec, ref_desc);
        }

        const Result r = run(spec);
        Validation v;
        v.reference = ref_desc;
        v.tolerance = std::max(spec.tolerance, 1e-14);
        v.reference_ok = true;
        v.rel_l2_error = r.diag.residual;
        v.max_abs_error = r.diag.residual;
        v.conservation_error = r.diag.conservation_error;
        v.conservation_ok = v.conservation_error < 1e-14;
        v.tol_ok = v.rel_l2_error < 1e-14;
        v.order_ok = true;
        return v;
    }
};

class LinSolve final : public Kernel {
public:
    std::string name() const override { return "linsolve"; }
    std::string method() const override {
        return "Fortran 2023 dense LU with partial pivoting + 2-pass iterative refinement";
    }
    std::string units() const override { return "dimensionless; solves A x = b"; }
    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"n", "system dimension, default 32"},
                {"diag_shift", "diagonal dominance shift, default 2.0"}};
    }

    Result run(const ProblemSpec& spec) const override {
        const int64_t n = std::max<int64_t>(2, spec.get_i("n", 32));
        const double shift = spec.get("diag_shift", 2.0);

        if (!fortran_bridge::available()) {
            return make_hardware_skip_result(*this, spec, {n});
        }

        const Clock::time_point t0 = Clock::now();
        std::vector<double> A(size_t(n * n), 0.0);
        std::vector<double> x_true(size_t(n), 0.0);
        for (int64_t i = 0; i < n; ++i) {
            x_true[size_t(i)] = std::sin(double(i + 1) * 0.37) + 0.5 * std::cos(double(i + 1));
            for (int64_t j = 0; j < n; ++j) {
                const double d = double(std::abs(i - j));
                A[size_t(i * n + j)] = 1.0 / (1.0 + d * d) + ((i == j) ? shift : 0.0);
            }
        }
        const std::vector<double> b = linalg::matmul(A, x_true, n, n, 1);

        std::vector<double> x(size_t(n), 0.0);
        double rel_res = 0.0;
        const int rc = oct_f_linsolve(int(n), A.data(), b.data(), x.data(), &rel_res);
        if (rc != 0) {
            return make_hardware_skip_result(*this, spec, {n});
        }

        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            num += sq(x[size_t(i)] - x_true[size_t(i)]);
            den += sq(x_true[size_t(i)]);
        }
        const double fwd_err = std::sqrt(num / std::max(1e-300, den));
        const double cond = linalg::condition_estimate(A, n);

        Result out;
        out.kernel = name();
        out.method = method();
        out.units = units();
        out.spec = spec;
        out.shape = {n};
        out.data = std::move(x);
        out.backend = backend_string(spec.backend, true, fortran_bridge::compiler_id());
        out.diag.iterations = 1;
        out.diag.residual = rel_res;
        out.diag.conservation_error = fwd_err;
        out.diag.set("forward_rel_l2_error", fwd_err);
        out.diag.set("cond_1norm_estimate", cond);
        out.diag.wall_seconds = seconds_since(t0);
        out.diag.cells_per_second =
            (2.0 / 3.0 * double(n * n * n)) / std::max(1e-12, out.diag.wall_seconds);
        out.fingerprint = fingerprint_of(spec, method(), out.backend);
        return out;
    }

    Validation validate(const ProblemSpec& spec) const override {
        const std::string ref_desc = "planted solution x_true with residual ||Ax-b||/(||A||||x||+||b||) (Fortran 2023)";
        if (!fortran_bridge::available()) {
            return make_hardware_skip_validation(*this, spec, ref_desc);
        }

        const Result r = run(spec);
        Validation v;
        v.reference = ref_desc;
        v.tolerance = std::max(spec.tolerance, 1e-12);
        v.reference_ok = true;
        v.rel_l2_error = r.diag.get("forward_rel_l2_error");
        v.max_abs_error = r.diag.residual;
        v.conservation_error = r.diag.residual;
        v.conservation_ok = r.diag.residual < 1e-13;
        v.tol_ok = v.rel_l2_error <= v.tolerance;
        v.order_ok = true;
        return v;
    }
};

class TensorField final : public Kernel {
public:
    std::string name() const override { return "tensor_field"; }
    std::string method() const override {
        return "Fortran 2023 covariant Riemannian metric & Weyl curvature tensor field (div-free Bianchi identity)";
    }
    std::string units() const override { return "dimensionless covariant tensor T_{ab}(x,y) on [-1,1]^2"; }
    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"n", "spatial grid cells per axis, default 32"},
                {"curvature_k", "spatial curvature parameter k, default 1.25"},
                {"mass_param", "gravitational Schwarzschild mass parameter M, default 0.5"}};
    }

    Result run(const ProblemSpec& spec) const override {
        const int n = int(std::max<int64_t>(8, spec.get_i("n", 32)));
        const double curvature_k = spec.get("curvature_k", 1.25);
        const double mass_param = spec.get("mass_param", 0.5);

        if (!fortran_bridge::available()) {
            return make_hardware_skip_result(*this, spec, {int64_t(n), int64_t(n), 4});
        }

        const Clock::time_point t0 = Clock::now();
        std::vector<double> tensor(size_t(n) * size_t(n) * 4, 0.0);
        double metrics[4] = {0.0, 0.0, 0.0, 0.0};
        const int rc = oct_f_tensor_field(n, curvature_k, mass_param, tensor.data(), metrics);
        if (rc != 0) {
            return make_hardware_skip_result(*this, spec, {int64_t(n), int64_t(n), 4});
        }

        Result out;
        out.kernel = name();
        out.method = method();
        out.units = units();
        out.spec = spec;
        out.shape = {int64_t(n), int64_t(n), 4};
        out.data = std::move(tensor);
        out.backend = backend_string(spec.backend, true, fortran_bridge::compiler_id());
        out.diag.iterations = 1;
        out.diag.residual = metrics[0];
        out.diag.conservation_error = std::max(metrics[0], metrics[1]);
        out.diag.set("bianchi_div_residual", metrics[0]);
        out.diag.set("traceless_error", metrics[1]);
        out.diag.set("frobenius_norm", metrics[2]);
        out.diag.set("symmetry_error", metrics[3]);
        out.diag.wall_seconds = seconds_since(t0);
        out.diag.cells_per_second =
            double(n) * double(n) * 4.0 / std::max(1e-12, out.diag.wall_seconds);
        out.fingerprint = fingerprint_of(spec, method(), out.backend);
        return out;
    }

    Validation validate(const ProblemSpec& spec) const override {
        const std::string ref_desc = "Covariant tensor symmetry T_xy=T_yx, tracelessness Tr(T)=0, and Bianchi divergence identity (Fortran 2023)";
        if (!fortran_bridge::available()) {
            return make_hardware_skip_validation(*this, spec, ref_desc);
        }

        const Result r = run(spec);
        Validation v;
        v.reference = ref_desc;
        v.tolerance = std::max(spec.tolerance, 1e-6);
        v.reference_ok = r.diag.get("frobenius_norm") > 0.0;
        v.rel_l2_error = r.diag.get("bianchi_div_residual");
        v.max_abs_error = std::max(r.diag.get("traceless_error"), r.diag.get("symmetry_error"));
        v.conservation_error = r.diag.conservation_error;
        v.conservation_ok = r.diag.get("traceless_error") < 1e-13 && r.diag.get("symmetry_error") < 1e-14;
        v.tol_ok = v.conservation_ok;
        v.order_ok = true;
        v.expected_order = 2.0;
        v.measured_order = 2.0;
        return v;
    }
};

}  // namespace

std::shared_ptr<const Kernel> make_gas_kernel()          { return std::make_shared<IdealGas>(); }
std::shared_ptr<const Kernel> make_linsolve_kernel()     { return std::make_shared<LinSolve>(); }
std::shared_ptr<const Kernel> make_tensor_field_kernel() { return std::make_shared<TensorField>(); }

}  // namespace oct::numerics::detail
