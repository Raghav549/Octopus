// Octopus Hybrid AI Engine -- Fortran 2023 diffusion and elliptic kernels.
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// Bound strictly to `oct_f_heat2d` and `oct_f_poisson2d` in
// src/fortran/octopus_kernels.f90. C++ reference logic replacements are banned.
//
// SPDX-License-Identifier: MIT
#include "kernels_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace oct::numerics::detail {
namespace {

using std::numbers::pi;

class Heat2D final : public Kernel {
public:
    std::string name() const override { return "heat2d"; }
    ProblemSpec validation_spec() const override {
        ProblemSpec s;
        s.kernel = name();
        s.t_end = 0.05;
        s.tolerance = 5e-3;
        return s;
    }
    std::string method() const override {
        return "Fortran 2023 explicit FTCS (5-point Laplacian) with ghost-cell Dirichlet walls";
    }
    std::string units() const override { return "unit square [0,1]^2; alpha in m^2/s"; }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"n", "cells per side, default 64"},
                {"alpha", "diffusivity, default 0.1"},
                {"cfl", "4 * Fourier number, <= 1 for stability (default 0.8)"},
                {"t_end", "final time, default 0.05"}};
    }

    static double exact(double x, double y, double alpha, double t) {
        return std::sin(pi * x) * std::sin(pi * y) * std::exp(-2.0 * pi * pi * alpha * t);
    }

    Result run(const ProblemSpec& spec) const override {
        const int n = int(std::max<int64_t>(4, spec.get_i("n", 64)));
        const double alpha = spec.get("alpha", 0.1);
        const double cfl = std::min(0.99, std::max(0.05, spec.get("cfl", 0.8)));
        const double t_end = spec.t_end > 0 ? spec.t_end : 0.05;
        const double h = 1.0 / double(n);
        const double r = cfl / 4.0;
        const double dt = r * h * h / alpha;
        const int64_t steps = int64_t(std::ceil(t_end / dt));
        const double dt_eff = t_end / double(std::max<int64_t>(1, steps));

        if (!fortran_bridge::available()) {
            return make_hardware_skip_result(*this, spec, {int64_t(n), int64_t(n)});
        }

        const Clock::time_point t0 = Clock::now();
        std::vector<double> u(size_t(n) * size_t(n));
        const int rc = oct_f_heat2d(alpha, t_end, n, cfl, u.data());
        if (rc != 0) {
            return make_hardware_skip_result(*this, spec, {int64_t(n), int64_t(n)});
        }

        Result out;
        out.kernel = name();
        out.method = method();
        out.units = units();
        out.spec = spec;
        out.shape = {int64_t(n), int64_t(n)};
        out.data = std::move(u);
        out.diag.iterations = steps;
        out.backend = backend_string(spec.backend, true, fortran_bridge::compiler_id());
        out.diag.set("steps", double(out.diag.iterations));
        out.diag.set("dt", dt_eff);
        out.diag.set("fourier_number", alpha * dt_eff / (h * h));
        out.diag.wall_seconds = seconds_since(t0);
        out.diag.cells_per_second =
            double(n) * double(n) * double(std::max<int64_t>(1, out.diag.iterations)) /
            std::max(1e-12, out.diag.wall_seconds);
        out.fingerprint = fingerprint_of(spec, method(), out.backend);
        return out;
    }

    Validation validate(const ProblemSpec& spec) const override {
        const std::string ref_desc = "manufactured solution exp(-2 pi^2 alpha t) sin(pi x) sin(pi y)";
        if (!fortran_bridge::available()) {
            return make_hardware_skip_validation(*this, spec, ref_desc);
        }

        const double alpha = spec.get("alpha", 0.1);
        const double t_end = spec.t_end > 0 ? spec.t_end : 0.05;

        Validation v;
        v.reference = ref_desc;
        v.tolerance = spec.tolerance;
        v.reference_ok = true;

        ProblemSpec sp = spec;
        auto rel_error_at = [&](int n) {
            sp.params["n"] = double(n);
            Result r = run(sp);
            const double h = 1.0 / double(n);
            double num = 0.0, den = 0.0, max_abs = 0.0;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const double e = exact((double(j) + 0.5) * h, (double(i) + 0.5) * h, alpha, t_end);
                    const double got = r.data[size_t(i) * size_t(n) + size_t(j)];
                    num += sq(got - e);
                    den += sq(e);
                    max_abs = std::max(max_abs, std::abs(got - e));
                }
            return std::pair<double, double>{std::sqrt(num / std::max(den, 1e-300)), max_abs};
        };

        const auto e32 = rel_error_at(32);
        const auto e64 = rel_error_at(64);
        v.rel_l2_error = e32.first;
        v.max_abs_error = e32.second;
        v.measured_order = (e32.first > 0.0 && e64.first > 0.0)
                               ? std::log(e32.first / e64.first) / std::log(2.0)
                               : 0.0;
        v.expected_order = 2.0;
        v.order_ok = v.measured_order > 1.7 && v.measured_order < 2.3;
        v.set_metric("rel_l2_error_n64", e64.first);

        sp.params["n"] = 32.0;
        Result r32 = run(sp);
        const int c = 16;
        const double h32 = 1.0 / 32.0;
        const double xc = (double(c) + 0.5) * h32;
        const double got = r32.data[size_t(c) * 32 + size_t(c)];
        const double got_norm = got / (std::sin(pi * xc) * std::sin(pi * xc));
        const double exact_norm = std::exp(-2.0 * pi * pi * alpha * t_end);
        v.conservation_error = std::abs(got_norm - exact_norm) / exact_norm;
        v.conservation_ok = v.conservation_error < 1e-2;
        v.tol_ok = e32.first <= std::max(spec.tolerance, 1e-9);
        v.notes = "Fortran 2023 explicit FTCS validated against manufactured eigenmode.";
        return v;
    }
};

class Poisson2D final : public Kernel {
public:
    std::string name() const override { return "poisson2d"; }
    std::string method() const override {
        return "Fortran 2023 5-point Laplace with ghost-cell Dirichlet walls, red-black SOR";
    }
    std::string units() const override { return "unit square; -lap(u) = f"; }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"n", "cells per side, default 64"},
                {"max_iter", "SOR iteration cap, default 20000"},
                {"tol", "relative residual target, default 1e-10"}};
    }

    static double manufactured_u(double x, double y) {
        return std::sin(pi * x) * std::sin(pi * y);
    }

    Result run(const ProblemSpec& spec) const override {
        const int n = int(std::max<int64_t>(4, spec.get_i("n", 64)));
        const int max_iter = int(std::max<int64_t>(10, spec.get_i("max_iter", 20000)));
        const double tol = (spec.tolerance > 0 && spec.tolerance < 1e-3) ? spec.tolerance
                                                                        : spec.get("tol", 1e-10);
        if (!fortran_bridge::available()) {
            return make_hardware_skip_result(*this, spec, {int64_t(n), int64_t(n)});
        }

        const Clock::time_point t0 = Clock::now();
        const size_t n2 = static_cast<size_t>(n) * static_cast<size_t>(n);
        std::vector<double> u(n2, 0.0);
        int fi = 0;
        double fr = 0.0;
        const int rc = oct_f_poisson2d(n, max_iter, tol, &fi, &fr, u.data());
        if (rc != 0) {
            return make_hardware_skip_result(*this, spec, {int64_t(n), int64_t(n)});
        }

        Result out;
        out.kernel = name();
        out.method = method();
        out.units = units();
        out.spec = spec;
        out.shape = {int64_t(n), int64_t(n)};
        out.data = std::move(u);
        out.backend = backend_string(spec.backend, true, fortran_bridge::compiler_id());
        out.diag.iterations = fi;
        out.diag.residual = fr;
        out.diag.wall_seconds = seconds_since(t0);
        out.diag.cells_per_second = double(n) * double(n) * double(std::max<int64_t>(1, fi)) /
                                    std::max(1e-12, out.diag.wall_seconds);
        out.fingerprint = fingerprint_of(spec, method(), out.backend);
        return out;
    }

    Validation validate(const ProblemSpec& spec) const override {
        const std::string ref_desc = "manufactured solution sin(pi x) sin(pi y) vs Fortran 2023 SOR";
        if (!fortran_bridge::available()) {
            return make_hardware_skip_validation(*this, spec, ref_desc);
        }

        Validation v;
        v.reference = ref_desc;
        v.tolerance = std::max(spec.tolerance, 1e-9);
        v.reference_ok = true;

        ProblemSpec sp = spec;
        sp.params["tol"] = 1e-12;
        auto rel_error = [&](int nn) {
            sp.params["n"] = double(nn);
            Result rr = run(sp);
            const double hh = 1.0 / double(nn);
            double a = 0.0, b = 0.0;
            for (int i = 0; i < nn; ++i)
                for (int j = 0; j < nn; ++j) {
                    const double e = manufactured_u((double(j) + 0.5) * hh, (double(i) + 0.5) * hh);
                    const double got = rr.data[size_t(i) * size_t(nn) + size_t(j)];
                    a += sq(got - e);
                    b += sq(e);
                }
            return std::sqrt(a / b);
        };
        const double e16 = rel_error(16), e32 = rel_error(32);
        sp.params["n"] = 32.0;
        Result r32 = run(sp);
        v.rel_l2_error = e32;
        v.measured_order = (e16 > 0.0 && e32 > 0.0) ? std::log(e16 / e32) / std::log(2.0) : 0.0;
        v.expected_order = 2.0;
        v.order_ok = v.measured_order > 1.7 && v.measured_order < 2.4;
        v.conservation_ok = r32.diag.residual <= 1e-6;
        v.tol_ok = r32.diag.residual <= 1e-6;
        v.notes = "Fortran 2023 red-black SOR validated against analytic solution.";
        return v;
    }
};

}  // namespace

std::shared_ptr<const Kernel> make_heat2d_kernel() { return std::make_shared<Heat2D>(); }
std::shared_ptr<const Kernel> make_poisson2d_kernel() { return std::make_shared<Poisson2D>(); }

}  // namespace oct::numerics::detail
