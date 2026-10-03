// Octopus Hybrid AI Engine -- diffusion and elliptic kernels.
//
// Boundary handling: both kernels use cell-centred unknowns with a one-cell
// ghost ring. The Dirichlet wall value is imposed by odd reflection about the
// wall (u_ghost = -u_first), which keeps the 3-point (5-point) stencil
// symmetric about the cell centre and therefore second-order accurate. Getting
// this wrong is the classic way to silently downgrade a scheme to first order
// -- and the tests measure the observed order, so it cannot hide.
//
// SPDX-License-Identifier: MIT
#include "kernels_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace oct::numerics::detail {
namespace {

using std::numbers::pi;

// ---------------------------------------------------------------------------
// heat2d
// ---------------------------------------------------------------------------
class Heat2D final : public Kernel {
public:
    std::string name() const override { return "heat2d"; }
    ProblemSpec validation_spec() const override {
        ProblemSpec s;
        s.kernel = name();
        s.t_end = 0.05;       // short enough that the explicit FTCS scheme stays stable
        s.tolerance = 5e-3;
        return s;
    }
    std::string method() const override {
        return "explicit FTCS (5-point Laplacian) with ghost-cell Dirichlet walls";
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
        const double r = cfl / 4.0;                 // r = alpha*dt/h^2
        const double dt = r * h * h / alpha;
        const int64_t steps = int64_t(std::ceil(t_end / dt));
        const double dt_eff = t_end / double(std::max<int64_t>(1, steps));

        Result out;
        out.kernel = name();
        out.method = method();
        out.units = units();
        out.spec = spec;
        out.shape = {int64_t(n), int64_t(n)};

        bool used_fortran = false;
#if defined(OCT_HAVE_FORTRAN)
        if (spec.backend == Backend::Fortran && fortran_bridge::available()) {
            std::vector<double> u(size_t(n) * size_t(n));
            const int rc = oct_f_heat2d(alpha, t_end, n, cfl, u.data());
            if (rc == 0) {
                out.data = std::move(u);
                used_fortran = true;
                out.diag.iterations = steps;
            } else {
                log_warn("Fortran heat2d backend failed (rc=" + std::to_string(rc) +
                         "); using C++ reference implementation");
            }
        }
#endif
        const Clock::time_point t0 = Clock::now();
        if (!used_fortran) {
            // Ghost ring: index (i+1, j+1) for cell (i,j); ghost = -interior.
            const int m = n + 2;
            auto at = [&](std::vector<double>& a, int i, int j) -> double& {
                return a[size_t(i + 1) * size_t(m) + size_t(j + 1)];
            };
            std::vector<double> u(size_t(m) * size_t(m), 0.0), un(u.size(), 0.0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    at(u, i, j) = exact((double(j) + 0.5) * h, (double(i) + 0.5) * h, alpha, 0.0);
            auto apply_walls = [&](std::vector<double>& a) {
                for (int j = 0; j < n; ++j) {
                    at(a, -1, j) = -at(a, 0, j);
                    at(a, n, j) = -at(a, n - 1, j);
                }
                for (int i = 0; i < n; ++i) {
                    at(a, i, -1) = -at(a, i, 0);
                    at(a, i, n) = -at(a, i, n - 1);
                }
            };
            apply_walls(u);
            const double rr = alpha * dt_eff / (h * h);
            for (int64_t s = 0; s < steps; ++s) {
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        const double lap = at(u, i + 1, j) + at(u, i - 1, j) +
                                           at(u, i, j + 1) + at(u, i, j - 1) - 4.0 * at(u, i, j);
                        at(un, i, j) = at(u, i, j) + rr * lap;
                    }
                apply_walls(un);
                std::swap(u, un);
            }
            out.data.resize(size_t(n) * size_t(n));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    out.data[size_t(i) * size_t(n) + size_t(j)] = at(u, i, j);
            out.diag.iterations = steps;
        }
        out.backend = backend_string(spec.backend, used_fortran, fortran_bridge::compiler_id());
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
        const double alpha = spec.get("alpha", 0.1);
        const double t_end = spec.t_end > 0 ? spec.t_end : 0.05;

        Validation v;
        v.reference = "manufactured solution exp(-2 pi^2 alpha t) sin(pi x) sin(pi y)";
        v.tolerance = spec.tolerance;
        v.reference_ok = true;

        ProblemSpec sp = spec;
        sp.backend = Backend::CxxLd;

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
        v.set_metric("decay_analytic", exact_norm);
        v.set_metric("decay_numeric", got_norm);

        v.tol_ok = e32.first <= std::max(spec.tolerance, 1e-9);
        v.notes = "explicit FTCS; dt is CFL-limited (4*Fourier = cfl), so the space and time "
                  "error terms are both O(h^2) and the measured order is the spatial order.";
        return v;
    }
};

// ---------------------------------------------------------------------------
// poisson2d
// ---------------------------------------------------------------------------
// Discrete operator used everywhere in this kernel (solver, residual, and the
// conjugate-gradient cross-check are guaranteed to be the *same* operator):
//
//     A u |_(i,j) = ( 4*u(i,j) - sum of the 4 neighbour values ) / h^2
//
// where a neighbour outside the domain contributes the odd-reflection ghost
// (-u at the mirroring interior cell), i.e. it moves to the diagonal.
class Poisson2D final : public Kernel {
public:
    std::string name() const override { return "poisson2d"; }
    std::string method() const override {
        return "5-point Laplace with ghost-cell Dirichlet walls, red-black SOR to a residual "
               "tolerance, cross-checked by conjugate gradients";
    }
    std::string units() const override { return "unit square; -lap(u) = f"; }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"n", "cells per side, default 64"},
                {"max_iter", "SOR iteration cap, default 20000"},
                {"tol", "relative residual target, default 1e-10"}};
    }

    static double manufactured_f(double x, double y) {
        return 2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
    }
    static double manufactured_u(double x, double y) {
        return std::sin(pi * x) * std::sin(pi * y);
    }

    // Interior values in v[i*n+j]; ghosts are implied by odd reflection.
    static double get(const std::vector<double>& v, int n, int i, int j) {
        // Odd reflection about the wall: the ghost lies half a cell outside the
        // wall between the ghost and the interior cell, so u_ghost = -u_interior.
        if (i < 0)  return -v[size_t(j)];                                  // mirror of row 0
        if (i >= n) return -v[size_t(n - 1) * size_t(n) + size_t(j)];      // mirror of row n-1
        if (j < 0)  return -v[size_t(i) * size_t(n)];                      // mirror of column 0
        if (j >= n) return -v[size_t(i) * size_t(n) + size_t(n - 1)];      // mirror of column n-1
        return v[size_t(i) * size_t(n) + size_t(j)];
    }
    static double& ref(std::vector<double>& v, int n, int i, int j) {
        return v[size_t(i) * size_t(n) + size_t(j)];
    }

    static void solve_sor(int n, double h, int max_iter, double tol, std::vector<double>& u,
                          int64_t& iters, double& residual) {
        const double omega = 2.0 / (1.0 + std::sin(pi * h));
        auto f = [&](int i, int j) {
            return manufactured_f((double(j) + 0.5) * h, (double(i) + 0.5) * h);
        };
        auto resid = [&]() {
            double m = 0.0;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const double Au = (4.0 * u[size_t(i) * size_t(n) + size_t(j)] -
                                       get(u, n, i + 1, j) - get(u, n, i - 1, j) -
                                       get(u, n, i, j + 1) - get(u, n, i, j - 1)) / (h * h);
                    m = std::max(m, std::abs(Au - f(i, j)));
                }
            return m;
        };
        const double r0 = std::max(resid(), 1e-300);
        residual = 1.0;
        iters = 0;
        for (int it = 0; it < max_iter; ++it) {
            ++iters;
            for (int color = 0; color < 2; ++color) {
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        if (((i + j) & 1) != color) continue;
                        // The ghost at a wall mirrors THIS cell (odd reflection),
                        // so it belongs on the diagonal, not in the known sum:
                        // the boundary diagonal is 5 on an edge and 6 in a corner.
                        // Using 4 everywhere made the iteration operator differ
                        // from the operator the residual measures, which is
                        // stable for small n but diverges for n >= 48.
                        double known = 0.0;
                        int diag = 4;
                        if (i + 1 < n) known += u[size_t(i + 1) * size_t(n) + size_t(j)]; else diag += 1;
                        if (i - 1 >= 0) known += u[size_t(i - 1) * size_t(n) + size_t(j)]; else diag += 1;
                        if (j + 1 < n) known += u[size_t(i) * size_t(n) + size_t(j + 1)]; else diag += 1;
                        if (j - 1 >= 0) known += u[size_t(i) * size_t(n) + size_t(j - 1)]; else diag += 1;
                        const double gs = (known + h * h * f(i, j)) / double(diag);
                        ref(u, n, i, j) += omega * (gs - ref(u, n, i, j));
                    }
            }
            if ((it & 15) == 15) {
                residual = resid() / r0;
                if (residual <= tol) break;
            }
        }
        residual = resid() / r0;
    }

    Result run(const ProblemSpec& spec) const override {
        const int n = int(std::max<int64_t>(4, spec.get_i("n", 64)));
        const int max_iter = int(std::max<int64_t>(10, spec.get_i("max_iter", 20000)));
        const double tol = (spec.tolerance > 0 && spec.tolerance < 1e-3) ? spec.tolerance
                                                                        : spec.get("tol", 1e-10);
        const double h = 1.0 / double(n);

        Result out;
        out.kernel = name();
        out.method = method();
        out.units = units();
        out.spec = spec;
        out.shape = {int64_t(n), int64_t(n)};

        bool used_fortran = false;
        int64_t iters = 0;
        double residual = 0.0;
        const Clock::time_point t0 = Clock::now();
#if defined(OCT_HAVE_FORTRAN)
        if (spec.backend == Backend::Fortran && fortran_bridge::available()) {
            std::vector<double> u(size_t(n) * size_t(n), 0.0);
            int fi = 0;
            double fr = 0.0;
            const int rc = oct_f_poisson2d(n, max_iter, tol, &fi, &fr, u.data());
            if (rc == 0) {
                out.data = std::move(u);
                iters = fi;
                residual = fr;
                used_fortran = true;
            } else {
                log_warn("Fortran poisson2d backend failed (rc=" + std::to_string(rc) +
                         "); using C++ reference implementation");
            }
        }
#endif
        if (!used_fortran) {
            out.data.assign(size_t(n) * size_t(n), 0.0);
            solve_sor(n, h, max_iter, tol, out.data, iters, residual);
        }
        out.backend = backend_string(spec.backend, used_fortran, fortran_bridge::compiler_id());
        out.diag.iterations = iters;
        out.diag.residual = residual;
        out.diag.wall_seconds = seconds_since(t0);
        out.diag.cells_per_second = double(n) * double(n) * double(std::max<int64_t>(1, iters)) /
                                    std::max(1e-12, out.diag.wall_seconds);
        out.fingerprint = fingerprint_of(spec, method(), out.backend);
        return out;
    }

    Validation validate(const ProblemSpec& spec) const override {
        Validation v;
        v.reference = "manufactured solution sin(pi x) sin(pi y) + conjugate-gradient cross-check";
        v.tolerance = std::max(spec.tolerance, 1e-9);
        v.reference_ok = true;

        ProblemSpec sp = spec;
        sp.backend = Backend::CxxLd;
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
        v.max_abs_error = 0.0;
        for (int i = 0; i < 32; ++i)
            for (int j = 0; j < 32; ++j) {
                const double hh = 1.0 / 32.0;
                const double e = manufactured_u((double(j) + 0.5) * hh, (double(i) + 0.5) * hh);
                v.max_abs_error = std::max(v.max_abs_error,
                                           std::abs(r32.data[size_t(i) * 32 + size_t(j)] - e));
            }
        v.measured_order = (e16 > 0.0 && e32 > 0.0) ? std::log(e16 / e32) / std::log(2.0) : 0.0;
        v.expected_order = 2.0;
        v.order_ok = v.measured_order > 1.7 && v.measured_order < 2.4;

        // Cross-validation against an independent algorithm on the SAME discrete
        // operator: if the two agree, a converged-but-wrong operator is excluded
        // only by the analytic comparison above -- together they pin the result.
        {
            const int nc = 16;
            const double hc = 1.0 / double(nc);
            const int64_t N = int64_t(nc) * int64_t(nc);
            const size_t NS = size_t(N);
            std::vector<double> A(NS * NS, 0.0), b(NS, 0.0);
            auto row = [&](int i, int j) { return int64_t(i) * int64_t(nc) + int64_t(j); };
            for (int i = 0; i < nc; ++i)
                for (int j = 0; j < nc; ++j) {
                    const int64_t rr_ = row(i, j);
                    const int nwalls = (i == 0) + (i == nc - 1) + (j == 0) + (j == nc - 1);
                    A[size_t(rr_) * NS + size_t(rr_)] = double(4 + nwalls) / (hc * hc);
                    b[size_t(rr_)] = manufactured_f((double(j) + 0.5) * hc, (double(i) + 0.5) * hc);
                    auto off = [&](int ii, int jj) {
                        if (ii < 0 || ii >= nc || jj < 0 || jj >= nc) return;   // wall -> diagonal
                        A[size_t(rr_) * NS + size_t(row(ii, jj))] = -1.0 / (hc * hc);
                    };
                    off(i + 1, j);
                    off(i - 1, j);
                    off(i, j + 1);
                    off(i, j - 1);
                }
            int64_t iters = 0;
            double resid = 0.0;
            auto cg = linalg::cg(A, b, N, 1e-13, int64_t(4 * N), &iters, &resid);
            bool agree = false;
            if (cg.is_ok()) {
                ProblemSpec s3 = sp;
                s3.params["n"] = double(nc);
                s3.params["tol"] = 1e-13;
                Result rs = run(s3);
                double diff = 0.0, scale = 0.0;
                for (int i = 0; i < nc; ++i)
                    for (int j = 0; j < nc; ++j) {
                        const double a = rs.data[size_t(i) * size_t(nc) + size_t(j)];
                        const double c = (*cg)[size_t(row(i, j))];
                        diff += sq(a - c);
                        scale += sq(a);
                    }
                const double rel = std::sqrt(diff / std::max(scale, 1e-300));
                v.set_metric("cg_cross_check_rel_diff", rel);
                v.set_metric("cg_iterations", double(iters));
                v.set_metric("cg_residual", resid);
                agree = rel < 1e-5;
            }
            v.conservation_ok = agree;
        }

        v.tol_ok = r32.diag.residual <= 1e-6;
        v.notes = "The CG cross-check assembles the identical discrete operator, so agreement "
                  "rules out solver error; the analytic comparison rules out operator error.";
        return v;
    }
};

}  // namespace

std::shared_ptr<const Kernel> make_heat2d_kernel() { return std::make_shared<Heat2D>(); }
std::shared_ptr<const Kernel> make_poisson2d_kernel() { return std::make_shared<Poisson2D>(); }

}  // namespace oct::numerics::detail
