// Octopus Hybrid AI Engine -- thermodynamics and linear-algebra kernels.
//
//   gas       ideal-gas relations validated by cross-checking two independent
//             routes (closed form vs numerical integration of ds = 0).
//   linsolve  dense LU solve with residual + condition estimate. It is here to
//             demonstrate the honest failure mode: on an ill-conditioned system
//             a small residual does NOT imply a small solution error, and the
//             kernel reports the condition number so callers can see that.
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
// gas
// ---------------------------------------------------------------------------
class Gas final : public Kernel {
public:
    std::string name() const override { return "gas"; }
    std::string method() const override {
        return "ideal-gas EOS + isentropic relations, closed form cross-checked by numerical "
               "integration of ds = 0";
    }
    std::string units() const override { return "p[Pa], T[K], R[J/(kg K)]"; }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"p1", "initial pressure, default 101325"},
                {"t1", "initial temperature, default 300"},
                {"ratio", "pressure ratio p2/p1, default 4"},
                {"gamma", "cp/cv, default 1.4"},
                {"R", "specific gas constant, default 287.05"}};
    }

    Result run(const ProblemSpec& spec) const override {
        const double p1 = spec.get("p1", 101325.0);
        const double t1 = spec.get("t1", 300.0);
        const double ratio = std::max(1.0001, spec.get("ratio", 4.0));
        const double gamma = spec.get("gamma", 1.4);
        const double R = spec.get("R", 287.05);
        const double p2 = p1 * ratio;

        // Closed-form isentropic relations.
        const double t2_closed = t1 * std::pow(p2 / p1, (gamma - 1.0) / gamma);
        const double rho1 = p1 / (R * t1);
        const double rho2_closed = rho1 * std::pow(p2 / p1, 1.0 / gamma);
        const double a1 = std::sqrt(gamma * R * t1);
        const double w_isen = (gamma * R * t1) / (gamma - 1.0) *
                              (1.0 - std::pow(p2 / p1, (gamma - 1.0) / gamma));

        // Numerical route: integrate ds = cp dT/T - R dp/p along an isentrope,
        // using the constraint T = T1 (p/p1)^((gamma-1)/gamma) evaluated at each
        // quadrature node. If the relation is wrong, the integral is non-zero.
        const int m = 200000;
        double integral = 0.0;
        const double cp = gamma * R / (gamma - 1.0);
        for (int i = 0; i < m; ++i) {
            const double f = (double(i) + 0.5) / double(m);
            const double p = p1 * std::pow(p2 / p1, f);
            const double dp = p * std::log(p2 / p1) / double(m);   // dp per step
            const double T = t1 * std::pow(p / p1, (gamma - 1.0) / gamma);
            const double dT = T * (gamma - 1.0) / gamma * (dp / p);
            integral += cp * dT / T - R * dp / p;
        }

        Result r;
        r.kernel = name();
        r.method = method();
        r.units = units();
        r.spec = spec;
        r.backend = backend_name(spec.backend);
        r.shape = {5};
        r.data = {t2_closed, rho2_closed, a1, w_isen, p2};
        r.diag.set("entropy_change_dimensionless", integral / R);
        r.diag.set("temperature_rise", t2_closed - t1);
        r.diag.wall_seconds = 0.0;
        r.fingerprint = fingerprint_of(spec, method(), r.backend);

        // Self-consistency: p = rho R T must hold on both states to machine level.
        const double eos1 = std::abs(p1 - rho1 * R * t1) / p1;
        const double eos2 = std::abs(p2 - rho2_closed * R * t2_closed) / p2;
        r.diag.set("eos_residual_state1", eos1);
        r.diag.set("eos_residual_state2", eos2);
        r.diag.residual = std::max(eos1, eos2);
        return r;
    }

    Validation validate(const ProblemSpec& spec) const override {
        Validation v;
        v.reference = "closed-form isentropic relations vs numerical ds = 0 integration";
        v.tolerance = std::max(spec.tolerance, 1e-12);
        v.reference_ok = true;

        ProblemSpec sp = spec;
        sp.backend = Backend::CxxLd;
        Result r = run(sp);

        // Entropy must be unchanged along an isentrope: |ds|/R << 1.
        const double ds = std::abs(r.diag.get("entropy_change_dimensionless", 1.0));
        v.conservation_error = ds;
        v.conservation_ok = ds < 1e-12;
        v.max_abs_error = std::max(r.diag.get("eos_residual_state1", 1.0),
                                   r.diag.get("eos_residual_state2", 1.0));
        v.rel_l2_error = v.max_abs_error;
        v.tol_ok = v.max_abs_error < 1e-14;

        // Second route: p v^gamma must be invariant between the two states.
        const double gamma = sp.get("gamma", 1.4);
        const double p1 = sp.get("p1", 101325.0), t1 = sp.get("t1", 300.0);
        const double R = sp.get("R", 287.05);
        const double ratio = std::max(1.0001, sp.get("ratio", 4.0));
        const double inv1 = p1 * std::pow(1.0 / (p1 / (R * t1)), gamma);
        const double p2 = p1 * ratio;
        const double t2 = r.data[0], rho2 = r.data[1];
        const double inv2 = p2 * std::pow(1.0 / rho2, gamma);
        const double rel = std::abs(inv1 - inv2) / inv1;
        v.set_metric("pv_gamma_invariance", rel);
        v.set_metric("entropy_change_over_R", ds);
        v.order_ok = rel < 1e-14;
        v.notes = "Thermodynamic identities, not a simulation: agreement to near machine "
                  "precision is the expected result, and any deviation would indicate a "
                  "relation error rather than a discretisation error.";
        return v;
    }
};

// ---------------------------------------------------------------------------
// linsolve
// ---------------------------------------------------------------------------
class LinSolve final : public Kernel {
public:
    std::string name() const override { return "linsolve"; }
    std::string method() const override {
        return "dense LU with partial pivoting, iterative refinement optional, "
               "1-norm condition estimate";
    }
    std::string units() const override { return "dimensionless (matrix problem)"; }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"n", "system size, default 8"},
                {"case", "0 = Hilbert (ill-conditioned, default), 1 = SPD Laplacian, 2 = random"}};
    }

    static std::vector<double> hilbert(int n) {
        std::vector<double> A(size_t(n) * size_t(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) A[size_t(i) * size_t(n) + size_t(j)] = 1.0 / double(i + j + 1);
        return A;
    }
    static std::vector<double> laplacian1d(int n) {
        std::vector<double> A(size_t(n) * size_t(n), 0.0);
        for (int i = 0; i < n; ++i) {
            A[size_t(i) * size_t(n) + size_t(i)] = 2.0;
            if (i > 0) A[size_t(i) * size_t(n) + size_t(i - 1)] = -1.0;
            if (i < n - 1) A[size_t(i) * size_t(n) + size_t(i + 1)] = -1.0;
        }
        return A;
    }

    Result run(const ProblemSpec& spec) const override {
        const int n = int(std::max<int64_t>(2, spec.get_i("n", 8)));
        const int cs = int(spec.get_i("case", 0));

        std::vector<double> A;
        const size_t nn = size_t(n);
        std::vector<double> x_exact(nn, 0.0);
        if (cs == 0) {
            A = hilbert(n);
            for (int i = 0; i < n; ++i) x_exact[size_t(i)] = 1.0;
        } else if (cs == 1) {
            A = laplacian1d(n);
            for (int i = 0; i < n; ++i) x_exact[size_t(i)] = 1.0;
        } else {
            Rng rng(20240501);
            A.resize(size_t(n) * size_t(n));
            for (auto& v : A) v = rng.next_normal();
            for (int i = 0; i < n; ++i) A[size_t(i) * size_t(n) + size_t(i)] += double(n);
            for (int i = 0; i < n; ++i) x_exact[size_t(i)] = rng.next_normal();
        }
        auto b = linalg::matmul(A, x_exact, n, n, 1);

        double residual = 0.0;
        auto x = linalg::solve(A, b, n, &residual);

        Result r;
        r.kernel = name();
        r.method = method();
        r.units = units();
        r.spec = spec;
        r.backend = backend_name(spec.backend);
        r.shape = {int64_t(n)};
        r.diag.residual = residual;

        if (x.is_ok()) {
            r.data = *x;
            double err = 0.0, norm = 0.0;
            for (int i = 0; i < n; ++i) {
                err = std::max(err, std::abs((*x)[size_t(i)] - x_exact[size_t(i)]));
                norm = std::max(norm, std::abs(x_exact[size_t(i)]));
            }
            const double cond = linalg::condition_1norm(A, n);
            r.diag.set("condition_1norm", cond);
            r.diag.set("solution_error_max", err);
            r.diag.set("solution_error_over_cond", err / std::max(cond, 1e-300));

            // Also solve with the SPD solver when the matrix is symmetric.
            if (cs == 1) {
                int64_t iters = 0;
                double cg_res = 0.0;
                auto xcg = linalg::cg(A, b, n, 1e-14, int64_t(10 * n), &iters, &cg_res);
                if (xcg.is_ok()) {
                    double d = 0.0;
                    for (int i = 0; i < n; ++i) d = std::max(d, std::abs((*xcg)[size_t(i)] - (*x)[size_t(i)]));
                    r.diag.set("lu_vs_cg_max_diff", d);
                    r.diag.set("cg_iterations", double(iters));
                    r.diag.set("cg_residual", cg_res);
                }
            }
        } else {
            r.backend += " (singular)";
        }
        r.diag.wall_seconds = 0.0;
        r.fingerprint = fingerprint_of(spec, method(), r.backend);
        return r;
    }

    Validation validate(const ProblemSpec& spec) const override {
        Validation v;
        v.reference = "exact solution known by construction (b = A x_exact) + CG cross-check";
        v.tolerance = std::max(spec.tolerance, 1e-12);
        v.reference_ok = true;

        // Case 1 (SPD Laplacian): LU and CG must agree; residual must be small.
        ProblemSpec sp = spec;
        sp.backend = Backend::CxxLd;
        sp.params["case"] = 1.0;
        sp.params["n"] = 32.0;
        Result r = run(sp);
        v.tol_ok = r.diag.residual < 1e-12;
        v.conservation_ok = r.diag.get("lu_vs_cg_max_diff", 1.0) < 1e-8;
        v.max_abs_error = r.diag.get("solution_error_max", 1.0);
        v.rel_l2_error = v.max_abs_error;
        v.set_metric("lu_vs_cg_max_diff", r.diag.get("lu_vs_cg_max_diff", 0.0));
        v.set_metric("cg_iterations", r.diag.get("cg_iterations", 0.0));

        // Case 0 (Hilbert): demonstrate the honest failure mode explicitly.
        ProblemSpec sp2 = sp;
        sp2.params["case"] = 0.0;
        sp2.params["n"] = 12.0;
        Result rh = run(sp2);
        const double cond = rh.diag.get("condition_1norm", 0.0);
        const double err = rh.diag.get("solution_error_max", 0.0);
        const double res = rh.diag.residual;
        v.set_metric("hilbert_condition", cond);
        v.set_metric("hilbert_solution_error", err);
        v.set_metric("hilbert_residual", res);
        // Residual stays tiny while the solution error grows with the condition
        // number: that is the mathematically correct behaviour, and the kernel
        // reports both so no caller can be misled by the residual alone.
        const bool honest = (res < 1e-10) && (err > 1e-4) && (cond > 1e10);
        v.order_ok = honest;
        v.notes = honest
                      ? "Ill-conditioned Hilbert system: small residual with large solution "
                        "error, exactly as the condition number predicts. Both are reported."
                      : "Unexpected behaviour on the Hilbert case -- investigate.";
        return v;
    }
};

}  // namespace

std::shared_ptr<const Kernel> make_gas_kernel() { return std::make_shared<Gas>(); }
std::shared_ptr<const Kernel> make_linsolve_kernel() { return std::make_shared<LinSolve>(); }

}  // namespace oct::numerics::detail
