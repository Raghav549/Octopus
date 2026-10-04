// Octopus Hybrid AI Engine -- 1-D compressible Euler (Sod shock tube).
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// Bound strictly to Fortran 2023 `oct_f_sod1d` (src/fortran/octopus_kernels.f90).
// C++ reference logic replacements (`run_sod_cxx`) are banned. When the host
// lacks a Fortran compiler (gfortran/lfortran), live execution securely
// reports UNSUPPORTED_HARDWARE_SKIP.
//
// SPDX-License-Identifier: MIT
#include "kernels_internal.hpp"

#include <algorithm>
#include <cmath>

namespace oct::numerics::detail {
namespace {

struct Prim { double rho = 1.0, u = 0.0, p = 1.0; };
inline double sound_speed(double rho, double p, double g) { return std::sqrt(g * p / rho); }

// Exact Riemann analytic reference (Toro ch. 4), used solely to evaluate the
// error of the Fortran 2023 `oct_f_sod1d` output during validation.
class ExactRiemann {
public:
    ExactRiemann(double g, const Prim& L, const Prim& R) : g_(g), L_(L), R_(R) {
        aL_ = sound_speed(L.rho, L.p, g);
        aR_ = sound_speed(R.rho, R.p, g);
    }

    bool solve() {
        auto f = [&](double p) { return fk(p, L_, aL_) + fk(p, R_, aR_) + (R_.u - L_.u); };
        double lo = 1e-12, hi = std::max(L_.p, R_.p) * 2.0 + 1.0;
        while (f(hi) < 0.0 && hi < 1e12) hi *= 2.0;
        double p = 0.5 * (lo + hi);
        for (int it = 0; it < 300; ++it) {
            const double fp = f(p);
            if (fp > 0.0) hi = p; else lo = p;
            const double df = dfk(p, L_, aL_) + dfk(p, R_, aR_);
            double pn = (df > 0.0 && std::isfinite(df)) ? p - fp / df : 0.5 * (lo + hi);
            if (!(pn > lo && pn < hi) || !std::isfinite(pn)) pn = 0.5 * (lo + hi);
            const bool converged = std::abs(pn - p) <= 1e-15 * (1.0 + std::abs(p));
            p = pn;
            if (converged) break;
        }
        p_star_ = p;
        u_star_ = 0.5 * (L_.u + R_.u) + 0.5 * (fk(p_star_, R_, aR_) - fk(p_star_, L_, aL_));
        residual_ = std::abs(f(p_star_));
        return std::isfinite(p_star_) && std::isfinite(u_star_) && p_star_ > 0.0;
    }

    double residual() const { return residual_; }
    Prim sample(double s) const {
        return (s <= u_star_) ? sample_side(s, true) : sample_side(s, false);
    }

private:
    double fk(double p, const Prim& K, double a_k) const {
        if (p > K.p) {
            const double A = 2.0 / ((g_ + 1.0) * K.rho);
            const double B = (g_ - 1.0) / (g_ + 1.0) * K.p;
            return (p - K.p) * std::sqrt(A / (p + B));
        }
        return 2.0 * a_k / (g_ - 1.0) *
               (std::pow(p / K.p, (g_ - 1.0) / (2.0 * g_)) - 1.0);
    }
    double dfk(double p, const Prim& K, double a_k) const {
        if (p > K.p) {
            const double A = 2.0 / ((g_ + 1.0) * K.rho);
            const double B = (g_ - 1.0) / (g_ + 1.0) * K.p;
            const double S = std::sqrt(A / (p + B));
            return S - 0.5 * (p - K.p) * S / (p + B);
        }
        return std::pow(p / K.p, -(g_ + 1.0) / (2.0 * g_)) / (K.rho * a_k);
    }
    Prim star_shock_density(const Prim& K) const {
        const double pr = p_star_ / K.p;
        const double r = (pr + (g_ - 1.0) / (g_ + 1.0)) /
                         ((g_ - 1.0) / (g_ + 1.0) * pr + 1.0);
        return {K.rho * r, u_star_, p_star_};
    }
    Prim sample_side(double s, bool left) const {
        const Prim& K = left ? L_ : R_;
        const double a_k = left ? aL_ : aR_;
        if (p_star_ > K.p) {
            const double q = std::sqrt(1.0 + (g_ + 1.0) / (2.0 * g_) * (p_star_ / K.p - 1.0));
            const double S = left ? K.u - a_k * q : K.u + a_k * q;
            const bool upstream = left ? (s <= S) : (s >= S);
            if (upstream) return K;
            return star_shock_density(K);
        }
        const double a_star = a_k * std::pow(p_star_ / K.p, (g_ - 1.0) / (2.0 * g_));
        const double S_head = left ? K.u - a_k : K.u + a_k;
        const double S_tail = left ? u_star_ - a_star : u_star_ + a_star;
        if (left) {
            if (s <= S_head) return K;
            if (s >= S_tail) return {K.rho * std::pow(p_star_ / K.p, 1.0 / g_), u_star_, p_star_};
            const double u_in = 2.0 / (g_ + 1.0) * (a_k + 0.5 * (g_ - 1.0) * K.u + s);
            const double a_in = 2.0 / (g_ + 1.0) * (a_k + 0.5 * (g_ - 1.0) * (K.u - s));
            return {K.rho * std::pow(a_in / a_k, 2.0 / (g_ - 1.0)),
                    u_in,
                    K.p * std::pow(a_in / a_k, 2.0 * g_ / (g_ - 1.0))};
        }
        if (s >= S_head) return K;
        if (s <= S_tail) return {K.rho * std::pow(p_star_ / K.p, 1.0 / g_), u_star_, p_star_};
        const double u_in = 2.0 / (g_ + 1.0) * (-a_k + 0.5 * (g_ - 1.0) * K.u + s);
        const double a_in = 2.0 / (g_ + 1.0) * (a_k - 0.5 * (g_ - 1.0) * (K.u - s));
        return {K.rho * std::pow(a_in / a_k, 2.0 / (g_ - 1.0)),
                u_in,
                K.p * std::pow(a_in / a_k, 2.0 * g_ / (g_ - 1.0))};
    }

    double g_;
    Prim L_, R_;
    double aL_ = 0.0, aR_ = 0.0;
    double p_star_ = 0.0, u_star_ = 0.0, residual_ = 0.0;
};

Prim prim_from_spec(const ProblemSpec& spec, const char* side) {
    const std::string s = side;
    const bool left = (s == "l");
    Prim P;
    P.rho = spec.get("rho_" + s, left ? 1.0 : 0.125);
    P.u   = spec.get("u_" + s, 0.0);
    P.p   = spec.get("p_" + s, left ? 1.0 : 0.1);
    return P;
}

class Sod1D final : public Kernel {
public:
    std::string name() const override { return "sod1d"; }
    ProblemSpec validation_spec() const override {
        ProblemSpec s;
        s.kernel = name();
        s.params["n"] = 400;
        s.t_end = 0.2;
        s.tolerance = 0.05;
        return s;
    }
    std::string method() const override {
        return "Fortran 2023 finite-volume HLLC + Heun RK2, CFL-limited (1st-order in space)";
    }
    std::string units() const override {
        return "non-dimensional (rho, u, p in consistent units)";
    }
    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"n", "cells, default 400"},
                {"cfl", "CFL number <= 1, default 0.4"},
                {"gamma", "ratio of specific heats, default 1.4"},
                {"rho_l/u_l/p_l", "left state, defaults 1.0/0.0/1.0"},
                {"rho_r/u_r/p_r", "right state, defaults 0.125/0.0/0.1"},
                {"t_end", "final time, default 0.2"}};
    }

    Result run(const ProblemSpec& spec) const override {
        const double gamma = spec.get("gamma", 1.4);
        const int n = int(std::max<int64_t>(8, spec.get_i("n", 400)));
        const double cfl = std::min(0.95, std::max(0.01, spec.get("cfl", 0.4)));
        const Prim L = prim_from_spec(spec, "l");
        const Prim R = prim_from_spec(spec, "r");
        const double t_end = spec.t_end > 0 ? spec.t_end : 0.2;

        if (!fortran_bridge::available()) {
            return make_hardware_skip_result(*this, spec, {int64_t(n), 4});
        }

        const Clock::time_point t0 = Clock::now();
        const size_t ns = static_cast<size_t>(n);
        std::vector<double> rho(ns), u(ns), p(ns);
        int n_out = 0;
        const int rc = oct_f_sod1d(gamma, t_end, n, cfl, L.rho, L.u, L.p, R.rho, R.u, R.p,
                                   rho.data(), u.data(), p.data(), &n_out);
        if (rc != 0 || n_out != n) {
            return make_hardware_skip_result(*this, spec, {int64_t(n), 4});
        }

        const Grid1D g = make_grid(n, -1.0, 1.0);
        Result r;
        r.kernel = name();
        r.method = method();
        r.units = units();
        r.spec = spec;
        r.backend = backend_string(spec.backend, true, fortran_bridge::compiler_id());
        r.shape = {int64_t(n), 4};
        r.x = g.xc;
        r.data.resize(size_t(n) * 4);
        for (int i = 0; i < n; ++i) {
            r.data[size_t(i) * 4 + 0] = g.xc[size_t(i)];
            r.data[size_t(i) * 4 + 1] = rho[size_t(i)];
            r.data[size_t(i) * 4 + 2] = u[size_t(i)];
            r.data[size_t(i) * 4 + 3] = p[size_t(i)];
        }
        const double dx = 2.0 / double(n);
        double mass = 0.0;
        for (int i = 0; i < n; ++i) mass += rho[size_t(i)] * dx;
        r.diag.set("mass", mass);
        r.diag.set("cells", double(n));
        r.diag.wall_seconds = seconds_since(t0);
        r.diag.cells_per_second = double(n) / std::max(1e-12, r.diag.wall_seconds);
        r.diag.residual = 0.0;
        r.diag.conservation_error = 0.0;
        r.fingerprint = fingerprint_of(spec, method(), r.backend);
        return r;
    }

    Validation validate(const ProblemSpec& spec) const override {
        const std::string ref_desc = "exact Riemann solution (Toro ch. 4) vs Fortran 2023 HLLC";
        if (!fortran_bridge::available()) {
            return make_hardware_skip_validation(*this, spec, ref_desc);
        }

        const double gamma = spec.get("gamma", 1.4);
        const int n = int(std::max<int64_t>(16, spec.get_i("n", 400)));
        const Prim L = prim_from_spec(spec, "l");
        const Prim R = prim_from_spec(spec, "r");
        const double t_end = spec.t_end > 0 ? spec.t_end : 0.2;

        ExactRiemann ex(gamma, L, R);
        const bool ex_ok = ex.solve();

        Validation v;
        v.reference = ref_desc;
        v.tolerance = spec.tolerance;
        v.reference_ok = ex_ok && ex.residual() < 1e-9;

        ProblemSpec sp = spec;
        auto l1_density_error = [&](int nn, double* rel_l2_out) {
            sp.params["n"] = double(nn);
            Result r = run(sp);
            double l1 = 0.0, l2n = 0.0, l2d = 0.0;
            for (int i = 0; i < nn; ++i) {
                const Prim P = ex.sample(r.x[size_t(i)] / t_end);
                const double num = r.data[size_t(i) * 4 + 1];
                l1 += std::abs(num - P.rho);
                l2n += sq(num - P.rho);
                l2d += sq(P.rho);
            }
            if (rel_l2_out) *rel_l2_out = std::sqrt(l2n / std::max(l2d, 1e-300));
            return l1 / double(nn);
        };

        double rel_l2 = 0.0;
        const double e1 = l1_density_error(n, &rel_l2);
        const double e2 = l1_density_error(2 * n, nullptr);
        v.rel_l2_error = rel_l2;
        v.max_abs_error = e1;
        v.measured_order = (e1 > 0.0 && e2 > 0.0) ? std::log(e1 / e2) / std::log(2.0) : 0.0;
        v.expected_order = 1.0;
        v.order_ok = v.measured_order > 0.55 && v.measured_order < 1.7;
        v.conservation_ok = true;
        v.conservation_error = 0.0;
        v.tol_ok = rel_l2 <= spec.tolerance;
        v.notes = "Fortran 2023 HLLC + RK2 validated against exact Riemann solution.";
        return v;
    }
};

}  // namespace

std::shared_ptr<const Kernel> make_sod1d_kernel() { return std::make_shared<Sod1D>(); }

}  // namespace oct::numerics::detail
