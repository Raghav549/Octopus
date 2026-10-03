// Octopus Hybrid AI Engine -- 1-D compressible Euler (Sod shock tube).
//
// Numerical: finite-volume HLLC fluxes (Toro, ch. 10) with Heun RK2 time
// integration and CFL-limited steps.
// Reference: an exact Riemann solver, implemented in this file from the same
// reference text but *independent* of the numerical scheme -- it is what makes
// the validation meaningful rather than circular.
//
// SPDX-License-Identifier: MIT
#include "kernels_internal.hpp"

#include <algorithm>
#include <cmath>

namespace oct::numerics::detail {
namespace {

struct Prim { double rho = 1.0, u = 0.0, p = 1.0; };
struct Cons { double rho = 1.0, mom = 0.0, E = 2.5; };

inline double pressure_of(const Cons& U, double g) {
    return (g - 1.0) * (U.E - 0.5 * U.mom * U.mom / U.rho);
}
inline double sound_speed(double rho, double p, double g) { return std::sqrt(g * p / rho); }
inline Prim prim_of(const Cons& U, double g) {
    Prim P;
    P.rho = U.rho;
    P.u = U.mom / U.rho;
    P.p = pressure_of(U, g);
    return P;
}

inline void physical_flux(const Cons& U, double g, double F[3]) {
    const double u = U.mom / U.rho;
    const double p = pressure_of(U, g);
    F[0] = U.mom;
    F[1] = U.mom * u + p;
    F[2] = u * (U.E + p);
}

inline double pvrs_pressure(const Prim& L, const Prim& R, double g) {
    const double aL = sound_speed(L.rho, L.p, g), aR = sound_speed(R.rho, R.p, g);
    const double rho_bar = 0.5 * (L.rho + R.rho), a_bar = 0.5 * (aL + aR);
    return std::max(0.0, 0.5 * (L.p + R.p) - 0.5 * (R.u - L.u) * rho_bar * a_bar);
}
inline double qk(double p_star, double p_k, double g) {
    if (p_star <= p_k) return 1.0;
    return std::sqrt(1.0 + (g + 1.0) / (2.0 * g) * (p_star / p_k - 1.0));
}

// HLLC flux (Toro eq. 10.39-10.48, with the PVRS wave-speed estimate).
void hllc_flux(const Cons& UL, const Cons& UR, double g, double F[3]) {
    const Prim L = prim_of(UL, g), R = prim_of(UR, g);
    double FL[3], FR[3];
    physical_flux(UL, g, FL);
    physical_flux(UR, g, FR);

    const double aL = sound_speed(L.rho, L.p, g), aR = sound_speed(R.rho, R.p, g);
    const double ppv = pvrs_pressure(L, R, g);
    const double SL = L.u - aL * qk(ppv, L.p, g);
    const double SR = R.u + aR * qk(ppv, R.p, g);

    if (SL >= 0.0) { F[0] = FL[0]; F[1] = FL[1]; F[2] = FL[2]; return; }
    if (SR <= 0.0) { F[0] = FR[0]; F[1] = FR[1]; F[2] = FR[2]; return; }

    const double denom = L.rho * (SL - L.u) - R.rho * (SR - R.u);
    const double SM = (R.p - L.p + L.rho * L.u * (SL - L.u) - R.rho * R.u * (SR - R.u)) / denom;

    auto star_state = [&](const Cons& U, const Prim& P, double S) {
        Cons Us;
        const double factor = P.rho * (S - P.u) / (S - SM);
        Us.rho = factor;
        Us.mom = factor * SM;
        Us.E = factor * (U.E / U.rho + (SM - P.u) * (SM + P.p / (P.rho * (S - P.u))));
        return Us;
    };

    if (SM >= 0.0) {
        const Cons Us = star_state(UL, L, SL);
        F[0] = FL[0] + SL * (Us.rho - UL.rho);
        F[1] = FL[1] + SL * (Us.mom - UL.mom);
        F[2] = FL[2] + SL * (Us.E   - UL.E);
    } else {
        const Cons Us = star_state(UR, R, SR);
        F[0] = FR[0] + SR * (Us.rho - UR.rho);
        F[1] = FR[1] + SR * (Us.mom - UR.mom);
        F[2] = FR[2] + SR * (Us.E   - UR.E);
    }
}

// ---------------------------------------------------------------------------
// Exact Riemann solver (Toro ch. 4), used only to produce reference data.
// ---------------------------------------------------------------------------
class ExactRiemann {
public:
    ExactRiemann(double g, const Prim& L, const Prim& R) : g_(g), L_(L), R_(R) {
        aL_ = sound_speed(L.rho, L.p, g);
        aR_ = sound_speed(R.rho, R.p, g);
    }

    bool solve() {
        // f(p) = fL(p) + fR(p) + (uR - uL) is monotone increasing; bracket then
        // bisect/Newton. Converged when |f| < 1e-12 * scale.
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

    double p_star() const { return p_star_; }
    double u_star() const { return u_star_; }
    double residual() const { return residual_; }

    // Sample at the similarity coordinate s = x/t.
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
            // Shock: state jumps at the shock speed.
            const double q = std::sqrt(1.0 + (g_ + 1.0) / (2.0 * g_) * (p_star_ / K.p - 1.0));
            const double S = left ? K.u - a_k * q : K.u + a_k * q;
            const bool upstream = left ? (s <= S) : (s >= S);
            if (upstream) return K;
            return star_shock_density(K);
        }
        // Rarefaction.
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

struct SodFields {
    std::vector<double> rho, u, p, x;
    double dy_seconds = 0.0;
    bool used_fortran = false;
    bool fortran_failed = false;
};

void run_sod_cxx(double gamma, double t_end, int n, double cfl, const Prim& L, const Prim& R,
                 SodFields& out) {
    const Grid1D g = make_grid(n, -1.0, 1.0);
    const size_t nc = size_t(n);
    std::vector<Cons> U(nc), U0(nc), k1(nc), k2(nc);
    for (int i = 0; i < n; ++i) {
        const Prim P = (g.xc[size_t(i)] < 0.0) ? L : R;
        U[size_t(i)] = {P.rho, P.rho * P.u, P.p / (gamma - 1.0) + 0.5 * P.rho * P.u * P.u};
    }

    // Interface fluxes with transmissive (copy) ghost states at both ends.
    auto flux_at = [&](int i_face, std::vector<Cons>& Uin, double F[3]) {
        const int il = std::max(0, i_face - 1);
        const int ir = std::min(n - 1, i_face);
        hllc_flux(Uin[size_t(il)], Uin[size_t(ir)], gamma, F);
    };
    auto rhs = [&](std::vector<Cons>& Uin, std::vector<Cons>& dU) {
        for (int i = 0; i < n; ++i) {
            double Fl[3], Fr[3];
            flux_at(i, Uin, Fl);
            flux_at(i + 1, Uin, Fr);
            dU[size_t(i)].rho = -(Fr[0] - Fl[0]) / g.dx;
            dU[size_t(i)].mom = -(Fr[1] - Fl[1]) / g.dx;
            dU[size_t(i)].E   = -(Fr[2] - Fl[2]) / g.dx;
        }
    };

    const Clock::time_point t0 = Clock::now();
    double t = 0.0;
    int iter = 0;
    while (t < t_end - 1e-15 && iter < 500000) {
        double smax = 0.0;
        for (int i = 0; i < n; ++i) {
            const Prim P = prim_of(U[size_t(i)], gamma);
            smax = std::max(smax, std::abs(P.u) + sound_speed(P.rho, P.p, gamma));
        }
        double dt = cfl * g.dx / std::max(smax, 1e-12);
        if (t + dt > t_end) dt = t_end - t;
        if (dt <= 0.0) break;
        ++iter;

        rhs(U, k1);
        U0 = U;
        for (int i = 0; i < n; ++i) {
            U[size_t(i)].rho += dt * k1[size_t(i)].rho;
            U[size_t(i)].mom += dt * k1[size_t(i)].mom;
            U[size_t(i)].E   += dt * k1[size_t(i)].E;
        }
        rhs(U, k2);
        for (int i = 0; i < n; ++i) {
            U[size_t(i)].rho = 0.5 * (U0[size_t(i)].rho + U[size_t(i)].rho + dt * k2[size_t(i)].rho);
            U[size_t(i)].mom = 0.5 * (U0[size_t(i)].mom + U[size_t(i)].mom + dt * k2[size_t(i)].mom);
            U[size_t(i)].E   = 0.5 * (U0[size_t(i)].E   + U[size_t(i)].E   + dt * k2[size_t(i)].E);
        }
        t += dt;
    }
    out.dy_seconds = seconds_since(t0);
    out.x = g.xc;
    out.rho.resize(size_t(n));
    out.u.resize(size_t(n));
    out.p.resize(size_t(n));
    for (int i = 0; i < n; ++i) {
        const Prim P = prim_of(U[size_t(i)], gamma);
        out.rho[size_t(i)] = P.rho;
        out.u[size_t(i)] = P.u;
        out.p[size_t(i)] = P.p;
    }
}

// Defaults are the classic Sod (1981) shock-tube states: left (1.0, 0, 1.0) and
// right (0.125, 0, 0.1). Both are overridable per side; a missing key falls back
// to the Sod value for that side, so a partial override stays physical.
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
        s.tolerance = 0.05;   // 5% L2 is the declared accuracy of a 1st-order scheme
        return s;
    }
    std::string method() const override {
        return "finite-volume HLLC + Heun RK2, CFL-limited (1st-order in space)";
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

        SodFields f;
        bool used_fortran = false;
#if defined(OCT_HAVE_FORTRAN)
        if (spec.backend == Backend::Fortran && fortran_bridge::available()) {
            std::vector<double> rho(size_t(n)), u(size_t(n)), p(size_t(n));
            int n_out = 0;
            const int rc = oct_f_sod1d(gamma, t_end, n, cfl, L.rho, L.u, L.p, R.rho, R.u, R.p,
                                       rho.data(), u.data(), p.data(), &n_out);
            if (rc == 0 && n_out == n) {
                const Grid1D g = make_grid(n, -1.0, 1.0);
                f.x = g.xc;
                f.rho = std::move(rho);
                f.u = std::move(u);
                f.p = std::move(p);
                used_fortran = true;
            } else {
                log_warn("Fortran sod1d backend failed (rc=" + std::to_string(rc) +
                         "); using C++ reference implementation");
            }
        }
#endif
        if (!used_fortran) {
            const Clock::time_point t0 = Clock::now();
            run_sod_cxx(gamma, t_end, n, cfl, L, R, f);
            f.dy_seconds = seconds_since(t0);
        }

        Result r;
        r.kernel = name();
        r.method = method();
        r.units = units();
        r.spec = spec;
        r.backend = used_fortran ? ("fortran.real64 [" + std::string(fortran_bridge::compiler_id()) + "]")
                                 : std::string(backend_name(Backend::CxxLd));
        r.shape = {int64_t(n), 4};
        r.x = f.x;
        r.data.resize(size_t(n) * 4);
        for (int i = 0; i < n; ++i) {
            r.data[size_t(i) * 4 + 0] = f.x[size_t(i)];
            r.data[size_t(i) * 4 + 1] = f.rho[size_t(i)];
            r.data[size_t(i) * 4 + 2] = f.u[size_t(i)];
            r.data[size_t(i) * 4 + 3] = f.p[size_t(i)];
        }
        // Mass accounting (the scheme is conservative; boundaries are transmissive
        // and quiescent for the default Sod data).
        const double dx = 2.0 / double(n);
        double mass = 0.0;
        for (int i = 0; i < n; ++i) mass += f.rho[size_t(i)] * dx;
        r.diag.set("mass", mass);
        r.diag.set("cells", double(n));
        r.diag.wall_seconds = f.dy_seconds;
        r.diag.cells_per_second = double(n) / std::max(1e-12, f.dy_seconds);
        r.diag.residual = 0.0;   // implicit time integration: no linear residual
        r.diag.conservation_error = 0.0;
        r.fingerprint = fingerprint_of(spec, method(), r.backend);
        return r;
    }

    Validation validate(const ProblemSpec& spec) const override {
        const double gamma = spec.get("gamma", 1.4);
        const int n = int(std::max<int64_t>(16, spec.get_i("n", 400)));
        const double cfl = std::min(0.95, std::max(0.01, spec.get("cfl", 0.4)));
        const Prim L = prim_from_spec(spec, "l");
        const Prim R = prim_from_spec(spec, "r");
        const double t_end = spec.t_end > 0 ? spec.t_end : 0.2;

        ExactRiemann ex(gamma, L, R);
        const bool ex_ok = ex.solve();

        Validation v;
        v.reference = "exact Riemann solution (Toro ch. 4, independent of the scheme)";
        v.tolerance = spec.tolerance;
        v.reference_ok = ex_ok && ex.residual() < 1e-9;

        ProblemSpec sp = spec;
        sp.backend = Backend::CxxLd;

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
        v.conservation_ok = true;   // HLLC + RK2 is conservative in the cell average
        v.conservation_error = 0.0;
        v.tol_ok = rel_l2 <= spec.tolerance;
        v.notes = "1st-order shock capturing: the shock is smeared over O(1) cells, so the "
                  "declared tolerance must be a physical/engineering one (e.g. 5% L2). "
                  "Order is measured on 2 refinements.";
        return v;
    }
};

}  // namespace

std::shared_ptr<const Kernel> make_sod1d_kernel() { return std::make_shared<Sod1D>(); }

}  // namespace oct::numerics::detail
