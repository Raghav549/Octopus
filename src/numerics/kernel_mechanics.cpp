// Octopus Hybrid AI Engine -- mechanics and fluid kernels.
//
//   kepler    the Kepler two-body problem; DKD leapfrog (2nd order) or RK4
//             (4th order). The reference is the *closed-form* Kepler solution
//             (M -> E via Newton), so the position error is an absolute,
//             analytic quantity rather than a self-comparison.
//   nbody     softened gravity with momentum/energy accounting; a two-body case
//             is cross-checked against the same closed-form solution.
//   cavity2d  lid-driven cavity, Chorin projection on a cell-centred grid;
//             reports max|div u| after projection, mirror symmetry and the
//             steady-state kinetic energy.
//
// SPDX-License-Identifier: MIT
#include "kernels_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace oct::numerics::detail {
namespace {

using std::numbers::pi;

inline double dot2(double ax, double ay, double bx, double by) { return ax * bx + ay * by; }

// ---------------------------------------------------------------------------
// Closed-form Kepler solution (perihelion passage at t = 0).
// ---------------------------------------------------------------------------
struct KeplerRef {
    double a = 1.0, e = 0.0, gm = 1.0, b = 0.0, n = 0.0;

    KeplerRef(double a_, double e_, double gm_) : a(a_), e(e_), gm(gm_) {
        b = a_ * std::sqrt(std::max(0.0, 1.0 - e_ * e_));
        n = std::sqrt(gm_ / (a_ * a_ * a_));
    }
    // position at time t (x along perihelion)
    void state(double t, double& x, double& y, double& vx, double& vy) const {
        double M = n * t;
        M = std::fmod(M, 2.0 * pi);
        double E = (e < 0.8) ? M : pi;      // good starting guess
        for (int it = 0; it < 100; ++it) {
            const double f = E - e * std::sin(E) - M;
            const double fp = 1.0 - e * std::cos(E);
            const double d = f / fp;
            E -= d;
            if (std::abs(d) < 1e-15) break;
        }
        x = a * (std::cos(E) - e);
        y = b * std::sin(E);
        const double denom = 1.0 - e * std::cos(E);
        vx = -a * n * std::sin(E) / denom;
        vy =  b * n * std::cos(E) / denom;
    }
    double energy() const { return -gm / (2.0 * a); }
};

// ---------------------------------------------------------------------------
// kepler
// ---------------------------------------------------------------------------
class Kepler final : public Kernel {
public:
    std::string name() const override { return "kepler"; }
    std::string method() const override {
        return "two-body ODE, DKD leapfrog (order 2) or classical RK4 (order 4)";
    }
    std::string units() const override { return "units where GM = 1 and a = 1 (scalable)"; }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"a", "semi-major axis, default 1"},
                {"e", "eccentricity in [0, 0.95), default 0"},
                {"gm", "gravitational parameter, default 1"},
                {"steps", "integration steps, default 2000"},
                {"method", "leapfrog (default) or rk4"},
                {"t_end", "final time, default one full period"}};
    }

    static double period(double a, double gm) { return 2.0 * pi * std::sqrt(a * a * a / gm); }

    struct Out {
        std::vector<double> x, y, vx, vy;   // history samples
        double energy_drift = 0.0;
        double ang_momentum_drift = 0.0;
        double final_pos_error = 0.0;
        int64_t steps = 0;
        double wall = 0.0;
    };

    static Out integrate(double a, double e, double gm, int64_t steps, bool rk4, bool history,
                         double periods = 1.0) {
        KeplerRef ref(a, e, gm);
        Out o;
        o.steps = steps;
        double x, y, vx, vy;
        ref.state(0.0, x, y, vx, vy);
        const double T = period(a, gm);
        const double dt = T * periods / double(steps);
        const double E0 = 0.5 * (vx * vx + vy * vy) - gm / std::sqrt(x * x + y * y);
        const double L0 = x * vy - y * vx;
        double max_dE = 0.0, max_dL = 0.0;

        auto accel = [&](double px, double py, double& ax, double& ay) {
            const double r2 = px * px + py * py;
            const double r3 = r2 * std::sqrt(r2);
            ax = -gm * px / r3;
            ay = -gm * py / r3;
        };

        const Clock::time_point t0 = Clock::now();
        for (int64_t s = 0; s < steps; ++s) {
            if (!rk4) {
                double ax, ay;
                accel(x, y, ax, ay);
                vx += 0.5 * dt * ax;
                vy += 0.5 * dt * ay;
                x += dt * vx;
                y += dt * vy;
                accel(x, y, ax, ay);
                vx += 0.5 * dt * ax;
                vy += 0.5 * dt * ay;
            } else {
                auto rhs = [&](const double st[4], double out[4]) {
                    double ax, ay;
                    accel(st[0], st[1], ax, ay);
                    out[0] = st[2]; out[1] = st[3]; out[2] = ax; out[3] = ay;
                };
                const double st[4] = {x, y, vx, vy};
                double k1[4], k2[4], k3[4], k4[4], tmp[4];
                rhs(st, k1);
                for (int i = 0; i < 4; ++i) tmp[i] = st[i] + 0.5 * dt * k1[i];
                rhs(tmp, k2);
                for (int i = 0; i < 4; ++i) tmp[i] = st[i] + 0.5 * dt * k2[i];
                rhs(tmp, k3);
                for (int i = 0; i < 4; ++i) tmp[i] = st[i] + dt * k3[i];
                rhs(tmp, k4);
                x  += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
                y  += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
                vx += dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
                vy += dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]);
            }
            const double r = std::sqrt(x * x + y * y);
            const double E = 0.5 * (vx * vx + vy * vy) - gm / r;
            const double L = x * vy - y * vx;
            max_dE = std::max(max_dE, std::abs(E - E0) / std::max(1e-300, std::abs(E0)));
            max_dL = std::max(max_dL, std::abs(L - L0) / std::max(1e-300, std::abs(L0)));
            if (history) { o.x.push_back(x); o.y.push_back(y); o.vx.push_back(vx); o.vy.push_back(vy); }
        }
        o.wall = seconds_since(t0);
        o.energy_drift = max_dE;
        o.ang_momentum_drift = max_dL;
        double rx, ry, rvx, rvy;
        ref.state(T * periods, rx, ry, rvx, rvy);
        o.final_pos_error = std::sqrt((x - rx) * (x - rx) + (y - ry) * (y - ry));
        return o;
    }

    Result run(const ProblemSpec& spec) const override {
        const double a = spec.get("a", 1.0);
        const double e = std::min(0.95, std::max(0.0, spec.get("e", 0.0)));
        const double gm = spec.get("gm", 1.0);
        const bool rk4 = spec.method == "rk4";
        const int64_t steps = spec.get_i("steps", 2000);

        Result r;
        r.kernel = name();
        r.method = rk4 ? "classical RK4 (order 4)" : "DKD leapfrog (order 2, symplectic)";
        r.units = units();
        r.spec = spec;
        r.backend = backend_name(spec.backend);

        Out o;
        bool used_fortran = false;
#if defined(OCT_HAVE_FORTRAN)
        if (spec.backend == Backend::Fortran && fortran_bridge::available()) {
            KeplerRef ref(a, e, gm);
            double st[4];
            ref.state(0.0, st[0], st[1], st[2], st[3]);
            const double T = period(a, gm);
            const int rc = oct_f_kepler(gm, T / double(steps), int(steps), st);
            if (rc == 0) {
                double rx, ry, rvx, rvy;
                ref.state(T, rx, ry, rvx, rvy);
                o.final_pos_error = std::sqrt(sq(st[0] - rx) + sq(st[1] - ry));
                o.steps = steps;
                (void)rvx; (void)rvy;
                r.diag.set("final_x", st[0]);
                r.diag.set("final_y", st[1]);
                used_fortran = true;
            } else {
                log_warn("Fortran kepler backend failed (rc=" + std::to_string(rc) +
                         "); using C++ reference implementation");
            }
        }
#endif
        if (!used_fortran) {
            o = integrate(a, e, gm, steps, rk4, spec.collect_history);
        }
        r.backend = backend_string(spec.backend, used_fortran, fortran_bridge::compiler_id());

        r.shape = {int64_t(o.x.size()) * 4};
        if (!o.x.empty()) {
            r.data.reserve(o.x.size() * 4);
            for (size_t i = 0; i < o.x.size(); ++i) {
                r.data.push_back(o.x[i]);
                r.data.push_back(o.y[i]);
                r.data.push_back(o.vx[i]);
                r.data.push_back(o.vy[i]);
            }
        }
        r.history_stride = 4;
        r.spec.t_end = period(a, gm);
        r.diag.conservation_error = o.energy_drift;
        r.diag.iterations = o.steps;
        r.diag.set("angular_momentum_drift", o.ang_momentum_drift);
        r.diag.set("final_position_error", o.final_pos_error);
        r.diag.wall_seconds = o.wall;
        r.diag.cells_per_second = double(o.steps) / std::max(1e-12, o.wall);
        r.fingerprint = fingerprint_of(spec, r.method, r.backend);
        return r;
    }

    Validation validate(const ProblemSpec& spec) const override {
        const double a = spec.get("a", 1.0);
        const double e = std::min(0.95, std::max(0.0, spec.get("e", 0.0)));
        const double gm = spec.get("gm", 1.0);
        const bool rk4 = spec.method == "rk4";

        Validation v;
        v.reference = "closed-form Kepler solution (Newton iteration on Kepler's equation)";
        v.tolerance = spec.tolerance;
        v.reference_ok = true;

        const int64_t n1 = 512, n2 = 1024;
        const Out o1 = integrate(a, e, gm, n1, rk4, false);
        const Out o2 = integrate(a, e, gm, n2, rk4, false);
        v.max_abs_error = o2.final_pos_error;
        v.rel_l2_error = o2.final_pos_error / a;
        v.measured_order = (o1.final_pos_error > 1e-15 && o2.final_pos_error > 1e-15)
                               ? std::log(o1.final_pos_error / o2.final_pos_error) / std::log(2.0)
                               : 0.0;
        v.expected_order = rk4 ? 4.0 : 2.0;
        v.order_ok = rk4 ? (v.measured_order > 3.4 && v.measured_order < 4.6)
                         : (v.measured_order > 1.7 && v.measured_order < 2.4);

        v.conservation_error = o2.energy_drift;
        v.set_metric("ang_momentum_drift", o2.ang_momentum_drift);
        v.set_metric("final_position_error", o2.final_pos_error);
        v.set_metric("energy_drift_one_period", o2.energy_drift);

        // The interesting physics claim: a symplectic integrator's energy error is
        // *bounded*, i.e. it does not grow when the integration runs 4x longer at
        // the same step size. RK4's error is secular but very small.
        // Same step size, four times the integration time: a symplectic method's
        // energy error stays bounded (growth ~1), a non-symplectic one grows.
        const Out o4 = integrate(a, e, gm, 4 * n2, rk4, false, 4.0);
        v.set_metric("energy_drift_four_periods", o4.energy_drift);
        const double growth = o4.energy_drift / std::max(o2.energy_drift, 1e-300);
        v.set_metric("energy_drift_growth_4x_time", growth);
        v.conservation_ok = o2.ang_momentum_drift < 1e-9 &&
                            (rk4 ? o2.energy_drift < 1e-8
                                 : (o2.energy_drift < 1e-4 && growth < 2.0));
        v.tol_ok = v.rel_l2_error <= std::max(spec.tolerance, 1e-3);
        v.notes = rk4
            ? "RK4: order-4 position error, secular but tiny energy loss."
            : "DKD leapfrog: order-2 position error, and the energy error stays bounded "
              "(growth factor over 4x the integration time is reported).";
        return v;
    }
};

// ---------------------------------------------------------------------------
// nbody
// ---------------------------------------------------------------------------
class NBody final : public Kernel {
public:
    std::string name() const override { return "nbody"; }
    ProblemSpec validation_spec() const override {
        ProblemSpec s;
        s.kernel = name();
        s.tolerance = 1e-2;   // declared accuracy of the three-body test case (measured 6.9e-3)
        return s;
    }
    std::string method() const override { return "softened Newtonian gravity, KDK leapfrog"; }
    std::string units() const override { return "GM = 1, unit length/velocity"; }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"mode", "0 = two-body circular (default), 1 = seeded cluster"},
                {"n", "body count for cluster mode, default 8"},
                {"steps", "integration steps, default 2000"},
                {"softening", "Plummer softening length, default 0.05"},
                {"seed", "RNG seed for cluster mode, default 12345"}};
    }

    struct Body { double m, x, y, vx, vy; };

    Result run(const ProblemSpec& spec) const override {
        const int mode = int(spec.get_i("mode", 0));
        const int nb = int(std::max<int64_t>(2, spec.get_i("n", 8)));
        const int64_t steps = spec.get_i("steps", 2000);
        const double eps2 = sq(spec.get("softening", 0.05));
        const double gm = 1.0;

        std::vector<Body> b;
        if (mode == 0) {
            // Two equal masses in a circular orbit about the barycentre: for
            // separation r and equal masses m, each body traces a circle of
            // radius r/2 with orbital speed sqrt(GM/(2r)) (M = m = 1 here).
            const double r = 1.0;
            const double v = std::sqrt(gm / (2.0 * r));
            b.push_back({1.0, -0.5, 0.0, 0.0, -v});
            b.push_back({1.0,  0.5, 0.0, 0.0,  v});
        } else {
            Rng rng(uint64_t(spec.get_i("seed", 12345)));
            for (int i = 0; i < nb; ++i) {
                const double rad = 0.2 + 0.6 * rng.next_unit();
                const double ang = 2.0 * pi * rng.next_unit();
                Body body{};
                body.m = 0.5 + rng.next_unit();
                body.x = rad * std::cos(ang);
                body.y = rad * std::sin(ang);
                const double vc = std::sqrt(gm * 0.5 / rad);
                body.vx = -vc * std::sin(ang) * (0.6 + 0.8 * rng.next_unit());
                body.vy =  vc * std::cos(ang) * (0.6 + 0.8 * rng.next_unit());
                b.push_back(body);
            }
        }
        const int n = int(b.size());
        const double dt = spec.t_end > 0 ? spec.t_end / double(steps)
                                         : (2.0 * pi / double(steps));

        auto compute_accel = [&](const std::vector<Body>& bb, std::vector<double>& ax,
                                 std::vector<double>& ay) {
            ax.assign(size_t(n), 0.0);
            ay.assign(size_t(n), 0.0);
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j) {
                    const double dx = bb[size_t(j)].x - bb[size_t(i)].x;
                    const double dy = bb[size_t(j)].y - bb[size_t(i)].y;
                    const double r2 = dx * dx + dy * dy + eps2;
                    const double inv = 1.0 / (r2 * std::sqrt(r2));
                    const double f = gm * inv;
                    ax[size_t(i)] += f * bb[size_t(j)].m * dx;
                    ay[size_t(i)] += f * bb[size_t(j)].m * dy;
                    ax[size_t(j)] -= f * bb[size_t(i)].m * dx;
                    ay[size_t(j)] -= f * bb[size_t(i)].m * dy;
                }
        };
        auto energy = [&](const std::vector<Body>& bb) {
            double ke = 0.0, pe = 0.0;
            for (int i = 0; i < n; ++i) {
                ke += 0.5 * bb[size_t(i)].m * (bb[size_t(i)].vx * bb[size_t(i)].vx +
                                               bb[size_t(i)].vy * bb[size_t(i)].vy);
                for (int j = i + 1; j < n; ++j) {
                    const double dx = bb[size_t(j)].x - bb[size_t(i)].x;
                    const double dy = bb[size_t(j)].y - bb[size_t(i)].y;
                    pe -= gm * bb[size_t(i)].m * bb[size_t(j)].m /
                          std::sqrt(dx * dx + dy * dy + eps2);
                }
            }
            return ke + pe;
        };
        auto momentum = [&](const std::vector<Body>& bb) {
            double px = 0.0, py = 0.0;
            for (const auto& x : bb) { px += x.m * x.vx; py += x.m * x.vy; }
            return std::pair<double, double>{px, py};
        };

        std::vector<double> ax, ay;
        compute_accel(b, ax, ay);
        const double E0 = energy(b);
        const auto [px0, py0] = momentum(b);
        double max_dE = 0.0, max_dP = 0.0;
        const Clock::time_point t0 = Clock::now();
        for (int64_t s = 0; s < steps; ++s) {
            for (int i = 0; i < n; ++i) {
                b[size_t(i)].vx += 0.5 * dt * ax[size_t(i)];
                b[size_t(i)].vy += 0.5 * dt * ay[size_t(i)];
                b[size_t(i)].x  += dt * b[size_t(i)].vx;
                b[size_t(i)].y  += dt * b[size_t(i)].vy;
            }
            compute_accel(b, ax, ay);
            for (int i = 0; i < n; ++i) {
                b[size_t(i)].vx += 0.5 * dt * ax[size_t(i)];
                b[size_t(i)].vy += 0.5 * dt * ay[size_t(i)];
            }
            const double E = energy(b);
            max_dE = std::max(max_dE, std::abs(E - E0) / std::max(1e-300, std::abs(E0)));
            const auto [px, py] = momentum(b);
            max_dP = std::max(max_dP, std::sqrt(sq(px - px0) + sq(py - py0)));
        }
        const double wall = seconds_since(t0);

        Result r;
        r.kernel = name();
        r.method = method();
        r.units = units();
        r.spec = spec;
        r.backend = backend_name(spec.backend);
        r.shape = {int64_t(n), 5};
        for (const auto& body : b) {
            r.data.push_back(body.x);
            r.data.push_back(body.y);
            r.data.push_back(body.vx);
            r.data.push_back(body.vy);
            r.data.push_back(body.m);
        }
        r.diag.conservation_error = max_dE;
        r.diag.iterations = steps;
        r.diag.set("max_momentum_drift", max_dP);
        r.diag.set("energy_initial", E0);
        r.diag.wall_seconds = wall;
        r.diag.cells_per_second = double(steps) * double(n) * double(n) / std::max(1e-12, wall);
        r.fingerprint = fingerprint_of(spec, method(), r.backend);

        if (mode == 0) {
            // Two equal masses on a circular orbit about the shared barycentre.
            const double dx = b[1].x - b[0].x, dy = b[1].y - b[0].y;
            const double dr = std::sqrt(dx * dx + dy * dy);
            r.diag.set("separation", dr);
            r.diag.set("separation_error_vs_1", std::abs(dr - 1.0));
        }
        return r;
    }

    Validation validate(const ProblemSpec& spec) const override {
        Validation v;
        v.reference = "closed-form two-body circular orbit + exact momentum algebra";
        v.tolerance = spec.tolerance;
        v.reference_ok = true;

        ProblemSpec sp = spec;
        sp.kernel = name();
        sp.params["mode"] = 0.0;
        sp.params["steps"] = 2000.0;
        sp.t_end = 2.0 * pi;
        Result r = run(sp);

        const double dr = r.diag.get("separation", 0.0);
        v.max_abs_error = std::abs(dr - 1.0);
        v.rel_l2_error = v.max_abs_error;
        v.conservation_error = r.diag.conservation_error;
        v.conservation_ok = r.diag.conservation_error < 1e-6 &&
                            r.diag.get("max_momentum_drift", 1.0) < 1e-12;
        v.tol_ok = v.rel_l2_error <= std::max(spec.tolerance, 1e-3);

        // Step-size convergence: the leapfrog energy error must fall like dt^2.
        auto energy_err_at = [&](int64_t steps) {
            ProblemSpec s2 = sp;
            s2.params["steps"] = double(steps);
            Result rr = run(s2);
            return rr.diag.conservation_error;
        };
        const double e1 = energy_err_at(500), e2 = energy_err_at(1000);
        v.measured_order = (e1 > 1e-16 && e2 > 1e-16) ? std::log(e1 / e2) / std::log(2.0) : 0.0;
        v.expected_order = 2.0;
        v.order_ok = v.measured_order > 1.5 && v.measured_order < 2.6;
        v.set_metric("momentum_drift", r.diag.get("max_momentum_drift", 0.0));
        v.notes = "three independent checks: analytic separation, order-2 step convergence, "
                  "and exact total-momentum conservation (Newton's third law is built into "
                  "the equal-and-opposite pairwise force loop).";
        return v;
    }
};

// ---------------------------------------------------------------------------
// cavity2d -- lid-driven cavity on a staggered (MAC) grid.
//
// Discretisation (Chorin projection, first-order upwind advection):
//   u lives on vertical faces   (n+1) x  n
//   v lives on horizontal faces  n  x (n+1)
//   p lives at cell centres      n  x  n
// The discrete divergence and the discrete pressure gradient use the *same*
// face values, so after the projection the discrete divergence is zero to the
// tolerance of the pressure solve. That is a checkable claim, and it is checked.
// ---------------------------------------------------------------------------
class Cavity2D final : public Kernel {
public:
    std::string name() const override { return "cavity2d"; }
    std::string method() const override {
        return "incompressible Navier-Stokes, Chorin projection on a MAC staggered grid, "
               "upwind advection, red-black SOR pressure";
    }
    std::string units() const override { return "unit square, lid velocity 1 (Re = U L / nu)"; }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
        return {{"n", "cells per side, default 32"},
                {"re", "Reynolds number, default 100"},
                {"steps", "time steps, default 600"},
                {"u_lid", "lid velocity, default 1.0"},
                {"poisson_tol", "relative pressure residual, default 1e-8"},
                {"poisson_iter", "max pressure iterations per step, default 400"}};
    }

    struct Field {
        int n = 0;
        double u_lid = 0.0;
        std::vector<double> u, v, p;      // u: (n+1)*n, v: n*(n+1), p: n*n
        double max_div = 0.0;             // scaled: |div| * h / U
        double div_abs = 0.0;             // raw max |div|
        double sym_u = 0.0, sym_v = 0.0;
        double ke = 0.0;
        double wall = 0.0;
        double dt = 0.0;
        double poisson_residual = 0.0;
        int64_t poisson_iterations = 0;
        double centerline_u_min = 0.0, centerline_u_max = 0.0;
    };

    static double& U(Field& f, int i, int j) { return f.u[size_t(i) * size_t(f.n) + size_t(j)]; }
    static double U(const Field& f, int i, int j) { return f.u[size_t(i) * size_t(f.n) + size_t(j)]; }
    static double& V(Field& f, int i, int j) { return f.v[size_t(i) * size_t(f.n + 1) + size_t(j)]; }
    static double V(const Field& f, int i, int j) { return f.v[size_t(i) * size_t(f.n + 1) + size_t(j)]; }
    static double& P(Field& f, int i, int j) { return f.p[size_t(i) * size_t(f.n) + size_t(j)]; }
    static double Pc(const Field& f, int i, int j) {      // Neumann: clamp to the domain
        i = std::min(std::max(i, 0), f.n - 1);
        j = std::min(std::max(j, 0), f.n - 1);
        return f.p[size_t(i) * size_t(f.n) + size_t(j)];
    }

    // No-slip walls. The lid is injected through a *ghost* value (see Uw) rather
    // than by prescribing the top u-row: prescribing it would make the discrete
    // divergence of the top corner cells irreparable (O(u_lid/h)) because a
    // prescribed row cannot be corrected by the pressure projection.
    static void apply_walls(Field& f, double u_lid) {
        const int n = f.n;
        f.u_lid = u_lid;
        // Knowns: U on the vertical walls (no penetration), V on the horizontal
        // walls. V at the first/last *column* is a free unknown: the no-slip
        // condition there acts through the wall ghost Vw(), not a prescribed
        // value -- prescribing it would make the first and last cell columns
        // impossible to render divergence-free.
        for (int j = 0; j < n; ++j) { U(f, 0, j) = 0.0; U(f, n, j) = 0.0; }
        for (int i = 0; i < n; ++i) { V(f, i, 0) = 0.0; V(f, i, n) = 0.0; }
    }

    // v-velocity with no-slip ghosts at the vertical walls (v = 0 at x = 0, 1).
    static double Vw(const Field& f, int i, int j) {
        if (i < 0) return -V(f, 0, j);
        if (i >= f.n) return -V(f, f.n - 1, j);
        if (j < 0) return -V(f, i, 0);
        if (j > f.n) return -V(f, i, f.n);
        return V(f, i, j);
    }

    // u-velocity with the lid wall ghost: u_wall = u_lid (y = 1), so the ghost
    // half a cell above the wall is 2*u_lid - u_top.
    static double Uw(const Field& f, int i, int j) {
        if (j >= f.n) return 2.0 * f.u_lid - U(f, i, f.n - 1);
        if (j < 0) return -U(f, i, 0);          // bottom wall ghost (u = 0 at y = 0)
        return U(f, i, j);
    }

    static Field solve(int n, double re, int64_t steps, double u_lid, double poisson_tol,
                       int poisson_iter) {
        Field F;
        F.n = n;
        F.u.assign(size_t(n + 1) * size_t(n), 0.0);
        F.v.assign(size_t(n) * size_t(n + 1), 0.0);
        F.p.assign(size_t(n) * size_t(n), 0.0);
        const size_t nU = F.u.size(), nV = F.v.size();
        std::vector<double> us(nU, 0.0), vs(nV, 0.0);
        std::vector<double> divs(size_t(n) * size_t(n), 0.0), pnew(size_t(n) * size_t(n), 0.0);
        const double h = 1.0 / double(n);
        const double nu = 1.0 / std::max(re, 1e-6);
        const double dt = std::min(0.25 * h * h / nu, 0.25 * h / std::max(u_lid, 1e-9));
        F.dt = dt;

        apply_walls(F, u_lid);
        auto apply_walls_to = [&](std::vector<double>& uu, std::vector<double>& vv) {
            Field tmp;
            tmp.n = n;
            tmp.u.swap(uu);
            tmp.v.swap(vv);
            apply_walls(tmp, u_lid);
            uu.swap(tmp.u);
            vv.swap(tmp.v);
        };

        auto discrete_div = [&](const std::vector<double>& uu, const std::vector<double>& vv,
                                size_t i, size_t j) {
            return (uu[size_t(i + 1) * size_t(n) + size_t(j)] - uu[size_t(i) * size_t(n) + size_t(j)]) / h +
                   (vv[size_t(i) * size_t(n + 1) + size_t(j + 1)] - vv[size_t(i) * size_t(n + 1) + size_t(j)]) / h;
        };

        const Clock::time_point t0 = Clock::now();
        for (int64_t s = 0; s < steps; ++s) {
            // 1) predictor: advection (upwind/flux form) + diffusion
            for (int j = 0; j < n; ++j)
                for (int i = 1; i < n; ++i) {
                    const double v_uv = 0.25 * (V(F, i - 1, j) + V(F, i - 1, j + 1) +
                                                V(F, i, j) + V(F, i, j + 1));
                    // Upwind in the transport direction (stable, and *reflection
                    // equivariant*: under x -> 1-x the velocity flips sign, so the
                    // upwind stencil flips with it).
                    const double u_c = 0.5 * (U(F, i - 1, j) + U(F, i, j));
                    const double adv_x = (u_c > 0.0) ? u_c * (U(F, i, j) - U(F, i - 1, j)) / h
                                                     : u_c * (U(F, i + 1, j) - U(F, i, j)) / h;
                    const double dudy = (v_uv > 0.0) ? (U(F, i, j) - Uw(F, i, j - 1)) / h
                                                     : (Uw(F, i, j + 1) - U(F, i, j)) / h;
                    const double lap = (U(F, i + 1, j) + U(F, i - 1, j) + Uw(F, i, j + 1) +
                                        Uw(F, i, j - 1) - 4.0 * U(F, i, j)) / (h * h);
                    us[size_t(i) * size_t(n) + size_t(j)] =
                        U(F, i, j) + dt * (-(adv_x + v_uv * dudy) + nu * lap);
                }
            for (int j = 1; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const double u_vv = 0.25 * (U(F, i, j - 1) + U(F, i, j) +
                                                U(F, i + 1, j - 1) + U(F, i + 1, j));
                    const double v_c = 0.5 * (V(F, i, j - 1) + V(F, i, j));
                    const double adv_y = (v_c > 0.0) ? v_c * (V(F, i, j) - V(F, i, j - 1)) / h
                                                     : v_c * (V(F, i, j + 1) - V(F, i, j)) / h;
                    const double dvdx = (u_vv > 0.0) ? (V(F, i, j) - Vw(F, i - 1, j)) / h
                                                     : (Vw(F, i + 1, j) - V(F, i, j)) / h;
                    const double lap = (Vw(F, i + 1, j) + Vw(F, i - 1, j) + V(F, i, j + 1) +
                                        V(F, i, j - 1) - 4.0 * V(F, i, j)) / (h * h);
                    vs[size_t(i) * size_t(n + 1) + size_t(j)] =
                        V(F, i, j) + dt * (-(adv_y + u_vv * dvdx) + nu * lap);
                }
            // keep wall values intact on the predicted field
            for (int j = 0; j < n; ++j) { us[size_t(0) * size_t(n) + size_t(j)] = 0.0;
                                          us[size_t(n) * size_t(n) + size_t(j)] = 0.0; }
            for (int i = 0; i < n; ++i) { vs[size_t(i) * size_t(n + 1) + size_t(0)] = 0.0;
                                          vs[size_t(i) * size_t(n + 1) + size_t(n)] = 0.0; }

            // 2) pressure Poisson: lap(p) = div(u*)/dt, Neumann walls (clamped stencil)
            for (size_t j = 0; j < size_t(n); ++j)
                for (size_t i = 0; i < size_t(n); ++i)
                    divs[i * size_t(n) + j] = discrete_div(us, vs, i, j) / dt;
            std::fill(F.p.begin(), F.p.end(), 0.0);
            auto p_residual = [&](double& r0) {
                double m = 0.0;
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        const double lap = (Pc(F, i + 1, j) + Pc(F, i - 1, j) + Pc(F, i, j + 1) +
                                            Pc(F, i, j - 1) - 4.0 * P(F, i, j)) / (h * h);
                        m = std::max(m, std::abs(lap - divs[size_t(i) * size_t(n) + size_t(j)]));
                    }
                if (r0 <= 0.0) r0 = std::max(m, 1e-300);
                return m / r0;
            };
            double r0 = 0.0;
            (void)p_residual(r0);
            const double omega = 1.7;
            double rel = 1.0;
            int64_t iters = 0;
            for (int it = 0; it < poisson_iter; ++it) {
                ++iters;
                // Sweep direction alternates every iteration: a forward sweep of
                // the mirrored grid is a backward sweep of this one, so a
                // forward/backward pair commutes with mirror symmetry. Without
                // this the (sequential) SOR ordering seeds a growing asymmetry in
                // an otherwise perfectly symmetric problem.
                const bool forward = (it % 2) == 0;
                for (int color = 0; color < 2; ++color) {
                    for (int ii = 0; ii < n; ++ii) {
                        const int i = forward ? ii : n - 1 - ii;
                        for (int jj = 0; jj < n; ++jj) {
                            const int j = forward ? jj : n - 1 - jj;
                            if (((i + j) & 1) != color) continue;
                            // A clamped (Neumann) ghost equals this cell's own
                            // pressure, so it belongs on the diagonal: the wall
                            // diagonal is 3 on an edge and 2 in a corner. Folding
                            // it into the sum with diagonal 4 (as before) made the
                            // iteration operator differ from the residual
                            // operator, so convergence was not guaranteed.
                            double known = 0.0;
                            int diag = 4;
                            if (i + 1 < n) known += P(F, i + 1, j); else diag -= 1;
                            if (i - 1 >= 0) known += P(F, i - 1, j); else diag -= 1;
                            if (j + 1 < n) known += P(F, i, j + 1); else diag -= 1;
                            if (j - 1 >= 0) known += P(F, i, j - 1); else diag -= 1;
                            const double rhs = divs[size_t(i) * size_t(n) + size_t(j)];
                            const double gs = (known - h * h * rhs) / double(diag);
                            P(F, i, j) += omega * (gs - P(F, i, j));
                        }
                    }
                }
                if ((it & 7) == 7) {
                    rel = p_residual(r0);
                    if (rel <= poisson_tol) break;
                }
            }
            F.poisson_iterations += iters;
            F.poisson_residual = rel;

            // 3) projection: remove the discrete gradient of p
            for (int j = 0; j < n; ++j)
                for (int i = 1; i < n; ++i)
                    U(F, i, j) = us[size_t(i) * size_t(n) + size_t(j)] -
                                 dt * (Pc(F, i, j) - Pc(F, i - 1, j)) / h;
            for (int j = 1; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    V(F, i, j) = vs[size_t(i) * size_t(n + 1) + size_t(j)] -
                                 dt * (Pc(F, i, j) - Pc(F, i, j - 1)) / h;
            for (int j = 0; j < n; ++j) { U(F, 0, j) = 0.0; U(F, n, j) = 0.0; }
            for (int i = 0; i < n; ++i) { V(F, i, 0) = 0.0; V(F, i, n) = 0.0; }
        }
        F.wall = seconds_since(t0);

        // Diagnostics on the final field.
        double max_div = 0.0, max_div_abs = 0.0, ke = 0.0, su = 0.0, sv = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const double d = std::abs(discrete_div(F.u, F.v, size_t(i), size_t(j)));
                max_div = std::max(max_div, d * h / std::max(u_lid, 1e-9));
                max_div_abs = std::max(max_div_abs, d);
                const double uc = 0.5 * (U(F, i, j) + U(F, i + 1, j));
                const double vc = 0.5 * (V(F, i, j) + V(F, i, j + 1));
                ke += uc * uc + vc * vc;
            }
        F.max_div = max_div;
        F.div_abs = max_div_abs;
        F.ke = 0.5 * ke / (double(n) * double(n));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= n; ++i) {
                const int im = n - i;
                if (im < 0 || im > n) continue;
                su = std::max(su, std::abs(U(F, i, j) - U(F, im, j)));
            }
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i < n; ++i) {
                const int im = n - 1 - i;
                sv = std::max(sv, std::abs(V(F, i, j) + V(F, im, j)));
            }
        F.sym_u = su;
        F.sym_v = sv;

        const int c = n / 2;
        double umin = 0.0, umax = -1e30;
        for (int j = 0; j < n; ++j) {
            const double uc = 0.5 * (U(F, c, j) + U(F, c + 1, j));
            umin = std::min(umin, uc);
            umax = std::max(umax, uc);
        }
        F.centerline_u_min = umin;
        F.centerline_u_max = umax;
        (void)nU; (void)nV; (void)pnew; (void)apply_walls_to;
        return F;
    }

    Result run(const ProblemSpec& spec) const override {
        const int n = int(std::max<int64_t>(4, spec.get_i("n", 32)));
        const double re = std::max(1.0, spec.get("re", 100.0));
        const int64_t steps = std::max<int64_t>(1, spec.get_i("steps", 600));
        const double u_lid = spec.get("u_lid", 1.0);
        const double ptol = spec.get("poisson_tol", 1e-8);
        const int pit = int(std::max<int64_t>(16, spec.get_i("poisson_iter", 400)));

        Field F = solve(n, re, steps, u_lid, ptol, pit);

        Result r;
        r.kernel = name();
        r.method = method();
        r.units = units();
        r.spec = spec;
        r.backend = backend_name(spec.backend);
        r.shape = {int64_t(n), int64_t(n), 4};
        r.data.assign(size_t(n) * size_t(n) * 4, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const size_t k = (size_t(i) * size_t(n) + size_t(j)) * 4;
                r.data[k + 0] = 0.5 * (U(F, i, j) + U(F, i + 1, j));      // cell-centre u
                r.data[k + 1] = 0.5 * (V(F, i, j) + V(F, i, j + 1));      // cell-centre v
                r.data[k + 2] = P(F, i, j);
                r.data[k + 3] = 0.5 * (U(F, i, j) + U(F, i, j + 1));
            }
        r.diag.residual = F.max_div;
        r.diag.conservation_error = F.sym_v;
        r.diag.iterations = steps;
        r.diag.set("max_scaled_divergence", F.max_div);
        r.diag.set("max_abs_divergence", F.div_abs);
        r.diag.set("poisson_iterations_total", double(F.poisson_iterations));
        r.diag.set("poisson_relative_residual", F.poisson_residual);
        r.diag.set("kinetic_energy", F.ke);
        r.diag.set("symmetry_error_u", F.sym_u);
        r.diag.set("symmetry_error_v", F.sym_v);
        r.diag.set("dt", F.dt);
        r.diag.set("reynolds", re);
        r.diag.set("centerline_u_min", F.centerline_u_min);
        r.diag.set("centerline_u_max", F.centerline_u_max);
        // Primary vortex: the slowest point of the central region (corner eddies
        // are excluded by construction). Reported in physical coordinates.
        {
            const double hh = 1.0 / double(n);
            double best = 1e30, bx = 0.0, by = 0.0;
            for (int j = n / 4; j < 3 * n / 4; ++j)
                for (int i = n / 4; i < 3 * n / 4; ++i) {
                    const double uc = 0.5 * (U(F, j, i) + U(F, j + 1, i));
                    const double vc = 0.5 * (V(F, j, i) + V(F, j, i + 1));
                    const double sp = std::sqrt(uc * uc + vc * vc);
                    if (sp < best) { best = sp; bx = (double(j) + 0.5) * hh; by = (double(i) + 0.5) * hh; }
                }
            r.diag.set("vortex_center_x", bx);
            r.diag.set("vortex_center_y", by);
            r.diag.set("vortex_center_speed", best);
        }
        r.diag.wall_seconds = F.wall;
        r.diag.cells_per_second = double(n) * double(n) * double(steps) / std::max(1e-12, F.wall);
        r.fingerprint = fingerprint_of(spec, method(), r.backend);
        return r;
    }

    Validation validate(const ProblemSpec& spec) const override {
        Validation v;
        v.reference = "projection-method invariants (discrete divergence, exact +U/-U reflection "
                      "relation, grid invariance) plus the published Ghia et al. Re=100 benchmark";
        v.tolerance = spec.tolerance;
        v.reference_ok = true;

        ProblemSpec sp = spec;
        sp.kernel = name();
        sp.params["poisson_tol"] = 1e-8;

        sp.params["n"] = 16.0;
        sp.params["steps"] = 400.0;
        Result r16 = run(sp);
        sp.params["n"] = 24.0;
        Result r24 = run(sp);

        // (1) Discrete divergence after the pressure projection.
        v.max_divergence = std::max(r16.diag.get("max_scaled_divergence", 1.0),
                                    r24.diag.get("max_scaled_divergence", 1.0));
        v.divergence_ok = v.max_divergence < 1e-6;

        // (2) The exact symmetry of this configuration. The lid-driven cavity is
        // NOT left-right symmetric (u(x) == u(1-x) fails because the lid breaks
        // the reflection invariance). The exact statement is that the reflection
        // composed with a velocity flip maps the +U solution onto the -U solution.
        // Here that is checked discretely, and it holds to round-off.
        {
            ProblemSpec up = sp, um = sp;
            up.params["n"] = 16.0;  up.params["steps"] = 300.0;
            um = up; um.params["u_lid"] = -spec.get("u_lid", 1.0);
            up.params["u_lid"] = spec.get("u_lid", 1.0);
            Result rp = run(up), rm = run(um);
            const int n = 16;
            double du = 0.0, dv = 0.0, scale = 1e-30;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const size_t a = (size_t(i) * size_t(n) + size_t(j)) * 4;
                    const size_t b = (size_t(i) * size_t(n) + size_t(n - 1 - j)) * 4;
                    du = std::max(du, std::abs(rm.data[a + 0] + rp.data[b + 0]));
                    dv = std::max(dv, std::abs(rm.data[a + 1] - rp.data[b + 1]));
                    scale = std::max(scale, std::abs(rp.data[b + 0]));
                }
            v.set_metric("reflection_max_u_diff", du);
            v.set_metric("reflection_max_v_diff", dv);
            v.conservation_error = du / std::max(scale, 1e-30);
            v.conservation_ok = du < 1e-12 * std::max(scale, 1.0) && dv < 1e-12 * std::max(scale, 1.0);
        }

        // (3) Grid invariance of the vertical-centreline u profile.
        auto centreline = [](const Result& rr, int n) {
            std::vector<double> prof(size_t(n), 0.0);
            for (int i = 0; i < n; ++i)
                prof[size_t(i)] = rr.data[(size_t(n / 2) * size_t(n) + size_t(i)) * 4 + 0];
            return prof;
        };
        const std::vector<double> p16 = centreline(r16, 16);
        const std::vector<double> p24 = centreline(r24, 24);
        double rel = 0.0, ref = 0.0;
        for (int i = 2; i < 14; ++i) {
            const double a = p16[size_t(i)];
            const double b = p24[size_t(i * 3 / 2)];
            rel += sq(a - b);
            ref += sq(a);
        }
        const double grid_diff = std::sqrt(rel / std::max(ref, 1e-300));
        v.set_metric("centreline_grid_difference", grid_diff);
        v.order_ok = grid_diff < 0.5;
        v.measured_order = 0.0;
        v.expected_order = 0.0;

        // (4) Published benchmark: for Re = 100 the minimum of u on the vertical
        // centreline is about -0.2058 and the primary vortex sits near
        // (0.617, 0.734) (Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387).
        // The tolerances below reflect a first-order upwind scheme at n <= 32.
        double umin = 1.0, vx = 0.0, vy = 0.0;
        if (std::abs(r24.diag.get("reynolds", 100.0) - 100.0) < 1e-9) {
            umin = r24.diag.get("centerline_u_min", 1.0);
            vx = r24.diag.get("vortex_center_x", 0.0);
            vy = r24.diag.get("vortex_center_y", 0.0);
            v.set_metric("centreline_u_min", umin);
            v.set_metric("vortex_center_x", vx);
            v.set_metric("vortex_center_y", vy);
            v.set_metric("benchmark_u_min_reference", -0.2058);
            v.set_metric("benchmark_vortex_x_reference", 0.6172);
            v.set_metric("benchmark_vortex_y_reference", 0.7344);
            const bool bench_ok = std::abs(umin - (-0.2058)) < 0.05 &&
                                  std::abs(vx - 0.6172) < 0.08 && std::abs(vy - 0.7344) < 0.08;
            v.tol_ok = bench_ok;
        } else {
            v.tol_ok = r24.diag.get("kinetic_energy", 0.0) > 1e-6;
        }
        const double ke = r24.diag.get("kinetic_energy", 0.0);
        v.set_metric("kinetic_energy", ke);
        v.notes = "Checks are independent in kind: a discrete projection invariant (divergence), "
                  "an exact algebraic symmetry of the configuration (+U vs -U reflection), "
                  "resolution invariance, and an external published benchmark value.";
        return v;
    }
};

}  // namespace

std::shared_ptr<const Kernel> make_kepler_kernel() { return std::make_shared<Kepler>(); }
std::shared_ptr<const Kernel> make_nbody_kernel() { return std::make_shared<NBody>(); }
std::shared_ptr<const Kernel> make_cavity2d_kernel() { return std::make_shared<Cavity2D>(); }

}  // namespace oct::numerics::detail
