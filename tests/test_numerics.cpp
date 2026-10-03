// Numerical kernel tests: correctness against independent references.
//
// These tests are deliberately strict: they assert the *measured* convergence
// order and the conservation diagnostics, not merely that the code ran.
// SPDX-License-Identifier: MIT
#include "harness.hpp"

#include "octopus/numerics.hpp"

#include <cmath>

using namespace oct;
using namespace oct::numerics;

namespace {

ProblemSpec base_spec(const std::string& kernel) {
    ProblemSpec s;
    s.kernel = kernel;
    s.backend = Backend::CxxLd;
    return s;
}

}  // namespace

OCT_TEST(numerics, kernel_library_inventory) {
    const auto& lib = KernelLibrary::instance();
    OCT_CHECK(lib.names().size() >= 8);
    for (const auto& n : {"sod1d", "heat2d", "poisson2d", "kepler", "nbody", "cavity2d", "gas", "linsolve"})
        OCT_CHECK_MSG(lib.find(n) != nullptr, n << " kernel must be registered");
    for (const auto& d : lib.describe_all()) OCT_NOTE(d);
    // Every kernel must declare its units and parameters (no silent unknowns).
    for (const auto& n : lib.names()) {
        auto k = lib.find(n);
        OCT_CHECK(!k->units().empty());
        OCT_CHECK(!k->method().empty());
        OCT_CHECK(!k->parameters().empty());
    }
}

OCT_TEST(numerics, sod1d_matches_exact_riemann) {
    auto k = KernelLibrary::instance().find("sod1d");
    ProblemSpec s = base_spec("sod1d");
    s.params["n"] = 400;
    s.t_end = 0.2;
    s.tolerance = 0.05;      // 5% L2: a physical tolerance for a 1st-order scheme
    Validation v = k->validate(s);
    OCT_NOTE("sod1d rel_l2=" << v.rel_l2_error << " order=" << v.measured_order);
    OCT_CHECK(v.reference_ok);                    // exact solver converged
    OCT_CHECK(v.tol_ok);                          // within declared tolerance
    OCT_CHECK(v.order_ok);                        // observed order near 1
    OCT_NEAR(v.measured_order, 1.0, 0.45);
}

OCT_TEST(numerics, sod1d_conserves_mass) {
    auto k = KernelLibrary::instance().find("sod1d");
    ProblemSpec s = base_spec("sod1d");
    s.params["n"] = 200;
    s.t_end = 0.2;
    Result r = k->run(s);
    const double mass = r.diag.get("mass", 0.0);
    // Default Sod: 1.0 left / 0.125 right over [-1,1] -> mass = 1.0*1 + 0.125*1.
    OCT_NEAR(mass, 1.125, 1e-12);
    OCT_CHECK(r.x.size() == 200);
    OCT_CHECK(r.data.size() == 200 * 4);
    // Physical admissibility: density and pressure must stay positive.
    for (int i = 0; i < 200; ++i) {
        OCT_CHECK(r.data[size_t(i) * 4 + 1] > 0.0);
        OCT_CHECK(r.data[size_t(i) * 4 + 3] > 0.0);
    }
}

OCT_TEST(numerics, sod1d_fortran_matches_cxx_when_available) {
    if (!KernelLibrary::instance().fortran_available()) OCT_SKIP("Fortran backend absent");
    auto k = KernelLibrary::instance().find("sod1d");
    ProblemSpec a = base_spec("sod1d");
    a.params["n"] = 200;
    a.t_end = 0.15;
    ProblemSpec b = a;
    b.backend = Backend::Fortran;
    Result ra = k->run(a), rb = k->run(b);
    OCT_CHECK(ra.data.size() == rb.data.size());
    double maxdiff = 0.0;
    for (size_t i = 0; i < ra.data.size(); ++i)
        maxdiff = std::max(maxdiff, std::abs(ra.data[i] - rb.data[i]));
    OCT_NOTE("C++ vs Fortran max abs difference: " << maxdiff);
    OCT_CHECK(maxdiff < 1e-8 * (1.0 + reduce::max(ra.data)));
}

OCT_TEST(numerics, heat2d_manufactured_solution_and_order) {
    auto k = KernelLibrary::instance().find("heat2d");
    ProblemSpec s = base_spec("heat2d");
    s.t_end = 0.05;
    s.tolerance = 5e-3;
    Validation v = k->validate(s);
    OCT_NOTE("heat2d rel_l2=" << v.rel_l2_error << " order=" << v.measured_order);
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.tol_ok);
    OCT_CHECK(v.order_ok);
    OCT_CHECK(v.conservation_ok);
    OCT_NEAR(v.measured_order, 2.0, 0.3);
}

OCT_TEST(numerics, poisson2d_residual_and_cg_crosscheck) {
    auto k = KernelLibrary::instance().find("poisson2d");
    ProblemSpec s = base_spec("poisson2d");
    s.params["n"] = 32;
    Validation v = k->validate(s);
    OCT_NOTE("poisson order=" << v.measured_order
            << " cg_diff=" << v.metric("cg_cross_check_rel_diff")
            << " cg_iters=" << v.metric("cg_iterations"));
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.order_ok);                    // O(h^2)
    OCT_CHECK(v.conservation_ok);             // SOR agrees with the CG solve
    OCT_CHECK(v.tol_ok);
    OCT_CHECK(v.metric("cg_cross_check_rel_diff") < 1e-6);
}

OCT_TEST(numerics, kepler_leapfrog_order_and_conservation) {
    auto k = KernelLibrary::instance().find("kepler");
    ProblemSpec s = base_spec("kepler");
    s.params["e"] = 0.3;
    s.tolerance = 1e-4;
    Validation v = k->validate(s);
    OCT_NOTE("kepler leapfrog order=" << v.measured_order
            << " energy_drift=" << v.conservation_error
            << " L_drift=" << v.metric("ang_momentum_drift"));
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.order_ok);         // measured order ~ 2
    OCT_CHECK(v.conservation_ok);  // bounded energy drift, tiny L drift
    OCT_CHECK(v.tol_ok);
}

OCT_TEST(numerics, kepler_rk4_reaches_fourth_order) {
    auto k = KernelLibrary::instance().find("kepler");
    ProblemSpec s = base_spec("kepler");
    s.method = "rk4";
    s.params["e"] = 0.3;
    s.tolerance = 1e-6;
    Validation v = k->validate(s);
    OCT_NOTE("kepler rk4 order=" << v.measured_order << " pos_err=" << v.max_abs_error);
    OCT_CHECK(v.order_ok);
    OCT_CHECK(v.measured_order > 3.4);
}

OCT_TEST(numerics, kepler_eccentric_orbit_closes) {
    auto k = KernelLibrary::instance().find("kepler");
    ProblemSpec s = base_spec("kepler");
    s.method = "rk4";
    s.params["e"] = 0.6;
    s.params["steps"] = 4096;
    Result r = k->run(s);
    // After exactly one period the body must return to perihelion (1-e, 0).
    const double pos_err = r.diag.get("final_position_error", 1.0);
    OCT_NOTE("e=0.6 orbit closure error after one period: " << pos_err);
    OCT_CHECK(pos_err < 1e-6);
    OCT_CHECK(r.diag.get("angular_momentum_drift", 1.0) < 1e-9);
}

OCT_TEST(numerics, nbody_conserves_momentum_exactly) {
    auto k = KernelLibrary::instance().find("nbody");
    ProblemSpec s = base_spec("nbody");
    Validation v = k->validate(s);
    OCT_NOTE("nbody sep_err=" << v.max_abs_error << " order=" << v.measured_order
            << " momentum_drift=" << v.metric("momentum_drift"));
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.conservation_ok);
    OCT_CHECK(v.order_ok);
    // Pairwise equal-and-opposite forces -> momentum conserved to round-off.
    OCT_CHECK(v.metric("momentum_drift") < 1e-12);
}

OCT_TEST(numerics, cavity2d_is_divergence_free_and_symmetric) {
    auto k = KernelLibrary::instance().find("cavity2d");
    ProblemSpec s = base_spec("cavity2d");
    s.params["n"] = 16;
    s.params["steps"] = 200;
    Validation v = k->validate(s);
    OCT_NOTE("cavity max_div=" << v.max_divergence << " sym=" << v.max_abs_error
            << " ke=" << v.metric("kinetic_energy"));
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.divergence_ok);
    OCT_CHECK(v.conservation_ok);
    OCT_CHECK(v.tol_ok);
}

OCT_TEST(numerics, gas_thermodynamics_identity) {
    auto k = KernelLibrary::instance().find("gas");
    ProblemSpec s = base_spec("gas");
    s.params["ratio"] = 4.0;
    Validation v = k->validate(s);
    OCT_NOTE("gas ds/R=" << v.conservation_error << " pv^gamma_rel="
            << v.metric("pv_gamma_invariance"));
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.conservation_ok);
    OCT_CHECK(v.tol_ok);
    OCT_CHECK(v.order_ok);
}

OCT_TEST(numerics, linsolve_residual_and_honest_ill_conditioning) {
    auto k = KernelLibrary::instance().find("linsolve");
    ProblemSpec s = base_spec("linsolve");
    Validation v = k->validate(s);
    OCT_NOTE("hilbert cond=" << v.metric("hilbert_condition")
            << " err=" << v.metric("hilbert_solution_error")
            << " resid=" << v.metric("hilbert_residual"));
    OCT_CHECK(v.tol_ok);              // SPD case: residual at machine level
    OCT_CHECK(v.conservation_ok);     // LU and CG agree
    OCT_CHECK(v.order_ok);            // Hilbert: small residual, large error, as predicted
    OCT_CHECK(v.metric("lu_vs_cg_max_diff") < 1e-8);
}

OCT_TEST(numerics, invalid_specs_are_rejected_not_guessed) {
    auto k = KernelLibrary::instance().find("sod1d");
    OCT_CHECK(k != nullptr);
    ProblemSpec s = base_spec("sod1d");
    s.params["n"] = 10;      // below the documented minimum -> clamped, must still run
    Result r = k->run(s);
    OCT_CHECK(r.data.size() >= 8 * 4);
    OCT_CHECK(std::isfinite(reduce::max(r.data)));

    // Kernels must be findable by exact name only: no silent fallback.
    OCT_CHECK(KernelLibrary::instance().find("sod2d") == nullptr);
    OCT_CHECK(KernelLibrary::instance().find("sod") == nullptr);
}
