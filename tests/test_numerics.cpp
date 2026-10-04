// Numerical kernel tests: Fortran 2023 Absolute Physics Core & Hardware-Skip Gate.
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// C++ reference logic replacements (`Backend::CxxLd`) are strictly banned.
// When `gfortran`/`lfortran` is available on the build host, the live Fortran
// 2023 numerical kernels are validated against exact mathematical invariants.
// When `gfortran`/`lfortran` is absent on the build host, the hardware-skip
// gate test verifies that every kernel reports `UNSUPPORTED_HARDWARE_SKIP`
// with zero C++ approximation data, and the live Fortran blocks are flagged
// explicitly as `[UNSUPPORTED_HARDWARE_SKIP]`.
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
    s.backend = Backend::Fortran;
    return s;
}

}  // namespace

OCT_TEST(numerics, kernel_library_inventory) {
    const auto& lib = KernelLibrary::instance();
    OCT_CHECK(lib.names().size() >= 9);
    for (const auto& n : {"sod1d", "heat2d", "poisson2d", "kepler", "nbody",
                          "cavity2d", "gas", "linsolve", "tensor_field"})
        OCT_CHECK_MSG(lib.find(n) != nullptr, n << " kernel must be registered");
    for (const auto& d : lib.describe_all()) OCT_NOTE(d);
    for (const auto& n : lib.names()) {
        auto k = lib.find(n);
        OCT_CHECK(!k->units().empty());
        OCT_CHECK(!k->method().empty());
        OCT_CHECK(!k->parameters().empty());
    }
}

OCT_TEST(numerics, hardware_skip_gate_bans_cxx_fallback) {
    const auto& lib = KernelLibrary::instance();
    for (const auto& name : lib.names()) {
        auto k = lib.find(name);
        OCT_CHECK(k != nullptr);
        ProblemSpec s = k->validation_spec();
        Result r = k->run(s);
        Validation v = k->validate(s);
        if (!lib.fortran_available()) {
            OCT_CHECK_MSG(r.hardware_skipped, name << " must set hardware_skipped when Fortran absent");
            OCT_CHECK_MSG(r.status.is_hardware_skip(), name << " must return UNSUPPORTED_HARDWARE_SKIP");
            OCT_CHECK_MSG(r.data.empty(), name << " must not substitute C++ fallback data");
            OCT_CHECK_MSG(v.hardware_skipped, name << " validation must set hardware_skipped");
            OCT_CHECK_MSG(r.backend.find("UNSUPPORTED_HARDWARE_SKIP") != std::string::npos,
                          name << " backend string must declare UNSUPPORTED_HARDWARE_SKIP");
        } else {
            OCT_CHECK(!r.hardware_skipped);
            OCT_CHECK(r.status.is_ok());
            OCT_CHECK(!r.data.empty());
            OCT_CHECK(v.accepted());
        }
    }
    // Kernels must be findable by exact name only: no silent fallback.
    OCT_CHECK(lib.find("sod2d") == nullptr);
    OCT_CHECK(lib.find("sod") == nullptr);
}

OCT_TEST_REQ_FORTRAN(numerics, sod1d_matches_exact_riemann) {
    auto k = KernelLibrary::instance().find("sod1d");
    ProblemSpec s = base_spec("sod1d");
    s.params["n"] = 400;
    s.t_end = 0.2;
    s.tolerance = 0.05;
    Validation v = k->validate(s);
    OCT_NOTE("sod1d rel_l2=" << v.rel_l2_error << " order=" << v.measured_order);
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.tol_ok);
    OCT_CHECK(v.order_ok);
    OCT_NEAR(v.measured_order, 1.0, 0.45);
}

OCT_TEST_REQ_FORTRAN(numerics, sod1d_conserves_mass) {
    auto k = KernelLibrary::instance().find("sod1d");
    ProblemSpec s = base_spec("sod1d");
    s.params["n"] = 200;
    s.t_end = 0.2;
    Result r = k->run(s);
    const double mass = r.diag.get("mass", 0.0);
    OCT_NEAR(mass, 1.125, 1e-12);
    OCT_CHECK(r.x.size() == 200);
    OCT_CHECK(r.data.size() == 200 * 4);
    for (int i = 0; i < 200; ++i) {
        OCT_CHECK(r.data[size_t(i) * 4 + 1] > 0.0);
        OCT_CHECK(r.data[size_t(i) * 4 + 3] > 0.0);
    }
}

OCT_TEST_REQ_FORTRAN(numerics, heat2d_manufactured_solution_and_order) {
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

OCT_TEST_REQ_FORTRAN(numerics, poisson2d_residual_and_order) {
    auto k = KernelLibrary::instance().find("poisson2d");
    ProblemSpec s = base_spec("poisson2d");
    s.params["n"] = 32;
    Validation v = k->validate(s);
    OCT_NOTE("poisson order=" << v.measured_order);
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.order_ok);
    OCT_CHECK(v.conservation_ok);
    OCT_CHECK(v.tol_ok);
}

OCT_TEST_REQ_FORTRAN(numerics, kepler_yoshida4_order_and_conservation) {
    auto k = KernelLibrary::instance().find("kepler");
    ProblemSpec s = base_spec("kepler");
    s.params["e"] = 0.5;
    s.tolerance = 1e-6;
    Validation v = k->validate(s);
    OCT_NOTE("kepler yoshida4 order=" << v.measured_order << " dE=" << v.conservation_error);
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.order_ok);
    OCT_CHECK(v.conservation_ok);
    OCT_CHECK(v.tol_ok);
}

OCT_TEST_REQ_FORTRAN(numerics, nbody_conserves_momentum_and_energy) {
    auto k = KernelLibrary::instance().find("nbody");
    ProblemSpec s = base_spec("nbody");
    Validation v = k->validate(s);
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.conservation_ok);
    OCT_CHECK(v.order_ok);
    OCT_CHECK(v.tol_ok);
}

OCT_TEST_REQ_FORTRAN(numerics, cavity2d_is_divergence_free) {
    auto k = KernelLibrary::instance().find("cavity2d");
    ProblemSpec s = base_spec("cavity2d");
    Validation v = k->validate(s);
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.divergence_ok);
    OCT_CHECK(v.conservation_ok);
    OCT_CHECK(v.tol_ok);
}

OCT_TEST_REQ_FORTRAN(numerics, gas_thermodynamics_identity) {
    auto k = KernelLibrary::instance().find("gas");
    ProblemSpec s = base_spec("gas");
    s.params["ratio"] = 4.0;
    Validation v = k->validate(s);
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.conservation_ok);
    OCT_CHECK(v.tol_ok);
    OCT_CHECK(v.order_ok);
}

OCT_TEST_REQ_FORTRAN(numerics, linsolve_residual_and_refinement) {
    auto k = KernelLibrary::instance().find("linsolve");
    ProblemSpec s = base_spec("linsolve");
    Validation v = k->validate(s);
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.tol_ok);
    OCT_CHECK(v.conservation_ok);
}

OCT_TEST_REQ_FORTRAN(numerics, tensor_field_symmetry_and_bianchi_identity) {
    auto k = KernelLibrary::instance().find("tensor_field");
    ProblemSpec s = base_spec("tensor_field");
    Validation v = k->validate(s);
    OCT_CHECK(v.reference_ok);
    OCT_CHECK(v.tol_ok);
    OCT_CHECK(v.conservation_ok);
}
