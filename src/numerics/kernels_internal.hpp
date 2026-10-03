// Octopus Hybrid AI Engine -- internal declarations shared by the numerical
// kernel translation units (not part of the public API).
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/numerics.hpp"

#include <memory>

#if defined(OCT_HAVE_FORTRAN)
extern "C" {
// Implemented in src/fortran/octopus_kernels.f90 (iso_c_binding interfaces).
int oct_f_sod1d(double gamma, double t_end, int n_cells, double cfl,
                double rho_l, double u_l, double p_l, double rho_r, double u_r, double p_r,
                double* out_rho, double* out_u, double* out_p, int* n_out);
int oct_f_heat2d(double alpha, double t_end, int n, double cfl, double* out_u);
int oct_f_poisson2d(int n, int max_iter, double tol, int* iters_out, double* residual_out,
                    double* out_u);
int oct_f_kepler(double gm, double dt, int n_steps, double* state_io);
int oct_f_compiler_id(char* buf, int len);
}
#endif

namespace oct::numerics::detail {

std::shared_ptr<const Kernel> make_sod1d_kernel();
std::shared_ptr<const Kernel> make_heat2d_kernel();
std::shared_ptr<const Kernel> make_poisson2d_kernel();
std::shared_ptr<const Kernel> make_kepler_kernel();
std::shared_ptr<const Kernel> make_nbody_kernel();
std::shared_ptr<const Kernel> make_cavity2d_kernel();
std::shared_ptr<const Kernel> make_gas_kernel();
std::shared_ptr<const Kernel> make_linsolve_kernel();

std::string fingerprint_of(const ProblemSpec& s, const std::string& method,
                           const std::string& backend);
void finish_result(Result& r, const ProblemSpec& spec, Clock::time_point t0,
                   double work_units, size_t data_elements);

inline double sq(double x) { return x * x; }

std::string backend_string(Backend requested, bool used_fortran,
                           const std::string& fortran_id);

// 1-D grid helper shared by the field kernels.
struct Grid1D {
    int n = 0;
    double x0 = 0.0, x1 = 1.0, dx = 0.0;
    std::vector<double> xc;
};
Grid1D make_grid(int n, double x0, double x1);

}  // namespace oct::numerics::detail
