// Octopus Hybrid AI Engine -- internal declarations for the Fortran 2023
// numerical kernel bindings.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/numerics.hpp"

#include <memory>

extern "C" {
// Exported by src/fortran/octopus_kernels.f90 (or src/fortran/fortran_hw_skip.cpp
// when gfortran/lfortran is absent on the compilation host).
int oct_f_compiler_available();
int oct_f_compiler_id(char* buf, int len);

int oct_f_sod1d(double gamma, double t_end, int n_cells, double cfl,
                double rho_l, double u_l, double p_l, double rho_r, double u_r, double p_r,
                double* out_rho, double* out_u, double* out_p, int* n_out);
int oct_f_heat2d(double alpha, double t_end, int n, double cfl, double* out_u);
int oct_f_poisson2d(int n, int max_iter, double tol, int* iters_out, double* residual_out,
                    double* out_u);
int oct_f_cavity2d(int n, double re, int steps, double u_lid, double poisson_tol,
                   int poisson_iter, double* out_data, double* out_metrics);
int oct_f_kepler(double gm, double dt, int n_steps, double* state_io);
int oct_f_nbody(int n_bodies, int steps, double dt, double eps2, double gm,
                double* bodies_io, double* max_de_out, double* max_dp_out);
int oct_f_gas(double p1, double t1, double ratio, double gamma, double r_gas,
              double* out_data, double* ds_over_r_out, double* eos_res_out);
int oct_f_linsolve(int n, const double* a_in, const double* b_in,
                   double* x_out, double* residual_out);
int oct_f_tensor_field(int n, double curvature_k, double mass_param,
                       double* out_tensor, double* out_metrics);
int oct_f_universe_field(int width, int height, double time_t,
                         double light_x, double light_y, double light_z,
                         double refl_ior, double reynolds,
                         double gm1, double gm2, double orbit_e,
                         double* out_coords, double* out_diag);
}

namespace oct::numerics::detail {

std::shared_ptr<const Kernel> make_sod1d_kernel();
std::shared_ptr<const Kernel> make_heat2d_kernel();
std::shared_ptr<const Kernel> make_poisson2d_kernel();
std::shared_ptr<const Kernel> make_kepler_kernel();
std::shared_ptr<const Kernel> make_nbody_kernel();
std::shared_ptr<const Kernel> make_cavity2d_kernel();
std::shared_ptr<const Kernel> make_gas_kernel();
std::shared_ptr<const Kernel> make_linsolve_kernel();
std::shared_ptr<const Kernel> make_tensor_field_kernel();

std::string fingerprint_of(const ProblemSpec& s, const std::string& method,
                           const std::string& backend);
void finish_result(Result& r, const ProblemSpec& spec, Clock::time_point t0,
                   double work_units, size_t data_elements);
Result make_hardware_skip_result(const Kernel& k, const ProblemSpec& spec,
                                 std::vector<int64_t> shape);
Validation make_hardware_skip_validation(const Kernel& k, const ProblemSpec& spec,
                                         std::string reference);

inline double sq(double x) { return x * x; }

std::string backend_string(Backend requested, bool used_fortran,
                           const std::string& fortran_id);

struct Grid1D {
    int n = 0;
    double x0 = 0.0, x1 = 1.0, dx = 0.0;
    std::vector<double> xc;
};
Grid1D make_grid(int n, double x0, double x1);

}  // namespace oct::numerics::detail
