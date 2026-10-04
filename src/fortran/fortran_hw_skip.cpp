// Octopus Hybrid AI Engine -- Strict Hardware-Skip Fortran 2023 ABI Gate.
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// This translation unit is linked ONLY when the compilation machine lacks a
// native Fortran 2023 compiler (gfortran / lfortran). It preserves the
// absolute structural C ABI symbols of `octopus_fortran_kernels` intact while
// refusing to substitute any C++ reference physics calculations. Every live
// Fortran execution block returns OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP (-99).
//
// SPDX-License-Identifier: MIT

#include <cstring>
#include <algorithm>

extern "C" {

static constexpr int OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP = -99;

int oct_f_compiler_available() {
    return 0;
}

int oct_f_compiler_id(char* buf, int len) {
    const char* id = "UNSUPPORTED_HARDWARE_SKIP (gfortran/lfortran absent; C++ fallback banned)";
    if (!buf || len < 1) return 1;
    const int n = std::min(len - 1, int(std::strlen(id)));
    std::memcpy(buf, id, size_t(n));
    buf[n] = '\0';
    return 0;
}

int oct_f_sod1d(double, double, int, double, double, double, double, double, double, double,
                double*, double*, double*, int* n_out) {
    if (n_out) *n_out = 0;
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

int oct_f_heat2d(double, double, int, double, double*) {
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

int oct_f_poisson2d(int, int, double, int* iters_out, double* residual_out, double*) {
    if (iters_out) *iters_out = 0;
    if (residual_out) *residual_out = 0.0;
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

int oct_f_cavity2d(int, double, int, double, double, int, double*, double*) {
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

int oct_f_kepler(double, double, int, double*) {
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

int oct_f_nbody(int, int, double, double, double, double*, double* max_de_out, double* max_dp_out) {
    if (max_de_out) *max_de_out = 0.0;
    if (max_dp_out) *max_dp_out = 0.0;
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

int oct_f_gas(double, double, double, double, double, double*, double* ds_out, double* eos_out) {
    if (ds_out) *ds_out = 0.0;
    if (eos_out) *eos_out = 0.0;
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

int oct_f_linsolve(int, const double*, const double*, double*, double* residual_out) {
    if (residual_out) *residual_out = 0.0;
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

int oct_f_tensor_field(int, double, double, double*, double*) {
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

int oct_f_universe_field(int, int, double, double, double, double, double, double,
                         double, double, double, double*, double*) {
    return OCT_F_ERR_UNSUPPORTED_HARDWARE_SKIP;
}

}  // extern "C"
