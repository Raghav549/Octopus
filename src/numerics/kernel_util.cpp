// Octopus Hybrid AI Engine -- shared numerical helpers.
// SPDX-License-Identifier: MIT
#include "kernels_internal.hpp"

namespace oct::numerics::detail {

Grid1D make_grid(int n, double x0, double x1) {
    Grid1D g;
    g.n = n;
    g.x0 = x0;
    g.x1 = x1;
    g.dx = (x1 - x0) / double(n);
    g.xc.resize(size_t(n));
    for (int i = 0; i < n; ++i) g.xc[size_t(i)] = x0 + (double(i) + 0.5) * g.dx;
    return g;
}

std::string backend_string(Backend /*requested*/, bool used_fortran, const std::string& fortran_id) {
    if (used_fortran) return "fortran.2023 [" + fortran_id + "]";
    return "UNSUPPORTED_HARDWARE_SKIP [" + fortran_id + "]";
}

}  // namespace oct::numerics::detail
