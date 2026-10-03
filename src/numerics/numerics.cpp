// Octopus Hybrid AI Engine -- numerical kernel library (registry + backend).
// SPDX-License-Identifier: MIT
#include "octopus/numerics.hpp"

#include "kernels_internal.hpp"

#include <algorithm>

namespace oct::numerics {

const char* backend_name(Backend b) noexcept {
    switch (b) {
        case Backend::CxxLd:   return "cxx.long-double";
        case Backend::Fortran: return "fortran.real64";
    }
    return "?";
}

double Diagnostics::get(std::string_view n, double dflt) const {
    for (const auto& kv : metrics) if (kv.first == n) return kv.second;
    return dflt;
}

Json Result::to_json(bool include_data) const {
    Json j;
    j.begin_object();
    j.field("kernel", kernel);
    j.field("method", method);
    j.field("backend", backend);
    j.field("units", units);
    j.field("fingerprint", fingerprint);
    j.key("shape");
    j.begin_array();
    for (int64_t d : shape) j.value(d);
    j.end_array();
    j.field("elements", int64_t(data.size()));
    j.field("t_end", spec.t_end);
    j.field("tolerance", spec.tolerance);
    j.key("diagnostics");
    j.begin_object();
    j.field("residual", diag.residual);
    j.field("conservation_error", diag.conservation_error);
    j.field("measured_order", diag.measured_order);
    j.field("iterations", diag.iterations);
    j.field("wall_seconds", diag.wall_seconds);
    for (const auto& kv : diag.metrics) j.field(kv.first, kv.second);
    j.end_object();
    if (include_data) {
        j.key("data");
        j.begin_array();
        for (double v : data) j.value(v);
        j.end_array();
        if (!x.empty()) {
            j.key("x");
            j.begin_array();
            for (double v : x) j.value(v);
            j.end_array();
        }
    }
    j.end_object();
    return j;
}

Array Result::as_array() const { return Array::from_f64(data, shape); }

Json Validation::to_json() const {
    Json j;
    j.begin_object();
    j.field("accepted", accepted());
    j.field("reference", reference);
    j.field("reference_ok", reference_ok);
    j.field("tolerance", tolerance);
    j.field("tol_ok", tol_ok);
    j.field("max_abs_error", max_abs_error);
    j.field("rel_l2_error", rel_l2_error);
    j.field("conservation_ok", conservation_ok);
    j.field("conservation_error", conservation_error);
    j.field("order_ok", order_ok);
    j.field("measured_order", measured_order);
    j.field("expected_order", expected_order);
    j.field("divergence_ok", divergence_ok);
    j.field("max_divergence", max_divergence);
    for (const auto& kv : metrics) j.field(kv.first, kv.second);
    if (!notes.empty()) j.field("notes", notes);
    j.end_object();
    return j;
}

namespace detail {

std::string fingerprint_of(const ProblemSpec& s, const std::string& method,
                           const std::string& backend) {
    hash::Sha256 h;
    h.update(s.kernel);
    h.update(method);
    h.update(backend);
    for (const auto& kv : s.params) {
        h.update(kv.first);
        h.update(&kv.second, sizeof(double));
    }
    h.update(&s.t_end, sizeof(double));
    h.update(&s.tolerance, sizeof(double));
    int64_t steps = s.steps;
    h.update(&steps, sizeof(steps));
    return h.finalize().hex();
}

void finish_result(Result& r, const ProblemSpec& spec, Clock::time_point t0,
                   double work_units, size_t data_elements) {
    r.spec = spec;
    r.diag.wall_seconds = seconds_since(t0);
    if (r.diag.cells_per_second == 0.0 && r.diag.wall_seconds > 0.0)
        r.diag.cells_per_second = work_units / r.diag.wall_seconds;
    r.fingerprint = fingerprint_of(spec, r.method, r.backend);
    (void)data_elements;
}

}  // namespace detail

KernelLibrary::KernelLibrary() {
    kernels_.push_back(detail::make_sod1d_kernel());
    kernels_.push_back(detail::make_heat2d_kernel());
    kernels_.push_back(detail::make_poisson2d_kernel());
    kernels_.push_back(detail::make_kepler_kernel());
    kernels_.push_back(detail::make_nbody_kernel());
    kernels_.push_back(detail::make_cavity2d_kernel());
    kernels_.push_back(detail::make_gas_kernel());
    kernels_.push_back(detail::make_linsolve_kernel());
    std::sort(kernels_.begin(), kernels_.end(),
              [](const auto& a, const auto& b) { return a->name() < b->name(); });
}

const KernelLibrary& KernelLibrary::instance() {
    static KernelLibrary lib;
    return lib;
}

std::shared_ptr<const Kernel> KernelLibrary::find(std::string_view name) const {
    for (const auto& k : kernels_) if (k->name() == name) return k;
    return nullptr;
}

std::vector<std::string> KernelLibrary::names() const {
    std::vector<std::string> v;
    for (const auto& k : kernels_) v.push_back(k->name());
    return v;
}

std::vector<std::string> KernelLibrary::describe_all() const {
    std::vector<std::string> v;
    for (const auto& k : kernels_)
        v.push_back(k->name() + " :: " + k->method() + " :: " + k->units());
    return v;
}

bool KernelLibrary::fortran_available() const { return fortran_bridge::available(); }
std::string KernelLibrary::fortran_compiler() const { return fortran_bridge::compiler_id(); }
std::string KernelLibrary::fortran_backend_id() const { return fortran_bridge::compiler_id(); }

namespace fortran_bridge {

bool available() {
#if defined(OCT_HAVE_FORTRAN)
    return true;
#else
    return false;
#endif
}

const char* compiler_id() {
    static const std::string id = [] {
#if defined(OCT_HAVE_FORTRAN)
        char buf[128] = {0};
        if (oct_f_compiler_id(buf, int(sizeof(buf))) == 0) return std::string(buf);
        return std::string("fortran.unknown");
#else
        return std::string("absent (no Fortran compiler found at configure time)");
#endif
    }();
    return id.c_str();
}

}  // namespace fortran_bridge

}  // namespace oct::numerics
