// Octopus Hybrid AI Engine -- Fortran 2023 numerical kernel library.
// SPDX-License-Identifier: MIT
#include "octopus/numerics.hpp"

#include "kernels_internal.hpp"

#include <algorithm>

namespace oct::numerics {

const char* backend_name(Backend b) noexcept {
    switch (b) {
        case Backend::Fortran: return "fortran.2023";
    }
    return "fortran.2023";
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
    j.field("hardware_skipped", hardware_skipped);
    j.field("status", to_string(status.code));
    if (!status.message.empty()) j.field("status_message", status.message);
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
    j.field("hardware_skipped", hardware_skipped);
    j.field("status", hardware_skipped ? "UNSUPPORTED_HARDWARE_SKIP" : (accepted() ? "ok" : "failed"));
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

Result make_hardware_skip_result(const Kernel& k, const ProblemSpec& spec,
                                 std::vector<int64_t> shape) {
    Result r;
    r.kernel = k.name();
    r.method = k.method();
    r.units = k.units();
    r.spec = spec;
    r.shape = std::move(shape);
    r.hardware_skipped = true;
    r.status = fortran_bridge::skip_status(k.name());
    r.backend = backend_string(spec.backend, false, fortran_bridge::compiler_id());
    r.fingerprint = fingerprint_of(spec, r.method, r.backend);
    return r;
}

Validation make_hardware_skip_validation(const Kernel& k, const ProblemSpec& spec,
                                         std::string reference) {
    Validation v;
    v.reference = std::move(reference);
    v.tolerance = spec.tolerance;
    v.hardware_skipped = true;
    v.notes = fortran_bridge::skip_status(k.name()).message;
    return v;
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
    kernels_.push_back(detail::make_tensor_field_kernel());
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
    return oct_f_compiler_available() != 0;
}

const char* compiler_id() {
    static const std::string id = [] {
        char buf[128] = {0};
        if (oct_f_compiler_id(buf, int(sizeof(buf))) == 0 && buf[0] != '\0')
            return std::string(buf);
        return std::string("UNSUPPORTED_HARDWARE_SKIP (gfortran/lfortran absent; C++ fallback banned)");
    }();
    return id.c_str();
}

Status skip_status(std::string_view kernel_name) {
    return Status::hardware_skip(
        "Fortran 2023 compiler (gfortran/lfortran) absent on host for kernel '" +
        std::string(kernel_name) + "'; C++ reference fallback is strictly banned");
}

}  // namespace fortran_bridge

}  // namespace oct::numerics
