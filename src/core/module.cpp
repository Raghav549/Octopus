// Octopus Hybrid AI Engine -- built-in module catalogue.
// SPDX-License-Identifier: MIT
#include "octopus/catalog.hpp"

#include "octopus/apl.hpp"
#include "octopus/intercal.hpp"
#include "octopus/lisp.hpp"
#include "octopus/llm.hpp"
#include "octopus/numerics.hpp"
#include "octopus/occam.hpp"
#include "octopus/piet.hpp"
#include "octopus/prolog.hpp"
#include "octopus/router.hpp"
#include "octopus/smalltalk.hpp"
#include "octopus/stackvm.hpp"
#include "octopus/supervisor.hpp"
#include "octopus/tokenizer.hpp"
#include "octopus/universe.hpp"

namespace oct {

namespace {

// The Fortran 2023 numerical kernel library exposed through its own module so
// the CLI and neural router can resolve "numeric.kernel.<name>" capabilities.
class NumericsModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "fortran.kernels";
        i.version = "2.0.0";
        i.language = "Fortran 2023 ISO_C_BINDING (C++ fallback banned)";
        i.role = "Absolute Physics Core: fluid dynamics, Kepler/N-body gravity, PDEs, "
                 "thermodynamics, linear algebra, and covariant tensor matrix fields";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"numeric.kernel"};
        for (const auto& n : numerics::KernelLibrary::instance().names())
            i.capabilities.push_back("numeric.kernel." + n);
        i.limitations = {
            numerics::KernelLibrary::instance().fortran_available()
                ? "Fortran 2023 backend linked natively; C++ reference fallback is banned"
                : "UNSUPPORTED_HARDWARE_SKIP: Fortran 2023 compiler (gfortran/lfortran) absent on "
                  "build host; live physics execution blocks are securely skipped (C++ fallback banned)",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }
    std::string describe() const override {
        return numerics::KernelLibrary::instance().fortran_available()
                   ? "Fortran 2023 Absolute Physics Core linked natively."
                   : "Fortran 2023 Absolute Physics Core [UNSUPPORTED_HARDWARE_SKIP: "
                     "gfortran/lfortran absent; C++ reference fallback banned].";
    }
    Status self_check() override {
        if (!numerics::KernelLibrary::instance().fortran_available()) {
            // Verify that every kernel strictly honours UNSUPPORTED_HARDWARE_SKIP
            // and never executes a C++ fallback approximation.
            for (const auto& name : numerics::KernelLibrary::instance().names()) {
                auto k = numerics::KernelLibrary::instance().find(name);
                if (!k) return Status::internal("numerics: kernel vanished: " + name);
                const numerics::ProblemSpec spec = k->validation_spec();
                const auto r = k->run(spec);
                const auto v = k->validate(spec);
                if (!r.hardware_skipped || !r.status.is_hardware_skip() || !v.hardware_skipped)
                    return Status::internal("numerics: kernel '" + name +
                                            "' did not flag UNSUPPORTED_HARDWARE_SKIP");
            }
            return Status::hardware_skip(
                "UNSUPPORTED_HARDWARE_SKIP: Fortran 2023 compiler (gfortran/lfortran) absent on host; "
                "all 9 kernels verified hardware-skip gate (C++ fallback banned)");
        }
        for (const auto& name : numerics::KernelLibrary::instance().names()) {
            auto k = numerics::KernelLibrary::instance().find(name);
            if (!k) return Status::internal("numerics: kernel vanished: " + name);
            const numerics::ProblemSpec spec = k->validation_spec();
            const auto v = k->validate(spec);
            if (!v.accepted())
                return Status::internal("numerics: kernel '" + name + "' failed validation: " + v.notes);
        }
        return Status::ok();
    }
};

}  // namespace

size_t register_builtin_modules(Registry& registry) {
    size_t n = 0;
    auto add = [&](std::shared_ptr<Module> m) {
        registry.add(std::move(m));
        ++n;
    };
    add(std::make_shared<NumericsModule>());
    add(router::make_router_module());
    add(apl::make_apl_module());
    add(prolog::make_prolog_module());
    add(lisp::make_lisp_module());
    add(stackvm::make_stackvm_module());
    add(tokenizer::make_tokenizer_module());
    add(smalltalk::make_smalltalk_module());
    add(occam::make_occam_module());
    add(intercal::make_intercal_module());
    add(piet::make_piet_module());
    add(supervisor::make_supervisor_module());
    add(llm::make_llm_module());
    add(universe::make_universe_module());
    return n;
}

}  // namespace oct
