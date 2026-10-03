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

namespace oct {

namespace {

// The numerical kernel library is exposed through its own module so the CLI can
// resolve "numeric.kernel.<name>" capabilities through the same registry.
class NumericsModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "fortran.kernels";
        i.version = "1.0.0";
        i.language = "C++20 reference + optional Fortran 2018 backend";
        i.role = "validated physics/maths kernels (fluids, gravity, thermodynamics)";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"numeric.kernel"};
        for (const auto& n : numerics::KernelLibrary::instance().names())
            i.capabilities.push_back("numeric.kernel." + n);
        i.limitations = {
            numerics::KernelLibrary::instance().fortran_available()
                ? "Fortran backend linked; per-kernel method is reported in every Result"
                : "no Fortran compiler at configure time: the C++ long-double reference kernels "
                  "run instead, and any request for the Fortran backend is labelled degraded in "
                  "the result's backend string",
            "no GPU/tensor-core path; the kernels are CPU-only by design",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }
    std::string describe() const override {
        return numerics::KernelLibrary::instance().fortran_available()
                   ? "Physics/maths kernels with the Fortran backend linked in."
                   : "Physics/maths kernels using the C++ long-double reference implementation "
                     "(Fortran unavailable; requests for it are reported as degraded).";
    }
    Status self_check() override {
        // Every kernel is validated against its declared reference; the module
        // fails if any kernel's own validation does not accept.
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
    return n;
}

}  // namespace oct
