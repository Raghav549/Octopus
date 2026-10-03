// Octopus Hybrid AI Engine -- built-in module catalogue.
//
// One call registers every compiled-in "language DNA" module with the engine's
// capability registry. Modules that could not be compiled in (for example the
// llama.cpp host without a prebuilt backend) still register, and report their
// limitation honestly from ModuleInfo::limitations.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/module.hpp"

namespace oct {

// Registers every built-in module and returns the number registered.
size_t register_builtin_modules(Registry& registry);

}  // namespace oct
