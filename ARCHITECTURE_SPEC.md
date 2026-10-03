# Octopus — Hybrid AI Engine Architecture Specification

## Mission

Octopus is a standalone, offline-first hybrid AI engine with a C++/llama.cpp native execution core and an SLM foundation. It combines specialized language-runtime-inspired subsystems behind stable C++ interfaces. The design must be measurable, testable, portable, and safe rather than depending on claims of magical correctness or self-modifying binaries.

## Core design

C++/llama.cpp is the execution mantle: model loading, tokenization, inference scheduling, CPU/GPU/UMA bindings, memory arenas, tensor execution, model lifecycle, and public CLI/API.

A capability router dispatches work by task type:

- numerical kernels -> Fortran-compatible numerical module
- array transformations -> APL-inspired array IR
- symbolic transformations/metaprogramming -> Lisp-style hygienic AST/data layer
- rule verification -> Prolog-style fact/rule engine
- parallel execution -> CSP/Occam-inspired task graph
- low-level utilities -> Forth-inspired stack VM abstraction
- actor/live components -> Smalltalk-inspired message objects
- compatibility/negative-test guard -> INTERCAL-inspired parser/DSL fuzz layer
- visualization -> Piet-inspired canvas IR
- ordinary language generation -> quantized SLM through llama.cpp

These are independent modules with explicit APIs and cannot silently mutate core executable code.

## Required correctness model

The SLM is never treated as an oracle for mathematics, physics, or factual verification. Numerical and symbolic answers are represented as structured computations where possible. Prolog-style verification can reject, flag, or request recomputation; it cannot establish truth about arbitrary real-world claims without trusted facts and validated rules.

Physics modules must declare numerical methods, precision, tolerances, units, boundary conditions, and validation references. Results must include uncertainty/error estimates where applicable.

## Memory and performance

All model dimensions and tensor metadata come from the actual GGUF/model configuration. No hard-coded model size.

The runtime must support:
- memory mapping and lazy model loading;
- bounded working-set memory;
- quantized storage;
- zero/minimal-copy tensor paths;
- asynchronous I/O;
- CPU SIMD and available GPU acceleration;
- hardware capability detection;
- deterministic scheduling modes;
- telemetry disabled by default and no cloud inference dependency.

## Security

Do not implement “immune to decompilation” claims or rely on obscurity as security.

Security goals are:
- signed/hashed model and module manifests;
- sandboxed plugin boundaries;
- capability-based permissions;
- no arbitrary runtime code generation in the core process;
- memory-safe interfaces;
- audit logs generated locally;
- opt-in developer diagnostics;
- fail-closed behavior for untrusted modules.

INTERCAL-inspired components are used as compatibility/fuzzing experiments, not cryptographic protection.

Forth-inspired facilities must not claim direct physical register access unless an explicit platform driver/backend grants that capability.

## Self-healing

Self-healing means supervised recovery:
- health checks;
- watchdogs;
- invariant checking;
- component restart;
- cache/index rebuild;
- rollback to a known-good module version.

The system must not rewrite executable code in-place during inference. Any update requires validation and an explicit reload boundary.

## Real physics / visualization

Piet is a visualization IR, not a physical truth mechanism.

A physics pipeline is:
problem specification -> validated numerical model -> solver -> error/residual checks -> structured result -> visualization IR.

APL-inspired arrays can optimize representation and movement, but every transformation must preserve declared shape and dtype semantics.

## Testing requirements

The repository must include:
1. clean-build tests;
2. model metadata tests;
3. llama.cpp integration tests;
4. tokenizer tests;
5. deterministic generation tests;
6. numerical kernel correctness tests against reference implementations;
7. array IR property tests;
8. rule-engine tests;
9. concurrency/race tests;
10. sandbox/security tests;
11. fuzz tests for parsers/DSLs;
12. end-to-end offline tests;
13. memory-budget tests;
14. benchmark harnesses with reproducible inputs.

No test may print PASS unless its underlying assertion executed.

## Initial implementation order

1. Create the C++ project and CMake structure.
2. Vendor/integrate llama.cpp through a pinned revision or documented build dependency.
3. Implement common tensor/task/result interfaces.
4. Implement model/config loader and capability registry.
5. Implement SLM inference adapter.
6. Implement numerical backend interface and a reference C++ implementation; add Fortran backend when compiler/toolchain is available.
7. Implement APL-inspired array IR.
8. Implement Prolog-style verifier.
9. Implement Lisp-style AST transformation layer with sandboxing.
10. Implement Occam/CSP task executor.
11. Implement Forth-style stack VM.
12. Implement Smalltalk-style actor/message runtime.
13. Implement Piet canvas IR.
14. Implement INTERCAL-style parser/fuzzer as a non-security module.
15. Build unified CLI and offline integration tests.
16. Benchmark on Apple Silicon, x86-64, and supported accelerator targets where available.

## Acceptance

Octopus is complete only when the implemented repository contains executable code, build instructions, tests, reference comparisons, reproducible benchmarks, and explicit capability/limitation reporting. Documentation alone is not completion.
