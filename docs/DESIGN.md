# Design notes — how the spec maps onto the code

[`../ARCHITECTURE_SPEC.md`](../ARCHITECTURE_SPEC.md) is the contract. This file records how
each of its requirements is realised, and — where a requirement could be read two ways —
which reading the code implements and why.

## Execution mantle

`llm::Host` (`include/octopus/llm.hpp`, `src/host/llama_host.cpp`) is the only place that
knows about models. It has two layers:

* **metadata** — `inspect()` parses the GGUF container directly (`src/core/gguf.cpp`) and
  returns real facts (architecture, layer/head/embedding counts, vocabulary size from the
  token array, parameter count and bytes-per-weight from the tensor table). This layer
  never needs llama.cpp and is what the tests and the CLI's `doctor` command exercise.
* **generation** — compiled only under `OCT_HAVE_LLAMA`. Without it, `load()` fails with
  `Status::unavailable`, and `--allow-stub` produces a deterministic, explicitly labelled
  answer (`"note":"deterministic stub, not model inference"`). There is no path that
  silently returns synthetic text as if it were model output.

## Correctness model ("no guessing")

Every numeric kernel returns a `Result` carrying `method`, `backend`, `units`, shape,
diagnostics (residual, conservation error, measured order, iterations, wall time) and a
SHA-256 `fingerprint` over the spec + method + backend. The word *correct* is only ever
produced by `Kernel::validate()`, which compares against an independently derived
reference (exact Riemann solution, manufactured solution, analytic conic, CG cross-check,
published Ghia benchmark) and against the declared tolerance.

`Kernel::validation_spec()` exists so a canonical, self-contained validation case travels
with the kernel: the CLI (`octopus kernels run <name>`), the module self-check and the
supervisor all use it instead of inventing parameters.

Prolog is the other half: `verify()` returns `Grounded` only when a derivation exists,
`Refuted` only for predicates the caller declared complete, and `Unknown` otherwise
(including every truncated search, which also reports `hit_limit`).

## Parallel reasoning

`occam::parallel_verify()` runs each strategy on its own thread, then accepts the *median*
only if every strategy that produced a value lies within the declared tolerance. If any
strategy errors, there is no verdict at all. Disagreement is never averaged into a value.
Measured wall time and speedup are returned so the claim "parallel" is quantifiable
rather than assumed.

## Self-healing = supervised recovery

`supervisor::Supervisor` owns checks (which may carry a watchdog deadline), Prolog
invariants, restart/rollback hooks, a failure counter per check and an append-only audit
chain. The action taken on failure is explicit and recorded: `Restart`, `Rollback`,
`Degrade` (no hook available) or `FailClosed`. On restart the generation counter advances;
rollback is offered to the host application through a hook (`set_rollback`) rather than
being performed behind its back.

The limits are deliberate and documented: a watchdog here *detects* an overrun after the
check returns (checks run on the calling thread), and recovery cannot survive a process
crash. That is why the code never claims "immortal".

## Security posture

Three layers, each with a documented ceiling:

1. **Capability registry** (`src/core/registry.cpp`) — modules advertise dotted
   capabilities; `dispatch()` resolves exact names then progressively shorter prefixes, and
   a module that fails its self-check is reported as failing rather than dropped.
2. **Guardrail** (`src/intercal/`) — envelopes with magic/nonce/ciphertext/SHA-256 tag and
   a hash-linked audit chain. Documented in `ModuleInfo::limitations` and
   `docs/LIMITATIONS.md` as obfuscation and tamper evidence, **not** a cipher; the fuzz test
   proves the *tamper-evidence* property (no accepted mutation), not secrecy.
3. **Honest failures** — anything that cannot be verified fails closed with a `Status`,
   never with a plausible-looking default.

## Array contract

`oct::Array` is a dense, dtype-tagged, row-major buffer with an explicit shape; the APL
layer is the expression language over it (right-to-left evaluation, scalar extension,
reductions, scans, inner/outer products, `⌹` delegated to `numerics::linalg::solve`). The
token-stream codec in the same module reports measured sizes, the single-byte pool size
and the Shannon entropy of the stream, and states that it is *not* entropy-optimal.

## Determinism and reproducibility

* Kernels are deterministic; `tests/test_engine.cpp` re-runs a kernel and requires a
  bit-identical result.
* LISP patching is generation-stamped with exact per-generation snapshots; rollback
  restores the snapshot rather than replaying a log.
* `tools/octbench.cpp` fixes every input, prints the build/host fingerprint and the result
  fingerprint per row, and reports min/median/max over repeats.
* `tests/run_tests.py` maps the 14 required test categories to executed tests and fails
  the report if a category has no test at all.

## Style rules that the codebase holds itself to

* A test only prints PASS if at least one assertion executed (`tests/harness.hpp`).
* Every module must supply non-empty `limitations`; the engine test asserts this for all
  twelve built-ins.
* Every result that ran a fallback path says `degraded` in its `backend` string.
* JSON fields that carry names are emitted as strings — a `const char*` overload on
  `Json::field` exists specifically because the pointer-to-bool conversion silently
  produced `true`/`false` (regression-tested in `tests/test_engine.cpp`).
