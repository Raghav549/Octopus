# Acceptance report

Everything below was executed in the environment described in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) (x86-64, 2 threads, g++ 12.2.0, no Fortran
compiler, no model weights, no network). Reproduce with:

```sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python3 tests/run_tests.py --no-build          # 50 passed, 0 failed, 1 skipped, 1159 assertions
python3 bench/bench.py --no-build --repeats 3  # bench/results/{raw.csv,summary.md}
./build/octopus selftest                       # 12/12 modules, each assertion executed
```

Measured suite result at this snapshot: **50 passed, 0 failed, 1 skipped, 1159
executed assertions**; **12/12 module self-checks pass**. The single skip is
`numerics.sod1d_fortran_matches_cxx_when_available` ("Fortran backend absent") — the
reason is printed, and the skip is converted to a failure by `run_tests.py --strict`.

## The 14 required test categories

| # | Category | Evidence (executed) | Status |
|---|---|---|---|
| 1 | clean-build tests | `cmake -B build && cmake --build build -j` from a clean tree; the runner fails the whole report if configure/build fails | **PASS** |
| 2 | model metadata tests | `integration.model_metadata_from_a_real_gguf_file` reads a real GGUF (50 257 tokens for the GPT-2 vocab fixture) and rejects a non-GGUF file | **PASS** |
| 3 | llama.cpp integration tests | `layers.llm_host_reports_what_it_cannot_do`: `backend()` reports `compiled=false`, `load()` without the stub fails closed, `load()` with `--allow-stub` returns a labelled deterministic answer. **The real llama.cpp path is not compiled here** — see LIMITATIONS | **PARTIAL (backend absent)** |
| 4 | tokenizer tests | `tokenizer.gpt2_matches_the_reference_fixture_exactly`: **46/46 cases and 473/473 tokens** vs the upstream fixture; degraded paths measured (deepseek-llm 0.978 cases / 0.996 tokens) and labelled | **PASS** |
| 5 | deterministic generation tests | `engine.results_are_reproducible_and_fingerprints_bind_the_spec` (bit-identical reruns, spec-bound SHA-256), `layers.llm_host...` (stub determinism). No real-model determinism is claimed | **PASS (stub only)** |
| 6 | numerical kernel correctness | 14 tests: shock tube vs exact Riemann solver; heat vs manufactured solution + measured order; Poisson vs analytic + CG cross-check; cavity projections + Ghia Re=100; Kepler vs analytic + order; N-body momentum; ideal-gas identities; LU vs CG + honest Hilbert conditioning | **PASS** |
| 7 | array IR property tests | APL: reduce/scan/reshape/transpose, inner and outer products vs hand computation, solve residual checked independently, malformed programs rejected | **PASS** |
| 8 | rule-engine tests | Prolog: `Grounded` only with a derivation, `Refuted` only for declared-complete predicates, `Unknown` + `hit_limit` for a truncated search; arithmetic unification (`X is 2+3*4-1` → 13) | **PASS** |
| 9 | concurrency/race tests | Occam channels/par_map under repetition and FIFO order; actor fault containment/restart over 8 repetitions | **PASS** |
| 10 | sandbox/security tests | capability registry: every advertised capability resolves to the module that lists it; a deliberately failing module is reported as failing; guardrail self-check proves tamper rejection and that the module documents "NOT a cipher" | **PASS** |
| 11 | fuzz tests for parsers/DSLs | fixed malformed-input corpus through the APL, Prolog, LISP and stack-VM parsers (no crash, errors returned as statuses); 384 single-bit guardrail mutations, **0 accepted** | **PASS** |
| 12 | end-to-end offline tests | `integration.offline_end_to_end_chain`: APL → Prolog → Occam → Piet → supervisor → audit chain, no network, no model | **PASS** |
| 13 | memory-budget tests | bounded stack-VM memory (out-of-range access faults instead of growing), bounded rendering grid for oversized input | **PASS (qualitative)** |
| 14 | benchmark harness | `tools/octbench` + `bench/bench.py`: fixed inputs, build/host fingerprint, min/median/max over repeats, result fingerprints; artifacts in `bench/results/` | **PASS** |

## Numerical evidence (selected, all from executed tests)

| Kernel | Reference | Measured | Gate |
|---|---|---|---|
| `sod1d` | exact Riemann solver | rel-L2 2.28e-2, order 0.66 (1st-order scheme), mass conserved over 403 checks | tol 5e-2, conservation |
| `heat2d` | manufactured solution | order 1.978, rel-L2 1.08e-4 (n=64) | order ∈ [1.7, 2.3] |
| `poisson2d` | analytic sine + CG cross-check | order 2.00209, residual ≤ 1e-6, SOR-vs-CG 4.24e-12 | independent-operator agreement |
| `cavity2d` | Ghia et al. Re=100 + discrete invariants | max scaled divergence 2.06e-9, ±lid reflection bit-exact, ke 0.028062 | div < 1e-6, reflection < 1e-12 |
| `kepler` | analytic conic | leapfrog order 2.00025, RK4 4.0835, energy drift 2.05e-5 | order, bounded drift |
| `nbody` | analytic 3-body separation | order 2.0219, momentum drift 1e-16 (0 printed) | order, momentum |
| `gas` | closed-form identities | `ds/R` error 2.37e-17, `pv^γ` 1.80e-16 | ≤ 1e-12 |
| `linsolve` | LU vs CG | SPD residual 5.5e-17, LU-vs-CG 4.2e-12 | ≤ 1e-8; Hilbert honesty gate |

Two defects were found *by these tests and benchmarks* and fixed in this snapshot, which is
the point of the exercise:

1. `poisson2d`'s red-black SOR folded the wall ghost into the neighbour sum while keeping a
   diagonal of 4. The iteration operator therefore differed from the operator the residual
   measures: stable for n ≤ 32, **divergent for n ≥ 48** (residual 4e+76 after the 20 000
   iteration cap). The boundary diagonal is now 5 on edges and 6 in corners; n=64 converges
   in 288 iterations to a 6.2e-10 relative residual, ~60× faster.
2. The same mistake was present in `cavity2d`'s pressure solve (Neumann ghosts);
   corrected the same way (diagonal 3 on edges, 2 in corners), after which the projection
   invariant tightened to 2.06e-9.

## Benchmark snapshot (2 threads, this container; `bench/results/summary.md`)

| case | min ms | median ms | notes |
|---|---:|---:|---|
| kernel sod1d n=400 | 11.5 | 11.8 | first-order HLLC, 1600 outputs |
| kernel heat2d n=64 | 0.37 | 0.45 | 103 steps |
| kernel poisson2d n=64 | 30.2 | 30.3 | 288 SOR iterations, residual 6.2e-10 |
| kernel kepler e=0.3 | 0.05 | 0.06 | 2000 steps |
| kernel nbody n=3 | 0.07 | 0.07 | 2000 steps |
| kernel gas | 10.6 | 10.6 | includes the numerical `ds` integration |
| kernel linsolve n=64 | 0.41 | 0.43 | |
| kernel cavity2d n=32 Re=100 | 1354 | 1362 | 600 projection steps |
| layer apl reduce (100 000) | 0.34 | 0.57 | |
| layer prolog path depth 200 | 550 | 570 | 599 inferences, limits stated in the harness |
| layer tokenizer BPE (44 B, GPT-2) | 0.018 | 0.022 | warm cache; first call builds ranks |
| layer intercal round-trip 64 KiB | 89.8 | 90.8 | encode + decode |
| layer piet render + PNG 512² | 9.0 | 9.3 | stored-deflate writer |

## What is *not* accepted (and is labelled as such everywhere)

* Real SLM inference: **not demonstrated** — no weights, no llama.cpp library in this
  build. `octopus version` says `"stub": true`; the CLI refuses to answer without
  `--allow-stub`.
* Fortran backend: **unexercised** — no Fortran compiler. Its test skips with a reason.
* Security: no cryptographic claims. Obfuscation + tamper evidence only; audit chain and
  fuzz results are measured, "immune to decompilation" is explicitly *not* claimed.
* Apple Silicon UMA / edge FPGA paths: **not implemented or tested** (no such hardware).
* Long-run stability beyond the tested windows (cavity to Re=800, Kepler over 4096 steps,
  N-body to t=2) is not claimed.

## Verdict

The engine builds from source, runs offline, exposes a machine-readable CLI, passes every
executed assertion in its own suite (50/50 with one documented skip), reports its own
capabilities and failures through the same interface it tests, and ships reproducible
benchmarks. The gaps above are stated rather than papered over; per the architecture spec,
this is a defensible partial delivery, not a claim of completeness.
