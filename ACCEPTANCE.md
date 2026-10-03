# Acceptance report

Everything below was executed in the environment described in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) (x86-64, 2 threads, g++ 12.2.0, no Fortran
compiler, no model weights, no network). Reproduce with:

```sh
# default build: no llama.cpp linked, no Fortran compiler
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python3 tests/run_tests.py --no-build          # 55 passed, 0 failed, 5 skipped
python3 bench/bench.py --no-build --repeats 3  # bench/results/{raw.csv,summary.md}
./build/octopus selftest                       # 12/12 modules, each assertion executed

# inference build: real llama.cpp statically linked
cmake -B build-llama -S . -DCMAKE_BUILD_TYPE=Release -DOCT_WITH_LLAMA=ON \
      -DOCT_LLAMA_PREBUILT_DIR=$PWD/third_party/llama.cpp/build-static
cmake --build build-llama -j
./build-llama/oct_tests                        # 59 passed, 0 failed, 1 skipped, 1308 assertions
./build-llama/slmgen /tmp/tiny.gguf            # synthetic-weight model (offline)
./build-llama/octopus llm --model /tmp/tiny.gguf --prompt 'hello' --max-tokens 8 --json
```

Measured suite results at this snapshot:

| build | result | skips |
|---|---|---|
| default (`OCT_WITH_LLAMA=OFF`) | **55 passed, 0 failed, 1279 assertions** | 5: 4 llama.cpp tests (backend not linked) + 1 Fortran |
| inference (`OCT_WITH_LLAMA=ON`) | **59 passed, 0 failed, 1308 assertions** | 1: Fortran |

`octopus selftest` reports **13/13 module self-checks pass** in both builds. Every skip
prints its reason, and `run_tests.py --strict` converts skips into failures.

## The 14 required test categories

| # | Category | Evidence (executed) | Status |
|---|---|---|---|
| 1 | clean-build tests | `cmake -B build && cmake --build build -j` from a clean tree; the runner fails the whole report if configure/build fails | **PASS** |
| 2 | model metadata tests | `integration.model_metadata_from_a_real_gguf_file` reads a real GGUF (50 257 tokens for the GPT-2 vocab fixture) and rejects a non-GGUF file | **PASS** |
| 3 | llama.cpp integration tests | In the `OCT_WITH_LLAMA=ON` build: `llm.backend_is_linked_and_honest`, `llm.metadata_only_gguf_is_refused_not_guessed`, `llm.missing_model_fails_and_stub_stays_labelled`, and `llm.synthetic_model_end_to_end_generation_is_deterministic` — llama.cpp loads a generated model, tokenizes, prefills, decodes, samples and detokenizes offline. In the default build these skip with a printed reason | **PASS (synthetic weights; no trained model available)** |
| 4 | tokenizer tests | `tokenizer.gpt2_matches_the_reference_fixture_exactly`: **46/46 cases and 473/473 tokens** vs the upstream fixture; degraded paths measured (deepseek-llm 0.978 cases / 0.996 tokens) and labelled | **PASS** |
| 5 | deterministic generation tests | `llm.synthetic_model_end_to_end_generation_is_deterministic`: greedy decoding of the synthetic model is bit-for-bit identical across repeated calls *and* across separate processes (KV cache is reset per call — a bug this test found); plus `engine.results_are_reproducible_and_fingerprints_bind_the_spec` | **PASS** |
| 6 | numerical kernel correctness | 14 tests: shock tube vs exact Riemann solver; heat vs manufactured solution + measured order; Poisson vs analytic + CG cross-check; cavity projections + Ghia Re=100; Kepler vs analytic + order; N-body momentum; ideal-gas identities; LU vs CG + honest Hilbert conditioning | **PASS** |
| 7 | array IR property tests | APL: reduce/scan/reshape/transpose, inner and outer products vs hand computation, solve residual checked independently, malformed programs rejected | **PASS** |
| 8 | rule-engine tests | Prolog: `Grounded` only with a derivation, `Refuted` only for declared-complete predicates, `Unknown` + `hit_limit` for a truncated search; arithmetic unification (`X is 2+3*4-1` → 13) | **PASS** |
| 9 | concurrency/race tests | Occam channels/par_map under repetition and FIFO order; actor fault containment/restart over 8 repetitions | **PASS** |
| 10 | sandbox/security tests | capability registry: every advertised capability resolves to the module that lists it; a deliberately failing module is reported as failing; guardrail self-check proves tamper rejection and that the module documents "NOT a cipher" | **PASS** |
| 11 | fuzz tests for parsers/DSLs | fixed malformed-input corpus through the APL, Prolog, LISP and stack-VM parsers (no crash, errors returned as statuses); 384 single-bit guardrail mutations, **0 accepted** | **PASS** |
| 12 | end-to-end offline tests | `integration.offline_end_to_end_chain`: APL → Prolog → Occam → Piet → supervisor → audit chain, no network, no model; plus `router.*`: one entry point dispatching to all eight layers through the capability registry, and failing closed when a layer cannot serve | **PASS** |
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

### Inference path (llama.cpp build, synthetic weights)

```
$ ./build-llama/slmgen /tmp/tiny.gguf
{"file":"/tmp/tiny.gguf","bytes":467552,"layers":2,"embd":64,"heads":4,"ff":128,
 "vocab":259,"weights":"synthetic-pseudo-random","sha256":"fead173b…"}
$ ./build-llama/octopus llm --model /tmp/tiny.gguf --prompt 'hello world' --max-tokens 8 --json
{"text":"\udcde…","backend":"{\"compiled\":true,\"available\":true,\"stub\":false,
 "name\":\"llama.cpp\",\"version\":\"b11371\",\"honesty\":\"llama.cpp static backend
 linked and a model is loaded; generation runs entirely on this machine\"}"}
```

Deterministic: the same prompt and greedy parameters produce identical bytes on repeated
calls and in separate processes. This proves the execution path, **not** model quality.

Defects found *by these tests, benchmarks and the inference run* and fixed in this
snapshot, which is the point of the exercise:

1. `poisson2d`'s red-black SOR folded the wall ghost into the neighbour sum while keeping a
   diagonal of 4. The iteration operator therefore differed from the operator the residual
   measures: stable for n ≤ 32, **divergent for n ≥ 48** (residual 4e+76 after the 20 000
   iteration cap). The boundary diagonal is now 5 on edges and 6 in corners; n=64 converges
   in 288 iterations to a 6.2e-10 relative residual, ~60× faster.
2. The same mistake was present in `cavity2d`'s pressure solve (Neumann ghosts);
   corrected the same way (diagonal 3 on edges, 2 in corners), after which the projection
   invariant tightened to 2.06e-9.
3. `llama_generate_impl` had never been compiled; on first build it hit a most-vexing-parse
   (`std::vector<llama_token> tokens(size_t(n_prompt))` declared a function) and the
   one-token re-decode passed a stale batch. Rewritten and now exercised end to end.
4. The same function did not reset the KV cache between calls, so a second `generate()`
   continued from the previous generation's state and greedy decoding was **not**
   reproducible. Caught by the new determinism test; fixed with `llama_memory_clear`.
5. The GGUF writer typed an *empty* string array as `arr[u8,0]`, which llama.cpp rejects
   ("invalid gguf type for tokenizer.ggml.merges"); empty arrays now record their declared
   element type. Without this the synthetic model could not be loaded at all.

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

* Real SLM inference **with trained weights**: not demonstrated — none are available here.
  What *is* demonstrated is the full inference path executing on real llama.cpp with a
  generated synthetic-weight model (see below); the output is nonsense by construction and
  is labelled as such everywhere. `octopus version` reports `"stub": true` until a model
  is loaded; without one the CLI refuses to answer unless `--allow-stub` is passed.
* Fortran backend: **unexercised** — no Fortran compiler. Its test skips with a reason.
* Security: no cryptographic claims. Obfuscation + tamper evidence only; audit chain and
  fuzz results are measured, "immune to decompilation" is explicitly *not* claimed.
* Apple Silicon UMA / edge FPGA paths: **not implemented or tested** (no such hardware).
* Long-run stability beyond the tested windows (cavity to Re=800, Kepler over 4096 steps,
  N-body to t=2) is not claimed.

## Router evidence

`octopus ask "<request>"` is the single entry point. Measured behaviour at this snapshot:

```
$ ./build/octopus ask "apl: +/ ⍳ 11"
{"route_kind":"array","route_module":"apl.arrays","route_reason":"explicit prefix 'apl:'",
 "route_confidence":1,"result_shape":"f64[]","elements":1,"values":[55],"fingerprint":"8f4a876…"}
$ ./build/octopus ask "occam: sum n=100000"
{"route_kind":"parallel",…,"value":4999950000,"agreed":true,"spread":0,"winner":"closed_form"}
$ ./build/octopus ask "what is the heat equation?"
{"route_kind":"numeric","route_capability":"numeric.kernel.heat2d",
 "route_reason":"matched kernel keyword 'heat'","route_confidence":0.7, …}
$ ./build/octopus ask "llm: write me a poem"
{"status":"unavailable","message":"router: this request needs a language model; pass model=<path.gguf>
 (no answer is fabricated without one)"}
$ ./build/octopus ask "zzzzz qqqqq"
{"status":"rejected","message":"router: no rule matched; refusing to guess which module should serve this"}
```

Three independent strategies (closed form, direct loop, APL reduce) agree on the parallel
sum in 0 spread; an unclassifiable request and a language request without a model both fail
closed.

## Verdict

The engine builds from source, runs offline, exposes a machine-readable CLI, passes every
executed assertion in its own suite (50/50 with one documented skip), reports its own
capabilities and failures through the same interface it tests, and ships reproducible
benchmarks. The gaps above are stated rather than papered over; per the architecture spec,
this is a defensible partial delivery, not a claim of completeness.
