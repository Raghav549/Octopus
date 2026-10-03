# Octopus

A standalone, offline-first **hybrid AI engine** written in C++20. Octopus combines a
numerical physics core, a byte-level BPE tokenizer, a Prolog-style verification layer, an
APL-style array IR, a supervised actor runtime and a set of honesty-preserving guardrails
behind stable C++ interfaces, with a command-line front end (`octopus`) that prints
machine-readable JSON.

The design contract is [`ARCHITECTURE_SPEC.md`](ARCHITECTURE_SPEC.md); the honest status of
every component is in [`ACCEPTANCE.md`](ACCEPTANCE.md) and
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). **Read those two before trusting any claim
here.**

## Build

Requirements: a C++20 compiler (g++ 11+ or clang 14+), CMake 3.16+, Python 3 (for the test
and benchmark drivers). No network access is required or used.

```sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Optional, auto-detected at configure time:

| Option | Default | Effect when enabled |
|---|---|---|
| `OCT_WITH_LLAMA` | `ON` (auto-detected) | links a prebuilt llama.cpp (`OCT_LLAMA_PREBUILT_DIR`) for real SLM inference |
| `OCT_WITH_FORTRAN` | `ON` (auto-detected) | compiles `src/fortran/octopus_kernels.f90` and uses it for the physics kernels |
| `OCT_BUILD_TESTS` | `ON` | builds `oct_tests` |
| `OCT_BUILD_TOOLS` | `ON` | builds `octopus`, `ggufgen`, `octbench` |

When an optional dependency is missing, Octopus **still builds**, falls back to the C++
reference implementation, and says so in every result (`backend` field) and in
`octopus doctor`. It never silently pretends the fast path ran.

## Quick start

```sh
./build/oct_tests                        # the full test suite
python3 tests/run_tests.py               # build + tests + coverage matrix
python3 bench/bench.py                   # reproducible benchmark -> bench/results/

./build/octopus version                  # engine + host + backend status (JSON)
./build/octopus doctor                   # what can and cannot run here
./build/octopus selftest                 # every module's executed self-check
./build/octopus modules                  # capability registry inventory
./build/octopus kernels list             # physics kernels and their methods

./build/octopus kernels run sod1d --json
./build/octopus apl 'x ← 3 2 ⍴ ⍳ 6'
./build/octopus prolog 'ancestor(X, Y)'
./build/octopus forth ': sq dup * ; 9 sq .'
./build/octopus tokenizer --model third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf --text "hello"
./build/slmgen /tmp/tiny.gguf                   # synthetic-weight tiny model (offline)
./build/octopus guard encode --key k --in secret.txt --out secret.bin
./build/octopus piet --in state.txt --out state.png
./build/octopus supervise --json
```

## What is in the box

| Layer | Source | Contract |
|---|---|---|
| Physics/maths kernels | `src/numerics/` | 8 validated kernels (shock tube, heat, Poisson, cavity, gravity, thermodynamics, linear solve); every result carries method, backend, residuals, measured order and a SHA-256 fingerprint |
| Tokenizer | `src/tokenizer/` | byte-level BPE read from the GGUF itself; **exact** match to the upstream GPT-2 fixture (46/46 cases, 473/473 tokens) |
| APL-style array IR | `src/apl/` | right-to-left evaluation, reshape/transpose/reduce/scan, inner & outer products, solve; lossless token-stream codec |
| Prolog-style logic | `src/prolog/` | SLD resolution with chronological backtracking; `Grounded` only with a derivation, `Refuted` only for predicates declared complete, otherwise `Unknown` |
| LISP-style self-modification | `src/lisp/` | Scheme subset with closures; every `patch()` is generation-stamped and `rollback()` restores the previous definition set exactly |
| Forth-style stack VM | `src/stackvm/` | compile/run/call, bounds-checked memory and explicit faults (no undefined behaviour) |
| Smalltalk-style runtime | `src/smalltalk/` | actors with supervision, bounded restart, fault containment and an event log |
| Occam-style parallelism | `src/occam/` | CSP channels, parallel strategies, consensus that fails closed on disagreement, measured speedup |
| INTERCAL-style guardrail | `src/intercal/` | obfuscated, tamper-evident envelopes and a hash-linked audit chain (**not** a cipher) |
| Piet-style visualisation | `src/piet/` | cognitive-state rendering to PNG (no zlib) / PPM, bounded quantisation, NaN marked black |
| Supervisor | `src/supervisor/` | health checks, watchdogs, Prolog invariants, restart/rollback hooks, audit chain |
| LLM host | `src/host/` | reads real GGUF metadata without llama.cpp; without the backend `generate()` is unavailable **or** an explicitly labelled deterministic stub; with `-DOCT_WITH_LLAMA=ON` it runs the real llama.cpp tokenize → prefill → decode → sample → detokenize path |
| Synthetic model writer | `src/host/synthetic_model.cpp`, `tools/slmgen.cpp` | writes a complete, loadable tiny llama GGUF with pseudo-random weights so the inference path can be exercised offline without downloading anything |

## Design rules that are enforced, not aspirational

* **No guessing.** A numeric answer is either validated against an independently derived
  reference or reported as degraded; a logic answer is `Unknown` unless it was derived.
* **No vacuous tests.** A test can only report PASS if at least one assertion executed
  (`tests/harness.hpp`); the runner prints the executed assertion count.
* **No unsupported security claims.** The guardrail layer is documented as obfuscation and
  tamper evidence; it is never described as unbreakable or as protection against a
  determined attacker.
* **Facts come from files.** Model metadata is read from GGUF key/values, never guessed
  from a file name or a parameter count.

## Repository layout

```
include/octopus/   public headers (one per subsystem + core/tensor/gguf/module/catalog)
src/               implementations, grouped by language DNA
src/fortran/       optional Fortran 2018 kernels (C ABI, auto-detected)
tools/             octopus (CLI), ggufgen (GGUF fixture writer), octbench (benchmark harness)
tests/             harness + per-category tests + run_tests.py driver
bench/             bench.py driver; results land in bench/results/
third_party/       pinned llama.cpp checkout + GGUF vocabulary fixtures (not vendored into git)
docs/              LIMITATIONS.md, example render, design notes
```

## Enabling real inference

```sh
./scripts/fetch_deps.sh                                     # clones llama.cpp @ b11371
cmake -B build-llama -S . -DCMAKE_BUILD_TYPE=Release -DOCT_WITH_LLAMA=ON \
      -DOCT_LLAMA_PREBUILT_DIR=$PWD/third_party/llama.cpp/build-static
cmake --build build-llama -j
./build-llama/slmgen /tmp/tiny.gguf                         # ~0.45 MiB, synthetic weights
./build-llama/octopus llm --model /tmp/tiny.gguf --prompt 'hello' --max-tokens 8 --json
```

With trained GGUF weights of your own, point `--model` at them instead. Weights produced
by `slmgen` are pseudo-random by construction: the text is nonsense and every report says
so — the value is that the whole path runs offline and greedily decoding is reproducible.

## Reproduction

```sh
python3 tests/run_tests.py --json /tmp/octopus_tests.json   # full matrix
python3 bench/bench.py --repeats 5                          # bench/results/{raw.csv,summary.md}
```

Benchmarks fix every input in `tools/octbench.cpp`, print the build/host fingerprint, and
report min/median/max over repeats together with the result fingerprints, so a run can be
diffed rather than trusted.

## License

MIT. See the SPDX headers in every source file.
