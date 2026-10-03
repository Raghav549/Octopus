# Limitations — what Octopus does **not** do

This file is part of the deliverable, not an apology. The architecture spec requires
explicit capability/limitation reporting, and every claim in `README.md` or
`ACCEPTANCE.md` is bounded by what is written here. If a limitation is discovered it is
added here *and* fixed or worked around in code; nothing on this list is hidden in a code
comment only.

## Environment this repository was assembled in

* x86-64 Linux, 2 CPU threads, ~3.85 GiB RAM, g++ 12.2.0, CMake 4.4.3, Python 3.
* **No Fortran compiler** (no `gfortran`, no `lfortran` binary). The Fortran backend is
  therefore *never compiled* here: `src/fortran/octopus_kernels.f90` is written and
  reviewed but unexercised, and the test that compares it against the C++ reference
  (`numerics.sod1d_fortran_matches_cxx_when_available`) **SKIPS with a printed reason**.
  Expect to debug that file on the first machine that has a Fortran compiler.
* **No trained model weights.** Hugging Face and GitHub release assets are unreachable
  from this environment, so no SLM (Llama-3-8B / Phi-3 4-bit or otherwise) is present.
  The `llm.host` module reads real GGUF metadata and, without llama.cpp linked,
  `generate()` returns `Status::unavailable` — or, only when `--allow-stub` is passed
  explicitly, a deterministic *labelled* stub.
* **What *is* demonstrated:** `tools/slmgen` writes a complete, loadable
  llama-architecture GGUF whose weights are deterministic pseudo-random numbers (2 layers,
  embedding 64, 4 heads, F32, ~0.45 MiB). In the `-DOCT_WITH_LLAMA=ON` build, llama.cpp
  loads it and the engine runs the full path — tokenize → prefill → decode → sample →
  detokenize — entirely offline. The tokens it produces are **nonsense**, because the
  weights are noise: the evidence is that the pipeline executes and that greedy decoding
  is bit-for-bit reproducible across calls and processes, not that the engine writes good
  text. The file states its own provenance (`octopus.weights = synthetic-pseudo-random`)
  and every tool that reports it labels it as synthetic.
* The llama.cpp checkout at tag `b11371` is present, and a static build of it
  (`libllama.a`, `libggml{,-base,-cpu}.a`) was produced in this environment. The
  `OCT_HAVE_LLAMA` path of `src/host/llama_host.cpp` compiles, links and runs against a
  generated synthetic model (see above); it has still never run against a *trained* model,
  so real-model behaviour (chat templates, long contexts, KV pressure) is untested.

## SLM / inference

* There is no inference engine in the default build. `octopus version` reports
  `{"compiled": false, "available": false, "stub": true}` and `octopus doctor` explains why.
* The stub backend is deterministic and explicitly labelled in every response
  (`"note":"deterministic stub, not model inference"`); it exists so the surrounding
  pipeline (metadata, tokenizer, routing, JSON contract) can be tested offline. It must
  never be presented as generation.
* No GPU, Metal, CUDA, ROCm, FPGA or UMA path is implemented. Memory-arena behaviour for
  Apple Silicon unified memory is not tested here (no such hardware).
* No chat templates, no grammar-constrained decoding, no speculative decoding, no
  KV-cache quantisation, no batching scheduler.
* 4-bit quantisation is only ever *reported* by reading GGUF metadata (`file_type`,
  bits-per-weight). Octopus does not dequantise or execute tensors.

## Tokenizer

* Exact for byte-level BPE vocabularies **whose `tokenizer.ggml.pre` is `gpt2`/`gpt-2`
  or absent**: the upstream GPT-2 fixture matches case-for-case and token-for-token
  (46/46 cases, 473/473 tokens).
* Other pre-tokenizer patterns (`deepseek-llm`, `qwen2`, `llama-bpe`, ...) fall back to
  GPT-2 pre-tokenisation. The encoder marks itself `degraded`, says so in `method`, and
  the measured agreement is reported (deepseek-llm fixture: 97.8% of cases, 99.6% of
  tokens — a measurement, not a guarantee). Nothing is silently assumed exact.
* SentencePiece vocabularies (no merges in the GGUF) use greedy longest-match, labelled
  `greedy.longest_match.degraded`.
* No added/special-token handling beyond bos/eos/pad ids, no chat templates, no
  normalisation (NFKC etc.), no `llama-bpe` regex port.
* The tokenizer is single-threaded; merge ranks are cached per `Vocab` instance.

## Numerics

* Kernels are CPU-only and use `long double` where the C++ reference path is active. The
  `backend` string always states which path produced a number.
* The shock tube (`sod1d`) is deliberately first-order in space (HLLC + Heun RK2 with
  upwind-limited interface states): measured order ≈ 0.58–0.66 in L1 and a 2.3% L2 error
  at t=0.2 with n=400. It is *not* a high-order solver and the module's `limitations`
  list says so.
* `cavity2d` is a projection method on a staggered grid with **upwind** advection. It is
  stable to Re = 800 in the current configuration, matches the Ghia et al. Re=100
  benchmark within the gates in `tests/test_numerics.cpp`, and its ±lid reflection
  relation is bit-exact — but it is first-order upwind in space and has no turbulence
  model, so it is not quantitatively predictive at high Reynolds number.
* `nbody` is softened Newtonian gravity with a KDK leapfrog; `kepler` is a two-body
  problem in units where GM = a = 1. No relativistic corrections, no long-term
  integrators, no symplectic refinement.
* `gas` evaluates closed-form thermodynamic identities; it is not a flow solver.
* `linsolve` is dense LU with partial pivoting plus an optional iterative refinement and
  a 1-norm condition estimate. Ill-conditioned systems are reported honestly: the Hilbert
  benchmark shows a tiny residual with a large solution error, because that is the truth
  about ill-conditioned systems.
* The 2-D kernels allocate `O(n²)` memory; the largest tested size is n=256 (≈0.5 MiB of
  doubles per field). There is no out-of-core or tiled execution.
* SOR iteration counts are problem- and size-dependent; the residual reported is the
  relative residual of the solved operator, not a discretisation error estimate.

## Logic and self-modification

* `prolog` implements SLD resolution with chronological backtracking. There is **no** cut,
  assert/retract, tabling, occurs-check, or unification of cyclic terms. Search limits
  (default 200 000 inferences, depth 256, one solution) are enforced and reported:
  a truncated search returns `Unknown` with `hit_limit = true`, never `Refuted`.
* `Refuted` requires the predicate to be declared complete by the caller
  (`declare_complete("parent/2")`); otherwise failure to prove is `Unknown` — the open
  world is respected.
* `lisp` is a Scheme subset (no continuations, no tail-call guarantee, no hygienic
  macros, no `call/cc`, depth limit 512). `patch()` requires an existing global and
  `rollback()` restores the exact previous definition set; it does **not** rewrite machine
  code, which the architecture spec forbids.
* `stackvm` is a bounds-checked interpreter, not a native code generator. It cannot touch
  registers or MMIO, and it says so in its `limitations`.

## Resilience and security

* Self-healing means **supervised recovery**: health checks, watchdogs, Prolog invariants,
  bounded restarts, rollback hooks and an audit chain. It is not in-place code patching
  and it cannot recover from a process-level crash (there is no separate supervisor
  process or checkpoint/restore in this build).
* The supervisor watchdog **detects** overruns after the fact (it measures the check's
  duration); it does not preempt a hung check, because checks run on the calling thread.
* The INTERCAL guardrail is **obfuscation plus tamper evidence, not cryptography**. The
  keystream is a public SHA-256 counter construction, the nonce is deterministic (so
  identical plaintext + key gives identical envelopes), there is no key management, and
  no claim of resistance to a determined attacker is made anywhere. INTERCAL is used as a
  deliberately cryptic encoding style, never as a security argument.
* Sandboxing is *capability-level*: modules advertise dotted capabilities in a registry
  and cannot call each other except through that interface. There is no OS-level sandbox,
  no seccomp, no separate process per module, and no signature verification of external
  plugins yet (the manifest hash field exists in `ModuleInfo` and is reported, but the
  loader only registers compiled-in modules in this snapshot).
* Fuzz coverage is bounded: 256–384 single-bit mutations per guardrail test, and a fixed
  list of malformed inputs for the parsers. This is a smoke-level fuzz, not exhaustive.

## Routing

* `Router::classify` is **rule-based, not learned**. Only an explicit prefix
  (`kernel:`, `apl:`, `llm:`, ...) is treated as a declaration with confidence 1.0;
  everything else is a keyword/grammar heuristic that reports the hint it matched and a
  confidence below 1.0, and an unmatched request is refused (`TaskKind::Unknown`). A
  heuristic route can therefore be wrong in principle — the confidence and the reason are
  printed precisely so a caller can see the basis for the choice.
* The router does not decompose compound requests into a task graph, does not chain
  verification automatically, and has no planner. Verifying a generated answer against a
  knowledge base is an explicit second request (`prolog:<claim>` with a `KnowledgeBase`).
* Actor tasks are not routed to: the router refuses to invent a message for an unnamed
  actor, and the runtime is driven through its own API.

## Parallelism

* Occam strategies are OS threads; the measured speedup is reported as measured and can
  be **below 1** for small workloads (thread creation dominates). There is no work
  stealing, affinity control, distributed execution or GPU offload, and no claim of
  linear scaling.
* Actor messages are delivered synchronously by the calling thread; there is no
  scheduler, mailbox prioritisation or persistence. Fault containment is per-handler.

## Visualisation

* Piet output is a quantised cognitive-state map (20 colours; index 18 = black =
  NaN/Inf marker). Quantisation error is bounded by the palette step and the bound is
  documented and tested (≤ 0.12 on [−1,1]); it is lossy and not a physical rendering.
* The PNG writer emits stored (uncompressed) deflate blocks, so files are large
  (~786 KiB for 512×512 RGB; `docs/piet_state_example.png` is 256×256, 196 947 bytes,
  sha256 `bb29e5b6f53f4dae7eeb9236957a476ba442dfe85e4a6661a9abc25b6254eabd`). No image decoding, no animation, no colour management.

## Build/packaging

* Only Linux and macOS are tested. Windows is untested (the code is portable C++20 but
  the CI has not run there).
* `third_party/llama.cpp` is a checkout, not a vendored submodule; it is expected to be
  present for vocabulary fixtures. `tests/run_tests.py` and the benchmark harness skip
  fixture-dependent cases with a printed reason when it is missing.
* Tests that need an optional dependency skip rather than fail; `run_tests.py --strict`
  turns skips into failures when you want a gate that permits no excuse.
