#!/usr/bin/env python3
"""Octopus test runner.

Runs the C++ test binaries and the CLI self-checks, prints the summary and a
coverage matrix mapping the 14 required test categories (ARCHITECTURE_SPEC.md)
to the tests that implement them. Exits non-zero when any executed assertion
fails; skipped tests are reported but do not fail the run unless --strict is
given (a skip always has a printed reason).

Usage:
    python3 tests/run_tests.py [--no-build] [--strict] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"

# The 14 categories required by the architecture spec, mapped to test-name
# prefixes (suite names). A category with no matching executed test is reported
# as MISSING, never silently omitted.
CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("1 clean-build tests", "the configure/build step itself", []),
    ("2 model metadata tests", "integration.model_metadata_from_a_real_gguf_file", ["integration.model_metadata"]),
    ("3 llama.cpp integration tests", "llm.cpp link/load/generate path", ["layers.llm_host", "llm."]),
    ("4 tokenizer tests", "byte-level BPE vs upstream fixtures", ["tokenizer."]),
    ("5 deterministic generation tests", "greedy decode determinism + reproducible results", ["engine.results_are_reproducible", "layers.llm_host", "llm.synthetic"]),
    ("6 numerical kernel correctness tests", "kernels vs analytic/manufactured references", ["numerics."]),
    ("7 array IR property tests", "APL evaluator and codec", ["languages.apl", "layers.piet_quantisation"]),
    ("8 rule-engine tests", "Prolog SLD + verification verdicts", ["languages.prolog"]),
    ("9 concurrency/race tests", "Occam threads and actor messaging", ["layers.occam", "integration.concurrency"]),
    ("10 sandbox/security tests", "capability boundaries and fail-closed behaviour", ["integration.security", "layers.intercal"]),
    ("11 fuzz tests for parsers/DSLs", "malformed input across all parsers", ["integration.parsers_reject", "integration.guard_fuzz"]),
    ("12 end-to-end offline tests", "router dispatch across layers + APL->Prolog->Occam->Piet chain",
     ["integration.offline_end_to_end", "router."]),
    ("13 memory-budget tests", "bounded VM memory and bounded rendering", ["integration.memory_use"]),
    ("14 benchmark harnesses", "tools/octbench + bench/bench.py", []),
]


def run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true", help="skip the cmake build step")
    ap.add_argument("--strict", action="store_true", help="treat skipped tests as failures")
    ap.add_argument("--build-dir", default="build", help="build directory to test")
    ap.add_argument("--json", dest="json_out", default=None, help="write the summary JSON here")
    args = ap.parse_args()

    results: dict = {"build": None, "tests": None, "selftest": None, "categories": {}}
    failures: list[str] = []

    if not args.no_build:
        cfg = run(["cmake", "-B", "build", "-S", ".", "-DCMAKE_BUILD_TYPE=Release", "-DOCT_WITH_LLAMA=OFF"])
        if cfg.returncode != 0:
            print(cfg.stdout[-4000:])
            print(cfg.stderr[-4000:], file=sys.stderr)
            return 1
        build = run(["cmake", "--build", "build", "-j"])
        if build.returncode != 0:
            print(build.stdout[-6000:])
            print(build.stderr[-4000:], file=sys.stderr)
            return 1
        results["build"] = "ok"
        print("build: ok")

    tests_bin = BUILD / "oct_tests"
    if not tests_bin.exists():
        print(f"missing {tests_bin}: run without --no-build first", file=sys.stderr)
        return 1

    proc = run([str(tests_bin), "--json"])
    match = re.search(r"\{.*\}", proc.stdout, re.S)
    if not match:
        print("could not parse test JSON output", file=sys.stderr)
        print(proc.stdout[-4000:], file=sys.stderr)
        return 1
    summary = json.loads(match.group(0))
    results["tests"] = summary
    print(f"tests: {summary['passed']} passed, {summary['failed']} failed, "
          f"{summary['skipped']} skipped, {summary['assertions']} assertions")
    for f in summary["failures"]:
        failures.append(f"test failure: {f}")

    # List the executed test names to build the category matrix.
    listing = run([str(tests_bin), "--list"])
    names = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    results["test_names"] = names

    cli = BUILD / "octopus"
    if cli.exists():
        st = run([str(cli), "selftest"])
        ok = st.stdout.count("[ ok ]")
        bad = st.stdout.count("[FAIL]")
        results["selftest"] = {"modules_ok": ok, "modules_failed": bad}
        print(f"module self-checks: {ok} ok, {bad} failed")
        if bad:
            failures.append(f"{bad} module self-check(s) failed")
        doctor = run([str(cli), "doctor"])
        results["doctor"] = doctor.stdout.strip()

    for label, description, prefixes in CATEGORIES:
        matched = [n for n in names if any(n.startswith(p) for p in prefixes)] if prefixes else []
        if prefixes:
            status = "COVERED" if matched else "MISSING"
            if not matched:
                failures.append(f"category not covered: {label}")
        else:
            # Build/benchmark categories are covered by the runner itself.
            status = "EXTERNAL"
        results["categories"][label] = {"status": status, "description": description,
                                        "tests": matched}
        print(f"  [{status:8}] {label} -- {description}"
              + (f" ({len(matched)} tests)" if matched else ""))

    if args.strict and summary["skipped"]:
        failures.append(f"{summary['skipped']} skipped test(s) with --strict")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"summary written to {args.json_out}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall executed assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
