#!/usr/bin/env python3
"""Octopus reproducible benchmark driver.

Runs tools/octbench (fixed inputs, min/median over repeats), captures the
machine/build fingerprint and writes results as CSV plus a markdown summary in
bench/results/. The same script on the same machine with the same build must
produce the same fingerprints; timings may vary and are reported as measured.

Usage:
    python3 bench/bench.py [--repeats N] [--no-build] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    if not args.no_build:
        cfg = run(["cmake", "-B", "build", "-S", ".", "-DCMAKE_BUILD_TYPE=Release",
                   "-DOCT_WITH_LLAMA=OFF"])
        if cfg.returncode != 0:
            print(cfg.stderr[-3000:], file=sys.stderr)
            return 1
        b = run(["cmake", "--build", "build", "-j", "--target", "octbench"])
        if b.returncode != 0:
            print(b.stderr[-4000:], file=sys.stderr)
            return 1

    bench = BUILD / "octbench"
    if not bench.exists():
        print("build/octbench missing (build the tools first)", file=sys.stderr)
        return 1

    cli = run([str(BUILD / "octopus"), "version"]) if (BUILD / "octopus").exists() else None
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = run([str(bench), "--repeats", str(args.repeats), "--require"])
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        return 1

    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    comments = [l for l in lines if l.startswith("#")]
    rows = list(csv.reader(io.StringIO("\n".join(l for l in lines if not l.startswith("#")))))
    header, data = rows[0], rows[1:]

    (out_dir / "raw.csv").write_text("\n".join(lines) + "\n")

    md = ["# Octopus benchmark results", ""]
    if cli is not None:
        md.append(f"`octopus version` -> `{cli.stdout.strip()}`")
        md.append("")
    md.extend(comments)
    md.extend(["", f"Repeats per case: {args.repeats} (min/median/max milliseconds)", ""])
    md.append("| category | name | params | min ms | median ms | max ms | detail |")
    md.append("|---|---|---|---:|---:|---:|---|")
    for r in data:
        if len(r) < 8:
            continue
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")
    md.append("")
    md.append("Reproduce with: `python3 bench/bench.py --repeats %d`" % args.repeats)
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwrote {out_dir/'raw.csv'} and {out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
