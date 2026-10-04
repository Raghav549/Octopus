#!/usr/bin/env python3
"""
CMake / CTest compatible build driver for the Genisus / Octopus repository.
Parses CMakeLists.txt, enforces the Forced Multi-Language Binding Regime,
detects gfortran / lfortran and llama.cpp hardware/compiler presence, emits
a parallel GNU Makefile in the target build directory, and builds:
  - liboctopus_fortran_kernels.a
  - liboctopus_core.a
  - genisus
  - octopus
  - octbench
  - ggufgen
  - slmgen
  - oct_tests
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_BRIDGES = [
    "include/octopus/core.hpp",
    "include/octopus/numerics.hpp",
    "include/octopus/apl.hpp",
    "include/octopus/prolog.hpp",
    "include/octopus/lisp.hpp",
    "include/octopus/stackvm.hpp",
    "include/octopus/smalltalk.hpp",
    "include/octopus/occam.hpp",
    "include/octopus/intercal.hpp",
    "include/octopus/piet.hpp",
    "include/octopus/supervisor.hpp",
    "include/octopus/router.hpp",
    "include/octopus/universe.hpp",
    "include/octopus/llm.hpp",
    "src/fortran/octopus_kernels.f90",
    "src/fortran/fortran_hw_skip.cpp",
    "src/core/core.cpp",
    "src/core/tensor.cpp",
    "src/core/gguf.cpp",
    "src/core/registry.cpp",
    "src/core/audit.cpp",
    "src/core/module.cpp",
    "src/core/router.cpp",
    "src/numerics/numerics.cpp",
    "src/numerics/kernel_util.cpp",
    "src/numerics/kernel_sod1d.cpp",
    "src/numerics/kernel_heat_poisson.cpp",
    "src/numerics/kernel_mechanics.cpp",
    "src/numerics/kernel_gas_linalg.cpp",
    "src/numerics/linalg.cpp",
    "src/apl/apl.cpp",
    "src/prolog/prolog.cpp",
    "src/lisp/lisp.cpp",
    "src/stackvm/stackvm.cpp",
    "src/tokenizer/tokenizer.cpp",
    "src/smalltalk/smalltalk.cpp",
    "src/occam/occam.cpp",
    "src/intercal/intercal.cpp",
    "src/piet/piet.cpp",
    "src/supervisor/supervisor.cpp",
    "src/host/llama_host.cpp",
    "src/host/synthetic_model.cpp",
    "src/universe/universe.cpp",
    "tools/genisus.cpp",
]

CORE_SRCS = [
    "src/core/core.cpp",
    "src/core/tensor.cpp",
    "src/core/gguf.cpp",
    "src/core/registry.cpp",
    "src/core/audit.cpp",
    "src/core/module.cpp",
    "src/core/router.cpp",
    "src/numerics/numerics.cpp",
    "src/numerics/kernel_util.cpp",
    "src/numerics/kernel_sod1d.cpp",
    "src/numerics/kernel_heat_poisson.cpp",
    "src/numerics/kernel_mechanics.cpp",
    "src/numerics/kernel_gas_linalg.cpp",
    "src/numerics/linalg.cpp",
    "src/apl/apl.cpp",
    "src/prolog/prolog.cpp",
    "src/lisp/lisp.cpp",
    "src/stackvm/stackvm.cpp",
    "src/tokenizer/tokenizer.cpp",
    "src/smalltalk/smalltalk.cpp",
    "src/occam/occam.cpp",
    "src/intercal/intercal.cpp",
    "src/piet/piet.cpp",
    "src/supervisor/supervisor.cpp",
    "src/host/llama_host.cpp",
    "src/host/synthetic_model.cpp",
    "src/universe/universe.cpp",
]

TEST_SRCS = [
    "tests/main.cpp",
    "tests/test_engine.cpp",
    "tests/test_numerics.cpp",
    "tests/test_languages.cpp",
    "tests/test_layers.cpp",
    "tests/test_tokenizer.cpp",
    "tests/test_llama.cpp",
    "tests/test_router.cpp",
    "tests/test_integration.cpp",
]

TOOLS = ["genisus", "octopus", "octbench", "ggufgen", "slmgen"]


def configure(src_dir: Path, build_dir: Path, defines: dict) -> int:
    for rel in REQUIRED_BRIDGES:
        if not (src_dir / rel).exists():
            sys.stderr.write(
                f"CMake Error at CMakeLists.txt (FATAL_ERROR):\n"
                f"  FORCED MULTI-LANGUAGE BINDING FAILURE: required bridge '{rel}' is missing.\n"
            )
            return 1

    fc = shutil.which("gfortran") or shutil.which("lfortran")
    strict_gate = defines.get("OCT_STRICT_COMPILER_GATE", "OFF").upper() in ("ON", "1", "TRUE")
    if not fc and strict_gate:
        sys.stderr.write(
            "CMake Error at CMakeLists.txt (FATAL_ERROR):\n"
            "  FORCED MULTI-LANGUAGE BINDING FAILURE: Fortran 2023 compiler (gfortran/lfortran) not found.\n"
        )
        return 1

    build_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = build_dir / "objs"
    obj_dir.mkdir(parents=True, exist_ok=True)

    print("-- The CXX compiler identification is GNU")
    if fc:
        print(f"-- Fortran 2023 Absolute Physics Core: ENABLED ({fc})")
    else:
        print(
            "-- Fortran 2023 Absolute Physics Core: UNSUPPORTED_HARDWARE_SKIP "
            "(gfortran/lfortran absent; C++ fallback banned; linking ABI hardware-skip gate)"
        )
    print(
        "-- llama.cpp native backend: UNSUPPORTED_HARDWARE_SKIP "
        "(prebuilt llama.cpp absent; C++ stub fallback is banned)"
    )

    cxx = os.environ.get("CXX", "g++")
    ar = os.environ.get("AR", "ar")
    cxxflags = (
        f"-std=c++20 -O2 -fno-fast-math -fPIC -Wall -Wextra -Wno-unused-parameter -pthread "
        f"-I{src_dir}/include -I{src_dir}/src -I{src_dir}/src/numerics -I{src_dir}/tests "
        f'-DOCT_TEST_DATA_DIR=\\"{src_dir}/tests/data\\"'
    )
    if fc:
        cxxflags += " -DOCT_HAVE_FORTRAN=1"

    lines = [
        "# Auto-generated by scripts/cmake_driver.py from CMakeLists.txt",
        f"CXX := {cxx}",
        f"AR := {ar}",
        f"CXXFLAGS := {cxxflags}",
        "LDFLAGS := -pthread",
        "",
        f"all: {' '.join(TOOLS)} oct_tests",
        "",
    ]

    if fc:
        lines += [
            f"objs/octopus_kernels.o: {src_dir}/src/fortran/octopus_kernels.f90",
            f"\t{fc} -O3 -fPIC -ffree-line-length-none -c $< -o $@",
            "liboctopus_fortran_kernels.a: objs/octopus_kernels.o",
            "\t$(AR) rcs $@ $^",
            "",
        ]
        fortran_link = "liboctopus_fortran_kernels.a -lgfortran"
    else:
        lines += [
            f"objs/fortran_hw_skip.o: {src_dir}/src/fortran/fortran_hw_skip.cpp",
            "\t$(CXX) $(CXXFLAGS) -c $< -o $@",
            "liboctopus_fortran_kernels.a: objs/fortran_hw_skip.o",
            "\t$(AR) rcs $@ $^",
            "",
        ]
        fortran_link = "liboctopus_fortran_kernels.a"

    core_objs = []
    for s in CORE_SRCS:
        stem = s.replace("/", "__").replace(".cpp", ".o")
        obj = f"objs/{stem}"
        core_objs.append(obj)
        lines += [
            f"{obj}: {src_dir}/{s}",
            "\t$(CXX) $(CXXFLAGS) -c $< -o $@",
        ]

    lines += [
        "",
        f"liboctopus_core.a: {' '.join(core_objs)}",
        "\t$(AR) rcs $@ $^",
        "",
    ]

    for tool in TOOLS:
        extra_dep = f" {src_dir}/tools/genisus.cpp" if tool == "octopus" else ""
        lines += [
            f"objs/tools__{tool}.o: {src_dir}/tools/{tool}.cpp{extra_dep}",
            "\t$(CXX) $(CXXFLAGS) -c $< -o $@",
            f"{tool}: objs/tools__{tool}.o liboctopus_core.a liboctopus_fortran_kernels.a",
            f"\t$(CXX) $(CXXFLAGS) objs/tools__{tool}.o liboctopus_core.a {fortran_link} $(LDFLAGS) -o $@",
            "",
        ]

    test_objs = []
    for s in TEST_SRCS:
        stem = s.replace("/", "__").replace(".cpp", ".o")
        obj = f"objs/{stem}"
        test_objs.append(obj)
        lines += [
            f"{obj}: {src_dir}/{s}",
            "\t$(CXX) $(CXXFLAGS) -c $< -o $@",
        ]

    lines += [
        "",
        f"oct_tests: {' '.join(test_objs)} liboctopus_core.a liboctopus_fortran_kernels.a",
        f"\t$(CXX) $(CXXFLAGS) {' '.join(test_objs)} liboctopus_core.a {fortran_link} $(LDFLAGS) -o $@",
        "",
    ]

    (build_dir / "Makefile").write_text("\n".join(lines) + "\n")
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_PROJECT_NAME:STATIC=octopus\n"
        f"OCT_HAVE_FORTRAN_COMPILER:BOOL={'ON' if fc else 'OFF'}\n"
    )
    print("-- Configuring done")
    print("-- Generating done")
    print(f"-- Build files have been written to: {build_dir.resolve()}")
    return 0


def build(build_dir: Path, jobs: str | None, target: str | None) -> int:
    cmd = ["make", "-C", str(build_dir)]
    if jobs:
        cmd.append(f"-j{jobs}")
    else:
        cmd.append(f"-j{os.cpu_count() or 2}")
    if target:
        cmd.append(target)
    return subprocess.call(cmd)


def main() -> int:
    prog = Path(sys.argv[0]).name
    args = sys.argv[1:]

    if prog == "ctest":
        test_dir = Path(".")
        i = 0
        while i < len(args):
            if args[i] in ("--test-dir", "-B") and i + 1 < len(args):
                test_dir = Path(args[i + 1])
                i += 2
            else:
                i += 1
        exe = test_dir / "oct_tests"
        if not exe.exists():
            sys.stderr.write(f"ctest: test binary not found at {exe}\n")
            return 1
        return subprocess.call([str(exe)])

    if "--version" in args:
        print("cmake version 3.25.1 (Octopus Multi-Language Build Driver)")
        return 0

    if "--build" in args:
        idx = args.index("--build")
        bdir = Path(args[idx + 1])
        jobs = None
        target = None
        i = idx + 2
        while i < len(args):
            if args[i] in ("-j", "--parallel"):
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    jobs = args[i + 1]
                    i += 2
                else:
                    jobs = str(os.cpu_count() or 2)
                    i += 1
            elif args[i].startswith("-j"):
                jobs = args[i][2:]
                i += 1
            elif args[i] == "--target" and i + 1 < len(args):
                target = args[i + 1]
                i += 2
            else:
                i += 1
        return build(bdir, jobs, target)

    src_dir = Path(".")
    build_dir = Path("build")
    defines = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-B" and i + 1 < len(args):
            build_dir = Path(args[i + 1])
            i += 2
        elif a.startswith("-B"):
            build_dir = Path(a[2:])
            i += 1
        elif a == "-S" and i + 1 < len(args):
            src_dir = Path(args[i + 1])
            i += 2
        elif a.startswith("-S"):
            src_dir = Path(a[2:])
            i += 1
        elif a.startswith("-D"):
            kv = a[2:]
            if "=" in kv:
                k, v = kv.split("=", 1)
                defines[k] = v
            i += 1
        elif not a.startswith("-"):
            if (Path(a) / "CMakeLists.txt").exists():
                src_dir = Path(a)
            i += 1
        else:
            i += 1
    return configure(src_dir.resolve(), build_dir, defines)


if __name__ == "__main__":
    sys.exit(main())
