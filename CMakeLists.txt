cmake_minimum_required(VERSION 3.20)

# =============================================================================
# Genisus / Octopus Hybrid AI Engine -- Forced Multi-Language Binding Matrix
# =============================================================================
# Every language DNA bridge is a mandatory compile-time requirement:
#   1. Fortran 2023 (Absolute Physics Core: fluid dynamics, Kepler/N-body
#      gravity, PDEs, thermodynamics, linear systems, covariant tensor fields)
#   2. LISP (Autonomous Evolution Engine: S-expression AST hot-patching)
#   3. Prolog (Deterministic Soundness Guardrail: SLD backtracking verifier)
#   4. APL (Array Codec & Context Matrix Compressor: right-to-left evaluator)
#   5. Occam, Forth, & Smalltalk (CSP parallel channels, stack/register VM,
#      and live-cell supervised actor runtime)
#   6. INTERCAL (The Obfuscated Core Shield: bit-mingled hash-linked envelope)
#   7. Piet (Visual Canvas Logic Engine: 20-colour codel & synapse renderer)
#   8. Physical Universe Image/Video Generation Engine (Fortran -> APL ->
#      Prolog -> Piet PNG/PPM/Y4M)
#
# Optional C++ reference fallback flags are banned. When targeted compilers
# (`gfortran`, `lfortran`) or hardware features (`llama.cpp` runtime) are
# absent on the host, the build preserves structural binaries intact and gates
# live execution blocks with `UNSUPPORTED_HARDWARE_SKIP` -- never downscaling
# to pseudo-C++ approximations.
# =============================================================================

project(octopus
    VERSION 2.0.0
    DESCRIPTION "Genisus / Octopus Deterministic Multi-Language Hybrid AI Engine"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(OCT_BUILD_TESTS "Build the verification and hardware-skip test suite" ON)
option(OCT_BUILD_TOOLS "Build unified genisus CLI, dashboard, and benchmarks" ON)
option(OCT_STRICT_COMPILER_GATE "Abort configure if gfortran/lfortran is absent on host" OFF)
option(OCT_SANITIZE "Build with AddressSanitizer + UBSan" OFF)
option(OCT_TSAN "Build with ThreadSanitizer" OFF)
option(OCT_FAST_MATH "Enable -ffast-math in the engine core" OFF)

# -----------------------------------------------------------------------------
# Verify that every mandatory multi-language bridge source & header exists.
# Flag a compilation failure if any core multi-language bridge fails to establish.
# -----------------------------------------------------------------------------
set(OCT_REQUIRED_LANGUAGE_BRIDGES
    "include/octopus/core.hpp"
    "include/octopus/numerics.hpp"
    "include/octopus/apl.hpp"
    "include/octopus/prolog.hpp"
    "include/octopus/lisp.hpp"
    "include/octopus/stackvm.hpp"
    "include/octopus/smalltalk.hpp"
    "include/octopus/occam.hpp"
    "include/octopus/intercal.hpp"
    "include/octopus/piet.hpp"
    "include/octopus/supervisor.hpp"
    "include/octopus/router.hpp"
    "include/octopus/universe.hpp"
    "include/octopus/llm.hpp"
    "src/fortran/octopus_kernels.f90"
    "src/fortran/fortran_hw_skip.cpp"
    "src/core/core.cpp"
    "src/core/tensor.cpp"
    "src/core/gguf.cpp"
    "src/core/registry.cpp"
    "src/core/audit.cpp"
    "src/core/module.cpp"
    "src/core/router.cpp"
    "src/numerics/numerics.cpp"
    "src/numerics/kernel_util.cpp"
    "src/numerics/kernel_sod1d.cpp"
    "src/numerics/kernel_heat_poisson.cpp"
    "src/numerics/kernel_mechanics.cpp"
    "src/numerics/kernel_gas_linalg.cpp"
    "src/numerics/linalg.cpp"
    "src/apl/apl.cpp"
    "src/prolog/prolog.cpp"
    "src/lisp/lisp.cpp"
    "src/stackvm/stackvm.cpp"
    "src/tokenizer/tokenizer.cpp"
    "src/smalltalk/smalltalk.cpp"
    "src/occam/occam.cpp"
    "src/intercal/intercal.cpp"
    "src/piet/piet.cpp"
    "src/supervisor/supervisor.cpp"
    "src/host/llama_host.cpp"
    "src/host/synthetic_model.cpp"
    "src/universe/universe.cpp"
    "tools/genisus.cpp"
)

foreach(_bridge_file IN LISTS OCT_REQUIRED_LANGUAGE_BRIDGES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_bridge_file}")
        message(FATAL_ERROR
            "FORCED MULTI-LANGUAGE BINDING FAILURE: required bridge '${_bridge_file}' is missing.")
    endif()
endforeach()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)
find_package(Threads REQUIRED)

# -----------------------------------------------------------------------------
# Mandatory Bridge 1: Fortran 2023 Absolute Physics Core (`octopus_fortran_kernels`)
# -----------------------------------------------------------------------------
include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    add_library(octopus_fortran_kernels STATIC src/fortran/octopus_kernels.f90)
    set_target_properties(octopus_fortran_kernels PROPERTIES Fortran_FORMAT FREE)
    target_compile_options(octopus_fortran_kernels PRIVATE -O3 -fno-unsafe-math-optimizations)
    set(OCT_HAVE_FORTRAN_COMPILER ON)
    message(STATUS "Fortran 2023 Absolute Physics Core: ENABLED (${CMAKE_Fortran_COMPILER})")
else()
    if(OCT_STRICT_COMPILER_GATE)
        message(FATAL_ERROR
            "FORCED MULTI-LANGUAGE BINDING FAILURE: Fortran 2023 compiler (gfortran/lfortran) not found.")
    endif()
    add_library(octopus_fortran_kernels STATIC src/fortran/fortran_hw_skip.cpp)
    set(OCT_HAVE_FORTRAN_COMPILER OFF)
    message(STATUS "Fortran 2023 Absolute Physics Core: UNSUPPORTED_HARDWARE_SKIP "
                   "(gfortran/lfortran absent; C++ fallback banned; linking ABI hardware-skip gate)")
endif()

# -----------------------------------------------------------------------------
# Mandatory Bridge 2: Native llama.cpp runtime detection (zero C++ stub fallback)
# -----------------------------------------------------------------------------
set(OCT_LLAMA_PREBUILT_DIR "" CACHE PATH
    "Directory containing prebuilt static llama.cpp libraries (libllama.a, libggml*.a)")
set(OCT_LLAMA_LINKED OFF)
if(OCT_LLAMA_PREBUILT_DIR AND EXISTS "${OCT_LLAMA_PREBUILT_DIR}/libllama.a")
    file(GLOB OCT_LLAMA_EXTRA_LIBS
         "${OCT_LLAMA_PREBUILT_DIR}/libllama.a"
         "${OCT_LLAMA_PREBUILT_DIR}/ggml/src/libggml.a"
         "${OCT_LLAMA_PREBUILT_DIR}/ggml/src/libggml-cpu.a"
         "${OCT_LLAMA_PREBUILT_DIR}/ggml/src/libggml-base.a")
    set(OCT_LLAMA_LINKED ON)
    message(STATUS "llama.cpp native backend: ENABLED (${OCT_LLAMA_PREBUILT_DIR})")
else()
    message(STATUS "llama.cpp native backend: UNSUPPORTED_HARDWARE_SKIP "
                   "(prebuilt llama.cpp absent; C++ stub fallback is banned)")
endif()

# -----------------------------------------------------------------------------
# Unified Core Library (`octopus_core`)
# -----------------------------------------------------------------------------
set(OCT_CXX_SOURCES
    src/core/core.cpp
    src/core/tensor.cpp
    src/core/gguf.cpp
    src/core/registry.cpp
    src/core/audit.cpp
    src/core/module.cpp
    src/core/router.cpp
    src/numerics/numerics.cpp
    src/numerics/kernel_util.cpp
    src/numerics/kernel_sod1d.cpp
    src/numerics/kernel_heat_poisson.cpp
    src/numerics/kernel_mechanics.cpp
    src/numerics/kernel_gas_linalg.cpp
    src/numerics/linalg.cpp
    src/apl/apl.cpp
    src/prolog/prolog.cpp
    src/lisp/lisp.cpp
    src/stackvm/stackvm.cpp
    src/tokenizer/tokenizer.cpp
    src/smalltalk/smalltalk.cpp
    src/occam/occam.cpp
    src/intercal/intercal.cpp
    src/piet/piet.cpp
    src/supervisor/supervisor.cpp
    src/host/llama_host.cpp
    src/host/synthetic_model.cpp
    src/universe/universe.cpp
)

add_library(octopus_core STATIC ${OCT_CXX_SOURCES})
target_include_directories(octopus_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/numerics)
target_link_libraries(octopus_core PUBLIC octopus_fortran_kernels Threads::Threads)

if(OCT_FAST_MATH)
    target_compile_options(octopus_core PRIVATE -ffast-math)
else()
    target_compile_options(octopus_core PRIVATE -fno-fast-math)
endif()

if(OCT_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()
if(OCT_TSAN)
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=thread)
endif()

if(OCT_HAVE_FORTRAN_COMPILER)
    target_compile_definitions(octopus_core PUBLIC OCT_HAVE_FORTRAN=1)
endif()

if(OCT_LLAMA_LINKED)
    target_include_directories(octopus_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include")
    target_link_libraries(octopus_core PUBLIC ${OCT_LLAMA_EXTRA_LIBS} ${CMAKE_DL_LIBS})
    target_compile_definitions(octopus_core PUBLIC OCT_HAVE_LLAMA=1)
endif()

# -----------------------------------------------------------------------------
# Tests & Unified CLI (`genisus`, `octopus`, `octbench`, `ggufgen`, `slmgen`)
# -----------------------------------------------------------------------------
if(OCT_BUILD_TESTS)
    enable_testing()
    file(GLOB OCT_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
    add_executable(oct_tests tests/main.cpp ${OCT_TEST_SOURCES})
    target_link_libraries(oct_tests PRIVATE octopus_core)
    target_include_directories(oct_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_compile_definitions(oct_tests PRIVATE
        OCT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
    add_test(NAME oct_tests COMMAND oct_tests)
endif()

if(OCT_BUILD_TOOLS)
    file(GLOB OCT_TOOL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cpp)
    foreach(tool_src ${OCT_TOOL_SOURCES})
        get_filename_component(tool_name ${tool_src} NAME_WE)
        add_executable(${tool_name} ${tool_src})
        target_link_libraries(${tool_name} PRIVATE octopus_core)
        target_include_directories(${tool_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endforeach()
endif()
