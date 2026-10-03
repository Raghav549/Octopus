cmake_minimum_required(VERSION 3.20)
project(octopus VERSION 0.2.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(OCT_WITH_LLAMA    "Integrate llama.cpp when a checkout is present in third_party/" ON)
option(OCT_WITH_FORTRAN  "Build the Fortran kernels when a compiler is available" ON)
option(OCT_BUILD_TESTS   "Build the test suite" ON)
option(OCT_BUILD_TOOLS   "Build CLI tools and benchmarks" ON)
option(OCT_SANITIZE      "Build with AddressSanitizer + UBSan" OFF)
option(OCT_TSAN          "Build with ThreadSanitizer" OFF)
option(OCT_FAST_MATH     "Enable -ffast-math in the engine core (OFF: reproducible results)" OFF)
set(OCT_LLAMA_PREBUILT_DIR "" CACHE PATH
    "Directory containing prebuilt static llama.cpp libraries (libllama.a, libggml*.a)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

set(OCT_CXX_SOURCES
  src/core/core.cpp
  src/core/tensor.cpp
  src/core/gguf.cpp
  src/core/registry.cpp
  src/core/audit.cpp
  src/core/module.cpp
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
)

add_library(octopus_core STATIC ${OCT_CXX_SOURCES})
target_include_directories(octopus_core
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/numerics)
find_package(Threads REQUIRED)
target_link_libraries(octopus_core PUBLIC Threads::Threads)

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

# ---------------------------------------------------------------------------
# llama.cpp integration (pinned revision; see third_party/LLAMA_CPP_REVISION)
# ---------------------------------------------------------------------------
if(OCT_WITH_LLAMA)
  if(OCT_LLAMA_PREBUILT_DIR AND EXISTS "${OCT_LLAMA_PREBUILT_DIR}/libllama.a")
    file(GLOB OCT_LLAMA_EXTRA_LIBS
         "${OCT_LLAMA_PREBUILT_DIR}/libllama.a"
         "${OCT_LLAMA_PREBUILT_DIR}/ggml/src/libggml.a"
         "${OCT_LLAMA_PREBUILT_DIR}/ggml/src/libggml-cpu.a"
         "${OCT_LLAMA_PREBUILT_DIR}/ggml/src/libggml-base.a")
    add_library(octopus_llama INTERFACE)
    target_include_directories(octopus_llama INTERFACE
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include")
    target_link_libraries(octopus_llama INTERFACE ${OCT_LLAMA_EXTRA_LIBS} Threads::Threads ${CMAKE_DL_LIBS})
    set_target_properties(octopus_llama PROPERTIES INTERFACE_COMPILE_DEFINITIONS "OCT_HAVE_LLAMA=1")
    target_link_libraries(octopus_core PUBLIC octopus_llama)
    target_compile_definitions(octopus_core PUBLIC OCT_HAVE_LLAMA=1)
    message(STATUS "Octopus: linking prebuilt llama.cpp from ${OCT_LLAMA_PREBUILT_DIR}")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
    set(GGML_CCACHE          OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
    target_compile_definitions(octopus_core PUBLIC OCT_HAVE_LLAMA=1)
    target_link_libraries(octopus_core PUBLIC llama)
    message(STATUS "Octopus: building llama.cpp from third_party/llama.cpp")
  else()
    message(STATUS "Octopus: llama.cpp not found -- SLM inference disabled. "
                   "Run scripts/fetch_deps.sh to enable it.")
    set(OCT_WITH_LLAMA OFF)
  endif()
endif()

# ---------------------------------------------------------------------------
# Fortran kernels (optional; the C++ reference implementation always works)
# ---------------------------------------------------------------------------
if(OCT_WITH_FORTRAN)
  include(CheckLanguage)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    add_library(octopus_fortran STATIC src/fortran/octopus_kernels.f90)
    set_target_properties(octopus_fortran PROPERTIES Fortran_FORMAT FREE)
    target_compile_options(octopus_fortran PRIVATE -O3 -fno-unsafe-math-optimizations)
    target_link_libraries(octopus_core PUBLIC octopus_fortran)
    target_compile_definitions(octopus_core PUBLIC OCT_HAVE_FORTRAN=1)
    message(STATUS "Octopus: Fortran backend enabled with ${CMAKE_Fortran_COMPILER_ID} "
                   "(${CMAKE_Fortran_COMPILER})")
  else()
    message(STATUS "Octopus: no Fortran compiler found -- numerical kernels use the C++ "
                   "long-double reference implementation (results reported as degraded).")
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests / tools
# ---------------------------------------------------------------------------
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

message(STATUS "Octopus ${PROJECT_VERSION}: build=${CMAKE_BUILD_TYPE}, "
               "llama=${OCT_WITH_LLAMA}, sanitizers=${OCT_SANITIZE}, tsan=${OCT_TSAN}")
