#!/usr/bin/env bash
# Apply build-host-only CMake overrides for modern CUDA arch (sm_86/89) + quieter
# deprecation flags. Safe to re-run (idempotent markers).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_CMAKE="$ROOT/src/GpGpu.cmake"
CMAKE_LISTS="$ROOT/CMakeLists.txt"
TOOLS_H="$ROOT/include/GpGpu/GpGpu_Tools.h"

python3 - "$GPU_CMAKE" "$CMAKE_LISTS" "$TOOLS_H" <<'PY'
from pathlib import Path
import sys

gpu, cmake_path, tools_path = map(Path, sys.argv[1:])
text = gpu.read_text(encoding="utf-8")
marker = "# RUNPOD MODERN CUDA ARCH OVERRIDE"
if marker not in text:
    probe = text.index("# verif if FoundCapa.exe exists --")
    arch = text.index('set(_cudaArch "${_outNVCC}")', probe)
    old_probe = text[probe:arch]
    text = (
        text[:probe]
        + f"""{marker}
if(DEFINED MICMAC_CUDA_ARCH AND NOT "${{MICMAC_CUDA_ARCH}}" STREQUAL "")
 set(_resultNVCC 0)
 set(_outNVCC "override")
else()
{old_probe}endif()

"""
        + text[arch:]
    )
    arch = text.index('set(_cudaArch "${_outNVCC}")')
    message = text.index('message("Cuda API Version', arch)
    old_arch = text[arch:message]
    text = (
        text[:arch]
        + f"""{marker}
if(DEFINED MICMAC_CUDA_ARCH AND NOT "${{MICMAC_CUDA_ARCH}}" STREQUAL "")
 set(cuda_arch_version "${{MICMAC_CUDA_ARCH}}")
 if("${{MICMAC_CUDA_ARCH}}" STREQUAL "120")
  set(cuda_arch_version_string "12.0")
  set(cuda_generation "Blackwell")
 elseif("${{MICMAC_CUDA_ARCH}}" STREQUAL "100")
  set(cuda_arch_version_string "10.0")
  set(cuda_generation "Blackwell")
 elseif("${{MICMAC_CUDA_ARCH}}" STREQUAL "90")
  set(cuda_arch_version_string "9.0")
  set(cuda_generation "Hopper")
 elseif("${{MICMAC_CUDA_ARCH}}" STREQUAL "89")
  set(cuda_arch_version_string "8.9")
  set(cuda_generation "Ada")
 elseif("${{MICMAC_CUDA_ARCH}}" STREQUAL "86")
  set(cuda_arch_version_string "8.6")
  set(cuda_generation "Ampere")
 elseif("${{MICMAC_CUDA_ARCH}}" STREQUAL "80")
  set(cuda_arch_version_string "8.0")
  set(cuda_generation "Ampere")
 else()
  message(FATAL_ERROR "Unsupported MICMAC_CUDA_ARCH=${{MICMAC_CUDA_ARCH}}")
 endif()
else()
{old_arch}endif()

"""
        + text[message:]
    )
    gpu.write_text(text, encoding="utf-8")
    print("patched GpGpu.cmake arch override")
else:
    print("GpGpu.cmake arch override already present")

compat = "RUNPOD CUDA 11.8 LEGACY API COMPAT"
cmake = cmake_path.read_text(encoding="utf-8")
if compat not in cmake:
    needle = '-Wno-error=stringop-overflow")'
    if needle in cmake:
        cmake = cmake.replace(
            needle,
            f'-Wno-error=stringop-overflow -Wno-error=deprecated-declarations") # {compat}',
            1,
        )
        cmake_path.write_text(cmake, encoding="utf-8")
        print("patched CMakeLists deprecation flags")
    else:
        print("WARN: CMakeLists overflow flag needle not found; continuing")
else:
    print("CMakeLists compat flags already present")

if tools_path.is_file():
    data = tools_path.read_bytes()
    legacy = b'"' + bytes([0xA4]) + b'"'
    if legacy in data:
        tools_path.write_bytes(data.replace(legacy, b'"*"', 1))
        print("patched GpGpu_Tools.h legacy char")
print("sm86_cmake_overrides_ok")
PY
