#!/usr/bin/env bash
# Standalone CUDA MicMac (mm3d) build for CI / offline packaging.
#
# Produces:
#   dist/micmac-gpgpu-sm<ARCHS>-<gitsha>.tar.gz
#   dist/micmac-gpgpu-sm<ARCHS>-<gitsha>.manifest.json
#
# Env:
#   CUDA_ARCH=86           primary arch for cmake MICMAC_CUDA_ARCH (default 86)
#   CUDA_ARCHS=86          comma/space list of SMs to embed, e.g. "80,86,89"
#                          default = CUDA_ARCH only. Fat binary = one mm3d, many SMs.
#   JOBS=N                 (default nproc)
#   BUILD_DIR=...          (default $ROOT/build-ci-sm${CUDA_ARCH})
#   PREFIX=...             (default $ROOT/install-ci-sm${CUDA_ARCH})
#   DIST_DIR=...           (default $ROOT/dist)
#   CUDA_HOME=...          (default /usr/local/cuda)
#   CUDA_SAMPLE_DIR=...    (auto-setup via ci/setup_cuda_samples.sh)
#   SKIP_SAMPLES=1         if CUDA_SAMPLE_DIR already set
#   APPLY_OVERRIDES=1      default 1 — apply modern-arch cmake patches
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
CUDA_ARCH="${CUDA_ARCH:-86}"
# Optional multi-SM fat binary: CUDA_ARCHS="80,86,89"
CUDA_ARCHS="${CUDA_ARCHS:-$CUDA_ARCH}"
# Normalize separators → spaces
CUDA_ARCHS_NORM="$(echo "$CUDA_ARCHS" | tr ',;/' ' ' | xargs)"
# Primary arch = first in list (or CUDA_ARCH)
PRIMARY_ARCH="$(echo "$CUDA_ARCHS_NORM" | awk '{print $1}')"
CUDA_ARCH="$PRIMARY_ARCH"
# Artifact label: sm86  or  sm80_86_89 (fat binary)
SM_LABEL="sm$(echo "$CUDA_ARCHS_NORM" | tr ' ' '_')"

JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-ci-${SM_LABEL}}"
PREFIX="${PREFIX:-$ROOT/install-ci-${SM_LABEL}}"
DIST_DIR="${DIST_DIR:-$ROOT/dist}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
APPLY_OVERRIDES="${APPLY_OVERRIDES:-1}"

# Build nvcc multi-gencode flags (SASS per SM). Newest SM also gets PTX for forward JIT.
NVCC_GENCODES=""
_LAST_A=""
for a in $CUDA_ARCHS_NORM; do
  case "$a" in
    ''|*[!0-9]*) echo "ERROR: invalid CUDA arch '$a' (digits only, e.g. 86)" >&2; exit 2 ;;
  esac
  NVCC_GENCODES+=" --generate-code=arch=compute_${a},code=sm_${a}"
  _LAST_A="$a"
done
# PTX for newest arch (helps slightly newer GPUs if driver can JIT)
if [[ -n "$_LAST_A" ]]; then
  NVCC_GENCODES+=" --generate-code=arch=compute_${_LAST_A},code=compute_${_LAST_A}"
fi
NVCC_GENCODES="$(echo "$NVCC_GENCODES" | xargs)"

echo "=== micmac-gpgpu CI build $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "ROOT=$ROOT"
echo "CUDA_ARCH(primary)=$CUDA_ARCH CUDA_ARCHS='$CUDA_ARCHS_NORM' SM_LABEL=$SM_LABEL"
echo "NVCC_GENCODES=$NVCC_GENCODES"
echo "JOBS=$JOBS"
echo "BUILD_DIR=$BUILD_DIR PREFIX=$PREFIX"
git rev-parse --short HEAD 2>/dev/null || true
git rev-parse HEAD 2>/dev/null || true
git log -1 --oneline 2>/dev/null || true

command -v nvcc >/dev/null || { echo "ERROR: nvcc not found (CUDA toolkit required)" >&2; exit 2; }
command -v cmake >/dev/null || { echo "ERROR: cmake missing" >&2; exit 2; }
nvcc --version | tail -1

if [[ "${SKIP_SAMPLES:-0}" != "1" ]]; then
  bash "$ROOT/ci/setup_cuda_samples.sh"
fi
CUDA_SAMPLE_DIR="${CUDA_SAMPLE_DIR:-$ROOT/ci/deps/cuda-samples}"
test -f "$CUDA_SAMPLE_DIR/common/inc/helper_cuda.h" || {
  echo "ERROR: helper_cuda.h missing under $CUDA_SAMPLE_DIR" >&2
  exit 2
}

if [[ "$APPLY_OVERRIDES" == "1" ]]; then
  bash "$ROOT/ci/apply_sm86_cmake_overrides.sh"
fi

# Preflight: pipeline / correl anchors present on this tip
test -f include/GpGpu/GpGpu_MultiThreadingCpu.h
grep -q 'HostIdleWaitForProgress\|RUNPOD_GPGPU_DIAG\|_gpgpu_sleep_us' \
  include/GpGpu/GpGpu_MultiThreadingCpu.h
test -f include/GpGpu/GpGpu_Pipeline.h
test -f include/GpGpu/GpGpu_SlotMap.h
test -f include/GpGpu/GpGpu_ImageLru.h

mkdir -p "$BUILD_DIR" "$PREFIX" "$DIST_DIR"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$(command -v g++ || command -v c++)}"

CMAKE_GEN=()
if command -v ninja >/dev/null 2>&1; then
  CMAKE_GEN=(-G Ninja)
fi

echo "=== cmake configure ==="
cmake -S "$ROOT" -B "$BUILD_DIR" "${CMAKE_GEN[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DWITH_QT5=0 \
  -DWITH_INTERFACE=OFF \
  -DCUDA_ENABLED=ON \
  -DWITH_OPENCL=OFF \
  -DWITH_OPEN_MP=OFF \
  -DBUILD_POISSON=OFF \
  -DBUILD_RNX2RTKP=OFF \
  -DWITH_ETALONPOLY=OFF \
  -DNO_X11=ON \
  -DWERROR=OFF \
  -DWITH_CCACHE=OFF \
  -DBUILD_ONLY_ELISE_MM3D=ON \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCUDA_SAMPLE_DIR="$CUDA_SAMPLE_DIR" \
  -DCUDA_CPP11THREAD_NOBOOSTTHREAD=ON \
  -DCUDA_FASTMATH=ON \
  -DMICMAC_CUDA_ARCH="$CUDA_ARCH" \
  -DCUDA_NVCC_FLAGS="$NVCC_GENCODES"

grep -q "CUDA_ENABLED:BOOL=ON" "$BUILD_DIR/CMakeCache.txt"

echo "=== build + install (jobs=$JOBS) ==="
cmake --build "$BUILD_DIR" --target install --parallel "$JOBS"

MM3D=""
for cand in "$PREFIX/bin/mm3d" "$ROOT/bin/mm3d" "$BUILD_DIR/bin/mm3d"; do
  if [[ -x "$cand" ]]; then
    MM3D="$cand"
    break
  fi
done
[[ -n "$MM3D" ]] || { echo "ERROR: mm3d not found after install" >&2; find "$PREFIX" -name mm3d 2>/dev/null | head; exit 3; }

# Ensure install prefix has bin/mm3d
mkdir -p "$PREFIX/bin"
if [[ "$(realpath "$MM3D")" != "$(realpath "$PREFIX/bin/mm3d" 2>/dev/null || echo none)" ]]; then
  cp -a "$MM3D" "$PREFIX/bin/mm3d"
fi
chmod +x "$PREFIX/bin/mm3d"

# Ship MicMac XML (required at runtime for Malt/MICMAC)
for d in XML_MicMac XML_GEN; do
  if [[ -d "$ROOT/include/$d" ]]; then
    mkdir -p "$PREFIX/include"
    rsync -a --delete "$ROOT/include/$d/" "$PREFIX/include/$d/" 2>/dev/null \
      || cp -a "$ROOT/include/$d" "$PREFIX/include/"
  fi
done

# Smoke: binary runs --help (no GPU needed)
set +e
"$PREFIX/bin/mm3d" 2>&1 | head -5
HELP_RC=${PIPESTATUS[0]}
set -e
# mm3d often returns non-zero without args; accept if it printed MicMac banner
if ! "$PREFIX/bin/mm3d" 2>&1 | head -20 | grep -qiE 'MicMac|MM3D|Ign'; then
  echo "WARN: mm3d banner check weak (rc=$HELP_RC)"
fi

# Optional: strings markers for pipeline / PR series
if command -v strings >/dev/null; then
  strings "$PREFIX/bin/mm3d" | grep -E 'MICMAC_GPU_PIPELINE|HostIdleWait|IMG_LRU|DeviceMemsetAsync' | head -10 || true
fi

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
STAMP="$(date -u +%Y%m%dT%H:%M:%SZ)"
NAME="micmac-gpgpu-${SM_LABEL}-${GIT_SHORT}"
TAR="$DIST_DIR/${NAME}.tar.gz"
MANIFEST="$DIST_DIR/${NAME}.manifest.json"

echo "=== package $TAR ==="
# Portable package root
PKG="$DIST_DIR/pkg-$NAME"
rm -rf "$PKG"
mkdir -p "$PKG"
rsync -a "$PREFIX/" "$PKG/" 2>/dev/null || cp -a "$PREFIX/." "$PKG/"

# Strip to shrink (keep mm3d runnable)
if command -v strip >/dev/null; then
  strip --strip-unneeded "$PKG/bin/mm3d" 2>/dev/null || true
fi

tar -C "$DIST_DIR" -czf "$TAR" "pkg-$NAME"
# Flatten: users extract and get bin/mm3d at top of tree named consistently
# Re-pack with stable top-level dir name
rm -rf "$DIST_DIR/${NAME}"
mv "$PKG" "$DIST_DIR/${NAME}"
tar -C "$DIST_DIR" -czf "$TAR" "$NAME"
rm -rf "$DIST_DIR/${NAME}"

SHA256="$(sha256sum "$TAR" | awk '{print $1}')"
SIZE="$(wc -c <"$TAR" | tr -d ' ')"

python3 - "$MANIFEST" <<PY
import json, os, sys
from datetime import datetime, timezone
path = sys.argv[1]
man = {
  "kind": "micmac-gpgpu-cuda-artifact",
  "generated_at": datetime.now(tz=timezone.utc).isoformat(),
  "git_sha": "$GIT_SHA",
  "git_short": "$GIT_SHORT",
  "git_branch": "$GIT_BRANCH",
  "cuda_arch_primary": "sm_$CUDA_ARCH",
  "cuda_archs": "$CUDA_ARCHS_NORM".split(),
  "sm_label": "$SM_LABEL",
  "nvcc_gencodes": "$NVCC_GENCODES",
  "cuda_toolkit": os.environ.get("CUDA_HOME", "$CUDA_HOME"),
  "artifact": os.path.basename("$TAR"),
  "sha256": "$SHA256",
  "bytes": int("$SIZE"),
  "mm3d_path_in_archive": "$NAME/bin/mm3d",
  "features_markers": [
    "MICMAC_GPU_PIPELINE",
    "MICMAC_GPU_LEGACY_PROCESS",
    "MICMAC_GPU_IMG_LRU",
    "MICMAC_GPU_POLL_US",
    "HostIdleWaitForProgress",
  ],
  "install_hint": "tar -xzf ARTIFACT && export PATH=\$PWD/$NAME/bin:\$PATH",
  "consumer_note": "Standalone CUDA mm3d. Fat binaries (multiple sm_XX) run on any listed arch; driver must support the GPU.",
}
open(path, "w").write(json.dumps(man, indent=2) + "\n")
print("wrote", path)
print(json.dumps(man, indent=2))
PY

echo "ARTIFACT=$TAR"
echo "MANIFEST=$MANIFEST"
echo "SHA256=$SHA256"
echo "BUILD_CUDA_SM86_OK"
