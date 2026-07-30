#!/usr/bin/env bash
# Fetch CUDA Samples helper headers for MicMac GpGpu (helper_cuda.h, etc.).
#
# MicMac expects:  ${CUDA_SAMPLE_DIR}/common/inc/helper_cuda.h
# NVIDIA layout (v11.8+): Common/helper_cuda.h  (no "inc/" subdir)
#
# Runtime GPU not required — compile-time includes only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CUDA_SAMPLES_DIR:-$ROOT/ci/deps/cuda-samples}"
TAG="${CUDA_SAMPLES_TAG:-v11.8}"

if [[ -f "$DEST/common/inc/helper_cuda.h" ]]; then
  echo "cuda-samples already present: $DEST/common/inc/helper_cuda.h"
  echo "CUDA_SAMPLE_DIR=$DEST"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "=== fetch cuda-samples $TAG (Common headers) ==="
git clone --depth 1 --filter=blob:none --sparse \
  --branch "$TAG" \
  https://github.com/NVIDIA/cuda-samples.git "$TMP/cuda-samples"

# v11.8 layout: Common/*.h  (capital C, flat — not Common/inc)
git -C "$TMP/cuda-samples" sparse-checkout set Common 2>/dev/null \
  || git -C "$TMP/cuda-samples" sparse-checkout set common 2>/dev/null \
  || true

SRC_DIR=""
if [[ -f "$TMP/cuda-samples/Common/helper_cuda.h" ]]; then
  SRC_DIR="$TMP/cuda-samples/Common"
elif [[ -d "$TMP/cuda-samples/Common/inc" && -f "$TMP/cuda-samples/Common/inc/helper_cuda.h" ]]; then
  SRC_DIR="$TMP/cuda-samples/Common/inc"
elif [[ -f "$TMP/cuda-samples/common/inc/helper_cuda.h" ]]; then
  SRC_DIR="$TMP/cuda-samples/common/inc"
elif [[ -f "$TMP/cuda-samples/common/helper_cuda.h" ]]; then
  SRC_DIR="$TMP/cuda-samples/common"
else
  echo "ERROR: helper_cuda.h not found under cuda-samples $TAG" >&2
  find "$TMP/cuda-samples" -name 'helper_cuda.h' 2>/dev/null | head -20 || true
  ls -la "$TMP/cuda-samples" || true
  ls -la "$TMP/cuda-samples/Common" 2>/dev/null || true
  exit 1
fi

# Normalize to MicMac's expected layout: common/inc/*.h
mkdir -p "$DEST/common/inc"
cp -a "$SRC_DIR"/*.h "$DEST/common/inc/" 2>/dev/null || true
# Some trees nest further
if [[ ! -f "$DEST/common/inc/helper_cuda.h" && -f "$SRC_DIR/helper_cuda.h" ]]; then
  cp -a "$SRC_DIR/helper_cuda.h" "$DEST/common/inc/"
fi

test -f "$DEST/common/inc/helper_cuda.h"
echo "CUDA_SAMPLE_DIR=$DEST"
echo "copied headers from $SRC_DIR -> $DEST/common/inc"
ls "$DEST/common/inc" | head -20
echo "setup_cuda_samples_ok"
