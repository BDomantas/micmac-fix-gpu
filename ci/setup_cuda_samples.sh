#!/usr/bin/env bash
# Fetch CUDA Samples "common/inc" headers (helper_cuda.h etc.) for MicMac GpGpu.
# Runtime GPU not required — compile-time includes only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CUDA_SAMPLES_DIR:-$ROOT/ci/deps/cuda-samples}"
TAG="${CUDA_SAMPLES_TAG:-v11.8}"

if [[ -f "$DEST/common/inc/helper_cuda.h" ]]; then
  echo "cuda-samples already present: $DEST"
  echo "CUDA_SAMPLE_DIR=$DEST"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "=== fetch cuda-samples $TAG (common/inc only) ==="
# Sparse checkout keeps CI disk light.
git clone --depth 1 --filter=blob:none --sparse \
  --branch "$TAG" \
  https://github.com/NVIDIA/cuda-samples.git "$TMP/cuda-samples"
git -C "$TMP/cuda-samples" sparse-checkout set Common/inc || \
  git -C "$TMP/cuda-samples" sparse-checkout set common/inc || true

# Layout variants across sample versions
if [[ -d "$TMP/cuda-samples/Common/inc" ]]; then
  mkdir -p "$DEST/common"
  cp -a "$TMP/cuda-samples/Common/inc" "$DEST/common/"
elif [[ -d "$TMP/cuda-samples/common/inc" ]]; then
  mkdir -p "$DEST/common"
  cp -a "$TMP/cuda-samples/common/inc" "$DEST/common/"
else
  echo "ERROR: could not find common/inc in cuda-samples" >&2
  find "$TMP/cuda-samples" -maxdepth 3 -type d | head -40
  exit 1
fi

test -f "$DEST/common/inc/helper_cuda.h"
echo "CUDA_SAMPLE_DIR=$DEST"
echo "setup_cuda_samples_ok"
