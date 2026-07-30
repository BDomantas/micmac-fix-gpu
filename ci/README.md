# CI: prebuilt CUDA MicMac (`mm3d`) for sm_86

Standalone packaging for **this fork only** (`BDomantas/micmac-fix-gpu`).  
Does **not** depend on `maps` or `maps-next` repos. Consumers (including maps-next GPU workers) can **download a binary** instead of recompiling on RunPod.

## What gets built

| Output | Description |
|--------|-------------|
| `micmac-gpgpu-sm86-<gitsha>.tar.gz` | Install tree with `bin/mm3d` + MicMac XML |
| `*.manifest.json` | sha256, git SHA, arch, install hint |
| GHCR image (optional) | `ghcr.io/<owner>/micmac-gpgpu-sm86:<sha>` thin Ubuntu + binary |

**CUDA arch default: sm_86** (Ampere A4500 / similar). Override with `CUDA_ARCH=89` for Ada.

Build is **compile-only** on GitHub Actions (no GPU on the runner). Runtime still needs a host with NVIDIA driver + matching CUDA userland.

## Workflow

File: [`.github/workflows/cuda-sm86-binaries.yml`](../.github/workflows/cuda-sm86-binaries.yml)

Triggers:

- Push to `gpu-instream`, `gpu-remaining-features-prbe-h`, `main`/`master`
- Tags `v*`
- Manual `workflow_dispatch` (can publish a draft release)

## Local / pod build (same script CI uses)

Inside a CUDA **devel** environment (e.g. `nvidia/cuda:11.8.0-devel-ubuntu22.04` or the aerial RunPod image):

```bash
# deps: cmake ninja g++ git python3 rsync libtiff-dev …
export CUDA_HOME=/usr/local/cuda   # or /opt/aerial-runtime-…/cuda-11.8
export CUDA_ARCH=86
bash ci/build_cuda_sm86.sh
# → dist/micmac-gpgpu-sm86-<sha>.tar.gz
```

## Pull without compiling

### Actions artifact

1. Open the successful **CUDA sm86 binaries** run on GitHub.
2. Download artifact `micmac-gpgpu-sm86-<fullsha>`.
3. Extract and put `bin` on `PATH`:

```bash
tar -xzf micmac-gpgpu-sm86-*.tar.gz
export PATH="$PWD/micmac-gpgpu-sm86-*/bin:$PATH"
mm3d
```

### Release asset (tags)

```bash
# example once a v* tag is pushed
curl -L -o mm.tgz \
  "https://github.com/BDomantas/micmac-fix-gpu/releases/download/vX.Y.Z/micmac-gpgpu-sm86-XXXX.tar.gz"
tar -xzf mm.tgz
```

### GHCR (for Docker worker images)

```bash
# after CI push succeeds
docker pull ghcr.io/bdomantas/micmac-gpgpu-sm86:<12-char-sha>
# or branch floating tag, e.g. gpu-remaining-features-prbe-h
docker pull ghcr.io/bdomantas/micmac-gpgpu-sm86:gpu-remaining-features-prbe-h
```

**maps-next integration (out of band):** GPU worker Dockerfile can:

```dockerfile
COPY --from=ghcr.io/bdomantas/micmac-gpgpu-sm86:<sha> /opt/micmac-gpgpu-sm86 /opt/micmac-gpgpu-sm86
ENV PATH="/opt/micmac-gpgpu-sm86/bin:${PATH}"
```

That stays in maps-next; this repo only publishes the binary/image.

## Feature surface on this tip

Pipeline / correl work from `gpu-remaining-features-prbe-h` (and ancestors), including:

- `MICMAC_GPU_PIPELINE` (default on) / `MICMAC_GPU_LEGACY_PROCESS`
- Stream-ordered correl path, events, dual slots, host CV wait
- `MICMAC_GPU_IMG_LRU` (default off)

## Notes / limits

- First CI run is long (~1–2h+) and disk-hungry; workflow frees space and uses Ninja + `BUILD_ONLY_ELISE_MM3D`.
- glibc must match consumer (Ubuntu 22.04 build → 22.04+ runtime preferred).
- Do not expect byte-identical DEMs across pipeline/legacy without documented float tolerance; see maps `docs/gpu-instream-results/`.
