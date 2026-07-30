# CI: prebuilt CUDA MicMac (`mm3d`) for sm_86

Standalone packaging for **this fork only** (`BDomantas/micmac-fix-gpu`).  
Does **not** depend on `maps` or `maps-next` repos. Consumers (including maps-next GPU workers) can **download a binary** instead of recompiling on RunPod.

## What gets built

| Output | Description |
|--------|-------------|
| `micmac-gpgpu-sm86-<gitsha>.tar.gz` | Per-SM package (`bin/mm3d` + MicMac XML) |
| `micmac-gpgpu-sm80-…`, `sm89-…` | Same for other SMs (CI matrix) |
| optional fat | `CUDA_ARCHS="80 86 89"` → one binary with all gencodes |
| `*.manifest.json` | sha256, git SHA, arch list, install hint |
| GHCR | `ghcr.io/<owner>/micmac-gpgpu-sm86:<sha>` (and sm80/sm89) |

### Architectures (per-SM **and** fat)

See [GPU_SM_MAP.md](./GPU_SM_MAP.md).

CI default (**CUDA 11.8** — required by MicMac legacy texture API):

| Package | SMs | Use for |
|---------|-----|---------|
| `micmac-gpgpu-sm80-…` | 80 | A100 |
| `micmac-gpgpu-sm86-…` | 86 | A4500 / Ampere pro |
| `micmac-gpgpu-sm89-…` | 89 | L40S, RTX 5000/5880 Ada, PRO 5000 Ada |
| `micmac-gpgpu-sm80_86_89-…` (**fat**) | 80+86+89 | one download for all of the above |

**Always publishes** (on push): Actions artifacts + Release **`cuda-latest`** + GHCR per-SM and `micmac-gpgpu-fat`.

**RTX 5090 / sm_120:** not yet — CUDA 12 removes `textureReference`; needs a code port (see GPU_SM_MAP.md).

**2× cards:** `CUDA_VISIBLE_DEVICES`, not fat.

Build is **compile-only** on GitHub Actions. Runtime needs NVIDIA driver + CUDA userland on the host.

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
