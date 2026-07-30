# GPU SKU → CUDA SM map

## Fat + per-SM packages (CUDA **12.8**)

Default CI builds **each** SM separately **and** a **fat** binary with all of:

| SM | Hardware |
|----|----------|
| **80** | **A100 PCIe** |
| **86** | A4500 / Ampere pro (existing pods) |
| **89** | **L40S**, **RTX 5000 Ada**, **RTX 5880 Ada**, **RTX PRO 5000 Ada** |
| **120** | **RTX 5090 / Blackwell** |

Artifact names: `micmac-gpgpu-sm80-…`, `…-sm86-…`, `…-sm89-…`, `…-sm120-…`, fat `micmac-gpgpu-sm80_86_89_120-…`.

Release tag: **`cuda-latest`**  
GHCR: `micmac-gpgpu-sm86:latest`, `micmac-gpgpu-fat:cuda-latest`, etc.

## Multi-GPU (2× cards)

Not a compile flag. One `mm3d` ≈ one GPU. For 2× RTX 5000 Ada / 5880 / 5090 use two processes / `CUDA_VISIBLE_DEVICES=0|1`.

## Texture API (CUDA 12)

MicMac GpGpu now uses **`cudaTextureObject_t` only** (no `texture<>` / `textureReference` / `cudaBindTextureToArray`).

Hot path (Phase C) already sampled via texobjs; host CMS / DecoratorImage / SData2Correl no longer compile the legacy bind path.

## History

CUDA 12.8 fat previously failed with:

```
textureReference / cudaBindTextureToArray not declared
```

Fixed by the CUDA 12 texture-object migration on this branch.
