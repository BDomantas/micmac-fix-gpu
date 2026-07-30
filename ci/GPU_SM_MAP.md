# GPU SKU → CUDA SM map

## Fat + per-SM packages (CUDA **11.8**)

Default CI builds **each** SM separately **and** a **fat** binary with all of:

| SM | Hardware |
|----|----------|
| **80** | **A100 PCIe** |
| **86** | A4500 / Ampere pro (existing pods) |
| **89** | **L40S**, **RTX 5000 Ada**, **RTX 5880 Ada**, **RTX PRO 5000 Ada** |

Artifact names: `micmac-gpgpu-sm80-…`, `…-sm86-…`, `…-sm89-…`, fat `micmac-gpgpu-sm80_86_89-…`.

Release tag: **`cuda-latest`**  
GHCR: `micmac-gpgpu-sm86:latest`, `micmac-gpgpu-fat:cuda-latest`, etc.

## Multi-GPU (2× cards)

Not a compile flag. One `mm3d` ≈ one GPU. For 2× RTX 5000 Ada / 5880 / 5090 use two processes / `CUDA_VISIBLE_DEVICES=0|1`.

## Not in fat yet: RTX 5090 / Blackwell (sm_120)

| Issue | Detail |
|-------|--------|
| Toolkit | sm_120 needs **CUDA ≥ 12.8** |
| MicMac code | Still uses **CUDA texture references** (`texture<>`, `cudaBindTextureToArray`) **removed in CUDA 12** |
| Hot path | Already has texture **objects** for correl, but host headers still compile the legacy path |

Until texture-ref is stubbed/ported for CUDA 12, **5090 cannot be in the fat binary**.  
Workaround: run 5090 jobs on a machine with a driver that… no — 5090 cannot execute sm_89 SASS. Need a real sm_120 build later.

## Why fat failed on CUDA 12.8

```
textureReference / cudaBindTextureToArray not declared
```

= CUDA 12 removed legacy texture API. Not a random ninja glitch.
