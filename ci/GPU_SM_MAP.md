# GPU SKU → CUDA SM map (for fat binary)

Fat default embeds: **sm_80 + sm_86 + sm_89 + sm_120**  
(CUDA toolkit ≥ **12.8** required for sm_120 / RTX 50-series.)

| Your hardware | Architecture | CUDA SM | Covered by fat? |
|---------------|--------------|---------|-----------------|
| **A100 PCIe** | Ampere | **sm_80** | yes |
| A4500 / GA10x class | Ampere | **sm_86** | yes (kept for existing pods) |
| **L40S** | Ada | **sm_89** | yes |
| **RTX 5000 Ada** | Ada | **sm_89** | yes |
| **RTX 5880 Ada** | Ada | **sm_89** | yes |
| **RTX PRO 5000** (Ada generation) | Ada | **sm_89** | yes |
| **RTX PRO 5000** (Blackwell workstation, if that SKU) | Blackwell | **sm_120** | yes |
| **RTX 5090** | Blackwell | **sm_120** | yes |

## Multi-GPU (2× cards)

**Not a compile flag.** One `mm3d` process typically uses **one** GPU (`CUDA_VISIBLE_DEVICES`).  
Two physical GPUs ⇒ run two processes or sequential jobs unless the app does multi-GPU (MicMac GPU path does not magically use 2× in one process via fat).

Fat only means: *same binary can execute on any listed SM*, not *one process drives N cards*.

## Artifact naming

- Fat: `micmac-gpgpu-sm80_86_89_120-<gitshort>.tar.gz`
- GHCR: `ghcr.io/<owner>/micmac-gpgpu-fat:<sha|branch|cuda-fat-latest>`
