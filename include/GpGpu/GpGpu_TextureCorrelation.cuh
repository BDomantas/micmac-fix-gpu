#ifndef CUDAREFTEXTURE
#define CUDAREFTEXTURE

#include "GpGpu_Defines.h"
#include "GpGpu_TextureTools.cuh"

// CUDA 12 capable path: no texture<> / textureReference (removed in CUDA 12).
// Sampling is exclusively via cudaTextureObject_t (Phase C / CUDA12 migration).

#define INTERPOLA LINEARINTER

// Device samplers — host g++ must not parse tex2DLayered (nvcc-only).
#if defined(__CUDACC__)

/// Sample images via texture object (multi-stream safe).
inline __device__ float GetImageValueObj(cudaTextureObject_t texImg, float2 ptProj, uint mZ)
{
#if	INTERPOLA == NEAREST
	return tex2DLayered<float>(texImg, ptProj.x, ptProj.y, (int)mZ);
#else
	return tex2DLayeredPtObj<float>(texImg, ptProj, (short)mZ);
#endif
}

/// Projection sample from per-slot texture object.
inline __device__ float2 GetProjectionObj(cudaTextureObject_t texProj, uint2 ptTer, uint sampProj, uint BZ)
{
#if (SAMPLETERR == 1)
    return tex2DLayeredPtObj<float2>(texProj, ptTer, 1, (short)BZ);
#else
    return tex2DLayeredPtObj<float2>(texProj, ptTer, (short)sampProj, (short)BZ);
#endif
}

#endif /* __CUDACC__ */

#endif /*CUDAREFTEXTURE*/
