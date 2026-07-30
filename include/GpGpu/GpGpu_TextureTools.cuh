#ifndef GPGPU_TEXTURETOOLS_CUH
#define GPGPU_TEXTURETOOLS_CUH

#include "GpGpu_Defines.h"
#include <cuda_runtime.h>
#include <cstring>

// ---------------------------------------------------------------------------
// Host-safe helpers (usable from .cpp and .cu). Create/destroy texture objects.
// ---------------------------------------------------------------------------

/// Create a texture object bound to a cudaArray (2D or layered).
inline cudaTextureObject_t GpGpuCreateTexObj(cudaArray * arr, bool linearFilter)
{
    if (!arr)
        return 0;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = linearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    return tex;
}

inline void GpGpuDestroyTexObj(cudaTextureObject_t & obj)
{
    if (obj)
    {
        cudaDestroyTextureObject(obj);
        obj = 0;
    }
}

// ---------------------------------------------------------------------------
// Device-only sampling (tex2D / tex2DLayered exist only under nvcc device compile).
// Must NOT be parsed by host g++ when this header is included from .cpp files.
// ---------------------------------------------------------------------------
#if defined(__CUDACC__)

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
	float w0(float a)
{
	return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
	float w1(float a)
{
	return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
	float w2(float a)
{
	return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
	float w3(float a)
{
	return (1.0f/6.0f)*(a*a*a);
}

__device__ float g0(float a)
{
	return w0(a) + w1(a);
}

__device__ float g1(float a)
{
	return w2(a) + w3(a);
}

__device__ float h0(float a)
{
	// note +0.5 offset to compensate for CUDA linear filtering convention
	return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

__device__ float h1(float a)
{
	return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

template<class T>
__device__
T cubicFilter(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

// Texture-object sampling (CUDA 5+; required for CUDA 12).
template<class T>
inline __device__ T tex2DLayeredPtObj(cudaTextureObject_t t, uint2 pt, short sample, short layer)
{
    return tex2DLayered<T>(t, (float)pt.x / sample + 0.5f, (float)pt.y / sample + 0.5f, layer);
}

template<class T>
inline __device__ T tex2DLayeredPtObj(cudaTextureObject_t t, float2 pt, short layer)
{
    return tex2DLayered<T>(t, pt.x + 0.5f, pt.y + 0.5f, layer);
}

#endif /* __CUDACC__ */

#endif //GPGPU_TEXTURETOOLS_CUH
