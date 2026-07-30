#include "GpGpu/GpGpu_ParamCorrelation.cuh"
#include "GpGpu/GpGpu_TextureTools.cuh"
#include "GpGpu/GpGpu_TextureCorrelation.cuh"
#include "GpGpu/SData2Correl.h"

#include <stdio.h>
#include <stdlib.h>
#include "GpGpu/GpGpu_Diag.h"

/// \file       GpGpuCudaCorrelation.cu
/// \brief      Kernel
/// \author     GC
/// \version    0.2
/// \date       mars 2013

static __constant__ invParamCorrel  invPc;

/// Phase C invPc model: host→__constant__ upload.
/// Pipeline path uses default stream order before enqueueing correl kernels on
/// work streams; full stream-ordered async would require cudaMemcpyToSymbolAsync
/// + event wait on each work stream (future refinement). Params rarely change mid-box.
extern "C" void CopyParamInvTodevice( pCorGpu param )
{
  checkCudaErrors(cudaMemcpyToSymbol(invPc, &param.invPC, sizeof(invParamCorrel)));
}

// Debug/preview path: CUDA12 texture-object sampling (was textureReference).
__global__ void projectionImageObj( HDParamCorrel HdPc, float* projImages, uint2* pRect,
                                    cudaTextureObject_t texImages, cudaTextureObject_t texProj)
{
    const uint2 ptHTer = make_uint2(blockIdx) *  blockDim.x + make_uint2(threadIdx);

    if (oSE(ptHTer,HdPc.dimHaloTer)) return;

    const ushort IdLayer = blockDim.z * blockIdx.z + threadIdx.z;

    const float2 ptProj  = GetProjectionObj(texProj, ptHTer, invPc.sampProj, IdLayer);

    const uint2  zoneImage = pRect[IdLayer];

    float* localImages = projImages + IdLayer * size(HdPc.dimHaloTer);

    localImages[to1D(ptHTer,HdPc.dimHaloTer)] =
        (oI(ptProj,0) || ptProj.x >= (float)zoneImage.x || ptProj.y >= (float)zoneImage.y)
            ? 1.f
            : GetImageValueObj(texImages, ptProj, threadIdx.z) / 2048.f;
}

extern "C" void	 LaunchKernelprojectionImage(pCorGpu &param, CuDeviceData3D<float>  &DeviImagesProj, Rect* pRect)
{
    // Legacy signature kept for linkage; body unused in production pipeline.
    (void)param; (void)DeviImagesProj; (void)pRect;
}

/// \brief Kernel fonction GpGpu Cuda — Phase C texture-object sampling (stream-safe).
/// Calcul les vignettes de correlation pour toutes les images
///
__global__ void correlationKernel(
    uint *dev_NbImgOk, ushort2 *ClassEqui, float* cachVig, uint2* pRect, uint2 nbActThrd,
    HDParamCorrel HdPc,
    cudaTextureObject_t texImages,
    cudaTextureObject_t texMaskImages,
    cudaTextureObject_t texMaskGlobal,
    cudaTextureObject_t texProj)
{

  extern __shared__ float cacheImg[];

  // Coordonn�es du terrain global avec bordure // __umul24!!!! A voir
  const uint2 ptHaloTer = make_uint2(blockIdx) * nbActThrd + make_uint2(threadIdx);

  // Si le point est hors du terrain, nous sortons du kernel
  if (oSE(ptHaloTer,HdPc.dimHaloTer)  ) return;

  // Obtenir la projection du point dans l'image (per-slot tex object)
  const float2 ptProj   = GetProjectionObj(texProj, ptHaloTer, invPc.sampProj, blockIdx.z);

  // Phase : obtention de la valeur dans l'image
  const uint	pitZ      = blockIdx.z / invPc.nbImages;
  const uint	pitCach   = pitZ * invPc.nbImages;
  const ushort	idImg     = blockIdx.z - pitCach; // ID image courante

  // DEBUT 2014 || taille de l'image
  const uint2  zoneImage = pRect[idImg];

  // Si la projection est en dehors de l'image on sort
  if (oI(ptProj,0) || ptProj.x >= (float)zoneImage.x || ptProj.y >= (float)zoneImage.y)
      return;

  cacheImg[sgpu::__mult<BLOCKDIM>(threadIdx.y) + threadIdx.x] = GetImageValueObj(texImages, ptProj, idImg);

  __syncthreads();

  // Point terrain local
  const int2 ptTer = make_int2(ptHaloTer) - make_int2(invPc.rayVig);

  // Nous traitons uniquement les points du terrain du bloque ou Si le processus est hors du terrain global, nous sortons du kernel

  // Sortir si threard inactif et si en dehors du terrain (� simplifier)
  if (oSE(threadIdx, nbActThrd + invPc.rayVig) || oI(threadIdx , invPc.rayVig) || oSE( ptTer, HdPc.dimTer) || oI(ptTer,0))
	return;

  // Faux mais fait le job! en limite d'image
  if ( oI( ptProj - invPc.rayVig.x-1, 0) || (ptProj.x + invPc.rayVig.x+1>= (float)zoneImage.x) || (ptProj.y + invPc.rayVig.x+1>= (float)zoneImage.y))
	  return;

  // Point terrain global
  int2 coorTer = ptTer + HdPc.rTer.pt0;

  if (tex2D<pixel>(texMaskGlobal, (float)coorTer.x, (float)coorTer.y) == 0) return;

  if (tex2DLayered<pixel>(texMaskImages, (float)coorTer.x, (float)coorTer.y, (int)idImg) == 0) return;

  const short2 c0	= make_short2(threadIdx) - invPc.rayVig;
  const short2 c1	= make_short2(threadIdx) + invPc.rayVig;

  // Intialisation des valeurs de calcul
  float aSV = 0.0f, aSVV = 0.0f;
  short2 pt;

  #pragma unroll // ATTENTION PRAGMA FAIT AUGMENTER LA quantit� MEMOIRE des registres!!!
  for (pt.y = c0.y ; pt.y <= c1.y; pt.y++)
  {

	  const float* cImg    = cacheImg +  sgpu::__mult<BLOCKDIM>(pt.y);
      #pragma unroll
      for (pt.x = c0.x ; pt.x <= c1.x; pt.x++)
      {
          const float val = cImg[pt.x];     // Valeur de l'image
          aSV  += val;                      // Somme des valeurs de l'image cte
          aSVV += (val*val);                // Somme des carr�s des vals image cte
      }
  }

  aSV   = fdividef(aSV,(float)invPc.sizeVig );

  aSVV  = fdividef(aSVV,(float)invPc.sizeVig );

  aSVV -=	(aSV * aSV);

  if ( aSVV <= invPc.mAhEpsilon) return;

  aSVV =	rsqrtf(aSVV); // racine carre inverse

  const uint pitchCachY = ptTer.y * invPc.dimVig.y ;

  const ushort iCla = ClassEqui[idImg].x;

  const ushort pCla = ClassEqui[iCla].y;

  const int  idN    = (pitZ * invPc.nbClass + iCla ) * HdPc.sizeTer + to1D(ptTer,HdPc.dimTer);

  const uint iCa    = atomicAdd( &dev_NbImgOk[idN], 1U) + pitCach + pCla;

  float* cache      = cachVig + (iCa * HdPc.sizeCach) + ptTer.x * invPc.dimVig.x - c0.x + (pitchCachY - c0.y)* HdPc.dimCach.x;

#pragma unroll
  for ( pt.y = c0.y ; pt.y <= c1.y; pt.y++)
    {
	  const float* cImg = cacheImg +  sgpu::__mult<BLOCKDIM>(pt.y);
      float* cVig = cache    + pt.y * HdPc.dimCach.x ;
#pragma unroll
      for ( pt.x = c0.x ; pt.x <= c1.x; pt.x++)

          cVig[ pt.x ] = (cImg[pt.x] -aSV)*aSVV;

    }
}
// Debug-only path retained for linkage; production uses correlationKernel texobjs.
extern "C" void	 LaunchKernelGetValueImages(pCorGpu &param,SData2Correl &data2cor)
{
    (void)param;
    (void)data2cor;
}
/// \brief Fonction qui lance les kernels de correlation (texture objects, slot s).
extern "C" void	 LaunchKernelCorrelation(const int s,cudaStream_t stream,pCorGpu &param,SData2Correl &data2cor)
{

    dim3	threads( BLOCKDIM, BLOCKDIM, 1);
    uint2	thd2D		= make_uint2(threads);
    uint2	nbActThrd	= thd2D - 2 * param.invPC.rayVig;
    uint2	block2D		= iDivUp(param.HdPc.dimHaloTer,nbActThrd);
    dim3	blocks(block2D.x , block2D.y, param.invPC.nbImages * param.ZCInter);

    const int slot = (s >= 0 && s < NSTREAM) ? s : 0;
    data2cor.EnsureTextureObjects((uint)slot);

    cudaTextureObject_t texImg  = data2cor.TexObjImages();
    cudaTextureObject_t texMImg = data2cor.TexObjMaskImages();
    cudaTextureObject_t texMGlb = data2cor.TexObjMaskGlobal();
    cudaTextureObject_t texProj = data2cor.TexObjProj((uint)slot);

    // Fallback: if objects missing, abort rather than silent global-tex race.
    if (!texImg || !texProj)
    {
        GPGPU_DIAG_ERR(
            "[GPGPU][PIPELINE] ERROR LaunchKernelCorrelation: missing texture objects "
            "img=%llu proj=%llu slot=%d\n",
            (unsigned long long)texImg, (unsigned long long)texProj, slot);
        abort();
    }

    correlationKernel<<<blocks, threads, BLOCKDIM * BLOCKDIM * sizeof(float), stream>>>(
        data2cor.DeviVolumeNOK(slot),
        data2cor.DeviClassEqui(),
        data2cor.DeviVolumeCache(slot),
        data2cor.DeviRect(),
        nbActThrd,
        param.HdPc,
        texImg, texMImg, texMGlb, texProj);
    getLastCudaError("Basic Correlation kernel failed (texture objects)");
}



/// \brief Multi-correlation via Huygens (atomics on vignette reduction).
///
/// Barrier contract (CUDA): every live thread in the block must execute the same
/// sequence of __syncthreads(). The stock kernel deadlocked because:
///   1) edge threads early-returned on oSE(ptCach) before the first barrier
///   2) threads with nImgOK <= 1 skipped the per-class barrier inside the loop
/// This rewrite uses predicates (inBounds / doClass / mainThread) so work is
/// masked but barriers stay uniform. Huygens math is unchanged (max-perf path).

template<ushort SIZE3VIGN > __global__ void multiCorrelationKernel(ushort2* classEqui,float *dTCost, float* cacheVign, uint* dev_NbImgOk, /*uint2 nbActThr,*/HDParamCorrel HdPc)
{

  __shared__ float aSV [ SIZE3VIGN   ][ SIZE3VIGN ];          // Somme des valeurs
  __shared__ float aSVV[ SIZE3VIGN   ][ SIZE3VIGN ];         // Somme des carr�s des valeurs
  __shared__ float resu[ SIZE3VIGN>>1 ][ SIZE3VIGN>>1 ];		// resultat

  __shared__ float cResu[ SIZE3VIGN>>1][ SIZE3VIGN>>1 ];		// resultat
  __shared__ uint nbIm[ SIZE3VIGN>>1][ SIZE3VIGN>>1 ];		// nombre d'images correcte

  // coordonn�es des threads // TODO uint2 to ushort2
  const uint2 t  = make_uint2(threadIdx);

  // Coordonn�es 2D du cache vignette
  const uint2 ptCach = make_uint2(blockIdx) * SIZE3VIGN + t;

  // Edge threads stay alive for barriers; they just skip memory work.
  const bool inBounds = !oSE(ptCach, HdPc.dimCach);

  // thTer is always valid from threadIdx (block is SIZE3VIGN x SIZE3VIGN).
  const uint2 thTer = t / invPc.dimVig;
  const bool mainThread = aEq(t - thTer * invPc.dimVig, 0);

  uint2 ptTer = make_uint2(0, 0);
  uint  ter   = 0;
  uint  iTer  = 0;
  if (inBounds)
  {
      ptTer = ptCach / invPc.dimVig;
      ter   = to1D(ptTer, HdPc.dimTer);
      iTer  = blockIdx.z * HdPc.sizeTer + ter;
  }

  // Multiple threads share thTer; redundant zero is intentional (same as stock).
  resu[thTer.y][thTer.x] = 0.0f;
  nbIm[thTer.y][thTer.x] = 0;

  __syncthreads();

  for (ushort iCla = 0; iCla < invPc.nbClass; ++iCla)
  {
      ushort nImgOK = 0;
      bool   doClass = false;

      if (inBounds)
      {
          const uint icTer = (blockIdx.z * invPc.nbClass + iCla) * HdPc.sizeTer + ter;
          nImgOK  = (ushort)dev_NbImgOk[icTer];
          doClass = (nImgOK > 1);
      }

      if (doClass)
      {
          aSV [t.y][t.x] = 0.0f;
          aSVV[t.y][t.x] = 0.0f;
          cResu[thTer.y][thTer.x] = 0.0f;

          const uint pitCla        = ((uint)classEqui[iCla].y) * HdPc.sizeCach;
          const uint pitLayerCache = blockIdx.z * HdPc.sizeCachAll + pitCla + to1D(ptCach, HdPc.dimCach);
          const float* caVi        = cacheVign + pitLayerCache;
          const uint limOK         = nImgOK * HdPc.sizeCach;

#pragma unroll
          for (uint i = 0; i < limOK; i += HdPc.sizeCach)
          {
              const float val = caVi[i];
              aSV[t.y][t.x]  += val;
              aSVV[t.y][t.x] += val * val;
          }

          atomicAdd(&(cResu[thTer.y][thTer.x]),
                    (aSVV[t.y][t.x] - fdividef(aSV[t.y][t.x] * aSV[t.y][t.x], (float)nImgOK)));
      }

      // Uniform barrier: every class iteration, every thread (doClass or not).
      __syncthreads();

      if (doClass && mainThread)
      {
          resu[thTer.y][thTer.x] +=
              (float)(1.0f - max(-1.0, min(1.0f, 1.0f - fdividef(cResu[thTer.y][thTer.x],
                                                                   ((float)(nImgOK - 1)) * (invPc.sizeVig))))) *
              nImgOK;
          nbIm[thTer.y][thTer.x] += nImgOK;
      }

      // Protect cResu zeroing on the next class against a racing mainThread read.
      __syncthreads();
  }

  if (inBounds && mainThread && (nbIm[thTer.y][thTer.x] != 0))
  {
      dTCost[iTer] = fdividef(resu[thTer.y][thTer.x], (float)nbIm[thTer.y][thTer.x]);
  }

}

template<ushort SIZE3VIGN > void LaunchKernelMultiCor(cudaStream_t stream, pCorGpu &param, SData2Correl &dataCorrel, const int s)
{
    //-------------	calcul de dimension du kernel de multi-correlation NON ATOMIC ------------
    // Phase B: volumes must match stream/slot s (never hard-code 0).
    const int slot = (s >= 0 && s < NSTREAM) ? s : 0;
    dim3	threads(SIZE3VIGN, SIZE3VIGN, 1);
    uint2	block2D	= iDivUp(param.HdPc.dimCach,SIZE3VIGN);
    dim3	blocks(block2D.x,block2D.y,param.ZCInter);

    multiCorrelationKernel<SIZE3VIGN><<<blocks, threads, 0, stream>>>(
        dataCorrel.DeviClassEqui(),
        dataCorrel.DeviVolumeCost(slot),
        dataCorrel.DeviVolumeCache(slot),
        dataCorrel.DeviVolumeNOK(slot),
        param.HdPc);
    getLastCudaError("Multi-Correlation NON ATOMIC kernel failed");
}

/// \brief Fonction qui lance les kernels de multi-Correlation n'utilisant pas des fonctions atomiques
/// \param s stream/slot index — multi-correl volumes must match correl volumes for this slot.
extern "C" void LaunchKernelMultiCorrelation(cudaStream_t stream, pCorGpu &param, SData2Correl &dataCorrel, const int s)
{
    const ushort ray = param.invPC.rayVig.x;
    if (ray == 1 || ray == 2)
        LaunchKernelMultiCor<SBLOCKDIM>(stream, param, dataCorrel, s);
    else if (ray == 3)
        LaunchKernelMultiCor<7*2>(stream, param, dataCorrel, s);
    else
    {
        // Stock silently no-oped unsupported rayVig — that looks like a hang downstream.
        GPGPU_DIAG_ERR(
                "[GPGPU][RUNPOD_GPGPU_DIAG] ERROR LaunchKernelMultiCorrelation: unsupported rayVig=%u "
                "(supported 1,2,3). Aborting instead of silent no-op.\n",
                (unsigned)ray);
        abort();
    }
}
