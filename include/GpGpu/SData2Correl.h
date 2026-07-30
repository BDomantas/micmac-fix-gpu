#ifndef _SDATA2CORREL_H
#define _SDATA2CORREL_H
/** @addtogroup GpGpuDoc */
/*@{*/

#include "GpGpu/GpGpu_ParamCorrelation.cuh"
#include <cuda_runtime.h>

/// \cond
#define SYNC    false
#define ASYNC   true
/// \endcond


///
/// \brief The cellules struct
/// Structure de cellules 3D
struct cellules
{
    ///
    /// \brief Zone
    /// La zone 2d
    ///
    Rect Zone;
    ///
    /// \brief Dz
    /// delta Z de la zone
    ///
	ushort Dz;

    cellules():
        Zone(MAXIRECT),
        Dz(INTERZ)
    {}
};

///
/// \brief The SData2Correl struct
///
struct SData2Correl
{

public:
    SData2Correl();

    ~SData2Correl();

	///
	/// \brief SetImages Initialise les images sur GPU
	///
    void    SetImages( float* dataImage, uint2 dimImage, int nbLayer );

	///
	/// \brief SetGlobalMask Initialise les masques sur GPU
	///
    void    SetGlobalMask( pixel* dataMask, uint2 dimMask );

	///
	/// \brief MemsetHostVolumeProj Initialise la memoire des projections par une valeur iDef
	///
    void    MemsetHostVolumeProj(int iDef, uint id = 0xFFFFFFFFu);

	///
	/// \brief HostVolumeCost
	///
    float*  HostVolumeCost(uint id);

	///
	/// \brief HostVolumeProj slot 0 legacy
	///
    float2* HostVolumeProj();

	/// Phase B: per-slot host projection ring (id < SIZERING / NSTREAM).
	float2* HostVolumeProj(uint id);

	uint2*	HostRect();

    uint*   DeviVolumeNOK(uint s);

    float*  DeviVolumeCache(uint s);

    float*  DeviVolumeCost(uint s);

	uint2*	DeviRect();

    void    copyHostToDevice(pCorGpu param, uint s = 0);

	/// Phase B: async H2D of projection slot s on cuda stream.
	void    copyHostToDeviceASync(pCorGpu param, uint s, cudaStream_t stream);

    void    CopyDevicetoHost(uint idBuf, uint s = 0);

	/// Phase B: async D2H of cost volume for stream slot s.
	void    CopyDevicetoHostASync(uint idBuf, uint s, cudaStream_t stream);

	/// Legacy name: no-op under texture objects (destroy handled on rebuild).
    void    UnBindTextureProj(uint s = 0);

	void    ReallocDeviceDataSlot(uint s, pCorGpu &param);
    /// PR-B: stream-ordered clear (cost + NIOk) — no default-stream hot path.
    void DeviceMemsetAsync(pCorGpu &param, uint s, cudaStream_t stream);

	/// Phase C / CUDA12: create cuda texture objects for images/masks/projections.
	void    EnsureTextureObjects(uint s = 0);
	void    DestroyTextureObjects();
	cudaTextureObject_t TexObjImages() const { return _texObjImages; }
	cudaTextureObject_t TexObjMaskImages() const { return _texObjMaskImages; }
	cudaTextureObject_t TexObjMaskGlobal() const { return _texObjMaskGlobal; }
	cudaTextureObject_t TexObjProj(uint s) const {
		return (s < (uint)NSTREAM) ? _texObjProj[s] : 0;
	}

    void    DeallocHostData();

    void    DeallocDeviceData();

    void    ReallocHostData(uint zInter, pCorGpu param);

    void    ReallocHostData(uint zInter, pCorGpu param, uint idBuff);

    void    ReallocDeviceData(pCorGpu &param);

    ushort2 *HostClassEqui();

	void    ReallocConstData(uint nbImages);

	void    SyncConstData();

	void	SetZoneImage(const ushort& idImage, const uint2& sizeImage, const ushort2& r);

    ushort2 *DeviClassEqui();

    void    SetMaskImages(pixel *dataMaskImages, uint2 dimMaskImage, int nbLayer);

private:

    void    ReallocDeviceData(int nStream, pCorGpu param);

    void    MallocInfo();

    CuHostData3D<float>         _hVolumeCost[SIZERING];
    /// Phase B: dual (SIZERING) host projection buffers for concurrent H2D.
    CuHostData3D<float2>        _hVolumeProj[SIZERING];

	CuUnifiedData3D<uint2>		_uRect;

	CuUnifiedData3D<ushort2>     _uClassEqui;


    CuDeviceData3D<float>       _d_volumeCost[NSTREAM];	// volume des couts
    CuDeviceData3D<float>       _d_volumeCach[NSTREAM];	// volume des calculs intermediaires
    CuDeviceData3D<uint>        _d_volumeNIOk[NSTREAM];	// nombre d'image correct pour une vignette

    ImageGpGpu<pixel,cudaContext>           _dt_GlobalMask;
    ImageLayeredGpGpu<float,cudaContext>    _dt_LayeredImages;
    ImageLayeredGpGpu<pixel,cudaContext>    _dt_LayeredMaskImages;
    ImageLayeredGpGpu<float2,cudaContext>   _dt_LayeredProjection[NSTREAM];

    // Texture objects only (CUDA 12 — no textureReference).
    cudaTextureObject_t         _texObjImages;
    cudaTextureObject_t         _texObjMaskImages;
    cudaTextureObject_t         _texObjMaskGlobal;
    cudaTextureObject_t         _texObjProj[NSTREAM];
    bool                        _texObjReady;

    void DeviceMemset(pCorGpu &param, uint s = 0);

    static cudaTextureObject_t CreateLayeredTexObj(cudaArray * arr, bool linearFilter);
    static cudaTextureObject_t Create2DTexObj(cudaArray * arr, bool linearFilter);
    static void DestroyTexObj(cudaTextureObject_t & obj);
};

/*@}*/
#endif
