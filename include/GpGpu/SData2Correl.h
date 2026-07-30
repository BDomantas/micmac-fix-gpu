#ifndef _SDATA2CORREL_H
#define _SDATA2CORREL_H
/** @addtogroup GpGpuDoc */
/*@{*/

#include "GpGpu/GpGpu_ParamCorrelation.cuh"
#include <cuda_runtime.h>

/// \cond
extern "C" textureReference&    getMaskGlobal();
extern "C" textureReference&	getMask();
extern "C" textureReference&	getImage();
extern "C" textureReference&	getProjection(int TexSel);
extern "C" textureReference&    getTexL_MaskImages();

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
	/// \param dataImage
	/// \param dimImage
	/// \param nbLayer
	///
    void    SetImages( float* dataImage, uint2 dimImage, int nbLayer );

	///
	/// \brief SetGlobalMask Initialise les masques sur GPU
	/// \param dataMask
	/// \param dimMask
	///
    void    SetGlobalMask( pixel* dataMask, uint2 dimMask );

	///
	/// \brief MemsetHostVolumeProj Initialise la memoire des projections par une valeur iDef
	/// \param iDef
	/// \param id slot (SIZERING); if >= SIZERING, fill all slots
	///
    void    MemsetHostVolumeProj(int iDef, uint id = 0xFFFFFFFFu);

	///
	/// \brief HostVolumeCost
	/// \param id
	/// \return le pointeur host du volume de corr�lation
	///
    float*  HostVolumeCost(uint id);

	///
	/// \brief HostVolumeProj
	/// \return le pointeur host du volume de projection (slot 0 legacy)
	///
    float2* HostVolumeProj();

	/// Phase B: per-slot host projection ring (id < SIZERING / NSTREAM).
	float2* HostVolumeProj(uint id);

	///
	/// \brief HostRect
	/// \return le pointeur des rectangles images
	///
	uint2*	HostRect();

	///
	/// \brief DeviVolumeNOK
	/// \param s
	/// \return  le pointeur device des volumes des images correctes
	///
    uint*   DeviVolumeNOK(uint s);

	///
	/// \brief DeviVolumeCache
	/// \param s
	/// \return le pointeur device du cache des vecteurs centr�s
	///
    float*  DeviVolumeCache(uint s);

	///
	/// \brief DeviVolumeCost
	/// \param s
	/// \return  Le pointeur device du volume de couts
	///
    float*  DeviVolumeCost(uint s);

	///
	/// \brief DeviRect
	/// \return Le pointeur device des rectangles images
	///
	uint2*	DeviRect();

	///
	/// \brief copyHostToDevice Copie les donn�es vers le device
	/// \param param
	/// \param s
	///
    void    copyHostToDevice(pCorGpu param, uint s = 0);

	/// Phase B: async H2D of projection slot s on cuda stream.
	void    copyHostToDeviceASync(pCorGpu param, uint s, cudaStream_t stream);

	///
	/// \brief CopyDevicetoHost Copie les donn�es vers le host
	/// \param idBuf
	/// \param s
	///
    void    CopyDevicetoHost(uint idBuf, uint s = 0);

	/// Phase B: async D2H of cost volume for stream slot s.
	void    CopyDevicetoHostASync(uint idBuf, uint s, cudaStream_t stream);

	///
	/// \brief UnBindTextureProj relacher les textures sur le device
	/// \param s
	///
    void    UnBindTextureProj(uint s = 0);

	/// Realloc + memset a single stream slot (avoid touching other in-flight slots).
	void    ReallocDeviceDataSlot(uint s, pCorGpu &param);

	/// Phase C: create cuda texture objects for images/masks/projections (stream-safe).
	void    EnsureTextureObjects(uint s = 0);
	void    DestroyTextureObjects();
	cudaTextureObject_t TexObjImages() const { return _texObjImages; }
	cudaTextureObject_t TexObjMaskImages() const { return _texObjMaskImages; }
	cudaTextureObject_t TexObjMaskGlobal() const { return _texObjMaskGlobal; }
	cudaTextureObject_t TexObjProj(uint s) const {
		return (s < (uint)NSTREAM) ? _texObjProj[s] : 0;
	}

	///
	/// \brief DeallocHostData Desalloue la m�moire host
	///
    void    DeallocHostData();

	///
	/// \brief DeallocDeviceData D�salloue la m�moire device
	///
    void    DeallocDeviceData();

	///
	/// \brief ReallocHostData r�alloue la m�moire host
	/// \param zInter
	/// \param param
	///
    void    ReallocHostData(uint zInter, pCorGpu param);

	///
	/// \brief ReallocHostData r�alloue la m�moire host
	/// \param zInter
	/// \param param
	/// \param idBuff
	///
    void    ReallocHostData(uint zInter, pCorGpu param, uint idBuff);

	///
	/// \brief ReallocDeviceData r�alloue la m�moire device
	/// \param param
	///
    void    ReallocDeviceData(pCorGpu &param);   

	///
	/// \brief HostClassEqui
	/// \return Le pointeur des classes d'�quivalence
	///
    ushort2 *HostClassEqui();

	///
	/// \brief ReallocConstData R�allocation des donn�es constantes
	/// \param nbImages
	///
	void    ReallocConstData(uint nbImages);

	///
	/// \brief SyncConstData Synchronise les donn�es constantes sur le device
	///
	void    SyncConstData();

	///
	/// \brief SetZoneImage D�finir les dimensions des images
	/// \param idImage
	/// \param sizeImage
	/// \param r
	///
	void	SetZoneImage(const ushort& idImage, const uint2& sizeImage, const ushort2& r);

	///
	/// \brief DeviClassEqui
	/// \return Le pointeur device des classes d'�quivalence
	///
    ushort2 *DeviClassEqui();

	///
	/// \brief SetMaskImages Transfert les masques sur le device
	/// \param dataMaskImages
	/// \param dimMaskImage
	/// \param nbLayer
	///
    void    SetMaskImages(pixel *dataMaskImages, uint2 dimMaskImage, int nbLayer);

private:

    void    ReallocDeviceData(int nStream, pCorGpu param);

    void    MallocInfo();

    textureReference& GetTeXProjection( int TexSel );

    CuHostData3D<float>         _hVolumeCost[SIZERING];
    /// Phase B: dual (SIZERING) host projection buffers for concurrent H2D.
    CuHostData3D<float2>        _hVolumeProj[SIZERING];

    // TODO il semblerait qu'un uint2 suffirai....
    ///
    /// \brief _hRect   HOST     gestion des bords d'images
    ///
//    CuHostData3D<Rect>          _hRect;
    ///
    /// \brief _dRect   Device   gestion des bords d'images
    ///
//    CuDeviceData3D<Rect>        _dRect;

	CuUnifiedData3D<uint2>		_uRect;

    ///
    /// \brief _hClassEqui HOST     gestion des classes d'images
    ///
	//CuHostData3D<ushort2>       _hClassEqui;
    ///
    /// \brief _dClassEqui DEVICE    gestion des classes d'images
    ///
	//CuDeviceData3D<ushort2>     _dClassEqui;
	CuUnifiedData3D<ushort2>     _uClassEqui;


    CuDeviceData3D<float>       _d_volumeCost[NSTREAM];	// volume des couts
    CuDeviceData3D<float>       _d_volumeCach[NSTREAM];	// volume des calculs interm�diaires
    CuDeviceData3D<uint>        _d_volumeNIOk[NSTREAM];	// nombre d'image correct pour une vignette

    ImageGpGpu<pixel,cudaContext>           _dt_GlobalMask;
    ImageLayeredGpGpu<float,cudaContext>    _dt_LayeredImages;
    ImageLayeredGpGpu<pixel,cudaContext>    _dt_LayeredMaskImages;
    ImageLayeredGpGpu<float2,cudaContext>   _dt_LayeredProjection[NSTREAM];

    textureReference&           _texMaskGlobal;
    textureReference&           _TexMaskImages;
    textureReference&           _texImages;
    textureReference&           _texProjections_00;
    textureReference&           _texProjections_01;

    // Phase C: texture objects (non-global bind model for multi-stream safety).
    cudaTextureObject_t         _texObjImages;
    cudaTextureObject_t         _texObjMaskImages;
    cudaTextureObject_t         _texObjMaskGlobal;
    cudaTextureObject_t         _texObjProj[NSTREAM];
    bool                        _texObjReady;

    void DeviceMemset(pCorGpu &param, uint s = 0);
    /// PR-B: stream-ordered clear (cost + NIOk) — no default-stream hot path.
    void DeviceMemsetAsync(pCorGpu &param, uint s, cudaStream_t stream);

    static cudaTextureObject_t CreateLayeredTexObj(cudaArray * arr, bool linearFilter);
    static cudaTextureObject_t Create2DTexObj(cudaArray * arr, bool linearFilter);
    static void DestroyTexObj(cudaTextureObject_t & obj);
};

/*@}*/
#endif
