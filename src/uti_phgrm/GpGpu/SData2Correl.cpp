#include "GpGpu/SData2Correl.h"

SData2Correl::SData2Correl():
    _texMaskGlobal(getMaskGlobal()),
    _TexMaskImages(getTexL_MaskImages()),
    _texImages(getImage()),
    _texProjections_00(getProjection(0)),
    _texProjections_01(getProjection(1)),
    _texObjImages(0),
    _texObjMaskImages(0),
    _texObjMaskGlobal(0),
    _texObjReady(false)
{
    _d_volumeCost[0].SetName("_d_volumeCost");
    _d_volumeCach[0].SetName("_d_volumeCach");
    _d_volumeNIOk[0].SetName("_d_volumeNIOk");
    _dt_GlobalMask.SetNameImage("_dt_GlobalMask");

    _dt_LayeredImages.CData3D::SetName("_dt_LayeredImages");
    _dt_LayeredMaskImages.CData3D::SetName("_dt_LayeredMaskImages");
    _dt_LayeredProjection->CData3D::SetName("_dt_LayeredProjection");

    for (int s = 0;s<NSTREAM;s++)
    {
        GpGpuTools::SetParamterTexture(GetTeXProjection(s));
        _texObjProj[s] = 0;
    }

    GpGpuTools::SetParamterTexture(_texImages);

    _texMaskGlobal.addressMode[0]	= cudaAddressModeBorder;
    _texMaskGlobal.addressMode[1]	= cudaAddressModeBorder;
    _texMaskGlobal.filterMode       = cudaFilterModePoint;
    _texMaskGlobal.normalized       = false;

    _TexMaskImages.addressMode[0]	= cudaAddressModeBorder;
    _TexMaskImages.addressMode[1]	= cudaAddressModeBorder;
    _TexMaskImages.filterMode       = cudaFilterModePoint;
    _TexMaskImages.normalized       = false;


    for (int i = 0; i < SIZERING; ++i)
    {
        _hVolumeCost[i].SetName("_hVolumeCost_0",i);
		_hVolumeCost[i].setPgLockMem(true);
        _hVolumeProj[i].SetName("_hVolumeProj", i);
        _hVolumeProj[i].setPgLockMem(true);
    }

}

SData2Correl::~SData2Correl()
{
    DestroyTextureObjects();
    DeallocDeviceData();
	DeallocHostData();
	_uRect.Dealloc();
	_uClassEqui.Dealloc();
}

void SData2Correl::MallocInfo()
{
    std::cout << "Malloc Info GpGpu\n";
    CGpGpuContext<cudaContext>::OutputInfoGpuMemory();
    _d_volumeCost[0].MallocInfo();
    _d_volumeCach[0].MallocInfo();
    _d_volumeNIOk[0].MallocInfo();
    _dt_GlobalMask.DecoratorImage<cudaContext>::MallocInfo();
    _dt_LayeredImages.CData3D::MallocInfo();
    _dt_LayeredMaskImages.CData3D::MallocInfo();
    _dt_LayeredProjection[0].CData3D::MallocInfo();
}

float *SData2Correl::HostVolumeCost(uint id)
{
    return _hVolumeCost[id].pData();
}

float2 *SData2Correl::HostVolumeProj()
{
    return _hVolumeProj[0].pData();
}

float2 *SData2Correl::HostVolumeProj(uint id)
{
    if (id >= (uint)SIZERING)
        id = 0;
    return _hVolumeProj[id].pData();
}

uint2 *SData2Correl::HostRect()
{
	return _uRect.hostData.pData();
}

ushort2 *SData2Correl::HostClassEqui()
{
	return _uClassEqui.hostData.pData();
}

void SData2Correl::DeallocHostData()
{
    for (int i = 0; i < SIZERING; ++i)
    {
            _hVolumeCost[i].Dealloc();
            _hVolumeProj[i].Dealloc();
    }
}

void SData2Correl::DeallocDeviceData()
{
    DestroyTextureObjects();
    checkCudaErrors( cudaUnbindTexture(&_texImages) );
    checkCudaErrors( cudaUnbindTexture(&_texMaskGlobal) );
    checkCudaErrors( cudaUnbindTexture(&_TexMaskImages) );

    for (int s = 0;s<NSTREAM;s++)
    {
        _d_volumeCach[s].Dealloc();
        _d_volumeCost[s].Dealloc();
        _d_volumeNIOk[s].Dealloc();
        _dt_LayeredProjection[s].Dealloc();
    }

    _dt_GlobalMask.Dealloc();
    _dt_LayeredImages.Dealloc();
    _dt_LayeredMaskImages.Dealloc();

}

textureReference &SData2Correl::GetTeXProjection(int TexSel)
{
    switch (TexSel)
    {
    case 0:
        return _texProjections_00;
    case 1:
        return _texProjections_01;
    default:
        return _texProjections_00;
    }
}

void SData2Correl::SetImages(float *dataImage, uint2 dimImage, int nbLayer)
{
#ifdef  NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A22B5);
#endif
    _dt_LayeredImages.CData3D::ReallocIfDim(dimImage,nbLayer);
    _dt_LayeredImages.copyHostToDevice(dataImage);
    _dt_LayeredImages.bindTexture(_texImages);
    // Phase C: rebuild image texture object after upload.
    DestroyTexObj(_texObjImages);
    _texObjImages = CreateLayeredTexObj(_dt_LayeredImages.GetCudaArray(), /*linear*/true);
#ifdef  NVTOOLS
	GpGpuTools::Nvtx_RangePop();
#endif
}

void SData2Correl::SetMaskImages(pixel *dataMaskImages, uint2 dimMaskImage, int nbLayer)
{
#ifdef  NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A22B5);
#endif
    _dt_LayeredMaskImages.CData3D::ReallocIfDim(dimMaskImage,nbLayer);
    _dt_LayeredMaskImages.copyHostToDevice(dataMaskImages);
    _dt_LayeredMaskImages.bindTexture(_TexMaskImages);
    DestroyTexObj(_texObjMaskImages);
    _texObjMaskImages = CreateLayeredTexObj(_dt_LayeredMaskImages.GetCudaArray(), /*linear*/false);
#ifdef  NVTOOLS
	GpGpuTools::Nvtx_RangePop();
#endif
}

void SData2Correl::SetGlobalMask(pixel *dataMask, uint2 dimMask)
{
	#ifdef  NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A2B51);
    #endif
    _dt_GlobalMask.DecoratorImage<cudaContext>::ReallocIfDim(dimMask,1);
    _dt_GlobalMask.copyHostToDevice(dataMask);
    _dt_GlobalMask.bindTexture(_texMaskGlobal);
    DestroyTexObj(_texObjMaskGlobal);
    _texObjMaskGlobal = Create2DTexObj(_dt_GlobalMask.GetCudaArray(), /*linear*/false);
	#ifdef  NVTOOLS
	GpGpuTools::Nvtx_RangePop();
	#endif
}

void SData2Correl::copyHostToDevice(pCorGpu param,uint s)
{
	#ifdef  NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF292CB0);
	#endif

    _dt_LayeredProjection[s].ReallocIfDim(param.dimSTer,param.invPC.nbImages * param.ZCInter);

    // Slot-aligned host proj: use same index as stream/idBuf ring.
    uint projSlot = (s < (uint)SIZERING) ? s : 0;
    _dt_LayeredProjection[s].copyHostToDevice(_hVolumeProj[projSlot].pData());

    _dt_LayeredProjection[s].bindTexture(GetTeXProjection(s));
    // Phase C: per-slot projection texture object.
    DestroyTexObj(_texObjProj[s]);
    _texObjProj[s] = CreateLayeredTexObj(_dt_LayeredProjection[s].GetCudaArray(), /*linear*/true);
	#ifdef  NVTOOLS
	GpGpuTools::Nvtx_RangePop();
	#endif
}

void SData2Correl::copyHostToDeviceASync(pCorGpu param, uint s, cudaStream_t stream)
{
    _dt_LayeredProjection[s].ReallocIfDim(param.dimSTer,param.invPC.nbImages * param.ZCInter);
    uint projSlot = (s < (uint)SIZERING) ? s : 0;
    _dt_LayeredProjection[s].copyHostToDeviceASync(_hVolumeProj[projSlot].pData(), stream);
    // Bind is host-side; must complete before kernel that samples this slot.
    _dt_LayeredProjection[s].bindTexture(GetTeXProjection(s));
    DestroyTexObj(_texObjProj[s]);
    _texObjProj[s] = CreateLayeredTexObj(_dt_LayeredProjection[s].GetCudaArray(), /*linear*/true);
}

void SData2Correl::CopyDevicetoHost(uint idBuf, uint s)
{
    _d_volumeCost[s].CopyDevicetoHost(_hVolumeCost[idBuf]);
}

void SData2Correl::CopyDevicetoHostASync(uint idBuf, uint s, cudaStream_t stream)
{
    _d_volumeCost[s].CopyDevicetoHostASync(_hVolumeCost[idBuf].pData(), stream);
}

void SData2Correl::UnBindTextureProj(uint s)
{
    checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))));
}

void SData2Correl::ReallocConstData(uint nbImages)
{
	_uClassEqui.ReallocIfDim(make_uint2(1,1),nbImages);
	_uRect.ReallocIfDim(make_uint2(1,1),nbImages*INTERZ);
}

void SData2Correl::SyncConstData()
{
	_uClassEqui.syncDevice();
	_uRect.syncDevice();
}

void SData2Correl::SetZoneImage(const ushort &idImage,const uint2 &sizeImage,const ushort2 &r)
{

	_uRect.hostData.pData()[idImage] = make_uint2(sizeImage.x - SAMPLETERR - r.x,sizeImage.y - SAMPLETERR - r.y);
}

void SData2Correl::ReallocHostData(uint zInter, pCorGpu param)
{
	#ifdef  NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFFAA0000);
	#endif
    for (int i = 0; i < SIZERING; ++i)
    {
        _hVolumeCost[i].ReallocIf(param.HdPc.dimTer,zInter);
        _hVolumeProj[i].ReallocIf(param.dimSTer,zInter*param.invPC.nbImages);
    }

	#ifdef  NVTOOLS
	GpGpuTools::Nvtx_RangePop();
	#endif
}

void SData2Correl::ReallocHostData(uint zInter, pCorGpu param, uint idBuff)
{
    _hVolumeCost[idBuff].ReallocIf(param.HdPc.dimTer,zInter);

    uint projSlot = (idBuff < (uint)SIZERING) ? idBuff : 0;
    _hVolumeProj[projSlot].ReallocIf(param.dimSTer,zInter*param.invPC.nbImages);

}

void SData2Correl::ReallocDeviceData(pCorGpu &param)
{
	#ifdef NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A2BB5);
	#endif
    for (int s = 0;s<NSTREAM;s++)
    {
        ReallocDeviceData(s, param);

        DeviceMemset(param,s);
    }
	#ifdef NVTOOLS
	GpGpuTools::Nvtx_RangePop();
	#endif
}

void SData2Correl::ReallocDeviceDataSlot(uint s, pCorGpu &param)
{
    if (s >= (uint)NSTREAM)
        s = 0;
    ReallocDeviceData((int)s, param);
    // PR-B: do not default-stream Memset here — BasicCorrelation clears on the work stream.
}

void    SData2Correl::DeviceMemset(pCorGpu &param, uint s)
{
	#ifdef NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A2BB5);
	#endif
    _d_volumeCost[s].Memset(param.invPC.IntDefault);

    _d_volumeNIOk[s].Memset(0);
	#ifdef NVTOOLS
	GpGpuTools::Nvtx_RangePop();
	#endif
}

void SData2Correl::DeviceMemsetAsync(pCorGpu &param, uint s, cudaStream_t stream)
{
	#ifdef NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A2BB5);
	#endif
    if (s >= (uint)NSTREAM)
        s = 0;
    _d_volumeCost[s].MemsetAsync(param.invPC.IntDefault, stream);
    _d_volumeNIOk[s].MemsetAsync(0, stream);
	#ifdef NVTOOLS
	GpGpuTools::Nvtx_RangePop();
	#endif
}

uint    *SData2Correl::DeviVolumeNOK(uint s){

    return _d_volumeNIOk[s].pData();

}

float   *SData2Correl::DeviVolumeCache(uint s){

    return _d_volumeCach[s].pData();

}

float   *SData2Correl::DeviVolumeCost(uint s){

    return _d_volumeCost[s].pData();

}

uint2 *SData2Correl::DeviRect()
{
	return _uRect.deviceData.pData();
}

ushort2 *SData2Correl::DeviClassEqui()
{
	return _uClassEqui.deviceData.pData();
}

void SData2Correl::ReallocDeviceData(int nStream, pCorGpu param)
{

    _d_volumeCost[nStream].ReallocIf(param.HdPc.dimTer,     param.ZCInter);

    _d_volumeCach[nStream].ReallocIf(param.HdPc.dimCach,    param.invPC.nbImages * param.ZCInter);

    _d_volumeNIOk[nStream].ReallocIf(param.HdPc.dimTer,     param.ZCInter * param.invPC.nbClass);
}

void SData2Correl::MemsetHostVolumeProj(int iDef, uint id)
{
    if (id < (uint)SIZERING)
    {
        _hVolumeProj[id].Memset(iDef);
        return;
    }
    for (int i = 0; i < SIZERING; ++i)
        _hVolumeProj[i].Memset(iDef);
}

cudaTextureObject_t SData2Correl::CreateLayeredTexObj(cudaArray * arr, bool linearFilter)
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
    checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    return tex;
}

cudaTextureObject_t SData2Correl::Create2DTexObj(cudaArray * arr, bool linearFilter)
{
    return CreateLayeredTexObj(arr, linearFilter);
}

void SData2Correl::DestroyTexObj(cudaTextureObject_t & obj)
{
    if (obj)
    {
        cudaDestroyTextureObject(obj);
        obj = 0;
    }
}

void SData2Correl::DestroyTextureObjects()
{
    DestroyTexObj(_texObjImages);
    DestroyTexObj(_texObjMaskImages);
    DestroyTexObj(_texObjMaskGlobal);
    for (int s = 0; s < NSTREAM; ++s)
        DestroyTexObj(_texObjProj[s]);
    _texObjReady = false;
}

void SData2Correl::EnsureTextureObjects(uint s)
{
    if (!_texObjImages && _dt_LayeredImages.GetCudaArray())
        _texObjImages = CreateLayeredTexObj(_dt_LayeredImages.GetCudaArray(), true);
    if (!_texObjMaskImages && _dt_LayeredMaskImages.GetCudaArray())
        _texObjMaskImages = CreateLayeredTexObj(_dt_LayeredMaskImages.GetCudaArray(), false);
    if (!_texObjMaskGlobal && _dt_GlobalMask.GetCudaArray())
        _texObjMaskGlobal = Create2DTexObj(_dt_GlobalMask.GetCudaArray(), false);
    if (s < (uint)NSTREAM && !_texObjProj[s] && _dt_LayeredProjection[s].GetCudaArray())
        _texObjProj[s] = CreateLayeredTexObj(_dt_LayeredProjection[s].GetCudaArray(), true);
    _texObjReady = true;
}
