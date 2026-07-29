#include <stdio.h> // RUNPOD_GPGPU_DIAG
#include <cuda_runtime.h> // RUNPOD_GPGPU_DIAG
#include "GpGpu/GpGpu_InterCorrel.h"
#include "GpGpu/GpGpu_Diag.h"
#include "GpGpu/GpGpu_AutoNbProc.h"
#include "GpGpu/GpGpu_Pipeline.h"
#include "GpGpu/GpGpu_Budget.h"

/// \brief Constructeur GpGpuInterfaceCorrel
GpGpuInterfaceCorrel::GpGpuInterfaceCorrel():
    CSimpleJobCpuGpu(true),
     NoMasked(false),
     copyInvParam(false)
{
    for (int s = 0;s<NSTREAM;s++)
        checkCudaErrors( cudaStreamCreate(GetStream(s)));

    freezeCompute();
}

GpGpuInterfaceCorrel::~GpGpuInterfaceCorrel()
{
    for (int s = 0;s<NSTREAM;s++)
        checkCudaErrors( cudaStreamDestroy(*(GetStream(s))));

}
void GpGpuInterfaceCorrel::ReallocHostData(uint interZ,ushort idBuff)
{
    _data2Cor.ReallocHostData(interZ,_param[idBuff],idBuff);
}

SData2Correl& GpGpuInterfaceCorrel::Data()
{
    return _data2Cor;
}

float* GpGpuInterfaceCorrel::VolumeCost(ushort id)
{
    return _data2Cor.HostVolumeCost(id);
}

void GpGpuInterfaceCorrel::IntervalZ(uint &interZ, int anZProjection, int aZMaxTer)
{
    // Keep correl slab within remaining Z and INTERZ cap.
    int remain = aZMaxTer - anZProjection;
    if (remain < 0)
        remain = 0;
    interZ = (uint)min(INTERZ, remain > 0 ? remain : (int)INTERZ);
    if (interZ < 1)
        interZ = 1;
}

uint2 &GpGpuInterfaceCorrel::DimTerrainGlob()
{
    return _m_DimTerrainGlob;
}

std::vector<cellules> &GpGpuInterfaceCorrel::MaskVolumeBlock()
{
    return _m_MaskVolumeBlock;
}

uint GpGpuInterfaceCorrel::InitCorrelJob(int Zmin, int Zmax)
{

    uint interZ = min(INTERZ, abs(Zmin - Zmax));

    if(UseMultiThreading())
    {
        ResetIdBuffer();
        SetPreComp(true);
    }

    return interZ;
}

/// \brief Initialisation des parametres constants
void GpGpuInterfaceCorrel::SetParameter(int nbLayer , ushort2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef, ushort nClass )
{

    if(!copyInvParam || _param[0].invPC.nbImages != (uint)nbLayer || _param[1].invPC.nbImages != (uint)nbLayer)
    {
        copyInvParam = true;
        _param[0].invPC.SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer,nClass);
        _param[1].invPC.SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer,nClass);
        CopyParamInvTodevice(_param[0]);
    }
}

void GpGpuInterfaceCorrel::BasicCorrelation()
{
    // Phase B: map host ring buffer id → CUDA stream/slot.
    const int s = (int)(GetIdBuf() % (ushort)NSTREAM);
    const int activeSlots = gpgpu_budget::ClampActiveSlots(NSTREAM);
    const int slot = (s < activeSlots) ? s : 0;
    cudaStream_t stream = *(GetStream(slot));

    GPGPU_DIAG_FULL(
        "[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation ENTER idBuf=%d slot=%d NSTREAM=%d pipeline=%d\n",
        (int)GetIdBuf(), slot, NSTREAM, gpgpu_pipeline::PipelineEnabled() ? 1 : 0);

    // Re-allocation for this stream slot only (preserve other in-flight volumes).
    Data().ReallocDeviceDataSlot((uint)slot, Param(GetIdBuf()));
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation after ReallocDeviceDataSlot\n");

    // Phase C: ensure texture objects exist for multi-stream-safe sampling path.
    Data().EnsureTextureObjects((uint)slot);

    // H2D proj on stream (async when pipeline on).
    if (gpgpu_pipeline::PipelineEnabled())
        Data().copyHostToDeviceASync(Param(GetIdBuf()), (uint)slot, stream);
    else
        Data().copyHostToDevice(Param(GetIdBuf()), (uint)slot);
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation after copyHostToDevice slot=%d\n", slot);

    // Host may prep next ring buffer after H2D is enqueued (pipeline) / done (legacy).
    SetPreComp(true);

    // Correl then multi-correl on the SAME stream (ordering without device-wide sync).
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation CorrelationGpGpu BEGIN slot=%d\n", slot);
    CorrelationGpGpu(GetIdBuf(), slot);

    // Multi does not need proj texture; unbind proj for this slot after correl.
    Data().UnBindTextureProj((uint)slot);

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation MultiCorrelationGpGpu BEGIN slot=%d\n", slot);
    MultiCorrelationGpGpu(GetIdBuf(), slot);

    // Peak VRAM while correl volumes still resident.
    gpgpu_auto::SampleGpuNow();

    // D2H cost on stream.
    if (gpgpu_pipeline::PipelineEnabled())
        Data().CopyDevicetoHostASync(GetIdBuf(), (uint)slot, stream);
    else
        Data().CopyDevicetoHost(GetIdBuf(), (uint)slot);

    // Host must see costs before consuming: stream sync (hot path) or full device for diag.
    if (GpgpuDiagFull() || !gpgpu_pipeline::PipelineEnabled())
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR after correl pipeline device sync: %s\n",
                    cudaGetErrorString(err));
    }
    else
    {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR after correl pipeline stream sync: %s\n",
                    cudaGetErrorString(err));
    }

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation EXIT slot=%d\n", slot);
}

cudaStream_t* GpGpuInterfaceCorrel::GetStream( int stream )
{
    return &(_stream[stream]);
}

void GpGpuInterfaceCorrel::simpleWork()
{
    BasicCorrelation();
}

void GpGpuInterfaceCorrel::freezeCompute()
{
    Param(0).HdPc.sizeCachAll = 0;
    Param(1).HdPc.sizeCachAll = 0;

    SetCompute(false);
}

bool  GpGpuInterfaceCorrel::TexturesAreLoaded()
{
    return _TexturesAreLoaded;
}

void GpGpuInterfaceCorrel::SetTexturesAreLoaded(bool load)
{
    _TexturesAreLoaded = load;
    NoMasked = false;
}

void GpGpuInterfaceCorrel::CorrelationGpGpu(ushort idBuf,const int s )
{
    LaunchKernelCorrelation(s, *(GetStream(s)),_param[idBuf], _data2Cor);
}

void GpGpuInterfaceCorrel::MultiCorrelationGpGpu(ushort idBuf, const int s)
{
    LaunchKernelMultiCorrelation( *(GetStream(s)),_param[idBuf],  _data2Cor, s);
}

pCorGpu& GpGpuInterfaceCorrel::Param(ushort idBuf)
{
    return _param[idBuf];
}

void GpGpuInterfaceCorrel::signalComputeCorrel(uint dZ)
{
    SetPreComp(false);
    SetCompute(true);
}
