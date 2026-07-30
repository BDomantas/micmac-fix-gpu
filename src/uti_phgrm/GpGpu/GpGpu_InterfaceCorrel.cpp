#include <stdio.h> // RUNPOD_GPGPU_DIAG
#include <cuda_runtime.h> // RUNPOD_GPGPU_DIAG
#include "GpGpu/GpGpu_InterCorrel.h"
#include "GpGpu/GpGpu_Diag.h"
#include "GpGpu/GpGpu_AutoNbProc.h"
#include "GpGpu/GpGpu_Pipeline.h"
#include "GpGpu/GpGpu_Budget.h"
#include "GpGpu/GpGpu_SlotMap.h"

/// \brief Constructeur GpGpuInterfaceCorrel
GpGpuInterfaceCorrel::GpGpuInterfaceCorrel():
    CSimpleJobCpuGpu(true),
     NoMasked(false),
     copyInvParam(false)
{
    for (int s = 0;s<NSTREAM;s++)
    {
        checkCudaErrors( cudaStreamCreate(GetStream(s)));
        _doneEvent[s] = 0;
        checkCudaErrors(cudaEventCreateWithFlags(&_doneEvent[s], cudaEventDisableTiming));
        _eventRecorded[s] = false;
    }

    freezeCompute();
}

GpGpuInterfaceCorrel::~GpGpuInterfaceCorrel()
{
    for (int s = 0;s<NSTREAM;s++)
    {
        if (_doneEvent[s])
        {
            cudaEventDestroy(_doneEvent[s]);
            _doneEvent[s] = 0;
        }
        checkCudaErrors( cudaStreamDestroy(*(GetStream(s))));
    }

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
        // Ensure host copy flag starts clear so first worker is not stuck.
        SetDataToCopy(false);
    }
    for (int s = 0; s < NSTREAM; ++s)
        _eventRecorded[s] = false;

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

int GpGpuInterfaceCorrel::MapSlotForIdBuf(ushort idBuf) const
{
    const int activeSlots = gpgpu_budget::GetActiveSlotsRuntime();
    return gpgpu_slot::MapHostToSlot((int)idBuf, activeSlots, NSTREAM);
}

void GpGpuInterfaceCorrel::WaitCorrelDone(ushort idBuf)
{
    const int slot = MapSlotForIdBuf(idBuf);
    if (slot < 0 || slot >= NSTREAM)
        return;
    if (!_eventRecorded[slot] || !_doneEvent[slot])
    {
        // Fallback: stream sync if event never recorded (legacy path).
        cudaError_t err = cudaStreamSynchronize(*(GetStream(slot)));
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR WaitCorrelDone stream sync: %s\n",
                    cudaGetErrorString(err));
        return;
    }
    cudaError_t err = cudaEventSynchronize(_doneEvent[slot]);
    if (err != cudaSuccess)
        GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR WaitCorrelDone event: %s\n",
                cudaGetErrorString(err));
}

void GpGpuInterfaceCorrel::BasicCorrelation()
{
    // PR-D: map host ring buffer id → CUDA stream/slot (no collapse when active_slots≥2).
    const int slot = MapSlotForIdBuf(GetIdBuf());
    cudaStream_t stream = *(GetStream(slot));
    const bool pipeline = gpgpu_pipeline::PipelineEnabled();

    GPGPU_DIAG_FULL(
        "[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation ENTER idBuf=%d slot=%d NSTREAM=%d "
        "active_slots=%d pipeline=%d\n",
        (int)GetIdBuf(), slot, NSTREAM,
        gpgpu_budget::GetActiveSlotsRuntime(),
        pipeline ? 1 : 0);

    // Re-allocation for this stream slot only (preserve other in-flight volumes).
    Data().ReallocDeviceDataSlot((uint)slot, Param(GetIdBuf()));
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation after ReallocDeviceDataSlot\n");

    // PR-B: stream-ordered clears (never default-stream cudaMemset on hot path).
    Data().DeviceMemsetAsync(Param(GetIdBuf()), (uint)slot, stream);

    // Phase C: ensure texture objects exist for multi-stream-safe sampling path.
    Data().EnsureTextureObjects((uint)slot);

    // H2D proj on stream (async when pipeline on).
    if (pipeline)
        Data().copyHostToDeviceASync(Param(GetIdBuf()), (uint)slot, stream);
    else
        Data().copyHostToDevice(Param(GetIdBuf()), (uint)slot);
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation after copyHostToDevice slot=%d\n", slot);

    // Host may prep next ring buffer after H2D is enqueued (pipeline) / done (legacy).
    SetPreComp(true);

    // Correl then multi-correl on the SAME stream (ordering without device-wide sync).
    // Phase C: correl samples texture objects — do NOT UnBindTextureProj while kernels
    // may still be in flight (stream-ordered). Multi-correl uses only device volumes.
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation CorrelationGpGpu BEGIN slot=%d\n", slot);
    CorrelationGpGpu(GetIdBuf(), slot);

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation MultiCorrelationGpGpu BEGIN slot=%d\n", slot);
    MultiCorrelationGpGpu(GetIdBuf(), slot);

    // Peak VRAM while correl volumes still resident.
    gpgpu_auto::SampleGpuNow();

    // D2H cost on stream.
    if (pipeline)
        Data().CopyDevicetoHostASync(GetIdBuf(), (uint)slot, stream);
    else
        Data().CopyDevicetoHost(GetIdBuf(), (uint)slot);

    // PR-C: record per-slot completion event after D2H.
    if (_doneEvent[slot])
    {
        cudaError_t eRec = cudaEventRecord(_doneEvent[slot], stream);
        if (eRec != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR cudaEventRecord: %s\n",
                    cudaGetErrorString(eRec));
        else
            _eventRecorded[slot] = true;
    }

    // Hot path: no full stream/device sync under pipeline (host WaitCorrelDone uses event).
    // FULL diag may force device sync for hang isolation. Legacy keeps old sync.
    if (GpgpuDiagFull() || !pipeline)
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR after correl pipeline device sync: %s\n",
                    cudaGetErrorString(err));
    }
    // else: return after enqueue; host consumes only after WaitCorrelDone(event).

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation EXIT slot=%d async=%d\n",
            slot, (pipeline && !GpgpuDiagFull()) ? 1 : 0);
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
