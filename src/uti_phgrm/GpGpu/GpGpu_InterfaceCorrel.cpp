#include <stdio.h> // RUNPOD_GPGPU_DIAG
#include <cuda_runtime.h> // RUNPOD_GPGPU_DIAG
#include "GpGpu/GpGpu_InterCorrel.h"
#include "GpGpu/GpGpu_Diag.h"
#include "GpGpu/GpGpu_AutoNbProc.h"

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
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation ENTER idBuf=%d\n", (int)GetIdBuf());

    // Re-allocation les structures de donnees si elles ont ete modifiees

    Data().ReallocDeviceData(Param(GetIdBuf()));
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation after ReallocDeviceData\n");

    // copie des donnees du host vers le device

    Data().copyHostToDevice(Param(GetIdBuf()));
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation after copyHostToDevice\n");

    // Indique que la copie est terminee pour le thread de calcul des projections
    SetPreComp(true);

    // Lancement du calcul de correlation
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation CorrelationGpGpu BEGIN\n");
    CorrelationGpGpu(GetIdBuf());
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR after CorrelationGpGpu: %s\n", cudaGetErrorString(err));
        else
            GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation CorrelationGpGpu END (sync ok)\n");
    }

    // relacher la texture de projection

    Data().UnBindTextureProj();

    // Lancement du calcul de multi-correlation
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation MultiCorrelationGpGpu BEGIN\n");
    MultiCorrelationGpGpu(GetIdBuf());
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR after MultiCorrelationGpGpu: %s\n", cudaGetErrorString(err));
        else
            GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation MultiCorrelationGpGpu END (sync ok)\n");
    }
    // Peak VRAM while correl volumes still resident (auto NbProc probe).
    gpgpu_auto::SampleGpuNow();

    // Copier les resultats de calcul des couts du device vers le host!

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation CopyDevicetoHost BEGIN\n");
    Data().CopyDevicetoHost(GetIdBuf());
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] BasicCorrelation EXIT\n");

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
    LaunchKernelMultiCorrelation( *(GetStream(s)),_param[idBuf],  _data2Cor);
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
