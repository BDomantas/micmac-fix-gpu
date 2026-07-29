#include <stdio.h> // RUNPOD_GPGPU_DIAG
#include <cuda_runtime.h> // RUNPOD_GPGPU_DIAG
#include "GpGpu/GpGpu_InterCorrel.h"

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
    // RUNPOD_GPGPU_DIAG
    fprintf(stderr, "[GPGPU][%s] BasicCorrelation ENTER idBuf=%d\n", "RUNPOD_GPGPU_DIAG", (int)GetIdBuf());
    fflush(stderr);

    // Re-allocation les structures de donnees si elles ont ete modifiees

    Data().ReallocDeviceData(Param(GetIdBuf()));
    fprintf(stderr, "[GPGPU][%s] BasicCorrelation after ReallocDeviceData\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);

    // copie des donnees du host vers le device

    Data().copyHostToDevice(Param(GetIdBuf()));
    fprintf(stderr, "[GPGPU][%s] BasicCorrelation after copyHostToDevice\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);

    // Indique que la copie est terminee pour le thread de calcul des projections
    SetPreComp(true);

    // Lancement du calcul de correlation
    fprintf(stderr, "[GPGPU][%s] BasicCorrelation CorrelationGpGpu BEGIN\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);
    CorrelationGpGpu(GetIdBuf());
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            fprintf(stderr, "[GPGPU][%s] ERROR after CorrelationGpGpu: %s\n", "RUNPOD_GPGPU_DIAG", cudaGetErrorString(err));
        else
            fprintf(stderr, "[GPGPU][%s] BasicCorrelation CorrelationGpGpu END (sync ok)\n", "RUNPOD_GPGPU_DIAG");
        fflush(stderr);
    }

    // relacher la texture de projection

    Data().UnBindTextureProj();

    // Lancement du calcul de multi-correlation
    fprintf(stderr, "[GPGPU][%s] BasicCorrelation MultiCorrelationGpGpu BEGIN\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);
    MultiCorrelationGpGpu(GetIdBuf());
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            fprintf(stderr, "[GPGPU][%s] ERROR after MultiCorrelationGpGpu: %s\n", "RUNPOD_GPGPU_DIAG", cudaGetErrorString(err));
        else
            fprintf(stderr, "[GPGPU][%s] BasicCorrelation MultiCorrelationGpGpu END (sync ok)\n", "RUNPOD_GPGPU_DIAG");
        fflush(stderr);
    }

    // Copier les resultats de calcul des couts du device vers le host!

    fprintf(stderr, "[GPGPU][%s] BasicCorrelation CopyDevicetoHost BEGIN\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);
    Data().CopyDevicetoHost(GetIdBuf());
    fprintf(stderr, "[GPGPU][%s] BasicCorrelation EXIT\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);

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
    SetDataToCopy(false);
    SetCompute(false);
    SetPreComp(false);
    //KillJob();
}

void GpGpuInterfaceCorrel::IntervalZ(uint &interZ, int anZProjection, int aZMaxTer)
{
    uint intZ = (uint)abs(aZMaxTer - anZProjection );
    if (interZ >= intZ  &&  anZProjection != (aZMaxTer - 1) )
        interZ = intZ;
}

float *GpGpuInterfaceCorrel::VolumeCost(ushort id)
{
    return UseMultiThreading() ? Data().HostVolumeCost(id) : Data().HostVolumeCost(0);
}

bool GpGpuInterfaceCorrel::TexturesAreLoaded()
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

SData2Correl &GpGpuInterfaceCorrel::Data()
{
    return _data2Cor;
}
