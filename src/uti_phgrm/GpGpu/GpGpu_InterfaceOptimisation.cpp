#include <stdio.h> // RUNPOD_GPGPU_DIAG
#include <cuda_runtime.h> // RUNPOD_GPGPU_DIAG
#include "GpGpu/GpGpu_InterOptimisation.h"
#include "GpGpu/GpGpu_Diag.h"
#include "GpGpu/GpGpu_AutoNbProc.h"
#include "GpGpu/GpGpu_Pipeline.h"
#include "GpGpu/GpGpu_Budget.h"

// Optional host callback run after optim kernel is enqueued (before stream drain).
// Used for cross-box host prefetch while GPU optim runs.
static void (*g_optimPrefetchFn)(void *) = 0;
static void *g_optimPrefetchCtx = 0;

extern "C" void GpGpu_SetOptimPrefetchHook(void (*fn)(void *), void *ctx)
{
    g_optimPrefetchFn = fn;
    g_optimPrefetchCtx = ctx;
}

InterfOptimizGpGpu::InterfOptimizGpGpu()
{
    _optStream = 0;
    checkCudaErrors(cudaStreamCreate(&_optStream));
    freezeCompute();
}

InterfOptimizGpGpu::~InterfOptimizGpGpu()
{
    if (_optStream)
        cudaStreamDestroy(_optStream);
    _optStream = 0;
}

void InterfOptimizGpGpu::Dealloc()
{
    _H_data2Opt.Dealloc();
    _D_data2Opt.Dealloc();

    _preFinalCost1D.Dealloc();
    _poInitCost.Dealloc();
}


void InterfOptimizGpGpu::Prepare(uint x, uint y, ushort penteMax, ushort NBDir,float zReg,float zRegQuad, ushort costDefMask,ushort costDefMaskTrans, bool hasMaskAuto)
{
    uint size = (uint)(1.5f*sqrt((float)x *x + y * y));

    _H_data2Opt.setDzMax(_poInitCost._maxDz);
    _D_data2Opt.setDzMax(_poInitCost._maxDz);

    ResetIdBuffer();
    SetPreComp(true);

    SetProgress(NBDir);

    _H_data2Opt.ReallocParam(size);
    _D_data2Opt.ReallocParam(size);    
    _D_data2Opt.setPenteMax(penteMax);
    _D_data2Opt.setZReg(zReg);
    _D_data2Opt.setZRegQuad(zRegQuad);
    _D_data2Opt.setCostDefMasked(costDefMask);
    _D_data2Opt.setCostTransMaskNoMask(costDefMaskTrans);
    _D_data2Opt.setHasMaskAuto(hasMaskAuto);

    _FinalDefCor.Fill(0);
    _preFinalCost1D.Fill(0);

}

void InterfOptimizGpGpu::optimisation()
{
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation ENTER idBuf=%d nbLines=%u\n",
            (int)GetIdBuf(), (unsigned)_H_data2Opt.nbLines());

    _D_data2Opt.SetNbLine(_H_data2Opt.nbLines());

    _D_data2Opt.setPenteMax(_H_data2Opt.penteMax());

    _H_data2Opt.ReallocOutputIf(_H_data2Opt.s_InitCostVol().GetSize(),_H_data2Opt.s_Index().GetSize(),GetIdBuf());

    _D_data2Opt.ReallocIf(_H_data2Opt);

    const bool pipe = gpgpu_pipeline::PipelineEnabled();

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation CopyHostToDevice BEGIN async=%d\n",
                    pipe ? 1 : 0);
    if (pipe)
        _D_data2Opt.CopyHostToDeviceASync(_H_data2Opt, GetIdBuf(), _optStream);
    else
        _D_data2Opt.CopyHostToDevice(_H_data2Opt,GetIdBuf());
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation CopyHostToDevice END\n");

    SetPreComp(true);

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation Gpu_OptimisationOneDirection BEGIN\n");
    Gpu_OptimisationOneDirection(_D_data2Opt, _optStream);

    // Phase E: while optim kernel runs on stream, allow host to prep next box.
    if (pipe && g_optimPrefetchFn)
    {
        GPGPU_DIAG_MIN("[GPGPU][PREFETCH] host next-box prep during optim stream\n");
        g_optimPrefetchFn(g_optimPrefetchCtx);
    }

    if (pipe && !GpgpuDiagFull())
    {
        cudaError_t err = cudaStreamSynchronize(_optStream);
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR after optim stream sync: %s\n",
                    cudaGetErrorString(err));
        else
            GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation END (stream sync ok)\n");
    }
    else
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR after Gpu_OptimisationOneDirection: %s\n",
                    cudaGetErrorString(err));
        else
            GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation Gpu_OptimisationOneDirection END (sync ok)\n");
    }
    gpgpu_auto::SampleGpuNow();

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation CopyDevicetoHost BEGIN async=%d\n",
                    pipe ? 1 : 0);
    if (pipe)
    {
        _D_data2Opt.CopyDevicetoHostASync(_H_data2Opt, GetIdBuf(), _optStream);
        cudaError_t err = cudaStreamSynchronize(_optStream);
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR after optim D2H stream sync: %s\n",
                    cudaGetErrorString(err));
    }
    else
        _D_data2Opt.CopyDevicetoHost(_H_data2Opt,GetIdBuf());
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation EXIT\n");
}

void InterfOptimizGpGpu::simpleWork()
{
    optimisation();
}

void InterfOptimizGpGpu::freezeCompute()
{
    _H_data2Opt.setNbLines(0);
    _D_data2Opt.setNbLines(0);

    SetDataToCopy(false);
    SetCompute(false);
    SetPreComp(false);
}
