#include <stdio.h> // RUNPOD_GPGPU_DIAG
#include <cuda_runtime.h> // RUNPOD_GPGPU_DIAG
#include "GpGpu/GpGpu_InterOptimisation.h"

InterfOptimizGpGpu::InterfOptimizGpGpu()
{
    //CreateJob();

    freezeCompute();
}

InterfOptimizGpGpu::~InterfOptimizGpGpu(){}

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
    // RUNPOD_GPGPU_DIAG
    fprintf(stderr, "[GPGPU][%s] optimisation ENTER idBuf=%d nbLines=%u\n",
            "RUNPOD_GPGPU_DIAG", (int)GetIdBuf(), (unsigned)_H_data2Opt.nbLines());
    fflush(stderr);

    _D_data2Opt.SetNbLine(_H_data2Opt.nbLines());

    _D_data2Opt.setPenteMax(_H_data2Opt.penteMax());

    _H_data2Opt.ReallocOutputIf(_H_data2Opt.s_InitCostVol().GetSize(),_H_data2Opt.s_Index().GetSize(),GetIdBuf());

    _D_data2Opt.ReallocIf(_H_data2Opt);

    fprintf(stderr, "[GPGPU][%s] optimisation CopyHostToDevice BEGIN\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);
    _D_data2Opt.CopyHostToDevice(_H_data2Opt,GetIdBuf());
    fprintf(stderr, "[GPGPU][%s] optimisation CopyHostToDevice END\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);

    SetPreComp(true);

    fprintf(stderr, "[GPGPU][%s] optimisation Gpu_OptimisationOneDirection BEGIN\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);
    Gpu_OptimisationOneDirection(_D_data2Opt);
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            fprintf(stderr, "[GPGPU][%s] ERROR after Gpu_OptimisationOneDirection: %s\n",
                    "RUNPOD_GPGPU_DIAG", cudaGetErrorString(err));
        else
            fprintf(stderr, "[GPGPU][%s] optimisation Gpu_OptimisationOneDirection END (sync ok)\n", "RUNPOD_GPGPU_DIAG");
        fflush(stderr);
    }

    fprintf(stderr, "[GPGPU][%s] optimisation CopyDevicetoHost BEGIN\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);
    _D_data2Opt.CopyDevicetoHost(_H_data2Opt,GetIdBuf());
    fprintf(stderr, "[GPGPU][%s] optimisation EXIT\n", "RUNPOD_GPGPU_DIAG");
    fflush(stderr);
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


