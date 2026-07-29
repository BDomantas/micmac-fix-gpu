#include <stdio.h> // RUNPOD_GPGPU_DIAG
#include <cuda_runtime.h> // RUNPOD_GPGPU_DIAG
#include "GpGpu/GpGpu_InterOptimisation.h"
#include "GpGpu/GpGpu_Diag.h"

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
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation ENTER idBuf=%d nbLines=%u\n",
            (int)GetIdBuf(), (unsigned)_H_data2Opt.nbLines());

    _D_data2Opt.SetNbLine(_H_data2Opt.nbLines());

    _D_data2Opt.setPenteMax(_H_data2Opt.penteMax());

    _H_data2Opt.ReallocOutputIf(_H_data2Opt.s_InitCostVol().GetSize(),_H_data2Opt.s_Index().GetSize(),GetIdBuf());

    _D_data2Opt.ReallocIf(_H_data2Opt);

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation CopyHostToDevice BEGIN\n");
    _D_data2Opt.CopyHostToDevice(_H_data2Opt,GetIdBuf());
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation CopyHostToDevice END\n");

    SetPreComp(true);

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation Gpu_OptimisationOneDirection BEGIN\n");
    Gpu_OptimisationOneDirection(_D_data2Opt);
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR after Gpu_OptimisationOneDirection: %s\n",
                    cudaGetErrorString(err));
        else
            GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation Gpu_OptimisationOneDirection END (sync ok)\n");
    }

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] optimisation CopyDevicetoHost BEGIN\n");
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


