#include "GpGpu/GpGpu_ParamCorrelation.cuh"

// dilateKernel historically sampled a never-bound textureReference (dead/broken).
// CUDA 12 path: compile-safe stub that produces zeros (API kept for linkage).

__global__ void dilateKernel(pixel* dataOut, int r, uint2 dim, uint2 dimH)
{
	(void)r;
	(void)dim;
	const int2 ptH = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
	                           blockIdx.y * blockDim.y + threadIdx.y);
	if (ptH.x >= (int)dimH.x || ptH.y >= (int)dimH.y)
		return;
	dataOut[to1D(make_uint2(ptH.x, ptH.y), dimH)] = 0;
}

extern "C" void dilateKernel(pixel* HostDataOut, short r, uint2 dim)
{
	(void)r;
	dim3	threads( BLOCKDIM, BLOCKDIM, 1);
	uint2	thd2D		= make_uint2(threads);
	uint2	block2D		= iDivUp(dim, thd2D);
	dim3	blocks(block2D.x , block2D.y,1);

	CuDeviceData2D<pixel> deviceDataOut;

	uint2 dimDM = dim + 2 * (uint)r;

	deviceDataOut.Realloc(dimDM);
	deviceDataOut.Memset(0);

	dilateKernel<<<blocks,threads>>>(deviceDataOut.pData(),r,dim,dimDM);
	getLastCudaError("DilateX kernel failed");

    deviceDataOut.DecoratorDeviceData::CopyDevicetoHost(HostDataOut);
	deviceDataOut.Dealloc();

}
