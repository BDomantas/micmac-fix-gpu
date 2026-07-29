// DEAD_CODE_NOT_IN_CMAKE: legacy MICMAC/Cuda path; not linked in GpGpu build.
// Phase C active correl uses GpGpu_Cuda_Correlation.cu texture objects.
// Kept for reference; simpleProjectionObj below is stream-safe if ever re-enabled.

__device__  inline float2 simpleProjection( uint2 size, uint2 ssize/*, uint2 sizeImg*/ ,uint2 coord, int L)
{
	const float2 cf		= make_float2(ssize) * make_float2(coord) / make_float2(size) ;
	const int2	 a		= make_int2(cf);
	const float2 uva	= (make_float2(a) + 0.5f) / (make_float2(ssize));
	const float2 uvb	= (make_float2(a+1) + 0.5f) / (make_float2(ssize));
	float2 ra, rb, Iaa;

	ra	= tex2DLayered( TexLay_Proj, uva.x, uva.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uva.y, L);
	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(cH.badVig);

	Iaa	= ((float)(a.x + 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra	= tex2DLayered( TexLay_Proj, uva.x, uvb.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uvb.y, L);

	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(cH.badVig);

	ra	= ((float)(a.x+ 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra = ((float)(a.y+ 1.0f) - cf.y) * Iaa + (cf.y - (float)(a.y)) * ra;
	/*ra = (ra + 0.5f) / (make_float2(sizeImg));*/

	return ra;
}


/*

Eviter les divergences dans la multi-corre

	__syncthreads();

	if ( mZ != 0 ) return;
	if (dev_NbImgOk[idN]<2) return;

	if (ptTer.x > sRVOK.pt0.x && ptTer.y > sRVOK.pt0.y && ptTer.x < sRVOK.pt1.x && ptTer.y < sRVOK.pt1.y)
		return;

	if (sRVOK.pt0.x != atomicMin(&(sRVOK.pt0.x),ptTer.x));
		atomicMin(&(rGVok->pt0.x),(int)sRVOK.pt0.x);

	if (sRVOK.pt0.y != atomicMin(&(sRVOK.pt0.y),ptTer.y));
		atomicMin(&(rGVok->pt0.y),(int)sRVOK.pt0.y);

	if (sRVOK.pt1.x != atomicMax(&(sRVOK.pt1.x),ptTer.x));
		atomicMax(&(rGVok->pt1.x),(int)sRVOK.pt1.x);

	if (sRVOK.pt1.y != atomicMax(&(sRVOK.pt1.y),ptTer.y));
		atomicMax(&(rGVok->pt1.y),(int)sRVOK.pt1.y);
		
		
*/

// Stream-safe variant (texture object) — preferred if this file is re-linked.
__device__ inline float2 simpleProjectionObj(cudaTextureObject_t texProj, uint2 size, uint2 ssize, uint2 coord, int L)
{
	const float2 cf		= make_float2(ssize) * make_float2(coord) / make_float2(size) ;
	const int2	 a		= make_int2(cf);
	const float2 uva	= (make_float2(a) + 0.5f) / (make_float2(ssize));
	const float2 uvb	= (make_float2(a+1) + 0.5f) / (make_float2(ssize));
	float2 ra, rb, Iaa;

	ra	= tex2DLayered<float2>(texProj, uva.x, uva.y, L);
	rb	= tex2DLayered<float2>(texProj, uvb.x, uva.y, L);
	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(-1.0f, -1.0f);

	Iaa	= ((float)(a.x + 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra	= tex2DLayered<float2>(texProj, uva.x, uvb.y, L);
	rb	= tex2DLayered<float2>(texProj, uvb.x, uvb.y, L);

	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(-1.0f, -1.0f);

	ra	= ((float)(a.x+ 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra = ((float)(a.y+ 1.0f) - cf.y) * Iaa + (cf.y - (float)(a.y)) * ra;
	return ra;
}

