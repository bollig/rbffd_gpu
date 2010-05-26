#ifndef _SCAN_WORKEFFICIENT_KERNEL_8_H_
#define _SCAN_WORKEFFICIENT_KERNEL_8_H_
#
#include "local_macros.h"

#define TIDX (__mul24(blockIdx.x,blockDim.x) + threadIdx.x)
#define TIDY (__mul24(blockIdx.y,blockDim.y) + threadIdx.y)
#define TWIDTH  (__mul24(gridDim.x,blockDim.x))
#define THEIGHT (__mul24(gridDim.y,blockDim.y))
#define ArrayID (TIDY*TWIDTH+TIDX)
#define MAKE_FLOAT4(arg) make_float4((arg), (arg), (arg), (arg))
#define MAKE_INT4(arg) make_int4((arg).x, (arg).y, (arg).z, (arg).w);

// Written by NVidia
// Modified by Gordon Erlebacher, Feb. 21, 2008

//----------------------------------------------------------------------
__global__ void scan_workefficient_8(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width)
// Version working with float4's again. Using floats is rather difficult
// hardcoded for 16x16 tiles (minimum size), with 256 threads per block
{
// edge=16, 64 threads: scan each row, one float per thread

    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float4 temp[];

	int numThreads = blockDim.x * blockDim.y;

	//if (blockIdx.x != 11) return; // block 13 has serial errors

	#if 1
	int last_share = edge*edge + ((edge*edge) >> LOG_NUM_BANKS);
	if (threadIdx.x == 0) {
		int4 ss = seeds[blockIdx.x];
		temp[last_share] = make_float4(ss.x+0.1,ss.y+0.1,ss.z+0.1,ss.w+0.1);
		//TMP(last_share) = make_float4(ss.x+0.1,ss.y+0.1,ss.z+0.1,ss.w+0.1);
	}
	__syncthreads();
	int4 seed = MAKE_INT4(temp[last_share]); // is int correct? Or must add 0.5?
	__syncthreads();
	#endif


	// get data from global memory (should be coalesced)

	int x = seed.x; 
	int y = seed.y;

	// edge should be part of the seed datastructure (per block)
	int edge2 = edge >> 2;
	int xorig = x - edge2; // one thread per float (edge float4's)
	int yorig = y - edge2; // edge rows

	// align xorig such that xorig is a multiple of 2 (speedup is evident)
	int shift = xorig - ((xorig >> 2) << 2);
	if (shift == 1) xorig -= 1;
	else if (shift == 2) xorig += 2;
	else if (shift == 3) xorig += 1;


	int flag1;
	float widthi = 1./width;

	int WW = width;  // array width (argument) (in float4)
	int HH = WW; // height of flat texture // MUST READ AS ARGUMENT

	int thread_id = threadIdx.x + blockDim.x * threadIdx.y;

	//--------------------
	// one iteration per row in the square tile
	// process 4 rows at a time

	__syncthreads();

		flag1 = 1;

		int xid = xorig + threadIdx.x; // xorig + 0...15
		if (xid < 0 || xid >= WW) flag1 = 0; 

		int yid1 = yorig + threadIdx.y;
		if (yid1 < 0 || yid1 >= HH) flag1 = 0;

 		int arrayid = xid + yid1*WW; 
		//temp[thread_id].x = float(xid*widthi);
		TMP(thread_id).x = float(xid*widthi);
		//temp[thread_id].y = float(yid1*widthi);
		TMP(thread_id).y = float(yid1*widthi);

		#if 1
		if (flag1 == 0) {
			//temp[thread_id] = make_float4(0.,0.,0.,0.);
			TMP(thread_id) = make_float4(0.,0.,0.,0.);
		} else {
			float f = g_idata[arrayid].w; // ERROR
			if (int(f+.1) != seed.w) {
				//temp[thread_id] = make_float4(0.,0.,0.,0.);
				TMP(thread_id) = make_float4(0.,0.,0.,0.);
			} else {
				//temp[thread_id].w = 1.0;
				TMP(thread_id).w = 1.0;
			}
		}
		#endif

	__syncthreads();


	#if 0
		sum[thread_id] = temp[thread_id];
		return;
	#endif

//	//--------------------
//
//// xorig - edge/2, xorig + edge/2 - 1
//
//// For the 16x16 case (hardcoded), the first pass with 64 threads can 
//// only handle 1/2 the domain (1024 floats = 16x16x4). The for loop that
//// follows had a thread handle two floats at a time, so can only handl
//// 1/2 the domain on each pass
//// manually treat each half of the domain

    int offset = 1;
	int ai, bi;
	int sz;

	#if 1
	sz = 128;  // hardcoded for 16x16=256 tile
    // build the sum in place up the tree
    for (int d = sz; d > 0; d >>= 1) {
        __syncthreads();

        if (thread_id < d) {
            ai = offset*(2*thread_id+1)-1;
            bi = offset*(2*thread_id+2)-1;

			//temp[bi].x += temp[ai].x;
			//temp[bi].y += temp[ai].y;
			//temp[bi].w += temp[ai].w;

			TMP(bi).x += TMP(ai).x;
			TMP(bi).y += TMP(ai).y;
			TMP(bi).w += TMP(ai).w;
        }

        offset <<= 1;
    }
	#endif

	__syncthreads();
	#if 0
		sum[thread_id] = temp[thread_id];
	return;
	#endif

	// Something wrong with the results

    // write results to global memory
    __syncthreads();
	if (threadIdx.x == (blockDim.x-1)) {
		int el = edge*edge-1;
		//float nbs = temp[el].w;
		float nbs = TMP(el).w;
		float nbs1 = 1./(nbs);
		if (nbs <= 0.5) nbs1 = 1.;
		//sum[blockIdx.x] = make_float4(temp[el].x*nbs1, temp[el].y*nbs1, 0., nbs); // orig
		sum[blockIdx.x] = make_float4(TMP(el).x*nbs1, TMP(el).y*nbs1, 0., nbs); // orig
	}
}
//----------------------------------------------------------------------
__global__ void scan_workefficient_8_larger(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width)
// Version working with float4's again. Using floats is rather difficult
// hardcoded for 16x16 tiles (minimum size), with 256 threads per block
{
// edge=16, 64 threads: scan each row, one float per thread

    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float4 temp[];

	int numThreads = blockDim.x * blockDim.y;

	//if (blockIdx.x != 11) return; // block 13 has serial errors

	#if 1
	int last_share = edge*edge + ((edge*edge) >> LOG_NUM_BANKS);
	if (threadIdx.x == 0) {
		int4 ss = seeds[blockIdx.x];
		temp[last_share] = make_float4(ss.x+0.1,ss.y+0.1,ss.z+0.1,ss.w+0.1);
		//TMP(last_share) = make_float4(ss.x+0.1,ss.y+0.1,ss.z+0.1,ss.w+0.1);
	}
	__syncthreads();
	int4 seed = MAKE_INT4(temp[last_share]); // is int correct? Or must add 0.5?
	__syncthreads();
	#endif


	// get data from global memory (should be coalesced)

	int x = seed.x; 
	int y = seed.y;

	// edge should be part of the seed datastructure (per block)
	int edge2 = edge >> 2;
	int xorig = x - edge; // one thread per float (edge float4's)
	int yorig = y - edge; // edge rows

	// align xorig such that xorig is a multiple of 2 (speedup is evident)
	int shift = xorig - ((xorig >> 2) << 2);
	if (shift == 1) xorig -= 1;
	else if (shift == 2) xorig += 2;
	else if (shift == 3) xorig += 1;


	int flag1;
	float widthi = 1./width;

	int WW = width;  // array width (argument) (in float4)
	int HH = WW; // height of flat texture // MUST READ AS ARGUMENT

	int thread_id = threadIdx.x + blockDim.x * threadIdx.y;

	//--------------------
	// one iteration per row in the square tile
	// process 4 rows at a time

	__syncthreads();

		flag1 = 1;

		int xid = xorig + (threadIdx.x << 2); // xorig + 0...15
		if (xid < 0 || xid >= WW) flag1 = 0; 

		int yid1 = yorig + (threadIdx.y << 2);
		if (yid1 < 0 || yid1 >= HH) flag1 = 0;

 		int arrayid = xid + yid1*WW; 
		TMP(thread_id).x = float(xid*widthi);
		TMP(thread_id).y = float(yid1*widthi);

		#if 1
		if (flag1 == 0) {
			TMP(thread_id) = make_float4(0.,0.,0.,0.);
		} else {
			float f = g_idata[arrayid].w; // ERROR
			if (int(f+.1) != seed.w) {
				TMP(thread_id) = make_float4(0.,0.,0.,0.);
			} else {
				TMP(thread_id).w = 1.0;
			}
		}
		#endif

	__syncthreads();


	#if 0
		sum[thread_id] = temp[thread_id];
		return;
	#endif

//	//--------------------
//
//// xorig - edge/2, xorig + edge/2 - 1
//
//// For the 16x16 case (hardcoded), the first pass with 64 threads can 
//// only handle 1/2 the domain (1024 floats = 16x16x4). The for loop that
//// follows had a thread handle two floats at a time, so can only handl
//// 1/2 the domain on each pass
//// manually treat each half of the domain

    int offset = 1;
	int ai, bi;
	int sz;

	#if 1
	sz = 128;  // hardcoded for 16x16=256 tile
    // build the sum in place up the tree
    for (int d = sz; d > 0; d >>= 1) {
        __syncthreads();

        if (thread_id < d) {
            ai = offset*(2*thread_id+1)-1;
            bi = offset*(2*thread_id+2)-1;

			//temp[bi].x += temp[ai].x;
			//temp[bi].y += temp[ai].y;
			//temp[bi].w += temp[ai].w;

			TMP(bi).x += TMP(ai).x;
			TMP(bi).y += TMP(ai).y;
			TMP(bi).w += TMP(ai).w;
        }

        offset <<= 1;
    }
	#endif

	__syncthreads();
	#if 0
		sum[thread_id] = temp[thread_id];
	return;
	#endif

	// Something wrong with the results

    // write results to global memory
    __syncthreads();
	if (threadIdx.x == (blockDim.x-1)) {
		int el = edge*edge-1;
		//float nbs = temp[el].w;
		float nbs = TMP(el).w;
		float nbs1 = 1./(nbs);
		if (nbs <= 0.5) nbs1 = 1.;
		//sum[blockIdx.x] = make_float4(temp[el].x*nbs1, temp[el].y*nbs1, 0., nbs); // orig
		sum[blockIdx.x] = make_float4(TMP(el).x*nbs1, TMP(el).y*nbs1, 0., nbs); // orig
	}
}
//----------------------------------------------------------------------

#endif
