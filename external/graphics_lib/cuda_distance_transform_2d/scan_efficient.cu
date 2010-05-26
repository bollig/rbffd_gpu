
#ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
#define _SCAN_WORKEFFICIENT_KERNEL_H_
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
__global__ void scan_workefficient_2(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width)
{
    // Dynamically allocated shared memory for scan kernels
#if 1
    extern  __shared__  float4 temp[];
	float4 zero = make_float4(0.,0.,0.,0.);

	//if (blockIdx.x != 2) return;

	int numThreads = blockDim.x * blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int thid = threadIdx.x + blockDim.x * threadIdx.y;
	temp[2*thid] = zero;
	temp[2*thid+1] = zero;


	int blockId = blockIdx.x;
	int4& seed = *(seeds+blockId);

	// compute 2D flat texture coordinate from 3D seed coordinate
	int x = seed.x; 
	int y = seed.y;

	// edge should be part of the seed datastructure (per block)
	int edge2 = edge >> 1;
	int xorig = x - edge2;
	int yorig = y - edge2;
	int xid  = xorig + threadIdx.x; // 2 elements per thread
	int yid1 = yorig + threadIdx.y;
	int yid2 = yorig + edge - 1 - threadIdx.y;
#endif

	int flag  = 1;
	int flag1 = 1;
	int flag2 = 1;


	int WW = width;  // array width (argument)
	if (xid < 0 || xid >= WW) flag = 0;
	int HH = WW; // height of flat texture // MUST READ AS ARGUMENT
	if (yid1 < 0 || yid1 >= HH) flag1 = 0;
	if (yid2 < 0 || yid2 >= HH) flag2 = 0;

	int arrayid1 = xid + yid1 * WW;
	int arrayid2 = xid + yid2 * WW;

    __syncthreads();


	// the data can be in arbitrary order in the shared array

	float4 f;
	//flag  = 1;
	//flag1 = 1;
	//flag2 = 1;

	if (flag == 1 && flag1 == 1) {
		f = g_idata[arrayid1];
	
		if (int(f.w) == seed.w)
		{
			f.x = xid;
			f.y = yid1;
			f.w = 1.;
			temp[2*thid] = f;
		}
	}

	if (flag == 1 && flag2 == 1) {
		f = g_idata[arrayid2];
		if (int(f.w) == seed.w)
		{
			f.x = xid;
			f.y = yid2;
			f.w = 1.;
			temp[2*thid+1] = f;
		}
	}

#if 0
	__syncthreads();
	g_idata[2*thid] = temp[2*thid];
	g_idata[2*thid+1] = temp[2*thid+1];
	//g_idata[2*thid] = make_float4(arrayid1, arrayid2, 1,1);
	//g_idata[2*thid+1] = make_float4(xid,yid1,arrayid1,WW); // ok
	//g_idata[2*thid+1] = make_float4(f.w,seed.w,0,0);
	//g_idata[2*thid] = (g_idata[arrayid1]);
	//g_idata[2*thid+1] = (g_idata[arrayid2]);
	//g_idata[2*thid] = make_float4(seed.x,seed.y,seed.z,seed.w);
	//g_idata[2*thid+1] = make_float4(seed.x,seed.y,seed.z,seed.w);
	return;
#endif

    int offset = 1;

#if 1

// xorig - edge/2, xorig + edge/2 - 1

#if 1
	#if 1
    // build the sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            temp[bi].x += temp[ai].x;
            temp[bi].y += temp[ai].y;
            temp[bi].z += temp[ai].z;
            temp[bi].w += temp[ai].w;
        }

        offset <<= 1;
    }
	#endif
	#
	// Something wrong with the results

    // write results to global memory
    __syncthreads();
	if (thid == (numThreads-1)) {
		float nbs = temp[n-1].w;
		float nbs1 = 1./(nbs*width);
		if (nbs == 0) nbs = 1.;
		sum[blockId] = make_float4(temp[n-1].x*nbs1, temp[n-1].y*nbs1, 0., nbs); //, nbs);
	}
#endif
#endif
}
//----------------------------------------------------------------------
// More efficient version of scan_workefficient_2 (more threads + remove non-coalesced reads)
__global__ void scan_workefficient_3(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float4 temp[];
	float* tempf = (float*) temp;

	//if (blockIdx.x != 2) return;

//	float* g_idata_f = (float*) g_idata;
	//float f1 = g_idata_f[0];

	int numThreads = blockDim.x * blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int thid = threadIdx.x + blockDim.x * threadIdx.y;

	// get data from global memory (should be coalesced)
	int thid2 = thid<<1;

	int blockId = blockIdx.x;

	int4 seed;

	if (thid == 0) {
		seed = *(seeds+blockId);
		temp[numThreads*2+1] = make_float4(seed.x,seed.y,seed.z,seed.w);
	}
	__syncthreads();
	seed = MAKE_INT4(temp[numThreads*2+1]);
	//int4 seed = make_int4(100,40,0,5);

	// compute 2D flat texture coordinate from 3D seed coordinate
	int x = seed.x; 
	int y = seed.y;

	// edge should be part of the seed datastructure (per block)
	int edge2 = edge >> 1;
	int xorig = x - edge2;
	int yorig = y - edge2;

	int flag1;

	int xid  = xorig + threadIdx.x; // 2 elements per thread
	int WW = width;  // array width (argument)
	int HH = WW; // height of flat texture // MUST READ AS ARGUMENT

	//--------------------
	for (int j=0; j < 1; j++) { // the loop added 2 registers (could be unrolled)
		__syncthreads();

		flag1 = 1;

		// need for each of the strings separately
		if (xid < 0 || xid >= WW) {
			flag1 = 0; 
		}

		temp[thid+j*numThreads] = g_idata[j*numThreads+thid];
		temp[thid+j*numThreads].w = 1.;

		int yid1 = yorig + threadIdx.y + j*numThreads;
		if (yid1 < 0 || yid1 >= HH) flag1 = 0;

 		int arrayid1 = xid + yid1 * WW;
    	__syncthreads();

	// the data can be in arbitrary order in the shared array

	//    CREATES uncoalesced  loads (HOW POSSIBLE?)
	// 1.1 ms if if statement is commented out
	// 1.8 ms if if statement is not commented out

	//return;

		if (flag1 == 0) {
			// creates incoherent loads
			temp[thid] = make_float4(0.,0.,0.,0.); 
		}

	} // end of for loop

	//return;
	//--------------------

    int offset = 1;

// xorig - edge/2, xorig + edge/2 - 1

    // build the sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;


			#if 1
            tempf[bi] += tempf[ai];
            tempf[bi+numThreads] += tempf[ai+numThreads];
            tempf[bi+numThreads << 1] += tempf[ai+numThreads << 1];
            tempf[bi+numThreads << 1 + numThreads] += tempf[ai+numThreads << 1 + numThreads];
			#endif
			#
			#if 0
            temp[bi].x += temp[ai].x;
            temp[bi].y += temp[ai].y;
            temp[bi].z += temp[ai].z;
            temp[bi].w += temp[ai].w;
			#endif
        }

        offset <<= 1;
    }

	// Something wrong with the results

    // write results to global memory
    __syncthreads();
	if (thid == (numThreads-1)) {
		float nbs = temp[n-1].w;
		float nbs1 = 1./(nbs*width);
		if (nbs == 0) nbs = 1.;
		sum[blockId] = make_float4(temp[n-1].x*nbs1, temp[n-1].y*nbs1, 0., nbs); //, nbs);
	}
}
//----------------------------------------------------------------------
// More efficient version of scan_workefficient_2 (more threads + remove non-coalesced reads)
__global__ void scan_test_incoherent(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float4 temp[];

	//if (blockIdx.x != 2) return;

	//float* g_idata_f = (float*) g_idata;

	int numThreads = blockDim.x * blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int thid = threadIdx.x + blockDim.x * threadIdx.y;


	temp[thid] = make_float4(0.,0.,0.,0.); 

	return;
}
//----------------------------------------------------------------------
// More efficient version of scan_workefficient_2 (more threads + remove non-coalesced reads)
// Use more threads by reading floats instead of float4
__global__ void scan_workefficient_4(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float4 temp[];
	float* tempf = (float*) temp;

	//if (blockIdx.x != 2) return;

	float* g_idata_f = (float*) g_idata;
	//float f1 = g_idata_f[0];

	// blockDim.x == edge (will generalize later)
	int numThreads = blockDim.x * blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int thid = threadIdx.x + blockDim.x * threadIdx.y;

	// get data from global memory (should be coalesced)
	int thid2 = thid<<1;

	int blockId = blockIdx.x;
	int4& seed = *(seeds+blockId);

	// compute 2D flat texture coordinate from 3D seed coordinate
	int x = seed.x; 
	int y = seed.y;

	// edge should be part of the seed datastructure (per block)
	int edge2 = edge >> 1;
	int xorig = x - edge2;
	int yorig = y - edge2;


	int flag1;

	int xid  = xorig + threadIdx.x; // 2 elements per thread
	int WW = width;  // array width (argument)
	int HH = WW; // height of flat texture // MUST READ AS ARGUMENT

	//--------------------
	int j = 0;
	//for (int j=0; j < 1; j++) { //} the loop added 2 registers (could be unrolled)
		__syncthreads();

		flag1 = 1;

		// need for each of the strings separately
		if (xid < 0 || xid >= WW) {
			flag1 = 0; 
		}

		int yid1 = yorig + threadIdx.y + j*numThreads;
 		int arrayid1 = xid + yid1 * WW;

		// 16 x 16 threads, tile: 16x16 float4 ==> 64 x 16 floats
		// break threads: 32 x 8


		int tid = threadIdx.x + blockDim.x * threadIdx.y;
		// tid = 0 ==. array[0,0] // column-major (Fortran)
		// tid = 1 ==. array[1,0]
		// tid = 15 ==. array[15,0]
		// tid = 16 ==. array[16,0]
		// tid = 17 ==. array[17,0]
		int warp_base = tid >> 5; // divide by 32 // array row  [0,...,7]
		int thread_in_warp = tid - (warp_base << 5);  // [0,...,31]

		// warp 0:   array[0,0]  --> array[31,0]
		// warp 1:   array[32,0] --> array[63,0]
		// warp 2:   array[0,1]  --> array[31,1]
		// warp 3:   array[32,1] --> array[63,1]
		// warp 4:   array[0,2]  --> array[31,2]
		// warp 5:   array[32,2] --> array[63,2]
		// warp 6:   array[0,3]  --> array[31,3]
		// warp 7:   array[32,3] --> array[63,3] // 4th row

		// There are 16 rows in the array. Create a loop: 
		// for (int i=0; i < 4; i++) {
		//   warp 0:   array[0,i*4] --> array[31,i*4]
		//   warp 7:   array[32,i*4+3] --> array[63,i*4]

		// Eventually generalize to more arrays

		// two warps per row

		// arrayid = thread_in_warp + warp_base * WW;
		// tempf[warp_base] = g_idata[array_id]     // floats (64 per row)
		// temp[thid+numThreads] = g_idata[array_id]


	// I could be exceeding memory bounds. So how to read coalesced without
		temp[thid] = g_idata[arrayid1];
		temp[thid].w = 1.;

		if (yid1 < 0 || yid1 >= HH) flag1 = 0;

    	__syncthreads();

	// the data can be in arbitrary order in the shared array

	//    CREATES uncoalesced  loads (HOW POSSIBLE?)
	// 1.1 ms if if statement is commented out
	// 1.8 ms if if statement is not commented out

	//return;

		#if 1 
		if (flag1 == 0) {
			// creates incoherent loads
			temp[thid] = make_float4(0.,0.,0.,0.); 
		}
		#endif

	//} // end of for loop
	//--------------------

	//return;

    int offset = 1;

// xorig - edge/2, xorig + edge/2 - 1

    // build the sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            temp[bi].x += temp[ai].x;
            temp[bi].y += temp[ai].y;
            temp[bi].z += temp[ai].z;
            temp[bi].w += temp[ai].w;
        }

        offset <<= 1;
    }

	// Something wrong with the results

    // write results to global memory
    __syncthreads();
	if (thid == (numThreads-1)) {
		float nbs = temp[n-1].w;
		float nbs1 = 1./(nbs*width);
		if (nbs == 0) nbs = 1.;
		sum[blockId] = make_float4(temp[n-1].x*nbs1, temp[n-1].y*nbs1, 0., nbs); //, nbs);
	}
}
//----------------------------------------------------------------------
__global__ void scan_workefficient_5(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width)
// More efficient version of scan_workefficient_2 (more threads + remove non-coalesced reads)
// Use more threads by reading floats instead of float4
{
// edge=16, 64 threads: scan each row, one float per thread

    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float4 temp[];
	float* tempf = (float*) temp;
	float* sumf = (float*) sum;

	// blockDim.x == edge (will generalize later)
	int numThreads = blockDim.x * blockDim.y;

//	if (blockIdx.x > 1000) return;

	#if 1
	//  SOMETHING NOT WORKING
	int last_share = edge*edge + ((edge*edge) >> LOG_NUM_BANKS);
	//int last_share = 0;
	if (threadIdx.x == 0) {
		int4 ss = seeds[blockIdx.x];
		temp[last_share] = make_float4(ss.x+0.1,ss.y+0.1,ss.z+0.1,ss.w+0.1);
	}
	__syncthreads();
	int4 seed = MAKE_INT4(temp[last_share]); // is int correct? Or must add 0.5?
	__syncthreads();
	#endif
	

	#if 0
	//int4 seed = make_int4(8,8,0,311); // TEST SEED
	int4 seed = seeds[blockIdx.x];
	#endif

	float* g_idata_f = (float*) g_idata;

	// get data from global memory (should be coalesced)

	int x = seed.x; 
	int y = seed.y;


	// edge should be part of the seed datastructure (per block)
	int edge2 = edge >> 2;
	int xorig = x - edge2; // one thread per float (edge float4's)
	int yorig = y - edge2; // edge rows

	// align xorig such that xorig is a multiple of 2 (speedup is evident)
	//xorig = (xorig >> 1) << 1; // xorig is divisble by 2^1
	int shift = xorig - ((xorig >> 2) << 2);
	if (shift == 1) xorig -= 1;
	else if (shift == 2) xorig += 2;
	else if (shift == 3) xorig += 1;
	//else return;

	int flag1;

	int WW = width;  // array width (argument) (in float4)
	int HH = WW; // height of flat texture // MUST READ AS ARGUMENT
	int xid  = 4*xorig + threadIdx.x; // measured in floats

	//--------------------
	// one iteration per row in the square tile
	for (int j=0; j < edge; j++) { // the loop added 2 registers (could be unrolled)
		__syncthreads();

		flag1 = 1;

		// need for each of the strings separately
		if (xid < 0 || xid >= (WW*4)) flag1 = 0; 

		int yid1 = yorig + j;
		if (yid1 < 0 || yid1 >= HH) flag1 = 0;

 		int arrayid1 = xid + yid1*WW*4; // WW*4 floats

		// I MUST ALSO CHECK THE SEED VALUE


		tempf[j*4*edge+threadIdx.x] = 0.;

		// crashes without this test
		if (flag1 != 0) {
			tempf[j*4*edge+threadIdx.x] = g_idata_f[arrayid1];
			//tempf[j*4*edge+threadIdx.x] = 0.; // very low overhead
		}


		//sumf[j*4*edge+threadIdx.x] = g_idata_f[arrayid1];
		//sumf[j*4*edge+threadIdx.x] = tempf[j*4*edge+threadIdx.x];
		//sumf[j*4*edge+threadIdx.x] = tempf[j*4*edge+threadIdx.x];
		//sumf[j*4*edge+threadIdx.x] = xorig;

		//if (j == 0) {
			//sum[threadIdx.x] = make_float4(seed.x,seed.y,seed.z,seed.w);
			//return;
		//}

    	__syncthreads();

	} // end of for loop

	//return;

	__syncthreads();

	float widthi = 1./width;

	for (int j=0; j < edge; j++) {
		__syncthreads();
		if (threadIdx.x < edge) {
			float f = temp[j*edge+threadIdx.x].w + 0.1;  // so that int() works
			if (int(f) != seed.w) {
				temp[j*edge+threadIdx.x] = make_float4(0.,0.,0.,0.);
			} else {
				temp[j*edge+threadIdx.x].x = (xorig+threadIdx.x) * widthi;
				temp[j*edge+threadIdx.x].y = (yorig+j) * widthi;
				temp[j*edge+threadIdx.x].w = 1.;
			}
		}
	}
	__syncthreads();
	#if 0
	for (int j=0; j < edge; j++) {
		sumf[j*4*edge+threadIdx.x] = tempf[j*4*edge+threadIdx.x];
	}
	return;
	#endif

	#if 0
	for (int j=0; j < edge; j++) { // the loop added 2 registers (could be unrolled)
		//sumf[j*4*edge+threadIdx.x] = g_idata_f[arrayid1];
		sumf[j*4*edge+threadIdx.x] = tempf[j*4*edge+threadIdx.x];
		//sumf[j*4*edge+threadIdx.x] = xorig;
		//sum[threadIdx.x] = make_float4(seed.x,seed.y,seed.z,seed.w);
	}
	return;
	#endif
	//--------------------

// xorig - edge/2, xorig + edge/2 - 1

// For the 16x16 case (hardcoded), the first pass with 64 threads can 
// only handle 1/2 the domain (1024 floats = 16x16x4). The for loop that
// follows had a thread handle two floats at a time, so can only handl
// 1/2 the domain on each pass

// manually treat each half of the domain

    int offset = 1;
	//====

	int tid, j, ai, bi;
	int mx = 8;

			tid = threadIdx.x >> 2;
			j = threadIdx.x - (tid << 2);


	for (int outer=0; outer < 3; outer++) { // HARDCODED
		for (int k=0; k < mx; k++) {
			__syncthreads();
			int off = k * 128 * (1 << outer); // HARDCODED

			ai = offset*(2*tid+1)-1;
			bi = offset*(2*tid+2)-1;

			ai = (ai << 2) + j;
			bi = (bi << 2) + j;

			tempf[bi+off] += tempf[ai+off];
		}
		mx >> 1;
		offset <<= 1;
	}
	//====
	#if 0
	for (int j=0; j < edge; j++) { // the loop added 2 registers (could be unrolled)
		//sumf[j*4*edge+threadIdx.x] = g_idata_f[arrayid1];
		sumf[j*4*edge+threadIdx.x] = tempf[j*4*edge+threadIdx.x];
		//sumf[j*4*edge+threadIdx.x] = xorig;
		//sum[threadIdx.x] = make_float4(seed.x,seed.y,seed.z,seed.w);
	}
	return;
	#endif
	#


	int sz = edge*edge / 2; //  (128 for 64 threads)

	#if 1
    // build the sum in place up the tree
    for (int d = sz>>1; d > 0; d >>= 1) {
        __syncthreads();

        if (threadIdx.x < d)      
        {
			//int tid = threadIdx.x >> 2; // thread id divided by 4
			//int j = threadIdx.x - (tid << 2); // 0,1,2,3

            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

			ai = (ai << 2) + j;
			bi = (bi << 2) + j;

            tempf[bi] += tempf[ai];
        }

        offset <<= 1;
    }
	#endif
	#
	#if 0
	for (int j=0; j < edge; j++) { // the loop added 2 registers (could be unrolled)
		//sumf[j*4*edge+threadIdx.x] = g_idata_f[arrayid1];
		sumf[j*4*edge+threadIdx.x] = tempf[j*4*edge+threadIdx.x];
		//sumf[j*4*edge+threadIdx.x] = xorig;
		//sum[threadIdx.x] = make_float4(seed.x,seed.y,seed.z,seed.w);
	}
	//return;
	#endif

	// Something wrong with the results

    // write results to global memory
    __syncthreads();
	if (threadIdx.x == (numThreads-1)) {
		int el = edge*edge-1;
		//int el = 0;
		float nbs = temp[el].w;
		//float nbs1 = 1./(nbs*width);
		float nbs1 = 1./(nbs);
		if (nbs <= 0.5) nbs1 = 1.;
		//sum[blockIdx.x] = make_float4(temp[el].x*nbs1, temp[el].y*nbs1, width, nbs);
		sum[blockIdx.x] = make_float4(temp[el].x*nbs1, temp[el].y*nbs1, 0., nbs); // orig
	}
}
//----------------------------------------------------------------------

__global__ void scan_workefficient_6(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width)
// More efficient version of scan_workefficient_2 (more threads + remove non-coalesced reads)
// Use more threads by reading floats instead of float4
// Remove bank conflicts (decrease serialized_warps)
{
// edge=16, 64 threads: scan each row, one float per thread

    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float4 temp[];
	float* tempf = (float*) temp;
	float* sumf = (float*) sum;

	// blockDim.x == edge (will generalize later)
	int numThreads = blockDim.x * blockDim.y;

//	if (blockIdx.x > 1000) return;

	#if 1
	//  SOMETHING NOT WORKING
	int last_share = edge*edge + ((edge*edge) >> LOG_NUM_BANKS);
	//int last_share = 0;
	if (threadIdx.x == 0) {
		int4 ss = seeds[blockIdx.x];
		temp[last_share] = make_float4(ss.x+0.1,ss.y+0.1,ss.z+0.1,ss.w+0.1);
	}
	__syncthreads();
	int4 seed = MAKE_INT4(temp[last_share]); // is int correct? Or must add 0.5?
	__syncthreads();
	#endif
	

	#if 0
	//int4 seed = make_int4(8,8,0,311); // TEST SEED
	int4 seed = seeds[blockIdx.x];
	#endif

	float* g_idata_f = (float*) g_idata;

	// get data from global memory (should be coalesced)

	int x = seed.x; 
	int y = seed.y;


	// edge should be part of the seed datastructure (per block)
	int edge2 = edge >> 2;
	int xorig = x - edge2; // one thread per float (edge float4's)
	int yorig = y - edge2; // edge rows

	// align xorig such that xorig is a multiple of 2 (speedup is evident)
	//xorig = (xorig >> 1) << 1; // xorig is divisble by 2^1
	int shift = xorig - ((xorig >> 2) << 2);
	if (shift == 1) xorig -= 1;
	else if (shift == 2) xorig += 2;
	else if (shift == 3) xorig += 1;
	//else return;

	int flag1;

	int WW = width;  // array width (argument) (in float4)
	int HH = WW; // height of flat texture // MUST READ AS ARGUMENT
	int xid  = 4*xorig + threadIdx.x; // measured in floats

	//--------------------
	// one iteration per row in the square tile
	for (int j=0; j < edge; j++) { // the loop added 2 registers (could be unrolled)
		__syncthreads();

		flag1 = 1;

		// need for each of the strings separately
		if (xid < 0 || xid >= (WW*4)) flag1 = 0; 

		int yid1 = yorig + j;
		if (yid1 < 0 || yid1 >= HH) flag1 = 0;

 		int arrayid1 = xid + yid1*WW*4; // WW*4 floats

		// I MUST ALSO CHECK THE SEED VALUE


		TMPF(j*4*edge+threadIdx.x) = 0.;

		// crashes without this test
		if (flag1 != 0) {
			TMPF(j*4*edge+threadIdx.x) = g_idata_f[arrayid1];
			//tempf[j*4*edge+threadIdx.x] = 0.; // very low overhead
		}


		//sumf[j*4*edge+threadIdx.x] = g_idata_f[arrayid1];
		//sumf[j*4*edge+threadIdx.x] = tempf[j*4*edge+threadIdx.x];
		//sumf[j*4*edge+threadIdx.x] = tempf[j*4*edge+threadIdx.x];
		//sumf[j*4*edge+threadIdx.x] = xorig;

		//if (j == 0) {
			//sum[threadIdx.x] = make_float4(seed.x,seed.y,seed.z,seed.w);
			//return;
		//}

    	__syncthreads();

	} // end of for loop

	//return;

	__syncthreads();

	float widthi = 1./width;

	for (int j=0; j < edge; j++) {
		__syncthreads();
		if (threadIdx.x < edge) {
			float f = temp[j*edge+threadIdx.x].w + 0.1;  // so that int() works
			if (int(f) != seed.w) {
				temp[j*edge+threadIdx.x] = make_float4(0.,0.,0.,0.);
			} else {
				temp[j*edge+threadIdx.x].x = (xorig+threadIdx.x) * widthi;
				temp[j*edge+threadIdx.x].y = (yorig+j) * widthi;
				temp[j*edge+threadIdx.x].w = 1.;
			}
		}
	}
	__syncthreads();
	#if 0
	for (int j=0; j < edge; j++) {
		sumf[j*4*edge+threadIdx.x] = TMPF(j*4*edge+threadIdx.x);
	}
	return;
	#endif

	#if 0
	for (int j=0; j < edge; j++) { // the loop added 2 registers (could be unrolled)
		//sumf[j*4*edge+threadIdx.x] = g_idata_f[arrayid1];
		sumf[j*4*edge+threadIdx.x] = TMPF(j*4*edge+threadIdx.x);
		//sumf[j*4*edge+threadIdx.x] = xorig;
		//sum[threadIdx.x] = make_float4(seed.x,seed.y,seed.z,seed.w);
	}
	return;
	#endif
	//--------------------

// xorig - edge/2, xorig + edge/2 - 1

// For the 16x16 case (hardcoded), the first pass with 64 threads can 
// only handle 1/2 the domain (1024 floats = 16x16x4). The for loop that
// follows had a thread handle two floats at a time, so can only handl
// 1/2 the domain on each pass

// manually treat each half of the domain

    int offset = 1;
	//====

	int mx = 8;
	for (int outer=0; outer < 3; outer++) { // HARDCODED
		for (int k=0; k < mx; k++) {
			__syncthreads();
			int off = k * 128 * (1 << outer); // HARDCODED
			int tid = threadIdx.x >> 2;
			int j = threadIdx.x - (tid << 2);
		
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;

			ai = (ai << 2) + j;
			bi = (bi << 2) + j;

			TMPF(bi+off) += TMPF(ai+off);
		}
		mx >> 1;
		offset <<= 1;
	}
	//====
	#if 0
	for (int j=0; j < edge; j++) { // the loop added 2 registers (could be unrolled)
		//sumf[j*4*edge+threadIdx.x] = g_idata_f[arrayid1];
		sumf[j*4*edge+threadIdx.x] = TMPF(j*4*edge+threadIdx.x);
		//sumf[j*4*edge+threadIdx.x] = xorig;
		//sum[threadIdx.x] = make_float4(seed.x,seed.y,seed.z,seed.w);
	}
	return;
	#endif


	int sz = edge*edge / 2; //  (128 for 64 threads)

	#if 1
    // build the sum in place up the tree
    for (int d = sz>>1; d > 0; d >>= 1) {
        __syncthreads();

        if (threadIdx.x < d)      
        {
			int tid = threadIdx.x >> 2; // thread id divided by 4
			int j = threadIdx.x - (tid << 2); // 0,1,2,3

            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

			ai = (ai << 2) + j;
			bi = (bi << 2) + j;

            TMPF(bi) += TMPF(ai);
        }

        offset <<= 1;
    }
	#endif
	#
	#if 0
	for (int j=0; j < edge; j++) { // the loop added 2 registers (could be unrolled)
		//sumf[j*4*edge+threadIdx.x] = g_idata_f[arrayid1];
		sumf[j*4*edge+threadIdx.x] = TMPF(j*4*edge+threadIdx.x);
		//sumf[j*4*edge+threadIdx.x] = xorig;
		//sum[threadIdx.x] = make_float4(seed.x,seed.y,seed.z,seed.w);
	}
	return;
	#endif

	// Something wrong with the results

    // write results to global memory
    __syncthreads();
	if (threadIdx.x == (numThreads-1)) {
		int el = edge*edge-1;
		//int el = 0;
		float nbs = temp[el].w;
		//float nbs1 = 1./(nbs*width);
		float nbs1 = 1./(nbs);
		if (nbs <= 0.5) nbs1 = 1.;
		//sum[blockIdx.x] = make_float4(temp[el].x*nbs1, temp[el].y*nbs1, width, nbs);
		sum[blockIdx.x] = make_float4(temp[el].x*nbs1, temp[el].y*nbs1, 0., nbs); // orig
	}
}
//----------------------------------------------------------------------
__global__ void scan_workefficient_7(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width)
// More efficient version of scan_workefficient_2 (more threads + remove non-coalesced reads)
// Use more threads by reading floats instead of float4
{
// edge=16, 64 threads: scan each row, one float per thread

    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float4 temp[];
	float* tempf = (float*) temp;
	float* sumf = (float*) sum;

	// blockDim.x == edge (will generalize later)
	int numThreads = blockDim.x * blockDim.y;

	//if (blockIdx.x != 11) return; // block 13 has serial errors

	#if 1
	//  SOMETHING NOT WORKING
	int last_share = edge*edge + ((edge*edge) >> LOG_NUM_BANKS);
	//int last_share = 0;
	if (threadIdx.x == 0) {
		int4 ss = seeds[blockIdx.x];
		temp[last_share] = make_float4(ss.x+0.1,ss.y+0.1,ss.z+0.1,ss.w+0.1);
	}
	__syncthreads();
	int4 seed = MAKE_INT4(temp[last_share]); // is int correct? Or must add 0.5?
	__syncthreads();
	#endif

	#if 0
	//int4 seed = make_int4(8,8,0,311); // TEST SEED
	int4 seed = seeds[blockIdx.x];
	#endif

	float* g_idata_f = (float*) g_idata;

	// get data from global memory (should be coalesced)

	int x = seed.x; 
	int y = seed.y;

	// edge should be part of the seed datastructure (per block)
	int edge2 = edge >> 2;
	int xorig = x - edge2; // one thread per float (edge float4's)
	int yorig = y - edge2; // edge rows

	// align xorig such that xorig is a multiple of 2 (speedup is evident)
	//xorig = (xorig >> 1) << 1; // xorig is divisble by 2^1
	int shift = xorig - ((xorig >> 2) << 2);
	if (shift == 1) xorig -= 1;
	else if (shift == 2) xorig += 2;
	else if (shift == 3) xorig += 1;
	//else return;

	int flag1;

	int WW = width;  // array width (argument) (in float4)
	int HH = WW; // height of flat texture // MUST READ AS ARGUMENT

	//--------------------
	// one iteration per row in the square tile
	for (int j=0; j < edge; j+=4) { // the loop added 2 registers (could be unrolled)
		__syncthreads();
//
		int subtid = j >> 4; // 0, 1, ..., numThreads/4
		int subrow = j - subtid;
//
		flag1 = 1;
//
//		// need for each of the strings separately
		int xid  = 4*xorig + threadIdx.x; // measured in floats
		if (xid < 0 || xid >= (WW*4)) flag1 = 0; 
//
		int yid1 = yorig + j + threadIdx.y;
		if (yid1 < 0 || yid1 >= HH) flag1 = 0;
//
 		int arrayid1 = xid + yid1*WW*4; // WW*4 floats
//
//		// I MUST ALSO CHECK THE SEED VALUE
//
//
		int jj = j+threadIdx.y;
		tempf[jj*4*edge+threadIdx.x] = 0.;
//
//		// crashes without this test
		if (flag1 != 0) {
			tempf[jj*4*edge+threadIdx.x] = g_idata_f[arrayid1];
			//tempf[jj*4*edge+threadIdx.x] = 0.; // very low overhead
		}
//
//
//		//sumf[jj*4*edge+threadIdx.x] = g_idata_f[arrayid1];
//		//sumf[jj*4*edge+threadIdx.x] = tempf[jj*4*edge+threadIdx.x];
//		//sumf[jj*4*edge+threadIdx.x] = tempf[jj*4*edge+threadIdx.x];
//		//sumf[jj*4*edge+threadIdx.x] = xorig;
//
//		//if (j == 0) {
//			//sum[threadIdx.x] = make_float4(seed.x,seed.y,seed.z,seed.w);
//			//return;
//		//}
//
    	__syncthreads();

	} // end of for loop
//
//
	__syncthreads();
//
	float widthi = 1./width;
	int thread_id = threadIdx.x + blockDim.x * threadIdx.y;

	// use float4
	// not the problem
	#if 0
	// 256 threads
	int tid = thread_id >> 2;
	int j4 = thread_id - (tid << 2);

	int (int j=0; j < 4; j++) {
		__syncthreads();
			if (j4 == 3) {
				float f = tempf[j*256 + tid + j4] + 0.1;
				if (int(f) != seed.w) {
					//tempf[j*256+thread_id] = 0.;
					temp[j*64+threadIdx.x] = make_float4(0.,0.,0.,0.);
				}
			}
	}
	#endif

	// use float4
	// not the problem
	#if 1
	for (int j=0; j < edge; j++) {   // takes 1.7 ms
		__syncthreads();
		if (threadIdx.x < edge && threadIdx.y == 0) {
			int tid = threadIdx.x;
			float f = temp[j*16+tid].w + 0.1;  // so that int() works
			if (int(f) != seed.w) {
				temp[j*16+tid] = make_float4(0.,0.,0.,0.); // cause of serialization
			} // else {
				//float4 g; 
				// Will do this later
				//g.x = (xorig+threadIdx.x) * widthi;
				//g.y = (yorig+j) * widthi;
				//g.z = 0.;
				//g.w = 1.;
				//temp[j*edge+threadIdx.x] = g;
			// }
		}
	}
	#endif

	// use float
	#if 0
	for (int j=0; j < edge; j++) {   // takes 1.7 ms
		__syncthreads();
		if (threadIdx.x < edge && threadIdx.y == 0) {
			int tid = thread_id >> 2;
			int j = thread_id - (tid << 2);
			float f = tempf[4*j*16+tid+j] + 0.1;  // so that int() works
			if (int(f) != seed.w) {
				tempf[4*j*16+tid] = 0.; 
				tempf[4*j*16+tid+1] = 0.; 
				tempf[4*j*16+tid+2] = 0.; 
				tempf[4*j*16+tid+3] = 0.; 
			} else {
				float4 g; 
				// Will do this later
				//g.x = (xorig+threadIdx.x) * widthi;
				//g.y = (yorig+j) * widthi;
				//g.z = 0.;
				//g.w = 1.;
				//temp[jj*edge+threadIdx.x] = g;
			}
		}
	}
	#endif
	__syncthreads();

	//return;
	#if 0
	for (int j=0; j < 4; j++) { // the loop added 2 registers (could be unrolled)
		sumf[j*256+thread_id] = tempf[j*256+thread_id];
	}
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
//
//// manually treat each half of the domain
//
    int offset = 1;
//	//====
//
	int tid, j4, ai, bi;

//	for (int j=0; j < 4; j++) { // the loop added 2 registers (could be unrolled)
		//sumf[j*256+thread_id] = tempf[j*256+thread_id];
	//}
	//return;

	tid = thread_id >> 2;
	j4 = thread_id - (tid << 2);

		for (int k=0; k < 2; k++) {
			__syncthreads();
			int off = k * 512;

			ai = offset*(2*tid+1)-1;
			bi = offset*(2*tid+2)-1;

			ai = (ai << 2) + j4;
			bi = (bi << 2) + j4;

			tempf[bi+off] += tempf[ai+off];
		}
		offset <<= 1;
	//====

	__syncthreads();
	#if 0
	for (int j=0; j < 4; j++) { // the loop added 2 registers (could be unrolled)
		sumf[j*256+thread_id] = tempf[j*256+thread_id];
	}
	return;
	#endif
//
//
	int sz = 512; // * edge*edge; //  (512 for 256 threads)

	//return;

//
	#if 1
    // build the sum in place up the tree
    for (int d = sz>>3; d > 0; d >>= 1) {
        __syncthreads();

        if (thread_id < (d*4))       // 4 subthreads per thread: 64*4 = 256
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

			ai = (ai << 2) + j4;
			bi = (bi << 2) + j4;

            tempf[bi] += tempf[ai];
        }

        offset <<= 1;
		//if (d == 0) break;
    }
	#endif
	#if 0
	for (int j=0; j < 4; j++) { // the loop added 2 registers (could be unrolled)
		sumf[j*256+thread_id] = tempf[j*256+thread_id];
	}
	return;
	#endif

	// Something wrong with the results

    // write results to global memory
    __syncthreads();
	if (threadIdx.x == (blockDim.x-1)) {
		int el = edge*edge-1;
		//int el = 0;
		float nbs = temp[el].w;
		//float nbs1 = 1./(nbs*width);
		float nbs1 = 1./(nbs);
		if (nbs <= 0.5) nbs1 = 1.;
		//sum[blockIdx.x] = make_float4(temp[el].x*nbs1, temp[el].y*nbs1, width, nbs);
		sum[blockIdx.x] = make_float4(temp[el].x*nbs1, temp[el].y*nbs1, 0., nbs); // orig
	}
}
//----------------------------------------------------------------------

#endif // #ifndef _SCAN_WORKEFFICIENT_KERNEL_H_

