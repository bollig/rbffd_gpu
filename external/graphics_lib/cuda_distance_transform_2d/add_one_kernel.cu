#ifndef _ADD_ONE_KERNEL_H_
#define _ADD_ONE_KERNEL_H_
#
#include <stdio.h>
//#include "scan_efficient.cu"
#include <vector>
#include <cutil.h>

#include "local_macros.h"


///////////////////////////////////////////////////////////////////////////////
// Work-efficient compute implementation of scan, one thread per 2 elements

__global__ void scan_workefficient(float4 *g_idata, float4 *sum, int n);
__global__ void scan_workefficient_2(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width);
__global__ void scan_workefficient_3(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width);
__global__ void scan_workefficient_4(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width);
__global__ void scan_workefficient_5(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width);
__global__ void scan_workefficient_6(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width);
__global__ void scan_workefficient_7(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width);
__global__ void scan_workefficient_8(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width);
__global__ void scan_workefficient_8_larger(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width);
__global__ void scan_test_incoherent(float4 *g_idata, float4 *sum, int4* seeds, int n, int edge, int width);

#define ATOMICS
//Comment this line and add '-arch sm_11' to the command line to enable atomic operations.
#undef ATOMICS

//----------------------------------------------------------------------
__global__ void
//transformKernel(float* g_odata, int width, int height)
addKernel(float* g_odata, float* o_data, int width, int height)
{
#if 1
    // calculate normalized texture coordinates
    //unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    //unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	//unsigned int indx = y*width + x;

    unsigned int x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	unsigned int indx = __mul24(y,width) + x;

	o_data[indx] = g_odata[indx] + 2.2f;
#endif
}
//----------------------------------------------------------------------
__global__ void
addArrayToArray(float* o_data, float* i1_data, float* i2_data, int width, int height)
{
    unsigned int x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	unsigned int indx = __mul24(y,width) + x;

#if 0
   __shared__ float blocki1[16*8];
   __shared__ float blocki2[16*8];

   unsigned int index_block = __mul24(threadIdx.y, blockDim.x) + threadIdx.x;
   blocki1[index_block] = i1_data[indx];
   blocki2[index_block] = i2_data[indx];
   __syncthreads();

   o_data[indx]  =  blocki1[index_block] + blocki2[index_block];
#else 
   o_data[indx]  =  i1_data[indx] + i2_data[indx];
#endif
}
//----------------------------------------------------------------------
#define nb_seeds    8*16
__global__ void
binEfficientKernel(int* data, int* bins, int width, int h, int nb_bins)
{
    unsigned int x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	unsigned int indx = __mul24(y,width) + x;

   // of course, in this test case, the integers are randomly distributed, 
   // but let us get an idea of efficiency anyway. 

   __shared__ int block[8][16];
   __shared__ int bins_s[nb_seeds]; // seeds 
   __shared__ int seed[nb_seeds]; // seeds 

	block[threadIdx.y][threadIdx.x] = data[indx];

	unsigned int loc = threadIdx.x + __mul24(threadIdx.y, blockDim.x);

	if (loc < nb_seeds) bins_s[loc] = 0;

	__syncthreads();

	unsigned int s = block[threadIdx.y][threadIdx.x];
    seed[s % nb_seeds] = s;

	// some of these threads are contending with each other. Is there a way
	// to lock shared memory?
    bins_s[s % nb_seeds] += 1;   // counter

	__syncthreads();

	 #if 0
	 if (loc < nb_seeds && bins_s[loc] > 0) {
	 	//bins[seed[loc]] += bins_s[loc];
	 	atomicAdd(bins+seed[loc], bins_s[loc]);
	 }
	 #endif
}
//----------------------------------------------------------------------
__global__ void
centroidVoronoiScanKernel(float4* data, float4* bins, int width, int h, int nb_bins, int4* newSeeds, int edge)
{
    unsigned int x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	unsigned int indx = __mul24(y,width) + x;

    // Dynamically allocated shared memory for scan kernels
    //extern  __shared__  float temp[];
}
//----------------------------------------------------------------------
__global__ void
centroidVoronoiKernel(float4* data, float4* bins, int width, int h, int nb_bins)
{
// Create one histogram per block. Make blocks do as much work as possible. 
// Follow by a histogram merge across blocks

// brute force appraoch
    unsigned int x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	unsigned int indx = __mul24(y,width) + x;

	// transform (x,y) into a seed
	float4 s = data[indx];
	//int sx = s.x * sz3; // hardcoded sz3d
	//int gx = 
	//int sy = s.y * sz3; // hardcoded sz3d
	//int sz = s.z * sz3; // hardcoded sz3d
	//int seed = sx + sz3*(sy + sz*sz3);
	int seed = s.w;

	//unsigned int ix = data[indx];
	int ix = seed;
	//atomicAdd(bins+ix, 1);
	//float4 yy = make_float4(1,1,1,1);
	float4 bb = bins[ix];
	bb.x = bb.x + s.x;
	bb.y = bb.y + s.y;
	bb.z = bb.z + s.z;
	bb.w = bb.w + 1.;
	bins[ix] = bb;   // Without atomic, I will have synchronization problems
	//bins[ix] = bins[ix] + yy;
}
//----------------------------------------------------------------------
__global__ void
binKernel(int* data, int* bins, int width, int h, int nb_bins)
{
    unsigned int x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	unsigned int indx = __mul24(y,width) + x;

	#if 0
	unsigned int ix = data[indx];
	atomicAdd(bins+ix, 1);
	//bins[ix] = bins[ix] + 1;
	#endif
}
//----------------------------------------------------------------------
#if 1
extern "C++"
void add(float* h, float* ho, int w, int he)
{
		addKernel<<<dim3(32,32), dim3(32,8)>>>(h, ho, w, he);
}
#endif
//----------------------------------------------------------------------
extern "C++"
void add(dim3& grid, dim3& block, float* h, float* ho, int w, int he)
{
		addKernel<<<grid, block>>>(h, ho, w, he);
}
//----------------------------------------------------------------------
extern "C++" void bins_call(dim3& grid , dim3& block, int* h, int* bins, int w, int he, int nb_bins) 
{
	binKernel<<<grid, block>>>(h, bins, w, he, nb_bins);
}
//----------------------------------------------------------------------
extern "C++" void bins_efficient_call(dim3& grid , dim3& block, int* h, int* bins, int w, int he, int nb_bins) 
{
	binEfficientKernel<<<grid, block>>>(h, bins, w, he, nb_bins);
}
//----------------------------------------------------------------------
extern "C++" void centroid_voronoi(dim3& grid , dim3& block, float4* h, float4* bins, int w, int he, 
    int nb_bins, int sz3d, std::vector<int4>& newSeeds, int edge) 
{
	edge = 16;
    const unsigned int shared_mem_size = sizeof(float4) * edge*edge; // square side: 2*edge
	printf(" shared mem size: %d\n", shared_mem_size);

	dim3 g(4,4,1);
	//dim3 b(edge/2, edge, 1);
	dim3 b(edge*edge/2, 1, 1);
	int nb_threads  = b.x * b.y * b.z;

	for (int i=0; i < newSeeds.size(); i++) {
		int4& s = newSeeds[i];
		printf("x,y,z,w= %d, %d, %d, %d\n", s.x, s.y, s.z, s.w);
	}
	exit(0);

	scan_workefficient<<<g, b, 2*shared_mem_size>>>(h, bins, nb_threads*2);
	//scan_workefficient<<<g, block>>>(h, bins, block.x*block.y*block.z);

	float4* h_h = (float4*) malloc(sizeof(float4)*nb_threads);
	float4* bins_h = (float4*) malloc(sizeof(float4)*nb_threads);

	cudaMemcpy(h_h, h, nb_threads*sizeof(float4), cudaMemcpyDeviceToHost);
	for (int i=0; i < 10; i++) {
		printf("h_h= %f, %f, %f, %f\n", h_h[i].x, h_h[i].y, h_h[i].z, h_h[i].w);
	}

	float tot = 0;
	for (int i=0; i < nb_threads; i++) {
		tot += h_h[i].w;
	}
	printf("total = %f\n", tot);


	printf("nb_threads= %d\n", nb_threads);
	printf("b size: %d\n", b.x*b.y*b.z);
	cudaMemcpy(bins_h, bins, nb_threads*sizeof(float4), cudaMemcpyDeviceToHost);
	//for (int j=0; j < nb_threads; j++) {
	for (int j=0; j < 1; j++) {
		printf("sum= %f, %f, %f, %f\n", bins_h[j].x, bins_h[j].y, bins_h[j].z, bins_h[j].w);
	}
	exit(0);
}
//----------------------------------------------------------------------
extern "C++" void centroid_voronoi_2(dim3& grid , dim3& block, float4* h, float4* bins, int w, int he, 
    int nb_bins, int sz3d, int4* newSeeds, int edge, int nbSeeds) 
{
	edge = 16; // tile 8 x 8 around seed: 0,1,2,3,4=seed,5,6,7  (seed-edge/2, seed+edge/2-1)

    const unsigned int shared_mem_size = sizeof(float4) * edge*edge + 4; // square side: 2*edge
	//shared_mem_size += 4;  // for   int4 seed
	printf(" shared mem size: %d\n", shared_mem_size);

	printf("nbSeeds= %d\n", nbSeeds);
	printf("w,he= %d, %d\n", w, he);

	//dim3 b(edge/2, edge, 1);
	dim3 b(edge, edge/2, 1); // two data elements per thread
	dim3 g(nbSeeds,1,1);
	int nb_threads  = b.x * b.y * b.z;

	printf("sizeof(float4)= %d\n", sizeof(float4)); // 16


	if (nb_threads*2 != edge*edge) {
		printf("error in nb_threads or edge\n");
		exit(0);
	}

	int width = w;
	printf("width= %d\n", width);

	float4* h_h = (float4*) malloc(sizeof(float4)*edge*edge);

	printf("nbSeeds= %d\n", nbSeeds);
	int4* newSeeds_d;
	CUDA_SAFE_CALL(cudaMalloc((void**) &newSeeds_d, sizeof(int4)*nbSeeds));
	CUDA_SAFE_CALL(cudaMemcpy(newSeeds_d, newSeeds, sizeof(int4)*nbSeeds, cudaMemcpyHostToDevice));

// Assume that width = height (SHOULD BE MADE MORE GENERAL)
	//scan_workefficient_2<<<g, b, 2*shared_mem_size>>>(h, bins, newSeeds_d, nb_threads*2, edge, width);

	// More efficient scan (hopefully)
	scan_workefficient_3<<<g, b, 2*shared_mem_size>>>(h, bins, newSeeds_d, nb_threads*2, edge, width);
	//scan_test_incoherent<<<g, b, 2*shared_mem_size>>>(h, bins, newSeeds_d, nb_threads*2, edge, width);

#if 1

	float4* bins_h = (float4*) malloc(sizeof(float4)*nbSeeds);

	CUDA_SAFE_CALL(cudaMemcpy(h_h, h, edge*edge*sizeof(float4), cudaMemcpyDeviceToHost));
	for (int i=0; i < edge*edge; i++) {
		printf("h_h= %f, %f, %f, %f\n", h_h[i].x, h_h[i].y, h_h[i].z, h_h[i].w);
	}
	//return;

	float tot = 0;
	for (int i=0; i < nb_threads; i++) {
		tot += h_h[i].w;
	}
	printf("total = %f\n", tot);


	printf("nbSeeds= %d\n", nbSeeds);
	printf("nb_threads= %d\n", nb_threads);
	printf("b size: %d\n", b.x*b.y*b.z);

	cudaMemcpy(bins_h, bins, nbSeeds*sizeof(float4), cudaMemcpyDeviceToHost);
	int count = 0;
	for (int j=0; j < nbSeeds; j++) { //}
		//printf("(%d) sum= %f, %f, %f, %f\n", j, bins_h[j].x, bins_h[j].y, bins_h[j].z, bins_h[j].w);
		count += (int) bins_h[j].w;
	}
	printf("total count= %d\n", count);
#endif
}
//----------------------------------------------------------------------
extern "C++" void centroid_voronoi_4(dim3& grid , dim3& block, float4* h, float4* bins, int w, int he, 
    int nb_bins, int sz3d, int4* newSeeds, int edge, int nbSeeds) 
{
	edge = 16; // tile 8 x 8 around seed: 0,1,2,3,4=seed,5,6,7  (seed-edge/2, seed+edge/2-1)

	unsigned int num_elements = edge*edge;
    unsigned int extra_space = num_elements / NUM_BANKS;

#ifdef ZERO_BANK_CONFLICTS
    extra_space += extra_space / NUM_BANKS;
#endif

    const unsigned int shared_mem_size = sizeof(float4)*num_elements + extra_space + 4; // in bytes


	printf(" shared mem size: %d\n", shared_mem_size);

	printf("nbSeeds= %d\n", nbSeeds);
	printf("w,he= %d, %d\n", w, he);

	//dim3 b(edge/2, edge, 1);
	dim3 b(edge, edge, 1); // two data elements per thread
	dim3 g(nbSeeds,1,1);
	int nb_threads  = b.x * b.y * b.z;

	printf("sizeof(float4)= %d\n", sizeof(float4)); // 16


	if (nb_threads != edge*edge) {
		printf("error in nb_threads or edge\n");
		exit(0);
	}

	int width = w;
	printf("width= %d\n", width);

	float4* h_h = (float4*) malloc(sizeof(float4)*edge*edge);

	printf("nbSeeds= %d\n", nbSeeds);
	int4* newSeeds_d;
	CUDA_SAFE_CALL(cudaMalloc((void**) &newSeeds_d, sizeof(int4)*nbSeeds));
	CUDA_SAFE_CALL(cudaMemcpy(newSeeds_d, newSeeds, sizeof(int4)*nbSeeds, cudaMemcpyHostToDevice));

// Assume that width = height (SHOULD BE MADE MORE GENERAL)

	scan_workefficient_4<<<g, b, shared_mem_size>>>(h, bins, newSeeds_d, nb_threads*2, edge, width);

#if 1

	float4* bins_h = (float4*) malloc(sizeof(float4)*nbSeeds);

	CUDA_SAFE_CALL(cudaMemcpy(h_h, h, edge*edge*sizeof(float4), cudaMemcpyDeviceToHost));
	//for (int i=0; i < edge*edge; i++) {
		//printf("h_h= %f, %f, %f, %f\n", h_h[i].x, h_h[i].y, h_h[i].z, h_h[i].w);
	//}
	//return;

	float tot = 0;
	for (int i=0; i < nb_threads; i++) {
		tot += h_h[i].w;
	}
	printf("total = %f\n", tot);


	printf("nbSeeds= %d\n", nbSeeds);
	printf("nb_threads= %d\n", nb_threads);
	printf("b size: %d\n", b.x*b.y*b.z);

	cudaMemcpy(bins_h, bins, nbSeeds*sizeof(float4), cudaMemcpyDeviceToHost);
	int count = 0;
	for (int j=0; j < nbSeeds; j++) { 
		//printf("(%d) sum= %f, %f, %f, %f\n", j, bins_h[j].x, bins_h[j].y, bins_h[j].z, bins_h[j].w);
		count += (int) bins_h[j].w;
	}
	printf("total count= %d\n", count);
#endif
}
//----------------------------------------------------------------------
extern "C++" void centroid_voronoi_5(dim3& grid , dim3& block, float4* h, float4* bins, int w, int he, 
    int nb_bins, int sz3d, int4* newSeeds, int edge, int nbSeeds) 
{
// Objective: remove all coalescing and bank conflicts
	edge = 16; // tile 8 x 8 around seed: 0,1,2,3,4=seed,5,6,7  (seed-edge/2, seed+edge/2-1)

	unsigned int num_elements = edge*edge;
    unsigned int extra_space = num_elements / NUM_BANKS;

#ifdef ZERO_BANK_CONFLICTS
    extra_space += extra_space / NUM_BANKS;
#endif

    const unsigned int shared_mem_size = sizeof(float4)*(num_elements + extra_space + 1); // in bytes


	printf(" shared mem size: %d\n", shared_mem_size);

	printf("nbSeeds= %d\n", nbSeeds);
	printf("w,he= %d, %d\n", w, he);

	dim3 b(4*edge, 1, 1); // 64 threads per block: 2 warps. Later, change to 128 threads. 
	//dim3 b(edge, edge, 1); // two data elements per thread
	dim3 g(nbSeeds,1,1);
	int nb_threads  = b.x * b.y * b.z;
	printf("block: %d, %d, %d\n", b.x, b.y, b.z);

	printf("sizeof(float4)= %d\n", sizeof(float4)); // 16

	if (nb_threads != 4*edge) {
		printf("error in nb_threads or edge\n");
		exit(0);
	}

	int width = w;
	printf("width= %d\n", width);

	float4* h_h = (float4*) malloc(sizeof(float4)*edge*edge);

	printf("nbSeeds= %d\n", nbSeeds);
	int4* newSeeds_d;
	CUDA_SAFE_CALL(cudaMalloc((void**) &newSeeds_d, sizeof(int4)*nbSeeds));
	CUDA_SAFE_CALL(cudaMemcpy(newSeeds_d, newSeeds, sizeof(int4)*nbSeeds, cudaMemcpyHostToDevice));

// Assume that width = height (SHOULD BE MADE MORE GENERAL)

	for (int i=0; i < 10; i++) {
		printf("newSeeds[%d]= %d, %d, %d, %d\n", i, newSeeds[i].x, newSeeds[i].y, newSeeds[i].z, newSeeds[i].w);
	}

	scan_workefficient_5<<<g, b, shared_mem_size>>>(h, bins, newSeeds_d, nb_threads, edge, width);
	//scan_workefficient_6<<<g, b, shared_mem_size>>>(h, bins, newSeeds_d, nb_threads, edge, width);
	//exit(0);

#if 1

	float4* bins_h = (float4*) malloc(sizeof(float4)*nbSeeds);

	CUDA_SAFE_CALL(cudaMemcpy(h_h, h, edge*edge*sizeof(float4), cudaMemcpyDeviceToHost));
	for (int i=0; i < 10; i++) {
		printf("h_h= %f, %f, %f, %f\n", h_h[i].x, h_h[i].y, h_h[i].z, h_h[i].w);
	}

	printf("nbSeeds= %d\n", nbSeeds);
	printf("nb_threads= %d\n", nb_threads);
	printf("b size: %d\n", b.x*b.y*b.z);

	cudaMemcpy(bins_h, bins, nbSeeds*sizeof(float4), cudaMemcpyDeviceToHost);
	int count = 0;
	for (int j=0; j < nbSeeds; j++) { 
		//if (j < 5) printf("(%d) sum= %f, %f, %f, %f\n", j, bins_h[j].x, bins_h[j].y, bins_h[j].z, bins_h[j].w);
		printf("(%d) sum= %f, %f, %f, %f\n", j, bins_h[j].x, bins_h[j].y, bins_h[j].z, bins_h[j].w);
		count += (int) bins_h[j].w;
	}
	printf("total count= %d\n", count);
#endif
}
//----------------------------------------------------------------------
extern "C++" float4* centroid_voronoi_7(float4* h, float4* bins, int w, int he, 
    int nb_bins, int sz3d, int4* newSeeds, int edge, int nbSeeds) 
// return 
{
// Objective: remove all coalescing and bank conflicts
	edge = 16; // tile 8 x 8 around seed: 0,1,2,3,4=seed,5,6,7  (seed-edge/2, seed+edge/2-1)

	unsigned int num_elements = edge*edge;
    unsigned int extra_space = num_elements / NUM_BANKS;

#ifdef ZERO_BANK_CONFLICTS
    extra_space += extra_space / NUM_BANKS;
#endif

    const unsigned int shared_mem_size = sizeof(float4)*(num_elements + extra_space + 1); // in bytes


#undef SCAN7

#ifdef SCAN7
	dim3 b(4*edge, 4, 1); // 256=4*64 threads per block: 2 warps. Later, change to 128 threads. 
#else
	dim3 b(edge, edge, 1); // 256=4*64 threads per block: 2 warps. Later, change to 128 threads. 
#endif

	//dim3 b(edge, edge, 1); // two data elements per thread
	dim3 g(nbSeeds,1,1);
	int nb_threads  = b.x * b.y * b.z;


	if (nb_threads != 4*4*edge) {
		printf("error in nb_threads or edge\n");
		exit(0);
	}

	int width = w;

	float4* h_h = (float4*) malloc(sizeof(float4)*edge*edge);

	int4* newSeeds_d;
	CUDA_SAFE_CALL(cudaMalloc((void**) &newSeeds_d, sizeof(int4)*nbSeeds));
	CUDA_SAFE_CALL(cudaMemcpy(newSeeds_d, newSeeds, sizeof(int4)*nbSeeds, cudaMemcpyHostToDevice));

// Assume that width = height (SHOULD BE MADE MORE GENERAL)

	for (int i=0; i < 10; i++) {
		printf("newSeeds[%d]= %d, %d, %d, %d\n", i, newSeeds[i].x, newSeeds[i].y, newSeeds[i].z, newSeeds[i].w);
	}

	//scan_workefficient_7<<<g, b, shared_mem_size>>>(h, bins, newSeeds_d, nb_threads, edge, width);
	scan_workefficient_8<<<g, b, shared_mem_size>>>(h, bins, newSeeds_d, nb_threads, edge, width);
	//scan_workefficient_8_larger<<<g, b, shared_mem_size>>>(h, bins, newSeeds_d, nb_threads, edge, width);

	float4* bins_h = (float4*) malloc(sizeof(float4)*nbSeeds);
	CUDA_SAFE_CALL(cudaMemcpy(bins_h, bins, nbSeeds*sizeof(float4), cudaMemcpyDeviceToHost));


#if 1
	printf("nbSeeds= %d\n", nbSeeds);
	printf("nb_threads= %d\n", nb_threads);
	printf("b size: %d\n", b.x*b.y*b.z);

	printf("--- GPU centroid histogram inside add_one_kernel.cu ----\n");
	int count = 0;
	for (int j=0; j < nbSeeds; j++) { 
		if (j < 20) printf("(%d) sum= %f, %f, %f, %f\n", j, bins_h[j].x, bins_h[j].y, bins_h[j].z, bins_h[j].w);
		//printf("(%d) sum= %f, %f, %f, %f\n", j, bins_h[j].x, bins_h[j].y, bins_h[j].z, bins_h[j].w);
		count += (int) bins_h[j].w;
	}
	printf("add_one: total count= %d\n", count);
#endif

	return bins_h;
}
//----------------------------------------------------------------------

#endif
