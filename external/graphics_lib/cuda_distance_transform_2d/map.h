#ifndef _MAP_H_
#define _MAP_H_

#include <vector_types.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cutil.h>
// contains necessary declarations because I am using templates
#include <cuda_runtime_api.h>
#include <vector>

extern "C++" void centroid_voronoi(dim3& grid , dim3& block, float4* h, float4* bins, int w, int he, 
	int nb_bins, int sz3d, std::vector<int4>& newSeeds, int edge);
extern "C++" void centroid_voronoi_2(dim3& grid, dim3& block, float4* h, float4* bins, int w, int he, 
	int nb_bins, int sz3d, int4* newSeeds, int edge, int nbSeeds);
extern "C++" void centroid_voronoi_4(dim3& grid, dim3& block, float4* h, float4* bins, int w, int he, 
	int nb_bins, int sz3d, int4* newSeeds, int edge, int nbSeeds);
extern "C++" void centroid_voronoi_5(dim3& grid, dim3& block, float4* h, float4* bins, int w, int he, 
	int nb_bins, int sz3d, int4* newSeeds, int edge, int nbSeeds);
extern "C++" float4* centroid_voronoi_7(float4* h, float4* bins, int w, int he, 
	int nb_bins, int sz3d, int4* newSeeds, int edge, int nbSeeds);

//extern "C++" void add(dim3&, dim3&, float*, float*, int, int); 
extern "C++" void add(dim3&, dim3&, float*, float*, int, int); 
extern "C++" void bins_call(dim3&, dim3&, int*, int*, int, int, int);
extern "C++" void bins_efficient_call(dim3&, dim3&, int*, int*, int, int, int);
extern "C++" void cudaInit();
//extern "C++" void mapBufferObject(void** ptr, unsigned int vbo);
extern "C++" void unmapBufferObject(unsigned int vbo);
extern "C++" void registerBufferObject(unsigned int vbo);
extern "C++" void unregisterBufferObject(unsigned int vbo);
extern "C" void copyFromDeviceToDevice(void* dst, const void* src, size_t count);
extern "C" void copyFromDeviceToHost(void* dst, const void* src, size_t count);

//----------------------------------------------------------------------
//extern "C++"
template <class T>
void createVBOonGPU(GLuint* vbo, int width, int height)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = width * height * sizeof(float);
    //unsigned int size = width * height * sizeof(T);
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*vbo));

    CUT_CHECK_ERROR_GL();
}
//----------------------------------------------------------------------
template <class T>
void mapBufferObject(T** ptr, unsigned int vbo)
{
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**) ptr, vbo));
}
//----------------------------------------------------------------------
//template <class T>
//void free(T* ptr)
//{
    //CUDA_SAFE_CALL(cudaFree((void*) ptr));
//}
//----------------------------------------------------------------------

#endif
