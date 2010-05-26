// includes, GL
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <cutil.h>
#include <stdio.h>

#include <cuda.h>
//#include "array_cuda_t.h"

#include "map.h"

//----------------------------------------------------------------------
extern "C++" 
void unmapBufferObject(unsigned int vbo)
{
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo));
}
//----------------------------------------------------------------------
extern "C++" 
void unregisterBufferObject(unsigned int vbo)
{
	CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo));
}
//----------------------------------------------------------------------
extern "C++" 
void registerBufferObject(unsigned int vbo)
{
	//printf("register: vbo= %d\n", vbo);
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo));
}
//----------------------------------------------------------------------
extern "C++" 
void cudaInit()
{
	CUT_DEVICE_INIT();
}
//----------------------------------------------------------------------
extern "C" 
void copyFromDeviceToDevice(void* dst, const void* src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}
//----------------------------------------------------------------------
extern "C" 
void copyFromDeviceToHost(void* dst, const void* src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}
//----------------------------------------------------------------------
