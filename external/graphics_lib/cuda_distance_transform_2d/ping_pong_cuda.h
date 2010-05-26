#ifndef _PING_PONG_CUDA_H_
#define _PING_PONG_CUDA_H_

#include <string>
#include <host_defines.h> // cuda
#include <vector_types.h> // cuda
#include "timege.h"

class PingPong;

class PingPongCuda
{
private:
	/// When manipulating pingpong outside this class instance, the changes are
	/// accessible to the instance. 
	PingPong* pingpong;
	GLenum pbo; // PBO id associated with CUDA array
	GLenum vbo; // VBO id associated with CUDA array
	GLenum fbo_id; // FBO attachment id 

	GE::Time* clock_tmp;
	GE::Time* clock_tmp1;
	GE::Time* clock_tmp2;
	GE::Time* clock_tmp3;

	/// pointer to linear CUDA array
	/// User is responsible for this array, which exists on the GPU
	__align__(256) float4* data4; // not clear this helps or is correct
	float* data; 

	int szx, szy; // width/height of cuda array and vbo/fbo/pingpong

public: 
	// Interface between PingPong object and cuda. Allows a cuda program 
	// to act on data in a vbo buffer created by some OpenGL program
    PingPongCuda(PingPong& ping);
    ~PingPongCuda();

	/// internal, type, type
	void setFormat();

    /// Set buffers to draw into this framebuffer object
    float* begin();

    /// Disable this framebuffer object, prevent drawing into 
    /// these buffers
    void end();

	/// draw last result drawn to the texture into the backbuffer
	/// Outstanding Shaders are disabled. The last shader used is not retained. 
	void toBackBuffer();

	/// OpenGL error checking
	void checkError(char* msg);   

	// print contents of buffer just written to
	void print(int i1, int j1, int w, int h);

	GLuint getPBO() { return pbo; }
	void printPBO(int x0, int y0, int w, int h);


private:
	void createPBO();
	void FBO_to_PBO();
	void FBO_to_PBO_faster();
	void PBO_to_FBO();
	float* cuda_register_and_map();;
};

#endif
