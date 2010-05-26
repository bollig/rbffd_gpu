#include "globals.h"
//#include "bench1.h"

Globals::Globals(int tex_size)
{
		printf("inside globals constructor\n");
		kTextureSize = tex_size; // default
		this->tex_size = tex_size;
		winx_size = 0;
		winy_size = 0;
		g_frame_count = 0;
		u = new utils();
		gl = new CG::GL(100); // maximum number of shader programs in use
		vel_size = 256;
}
//----------------------------------------------------------------------
// should go into more general library
CG::Program& Globals::enableShader(GLuint id)
{
	if (id <= 0) {
		printf("enableShader: shader %d should be > 0\n", id);
		exit(0);
	}
	glUseProgram(id); // error
	return getShader(id);
}
//----------------------------------------------------------------------
void Globals::disableShaders()
{
	glUseProgram(0);
}
//----------------------------------------------------------------------
