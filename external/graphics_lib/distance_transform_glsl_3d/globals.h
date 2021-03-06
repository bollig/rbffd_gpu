#ifndef _GLOBALS_H_
#define _GLOBALS_H_

#include <stdio.h>
#include "utils.h"
#include "gl_state.h"
#include "glincludes.h"

class utils;
class TextureOGL;

// Avoids paying the cost of file inclusion
// Avoid potential naming conflicts

class Globals {
private:
	//typedef unsigned int GLuint;

public:

// benchmark codes
	GLuint p_passthrough; // for AbstractLic
	// texture sample in x,y
	GLuint voronoi_gpu_3d;
	// texture sample in x,y, and z
	GLuint voronoi_gpu_3d_full;
	// draw seeds (translate 3D to 2D coordinates)
	GLuint draw_seed;
	GLuint draw_seed_final; // seeds after reset of the image is drawn
	GLuint color_voronoi; // final coloring of voronoi mesh

public:
	utils *u;
    int if_bind;
    int if_float; 

	CG::GL* gl;

	int kTextureSize; // texture sizes
	int tex_size; // texture sizes
	int  g_frame_count;
	int winx_size;
	int winy_size;
	int vel_size;
	int main_window; // needed if using multiple windows

	TextureOGL* vel_tex; // reference velocity field, mostly used during quad creation

public:

	Globals(int texSize=0);

	void setWinSize(int wx, int wy) {
		winx_size = wx;
		winy_size = wy;
	}

	void setTextureSize(int sz) {
		kTextureSize = sz;
		tex_size = sz;
	}

	int getSize() {
		return kTextureSize;
	};

	CG::Program& getShader(GLuint shader_id) {
		return *gl->getShader(shader_id);
	}

	CG::Program& enableShader(GLuint id);

	void disableShaders();
};

#endif
