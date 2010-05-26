#ifndef _DISTANCE_MAP_H_
#define _DISTANCE_MAP_H_

#include "abstract_lic.h"
#include "framebufferObject.h"
#include "tex_ogl.h"

class DistanceTransform : public AbstractLic
{
private:
	Globals *g;
	int stepLength;
	int tex_size;
	int curTex;
	utils u;
	TexOGL* quad1_tex;
	TexOGL* pos_tx[2]; // to hold data related to Voronoi mesh
	GLuint quad1;
	FramebufferObject* fbo_pos;
	AbstractLic* alic;
public:
	DistanceTransform(Globals* g, int tex_size);
	~DistanceTransform();
	void setupFBOs();
	void setupTextures();
	void run();
	void updateTexture();
	TexOGL* getFinalTexture();
};

#endif

