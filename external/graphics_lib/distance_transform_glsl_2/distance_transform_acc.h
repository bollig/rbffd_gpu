#ifndef _DISTANCE_TRANSFORM_ACC_H_
#define _DISTANCE_TRANSFORM_ACC_H_

//#include "abstract_lic.h"
//#include "framebufferObject.h"
#include <vector>
#include <math.h>
#include "tex_ogl.h"
#include "ping_pong.h"
#include "vbo.h"

class Globals;

class DistanceTransformAcc //: public AbstractLic
{
public:
	struct Seed {
		float x, y;
	};
	struct POINT3 {
		float x, y, z;
	};
private:
	int nb_pingpong_levels;
	VBO<POINT3, POINT3>* vbo;
	std::vector<Seed> seeds;
	std::vector<POINT3> seed_pts;
	std::vector<POINT3> seed_col;
	PingPong* pp;
	PingPong* pph; // buffers half the size of pp
	std::vector<PingPong*> pps; // collection of ping-pongs
	Globals* g;
	int stepLength;
	int tex_size;
	int curTex;
	utils u;
	TexOGL* quad1_tex;
	TexOGL* pos_tx[2]; // to hold data related to Voronoi mesh
	GLuint quad1;
	//FramebufferObject* fbo_pos;
	//AbstractLic* alic;
public:
	DistanceTransformAcc(Globals* g, int tex_size);
	~DistanceTransformAcc();
	void setupTextures();
	void run();
	void updateTexture();
	void updateTexture(PingPong* ping);
	void updateTexture(PingPong* ping, TexOGL& tex);
	void computeSeeds(int nb);
	void drawSeeds(PingPong& ping);
	void resetTextureWithSeeds();
	PingPong* PingPongFactory(int tex_size);
	void drawVoronoi(PingPong* pph);
	void drawVoronoi(PingPong* pph, TexOGL& tex);
	void drawSeedsNoBegin(PingPong& ping, TexOGL& tex);
};

#endif

