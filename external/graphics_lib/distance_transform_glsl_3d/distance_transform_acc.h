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
		void setValue(float x_, float y_, float z_) {
			x = x_;
			y = y_;
			z = z_;
		}
	};
private:
	int nb_pingpong_levels;
	VBO<POINT3, POINT3>* vbo;
	VBO<POINT3, POINT3>* vboQuad; // each seed point is a quad
	std::vector<Seed> seeds;
	std::vector<POINT3> seed_pts;
	std::vector<POINT3> seed_col;
	std::vector<POINT3> seed_pts_quad; // each seed is a quad
	std::vector<POINT3> seed_col_quad;
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
	// each seed is a 2x2 quad instead of a point
	void drawSeedsQuad(PingPong& ping);
	void drawSeedsFinal(PingPong& ping);
	void drawSeedsNoBegin(PingPong& ping, TexOGL& tex);
	void resetTextureWithSeeds();
	PingPong* PingPongFactory(int tex_size);
	void drawVoronoi(PingPong* pph, int nsteps);
	void drawVoronoi(PingPong* pph, TexOGL& tex);
	void colorVoronoi(PingPong& pph);
};

#endif

