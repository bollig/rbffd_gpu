#ifndef _DISTANCE_TRANSFORM_ACC_H_
#define _DISTANCE_TRANSFORM_ACC_H_

//#include "abstract_lic.h"
//#include "framebufferObject.h"
#include <vector>
#include <math.h>
#include "tex_ogl.h"
#include "ping_pong.h"
#include "vbo.h"
#include "local_types.h"

class Globals;

class DistanceTransformAcc //: public AbstractLic
{
public:
	//struct Seed {
		//float x, y;
	//};
	struct POINT4 {
		float x, y, z, w;
	};
private:
	int nb_pingpong_levels;
	VBO<POINT4, POINT4>* vbo;
	std::vector<Seed> seeds;
	std::vector<float> float_seeds;
	std::vector<POINT4> seed_pts;
	std::vector<POINT4> seed_col;
	PingPong* pp;
	PingPong* pph; // buffers half the size of pp
	std::vector<PingPong*> pps; // collection of ping-pongs
	PingPong* cur_ping_pong;
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
	PingPong* PingPongFactory(int tex_size);
	void drawVoronoi(PingPong* pph, int nbSteps);
	void drawVoronoi(PingPong* pph, TexOGL& tex);
	void drawSeedsNoBegin(PingPong& ping, TexOGL& tex);

	PingPong& getCurPingPong() { return *cur_ping_pong; }

	/// seeds is under control of this instannce 
	/// Returning reference  guarantees that changes in the main program
	/// affect the datastructure in the class
	std::vector<Seed>& getSeeds() { return seeds; }

	std::vector<float>& getFloatSeeds() { return float_seeds; }
	//int getSize3d() { return sz3d; }
};

#endif

