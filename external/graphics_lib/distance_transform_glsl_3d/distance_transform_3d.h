#ifndef _DISTANCE_TRANSFORM_3D_H_
#define _DISTANCE_TRANSFORM_3D_H_

//#include "abstract_lic.h"
//#include "framebufferObject.h"
#include <vector>
#include <stdio.h>
#include <math.h>
#include "tex_ogl.h"
#include "tex_ogl_1d.h"
#include "ping_pong.h"
#include "vbo.h"
#include "Vec3i.h"

class Globals;

class DistanceTransform3D //: public AbstractLic
{
public:
	struct Texsz {
		int gx, gy;   // 2D grid size
		int szx, szy; // 2D texture size
		int sz3d;
		float hx, hy; // width/height of ortho (1,1) or (1,0.5)
		Texsz(int sz3d_, int gx_, int gy_) : sz3d(sz3d_), gx(gx_), gy(gy_)
		{ 
			szx = sz3d_*gx;
			szy = sz3d_*gy;
			hx = 1.;
			hy = (gx == gy) ? hx : 0.5 * hx;
		}
		void print() {
			printf("%d, szx,szy= %d, %d, gx,gy= %d, %d, sz3d= %d\n", sz3d, szx, szy, gx, gy, sz3d);
		}
	};
	struct Seed {
		float x, y, z;
	};
	struct POINT3 {
		float x, y, z;
	};
	struct IVEC2 { 
		int x, y; 
		void print() {
			printf("IVEC2: x,y= %d, %d\n", x, y);
		}
	};

private:
	Texsz* curTexSz;
	int i_level;  // current level in pyramid structure
	std::vector<Texsz> texsz;
	std::vector<IVEC2> offsets;
	int nb_pingpong_levels;
	VBO<POINT3, POINT3>* vbo;
	std::vector<VBO_T<POINT3, POINT3>* >* vbo_list;
	std::vector<Seed> seeds;
	std::vector<POINT3> seed_pts;
	std::vector<POINT3> seed_col;

	std::vector<VBO_T<POINT3, POINT3>* >* vbo_interp_list;

	// more efficient passing of slice hights to the GPU
	vector<int> koffsets;
	vector<float> k0v;
	TexOGL1D* k0_offsetTex;

	PingPong* pp;
	PingPong* pph; // buffers half the size of pp
	std::vector<PingPong*> pps; // collection of ping-pongs
	Globals* g;
	int stepLength;
	//int tex_size; /// equivalent 2D texture size
	//int szx, szy; /// size of 2D texture
	int gx, gy; /// 2D grid size (number of smaller textures in each direction
	/// full 2D texture is of size: (gx*sz3d) x (gy*sz3d) = szx x szy
	int sz3d; /// 3D texture size 
	int curTex;
	utils u;
	TexOGL* quad1_tex;
	TexOGL* pos_tx[2]; // to hold data related to Voronoi mesh
	GLuint quad1;
	//FramebufferObject* fbo_pos;
	//AbstractLic* alic;
public:
	DistanceTransform3D(Globals* g, int tex_size);
	~DistanceTransform3D();
	void setupTextures();
	void run();
	void runFull();
	void updateTexture();
	void updateTexture(PingPong* ping);
	void updateTextureFull(PingPong* ping, int step);
	void updateTextureFull_vbo(PingPong* ping, int step);
	void computeSeeds(int nb);
	void drawSeeds(PingPong& ping);
	void drawSeedsToAllTiles(PingPong& ping, int ix);
	void resetTextureWithSeeds();
	PingPong* PingPongFactory(int tex_size_x, int tex_size_y);
	void drawVoronoi(PingPong* pph, int sz); // width of 2D texture
	void computeOffsets();

	/// Given a 3D slice, compute the offset into the 3D texture
	Vec3i offset(int k);

	/// Given a point in the 2D plane (rect. coord.), compute
	/// the point in 3D space (rect. coord)
	void two2three(int x, int y);

	/// Given a point in the 2D plane in [0,1], compute
	/// the point in 3D space [0,1]
	void two2three(float x, float y);

	void drawQuad(int szc3d, int szf3d, int ic, int jc, int i, int j);
	void interpolate(int szc3d, int szf3d, PingPong& pc, PingPong& pf);
	void interpolateFull(int szc3d, int szf3d, PingPong& pc, PingPong& pf);
	void ijOffset(int k, int gx, int* i, int *j);
	void drawSeeds3D(PingPong& ping, int ix);

	/// create an arary of Vertex Buffer Objects (quads), one per level.
	void computeQuadsArray();
	void computeOddQuad(int n, int m, float hx, float hy);
	void computeInterpolationQuads();
	void computeInterpolationQuad(int klevel, int n, int m, float hx, float hy);
	void drawVoronoiFull(PingPong* pph, int sz, int steps);
	void setup_k0_structures();
};

#endif

