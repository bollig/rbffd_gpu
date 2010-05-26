#include "distance_transform_3d.h"
#include "textures.h"
#include "textures_1d.h"
#include "globals.h"
#include "Array3D.h"
#include "math.h"
#include "vbo.h"

// Accelerated version

// 1. create a texture of size tex_size/2 with the same seed points  (create a shader
//  for this
//

//----------------------------------------------------------------------
DistanceTransform3D::DistanceTransform3D(Globals* g, int tex_size_3d) 
{
	this->g = g;
	this->sz3d = tex_size_3d;

// texsz[i] gives characteristics of 3D texture of size (2^i)^3
// We will start with texsz[m] and go down to texsz[1] in the multigrid 
// approach

	texsz.push_back(Texsz(1,	1, 	1));   // sz3d, gx, gy
	texsz.push_back(Texsz(2,  	2, 	1));
	texsz.push_back(Texsz(4, 	2, 	2));
	texsz.push_back(Texsz(8, 	4, 	2));
	texsz.push_back(Texsz(16, 	4, 	4));
	texsz.push_back(Texsz(32, 	8, 	4));
	texsz.push_back(Texsz(64, 	8, 	8));
	texsz.push_back(Texsz(128, 	16, 8));
	texsz.push_back(Texsz(256, 	16, 16));
	//texsz.push_back(Texsz(512, 	32, 16));
	//texsz.push_back(Texsz(1024, 32, 32));
	//texsz.push_back(Texsz(2048, 64, 32));

	for (int i=0; i < texsz.size(); i++) {
		texsz[i].print();
	}

	// table of predefined correspondances between 3D and 2D textures, going 
	// from 2^3 up to 4000^3

	setup_k0_structures(); // before setupTextures
	
	vbo = new VBO<POINT3, POINT3>();
	vbo_list = new vector<VBO_T<POINT3, POINT3>* > [15]; // more than enough
	vbo_interp_list = new vector<VBO_T<POINT3, POINT3>* > [15]; // more than enough
	setupTextures(); 

	computeQuadsArray();
	computeInterpolationQuads();
//	printf("vbo_list.size= %d\n", vbo_list->size()); exit(0);
	//printf("vbo_list size: %d\n", vbo_list->size()); exit(0);
}
//----------------------------------------------------------------------
DistanceTransform3D::~DistanceTransform3D()
{	
	delete pp;
	delete pph;
}
//----------------------------------------------------------------------
#define	SWAPBUF(ping, msg) \
	printf("i= %d, %s\n", i, msg); \
	fflush(stdout); \
	ping->toBackBuffer(); \
	glutSwapBuffers(); \
	sleep(nbsec);

#if 0
#define PRINT(msg, ping, i)  \
	sleep(nbsec);
#else
#define PRINT(msg, ping, i) \
	glFinish(); \
	printf("---- %s ---------------\n", msg); \
	ping->print(0,0,texsz[i].szx,texsz[i].szy);  \
	//ping->print(texsz[i].szx-11, texsz[i].szy-11, 3, 3); \
	//printf("-------------------\n"); \
	//ping->print(texsz[i].szx-3, texsz[i].szy-3, 3, 3); \
	//sleep(nbsec);
	;
#endif

	 
//----------------------------------------------------------------------
void DistanceTransform3D::runFull()
{
	PingPong* pc;
	PingPong* pf;
	int nbsec = 2;

// First texture
	int i; // higher i corresonds to finer levels
	//printf("nb pg levels: %d\n", nb_pingpong_levels); 
	int i_finest = nb_pingpong_levels - 1;
	i = i_finest;
	i = i_finest-1;
	pc = pps[i];
	int sz3 = texsz[i].szx;
	curTexSz = &texsz[i]; // use as global: BAD
	i_level = i;
//	printf("3D size: sz3= %d\n", texsz[i].sz3d);
	drawSeeds3D(*pc, i); // seems ok
	//PRINT("after drawSeeds3D", pc, i); 
	// works fine, but that is too many sweeps
	int nbl;
	// converges in one shot. But Something is not right!! It must be
	// with the seed location
	//nbl = log2(pc->getTexture().getWidth()+.001);
	// number of sweeps should be based on sz3d of tile. 
	// But does not work!!
	// Propagation now takes 2-3 iterations. Not yet perfect. 
	// with 100 seeds, takes 9 iterations (frames) to converget
	nbl =  log2(curTexSz->sz3d+.001); 
	//nbl =  3;

//	printf("**** runFull: nbl= %d, i_level= %d\n", nbl, i_level);
	drawVoronoiFull(pc, sz3, nbl);
	//PRINT("after drawVoronoiFull", pc, i); exit(0);
	//SWAPBUF(pc, "after initial drawVoronoi"); 
	//pc->toBackBuffer(); return; 
	pf = pc;

	while (1) {
		if (i == (nb_pingpong_levels-1)) {
			//printf("exit final stage\n");
	 		//SWAPBUF(pf, "after final stage"); 
			pf->toBackBuffer();
			return;
		}
		//printf("increment i\n");
		i++;
		pf = pps[i];
		int szc3d = texsz[i-1].sz3d;
		int szf3d = texsz[i].sz3d;
		//printf("3D size: sz3= %d\n", sz3);
		//printf("before szc3d= %d, szf3d= %d\n", szc3d,szf3d);
		//printf("before i-1= %d\n", i-1);
		//printf("before   i= %d\n",   i);
		i_level = i;
		//printf("3D size: sz3= %d\n", texsz[i].sz3d);
		interpolateFull(szc3d, szf3d, *pc, *pf); // does not work!
		//printf("after interpolateFull\n");
	 	//SWAPBUF(pf, "after interpolateFull"); 
		pf->undoSwap(); // allows me to draw into same buffer as previous draw
	    curTexSz = &texsz[i]; // use as global. BAD
		drawSeeds3D(*pf, i); // seems ok
	 	//SWAPBUF(pf, "after drawSeeds3D"); 
		drawVoronoiFull(pf, texsz[i].szx, 1); // of course the 2D texture is not square
		pc = pf;
	}
}
//----------------------------------------------------------------------
void DistanceTransform3D::drawVoronoiFull(PingPong* pph, int sz, int steps)
{
	int nbl;
	int nbsec = 2;
	//int nbl = log2(curTexSz->sz3d+.001) + 3;
	//printf("drawVoronoiFull, nbl= %d\n", nbl);

	//stepLength= sz / 2;

	nbl = steps;
	stepLength = (int) pow(2., steps-1);
	//printf("nbl= %d\n", nbl);
	//stepLength= 2;

	//nbl = 2; // sz3=16=2^4, so of course the results are correct!

	for (int i=0; i < nbl; i++) {
		//printf("drawVoronoiFull, i=%d, nsteps= %d, stepLength= %d\n", i, steps, stepLength);
		updateTextureFull_vbo(pph, stepLength);
		//PRINT("step: ", pph, i_level); exit(0);
		stepLength /= 2;
		if (stepLength < 0) {
			printf("stepLength < 1, RESET to 1\n");
			stepLength = 1;
		}
	}
	return;
}
//----------------------------------------------------------------------
void DistanceTransform3D::drawVoronoi(PingPong* pph, int sz)
{
#if 0
	//printf("stepLength= %d\n", stepLength);
	//printf("nbl= %d\n", nbl);
	int nbl = log2(pph->getTexture().getWidth()) + .001;

	stepLength= sz / 2;
	stepLength= 1.;
	nbl = 0;

	for (int i=0; i < nbl; i++) {
	//	printf("step: %d\n", stepLength);
		updateTexture(pph);
		stepLength /= 2;
		if (stepLength < 1) stepLength = 1;
	}
	return;
#endif
}
//----------------------------------------------------------------------
void DistanceTransform3D::resetTextureWithSeeds() { }
//----------------------------------------------------------------------
void DistanceTransform3D::updateTextureFull_vbo(PingPong* ping, int step)
{
	u.checkError("*** 0 quads");
	TexOGL& tex = ping->getTexture();
	u.checkError("*** 0a quads");

	ping->begin();
	u.checkError("*** 0b quads");
	glEnable(tex.getTarget());
	u.checkError("*** 0c quads");
	tex.bind();
	u.checkError("1 quads"); // no error
	CG::Program& pg = g->enableShader(g->voronoi_gpu_3d_full);
	u.checkError("2 quads");   // ERROR

	//printf("step: %f\n", (float) stepLength);
	//printf("update: szx,szy,sz3d= %d, %d, %d\n", tex.getWidth(), tex.getHeight(), sz3d);
	pg.set_param1("stepLength", stepLength);
	u.checkError("3 quads");

	pg.set_param1("tex_size", (float) tex.getWidth());
	u.checkError("4 quads");
	pg.set_param1("szx", (float) tex.getWidth());
	u.checkError("4 quads");
	pg.set_param1("szy", (float) tex.getHeight());
	u.checkError("5 quads");
	pg.set_param1("szy", (float) tex.getHeight());
	//printf("sz3d: %d\n", curTexSz->sz3d); exit(0);
	u.checkError("6 quads");
	pg.set_param1("sz3d", curTexSz->sz3d);
	u.checkError("7 quads");
	pg.set_tex("texture", tex, 0);
	u.checkError("8 quads");

	int sz = tex.getWidth(); 
	int szx = tex.getWidth(); 
	int szy = tex.getHeight(); 

	// scan list of tiles on flat texture
	int gx = curTexSz->gx;
	int gy = curTexSz->gy;
	int sz3 = curTexSz->sz3d;

	//printf("gx*sz3= %d, gy*sz3= %d\n", gx*sz3, gy*sz3);

	// Should not be required!
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0., (float) gx*sz3, 0., (float) gy*sz3); // mapped to fbo size (i.e., the texture)
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	int k0,km, kp;
	// I need to pass k0 to the shader because if I compute it on a quad, 
	// it will have different values at the different vertices, which will create
	// problems in the fragment shader. 
	
	// Create 1D texture with all k0's for each level (one per tile). 
	// for each level, one needs gx*gy values of k0. Very little memory required. 
	// I also need offets into this array. This offset will be passed to the shader
	// as a Uniform variable. 
	// Ideally, the 1D texture should be a RECTANGULAR Texture. 

	//printf("vbo_list size: %d\n", vbo_list->size()); exit(0);
	u.checkError("10 quads");
	//printf("vbo: i_level= %d\n", i_level);
	VBO_T<POINT3,POINT3>* vboq = (*vbo_list)[i_level];
	u.checkError("11 quads");
	int nb_quads = curTexSz->gx * curTexSz->gy;
	//printf("vboq= %d\n", vboq);
	//printf("nb quads: %d\n", nb_quads);

	//printf("sz3d: %d\n", curTexSz->sz3d); exit(0);
	pg.set_param1("sz3d", curTexSz->sz3d);
	pg.set_tex("texture", tex, 0); 
	u.checkError("11a quads"); // no error

	vboq->draw(GL_QUADS, 4*nb_quads); // 2nd arg = nb elements
	u.checkError("12 quads"); // error
	//printf("after vboq: gx,gy= %d, %d\n", gx, gy);

	#if 0
	glBegin(GL_QUADS);

	for (int j=0; j < gy; j++) {
	for (int i=0; i < gx; i++) {
		k0 = i + j*gx;
		//km = k0 - stepLength;
		//kp = k0 + stepLength;
		//printf("k0= %d\n", k0);
		//pg.set_param1("k0", k0);
		//pg.set_param1("kp", kp);
		//pg.set_param1("km", km);

			float x0 = i*sz3;
			float y0 = j*sz3;

			glTexCoord3d(x0, y0, k0);
			glVertex2f(x0, y0);

			glTexCoord3d(x0+sz3, y0, k0);
			glVertex2f(  x0+sz3, y0);

			glTexCoord3d(x0+sz3, y0+sz3, k0);
			glVertex2f(  x0+sz3, y0+sz3);

			glTexCoord3d(x0, y0+sz3, k0);
			glVertex2f(  x0, y0+sz3);
	}}
	glEnd();

	#endif

	// should not be required
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();


	glDisable(tex.getTarget());
	glUseProgram(0);
	ping->end();
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void DistanceTransform3D::updateTextureFull(PingPong* ping, int step)
{
	TexOGL& tex = ping->getTexture();

	ping->begin();
	glEnable(tex.getTarget());
	tex.bind();
	CG::Program& pg = g->enableShader(g->voronoi_gpu_3d_full);

	//printf("step: %f\n", (float) stepLength);
	//printf("update: szx,szy,sz3d= %d, %d, %d\n", tex.getWidth(), tex.getHeight(), sz3d);
	pg.set_param1("stepLength", (float) stepLength);
	pg.set_param1("tex_size", (float) tex.getWidth());
	pg.set_param1("szx", (float) tex.getWidth());
	pg.set_param1("szy", (float) tex.getHeight());
	//printf("sz3d: %d\n", curTexSz->sz3d); exit(0);
	pg.set_param1("sz3d", curTexSz->sz3d);
	pg.set_tex("texture", tex, 0);
	//pg.set_tex("k0_tex", *k0_offsetTex, 0);

	int sz = tex.getWidth(); 
	int szx = tex.getWidth(); 
	int szy = tex.getHeight(); 

	// scan list of tiles on flat texture
	int gx = curTexSz->gx;
	int gy = curTexSz->gy;
	int sz3 = curTexSz->sz3d;


	// Should not be required!
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	//gluOrtho2D(0., 1., 0., 1.); // mapped to fbo size (i.e., the texture)
	gluOrtho2D(0., gx*sz3, 0., gy*sz3); // mapped to fbo size (i.e., the texture)
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	int k0,km, kp;
	// I need to pass k0 to the shader because if I compute it on a quad, 
	// it will have different values at the different vertices, which will create
	// problems in the fragment shader. 
	
	// Create 1D texture with all k0's for each level (one per tile). 
	// for each level, one needs gx*gy values of k0. Very little memory required. 
	// I also need offets into this array. This offset will be passed to the shader
	// as a Uniform variable. 
	// Ideally, the 1D texture should be a RECTANGULAR Texture. 

	glBegin(GL_QUADS);

	for (int j=0; j < gy; j++) {
	for (int i=0; i < gx; i++) {
		k0 = i + j*gx;
		//km = k0 - stepLength;
		//kp = k0 + stepLength;
		//printf("k0= %d\n", k0);
		//pg.set_param1("k0", k0);
		//pg.set_param1("kp", kp);
		//pg.set_param1("km", km);

			float x0 = i*sz3;
			float y0 = j*sz3;

			glTexCoord3d(x0, y0, k0);
			glVertex2f(x0, y0);

			glTexCoord3d(x0+sz3, y0, k0);
			glVertex2f(  x0+sz3, y0);

			glTexCoord3d(x0+sz3, y0+sz3, k0);
			glVertex2f(  x0+sz3, y0+sz3);

			glTexCoord3d(x0, y0+sz3, k0);
			glVertex2f(  x0, y0+sz3);
	}}
	glEnd();

	// should not be required
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();


	glDisable(tex.getTarget());
	glUseProgram(0);
	ping->end();
}
//----------------------------------------------------------------------
void DistanceTransform3D::updateTexture(PingPong* ping)
{
	TexOGL& tex = ping->getTexture();

	ping->begin();
	glEnable(tex.getTarget());
	tex.bind();
	CG::Program& pg = g->enableShader(g->voronoi_gpu_3d);

	//printf("step: %f\n", (float) stepLength);
	//printf("update: szx,szy,sz3d= %d, %d, %d\n", tex.getWidth(), tex.getHeight(), sz3d);
	pg.set_param1("stepLength", (float) stepLength);
	pg.set_param1("tex_size", (float) tex.getWidth());
	pg.set_param1("szx", (float) tex.getWidth());
	pg.set_param1("szy", (float) tex.getHeight());
	//printf("sz3d: %d\n", curTexSz->sz3d); exit(0);
	pg.set_param1("sz3d", curTexSz->sz3d);
	pg.set_tex("texture", tex, 0);

	int sz = tex.getWidth(); 
	int szx = tex.getWidth(); 
	int szy = tex.getHeight(); 
	//if (tex.getHeight() != sz) {
		//printf("updateTexture: texture should be square for now\n");
		//exit(0);
	//}

	glBegin(GL_QUADS);
		glTexCoord2d(0., 0.);
		glVertex2f(0., 0.);
		glTexCoord2d(szx, 0.);
		glVertex2f(1., 0.);
		glTexCoord2d(szx, szy);
		glVertex2f(1., 1.);
		glTexCoord2d(0, szy);
		glVertex2f(0., 1.);
	glEnd();

	glDisable(tex.getTarget());
	glUseProgram(0);
	ping->end();
}
//----------------------------------------------------------------------
void DistanceTransform3D::computeSeeds(int nb)
{
	Seed seed;
	POINT3 pt;

	#if 0
	pt.x = pt.y = pt.z = 0.01; // point is not there!
	seed_pts.push_back(pt); seed_col.push_back(pt);
	pt.x = pt.y = pt.z = 0.99; // point is not there!
	seed_pts.push_back(pt); seed_col.push_back(pt);
	#endif

	#if 1
	srand(13); // seed for random number generator

	for (int i=0; i < nb; i++) {
		//seed.x = u.rand_float();
		//seed.y = u.rand_float();
		//seed.z = u.rand_float();
		seed.x = rand() / (double) RAND_MAX;
		seed.y = rand() / (double) RAND_MAX;
		seed.z = rand() / (double) RAND_MAX;
		seeds.push_back(seed);

		pt.x = seed.x;
		pt.y = seed.y;
		pt.z = seed.z;
		seed_pts.push_back(pt);
		seed_col.push_back(pt);

		printf("x,y,z= %f, %f %f\n", pt.x, pt.y, pt.z);
	}
	#endif

	vbo->create(&seed_pts, &seed_col);
}
//----------------------------------------------------------------------
void DistanceTransform3D::computeQuadsArray()
{
	// k is size of a given slice

	for (int k=0; k < texsz.size(); k++) {
		Texsz& tx = texsz[k];
		int sz = tx.sz3d;
		i_level = k;
		printf("computeQuadsArray: i_level= %d\n", i_level);
		computeOddQuad(tx.gx, tx.gy, tx.hx, tx.hy);
	}
	//exit(0);
}
//----------------------------------------------------------------------
void DistanceTransform3D::computeInterpolationQuads()
{
	for (int k=0; k < texsz.size()-1; k++) {
		Texsz& tx = texsz[k];
		int sz = tx.sz3d;
		i_level = k;
		if (sz >= sz3d) break;
		printf("computeQuadsArray: i_level= %d\n", i_level);
		computeInterpolationQuad(i_level, tx.gx, tx.gy, tx.hx, tx.hy);
	}
	//exit(0);
}
//----------------------------------------------------------------------
// n: number of quads in x direction (gx)
// m: number of quads in y direction (gx)
// Interpolate from coarse to fine grid
void DistanceTransform3D::computeInterpolationQuad(int k_level, int m, int n, float hx, float hy)
{
	// MAKE SURE THAT i_level is defined
	Texsz& txc = texsz[k_level];
	Texsz& txf = texsz[k_level+1];

	// create list of VBOs interpolation
	POINT3 pt[4], tex[4];
	vector<POINT3>* pts = new vector<POINT3>;
	vector<POINT3>* texs = new vector<POINT3>;

	VBO_T<POINT3,POINT3>* vbo = new VBO_T<POINT3,POINT3>();

	for (int k=0; k < 4; k++) {
		pt[k].z = 0.;
		tex[k].z = 0.;
	}

	//------------------------------------
	int xv, yv;
	int xt, yt;

		int szc3d = txc.sz3d;
		int szf3d = txf.sz3d;
		int gxc = txc.gx;
		int gyc = txc.gy;
		int gxf = txf.gx;
		int gyf = txf.gy;
		printf("gxc= %d, gyc= %d\n", gxc, gyc);
		printf("gxf= %d, gyf= %d\n", gxf, gyf);

		int ic, jc;
		int if1, jf1;
		int if2, jf2;
		int xv1, yv1;
		int xv2, yv2;

		printf("szc3d= %d\n", szc3d);
		for (int kc=0; kc < szc3d; kc++) {
			ijOffset(kc, gxc, &ic, &jc);
			ijOffset(2*kc, gxf, &if1, &jf1);
			if2 = if1 + 1; // column
			jf2 = jf1; // row
			//printf("ic,jc= %d, %d\n", ic, jc);
			//printf("if1,jf1= %d, %d\n", if1, jf1);
			//printf("if2,jf2= %d, %d\n", if2, jf2);

			xt = ic*szc3d;
			yt = jc*szc3d;
			xv1 = if1*szf3d; // assumes that szc3d*2 = szf3d
			yv1 = jf1*szf3d;
			xv2 = if2*szf3d; // assumes that szc3d*2 = szf3d
			yv2 = jf2*szf3d;
			pt[0].x = (float) xv1;	 		pt[0].y = (float) yv1;
			pt[1].x = (float) xv1+szf3d; 	pt[1].y = (float) yv1;
			pt[2].x = (float) xv1+szf3d; 	pt[2].y = (float) yv1+szf3d;
			pt[3].x = (float) xv1; 			pt[3].y = (float) yv1+szf3d;

			tex[0].x = (float) xt;	 		tex[0].y = (float) yt;
			tex[1].x = (float) xt+szc3d; 	tex[1].y = (float) yt;
			tex[2].x = (float) xt+szc3d; 	tex[2].y = (float) yt+szc3d;
			tex[3].x = (float) xt; 			tex[3].y = (float) yt+szc3d;

			for (int k=0; k < 4; k++) {
				pts->push_back(pt[k]); // 2nd quad
				texs->push_back(tex[k]);
			}

			pt[0].x = (float) xv2;	 		pt[0].y = (float) yv1;
			pt[1].x = (float) xv2+szf3d; 	pt[1].y = (float) yv1;
			pt[2].x = (float) xv2+szf3d; 	pt[2].y = (float) yv1+szf3d;
			pt[3].x = (float) xv2; 			pt[3].y = (float) yv1+szf3d;

			for (int k=0; k < 4; k++) {
				pts->push_back(pt[k]); // 2nd quad
				texs->push_back(tex[k]);
			}
		}

		vbo->create(pts, texs);
		vbo_interp_list->push_back(vbo);
}
//----------------------------------------------------------------------
// n: number of quads in x direction
// m: number of quads in y direction
void DistanceTransform3D::computeOddQuad(int m, int n, float hx, float hy)
{
	POINT3 pt[4], tex[4];
	vector<POINT3>* pts = new vector<POINT3>;
	vector<POINT3>* texs = new vector<POINT3>;


	// glOrtho: [0,1] x [0,1]
	int sz3 = texsz[i_level].sz3d;
	float dx = sz3; //*hx/n; // dx always equal to hy: tiles are squares
	float dy = sz3; //*hy/m;

	//printf("computeOddQuad: m,n= %d, %d, sz3= %d\n", m,n, sz3);

	int gx = texsz[i_level].gx;
	float sx = (float) texsz[i_level].szx;
	float sy = (float) texsz[i_level].szy;

	//printf("i_level= %d\n", i_level);
	//printf("gx= %d\n", gx);
	//printf("dx,dy= %f, %f\n", dx, dy);
	//printf("sx,sy= %d, %d\n", (int) sx, (int) sy);


	VBO_T<POINT3,POINT3>* vbo = new VBO_T<POINT3,POINT3>();

	pt[0].z = 0.;
	pt[1].z = 0.;
	pt[2].z = 0.;
	pt[3].z = 0.;

	printf("dx*m= %d, dy*n= %d\n", (int) dx*m, (int) dy*n);

	for (int j=0; j < n; j++) {
	for (int i=0; i < m; i++) {
		pt[0].x = (i+0)*dx;	  pt[0].y = (j+0)*dy;
		pt[1].x = (i+1)*dx;   pt[1].y = (j+0)*dy;
		pt[2].x = (i+1)*dx;   pt[2].y = (j+1)*dy;
		pt[3].x = (i+0)*dx;	  pt[3].y = (j+1)*dy;

		float kf0 = (float) (i + j*m);
		//printf("kf0= %f\n", kf0);

		// tex should be between 0 and n : rescale inside the shader
		for (int k=0; k < 4; k++) {
			tex[k].x = pt[k].x;
			tex[k].y = pt[k].y;
			tex[k].z = kf0;
		}
		for (int k=0; k < 4; k++) {
			pts->push_back(pt[k]);
			texs->push_back(tex[k]);
			//printf("tex[k].z= %f\n", tex[k].z);
		}
	}}

	vbo->create(pts, texs);
	vbo_list->push_back(vbo);

	// delete is safe once the points are in the VBO
	//delete pts; // CANNOT DELETE: WE ARE WORKING WITH POINTERS
	//delete cols;
}
//----------------------------------------------------------------------
void DistanceTransform3D::drawSeedsToAllTiles(PingPong& ping, int ix)
{
	Texsz& tx = texsz[ix];
	float gx = (float) tx.gx;
	float gy = (float) tx.gy;

	glUseProgram(0);
	glDisable(ping.getTexture().getTarget());
	glDisable(GL_TEXTURE_2D); // should not be required
	ping.begin();

	// Should not be required!
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0., 1., 0., 1.); // mapped to fbo size (i.e., the texture)
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	// I need to draw all points into each tile of this
	// texture. For this I need to translate and scale
	// before each draw. Points are scaled between [0,1]
	// in x,y,z. Apply a glScale(1/gx,1/gy), and 
	// translate to position over each tile

	// potential problem: seeds at the same x,y location might
	// overwrite the correct seed at level k (BAD)

	//printf("gx,gy= %f, %f,     %f, %f\n", gx, gy, gx*tx.sz3d, gy*tx.sz3d);

	for (int j=0; j < gy; j++) {
	for (int i=0; i < gx; i++) {
		glPushMatrix();
		glTranslated(i/gx, j/gy, 0.);
		glScalef(1./gx, 1./gy, 1.);
		vbo->draw(GL_POINTS, seed_pts.size());
		glPopMatrix();
	}}

	// should not be required
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	ping.end();
}
//----------------------------------------------------------------------
void DistanceTransform3D::drawSeeds3D(PingPong& ping, int ix)
{
	glUseProgram(0); // should not be required
	TexOGL& tex = ping.getTexture();
	glDisable(tex.getTarget());
	ping.begin();

	CG::Program& pg = g->enableShader(g->draw_seed);

	//printf("---------------------------------------\n");
	//printf("width, height= %d, %d\n", tex.getWidth(), tex.getHeight());
	//printf("ix= %d, curTexSz->sz3d= %d\n", ix, curTexSz->sz3d);
	//printf("   curTexSz->szx/y= %d, %d\n", curTexSz->szx, curTexSz->szy);

	pg.set_param1("szx", (float) tex.getWidth());
	pg.set_param1("szy", (float) tex.getHeight());
	pg.set_param1("sz3d", curTexSz->sz3d);

	if (tex.getWidth() != curTexSz->szx || tex.getHeight() != curTexSz->szy) {
		printf("mismatch in texture sizes\n");
		exit(0);
	}

	//printf("nb seeds: %d\n", seed_pts.size());

	// Should not be required!
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	//gluOrtho2D(0., 1., 0., 1.);
	gluOrtho2D(0., tex.getWidth(), 0., tex.getHeight());
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	vbo->draw(GL_POINTS, seed_pts.size());

	// should not be required
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	ping.end();
	glUseProgram(0);
}
//----------------------------------------------------------------------
void DistanceTransform3D::drawSeeds(PingPong& ping)
{
	glUseProgram(0);
	glDisable(ping.getTexture().getTarget());
	glDisable(GL_TEXTURE_2D); // should not be required
	ping.begin();

	// Should not be required!
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0., 1., 0., 1.);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	vbo->draw(GL_POINTS, seed_pts.size());

	// should not be required
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	ping.end();
}
//----------------------------------------------------------------------
void DistanceTransform3D::setupTextures()
{
	// I get one more domain than the number of seeds. Correct?
	int nb_seeds = 50;
	computeSeeds(nb_seeds);

	int ix = (int) log2(sz3d+.001);
	Texsz& tz = texsz[ix];

	Textures tx(tz.szx, tz.szy);
	printf("setupTextures: %d, %d\n", tz.szx, tz.szy);

	//GLuint internal_format = GL_RGBA;
	GLuint internal_format = FLOAT_BUFFER;
	GLenum format 		= GL_RGBA;
	GLenum data_type 	= GL_FLOAT;
	//GLenum target = GL_TEXTURE_2D;
	GLenum target = TARGET;
	tx.setTarget(target);
	tx.setFormat(internal_format, format, data_type);

	//TexOGL* tex = tx.userDefined(a);
	TexOGL* tex = tx.createEmpty();
	pp = new PingPong(tex);

	// cannot reuse tex since only pointers are manipulated. There are no copies.

	// Create hierarchy of pingpong textures
	int nbl = 0;

	//printf("sz3d= %d\n", sz3d);
	int isz =  log2(sz3d+.001) + 1;
	//printf("isz= %d, %f\n", isz, log2(sz3d));

	for (int i=0; i < isz; i++) {
		Texsz& tx = texsz[i];
		//printf("i= [%d], szx,szy= %d, %d\n", i, tx.szx, tx.szy);
		//printf("... gx,gy= %d, %d\n", tx.gx, tx.gy);
		pps.push_back(PingPongFactory(tx.szx, tx.szy));
		nbl++;
	}
	//exit(0);

	//while (sz >= 2) {
		//pps.push_back(PingPongFactory(sz, sz));
		//sz /= 2;
		//nbl++;
	//}
	nb_pingpong_levels = nbl;
	printf("nbLevels: %d\n", nb_pingpong_levels);
	printf("pps size = %d\n", pps.size());

	// k0 offset structures
	Textures1D tx1d(k0v.size());
	tx1d.setFormat(internal_format, format, data_type);

	Array3D a(k0v.size());
	for (int i=0; i < k0v.size(); i++) {
		a(i) = k0v[i];
	}
	k0_offsetTex = tx1d.userDefined(a);
}
//----------------------------------------------------------------------
PingPong* DistanceTransform3D::PingPongFactory(int tex_size_x, int tex_size_y)
{
	Textures tx(tex_size_x, tex_size_y);
	//GLuint internal_format = GL_RGBA;
	GLuint internal_format = FLOAT_BUFFER;
	GLenum format 		= GL_RGBA;
	GLenum data_type 	= GL_FLOAT;
	//GLenum target = GL_TEXTURE_2D;
	GLenum target = TARGET;
	tx.setTarget(target);
	tx.setFormat(internal_format, format, data_type);

	// Next two lines ONLY work for FLOAT textures
	TexOGL* tex1 = tx.createOneColor(100.,100.,100.);
	TexOGL* tex2 = tx.createOneColor(100.,100.,100.);

	PingPong* pp = new PingPong(tex1, tex2);
	pp->point();
	pp->printInfo("pps: ");

	return pp;
}
//----------------------------------------------------------------------
Vec3i DistanceTransform3D::offset(int k)
{
	// 2D texture is a square tiling of 3D
	// offset is (x,y) location in 2D texture of beginning of 
	//    slice k in the 3D texture. 
	// Currently we have assumed that the 3D texture is cubic and that
	//     its dimension is a perfect square, in order to create a
	//     square 2D texture. 

	int nz = (int) sqrt(sz3d);
	int ox = k/nz;
	int oy = k - ox*nz;
	return Vec3i(ox*sz3d, oy*sz3d, 0.);
}
//----------------------------------------------------------------------
void DistanceTransform3D::computeOffsets()
{
	IVEC2 v;
	int nz = (int) sqrt(sz3d);

	for (int k=0; k < sz3d; k++) {
		v.x = k / nz;
		v.y = (k-v.x*nz)*sz3d;
		v.x *= sz3d;
		offsets.push_back(v);
	}
}
//----------------------------------------------------------------------
void DistanceTransform3D::two2three(int x, int y)
{
   int nz = sqrt(sz3d);

   int ox = x / sz3d; // integer division
   int oy = y / sz3d; // integer division

   int x3 = x - ox*sz3d;
   int y3 = y - oy*sz3d;
   int z3 = oy*nz + ox;
}
//----------------------------------------------------------------------
// tex_size should not appear in this routine. 
void DistanceTransform3D::two2three(float x, float y)
{
   #if 0
   int nz = sqrt(sz3d);

   float xx = x * tex_size;
   float yy = y * tex_size;

   int ox = (int) xx / sz3d; // integer division
   int oy = (int) yy / sz3d; // integer division

   float x3 = xx - ox*sz3d;
   float y3 = yy - oy*sz3d;
   float z3 = oy*nz + ox;

   float xx3 = x3 / sz3d;
   float yy3 = y3 / sz3d;
   float zz3 = z3 / sz3d;
   #endif
}
//----------------------------------------------------------------------
// do Ortho outside
void DistanceTransform3D::drawQuad(int szc3d, int szf3d, int ic, int jc, int ifi, int jfi)
{
	// Should not be required!

	int xv, yv;
	int xt, yt;

	//printf("  x0,y0= %d, %d, sz3d= %d\n",  x0, y0, sz3d);

	glBegin(GL_QUADS);
		xt = ic*szc3d;
		yt = jc*szc3d;
		xv = ifi*szf3d; // assumes that szc3d*2 = szf3d
		yv = jfi*szf3d;
		glTexCoord2f(xt,      yt);
		glVertex2i(xv,      yv);
		glTexCoord2f(xt+szc3d, yt);
		glVertex2i(xv+szf3d, yv);
		glTexCoord2f(xt+szc3d, yt+szc3d);
		glVertex2i(xv+szf3d, yv+szf3d);
		glTexCoord2f(xt,      yt+szc3d);
		glVertex2i(xv,      yv+szf3d);
	glEnd();
}
//----------------------------------------------------------------------
// Assuming I have the Voronoi mesh at level k (coarse), interpolate
// the Voronoi mesh onto level k+1 (finer)
// pc, pf: coarse and fine pingpong
// szc3d: 3D grid size corresponding to the coarase mesh (pc)
void DistanceTransform3D::interpolate(int szc3d, int szf3d,  PingPong& pc, PingPong& pf)
{
	pf.begin();
		pc.getTexture().bind();
		glEnable(pc.getTexture().getTarget());
		int icz = log2(szc3d+.001);
		int ifz = log2(szf3d+.001);
		//printf("szc3d, szf3d= %d, %d\n", szc3d, szf3d);
		//printf("icz, ifz= %d, %d\n", icz, ifz);
		Texsz& txc = texsz[icz];
		Texsz& txf = texsz[ifz];

	// Draw individual quads
	//printf("vbo_list size: %d\n", vbo_list->size());
	//printf("ix= %d\n", ix);
	//VBO<POINT3,POINT3>* vbo = (*vbo_list)[ix];

		if (2*txc.sz3d != txf.sz3d) {
			printf("Does not work unless successive texture sizes are multiples of two\n");
			exit(0);
		}

		int ic, jc; // index into coarse grid
		int ifine, jfine; // index into fine   grid
		int if1, jf1;
		int if2, jf2;

		//printf("coarse gx,gy= %d, %d\n", txc.gx, txc.gy);
		//printf("fine gx,gy= %d, %d\n", txf.gx, txf.gy);

		// should not be required
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
	
		gluOrtho2D(0., txf.szx, 0., txf.szy);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		//printf("szx, szy= %d, %d\n", txf.szx, txf.szy); exit(0);

		for (int kc=0; kc < szc3d; kc++) {
			ijOffset(kc, txc.gx, &ic, &jc);
			//printf("--- ic,jc= %d, %d, kc= %d\n", ic, jc, kc);
			ijOffset(2*kc, txf.gx, &if1, &jf1);
			//ijOffset(2*kc+1, txf.gx, &jf1, &jf2);
			if2 = if1 + 1;
			jf2 = jf1;
			//printf("    if1,jf1= %d, %d\n", if1, jf1);
			//printf("    if2,jf2= %d, %d\n", if2, jf2);
			drawQuad(szc3d, szf3d, ic, jc, if1, jf1);
			drawQuad(szc3d, szf3d, ic, jc, if2, jf2);
		}

		//printf("gxc,gyc= %d, %d\n", txc.gx, txc.gy);
		//printf("gxf,gyf= %d, %d\n", txf.gx, txf.gy);
		//printf("szc3d,szf3d= %d, %d\n", szc3d, szf3d);

		// should not be required
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

	pf.end();
}
//----------------------------------------------------------------------
// Assuming I have the Voronoi mesh at level k (coarse), interpolate
// the Voronoi mesh onto level k+1 (finer)
// pc, pf: coarse and fine pingpong
// szc3d: 3D grid size corresponding to the coarase mesh (pc)
// Full 3D implementation onto flat texture
void DistanceTransform3D::interpolateFull(int szc3d, int szf3d,  PingPong& pc, PingPong& pf)
{
	//printf("interpolateFull: szc3d=%d, szf3d= %d\n", szc3d, szf3d);

	pf.begin();
		pc.getTexture().bind();
		glEnable(pc.getTexture().getTarget());
		int icz = log2(szc3d+.001);
		int ifz = log2(szf3d+.001);
		//printf("szc3d, szf3d= %d, %d\n", szc3d, szf3d);
		//printf("icz, ifz= %d, %d\n", icz, ifz);
		Texsz& txc = texsz[icz];
		Texsz& txf = texsz[ifz];

		if (2*txc.sz3d != txf.sz3d) {
			printf("Does not work unless successive texture sizes are multiples of two\n");
			exit(0);
		}

		int ic, jc; // index into coarse grid
		int ifine, jfine; // index into fine   grid
		int if1, jf1;
		int if2, jf2;

		//printf("coarse gx,gy= %d, %d\n", txc.gx, txc.gy);
		//printf("fine gx,gy= %d, %d\n", txf.gx, txf.gy);

		// should not be required
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
	
		//printf("... szx, szy= %d, %d\n", txf.szx, txf.szy);
		gluOrtho2D(0., txf.szx, 0., txf.szy);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();


#if 1
	// argument should correspond to the coarser level
	VBO_T<POINT3,POINT3>* vboq = (*vbo_interp_list)[i_level-1];
	//printf("i_level = %d\n",i_level);
	int gxc = texsz[i_level-1].gx;
	int gyc = texsz[i_level-1].gy;
	int nb_quads = gxc*gyc;

	//pg.set_param1("sz3d", curTexSz->sz3d);
	//pg.set_tex("texture", tex, 0); 

	//printf("interpolate: 3d size coarse: %d\n", texsz[i_level-1].sz3d);
	//printf("interpolate: nb_quads: %d (%d,%d)\n", nb_quads, gxc, gyc);
	vboq->draw(GL_QUADS, 8*nb_quads); // 2nd arg = nb elements

#else

		for (int kc=0; kc < szc3d; kc++) {
			ijOffset(kc, txc.gx, &ic, &jc);
			//printf("--- ic,jc= %d, %d, kc= %d\n", ic, jc, kc);
			ijOffset(2*kc, txf.gx, &if1, &jf1);
			//ijOffset(2*kc+1, txf.gx, &jf1, &jf2);
			if2 = if1 + 1;
			jf2 = jf1;
			//printf("    if1,jf1= %d, %d\n", if1, jf1);
			//printf("    if2,jf2= %d, %d\n", if2, jf2);
			drawQuad(szc3d, szf3d, ic, jc, if1, jf1);
			drawQuad(szc3d, szf3d, ic, jc, if2, jf2);
		}
#endif

		//printf("gxc,gyc= %d, %d\n", txc.gx, txc.gy);
		//printf("gxf,gyf= %d, %d\n", txf.gx, txf.gy);
		//printf("szc3d,szf3d= %d, %d\n", szc3d, szf3d);

		// should not be required
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

	pf.end();
}
//----------------------------------------------------------------------
void DistanceTransform3D::ijOffset(int k, int gx, int* i, int *j)
{
	*j = k / gx;
	*i = k - *j * gx;
}
//----------------------------------------------------------------------
void DistanceTransform3D::setup_k0_structures()
{
	int offset = 0;
	int k0;

	for (int k=0; k < texsz.size(); k++) {
		koffsets.push_back(0);
		int sz3d = texsz[k].sz3d;
		int szx =  texsz[k].szx;
		int szy =  texsz[k].szy;
		int gx =  texsz[k].gx;

		for (int j=0; j < gy; j++) {
		for (int i=0; i < gx; i++) {
			k0 = i + j*gx;
			k0v.push_back((float) k0);
		}}

		offset += gx*gy;
	}
}
//----------------------------------------------------------------------
