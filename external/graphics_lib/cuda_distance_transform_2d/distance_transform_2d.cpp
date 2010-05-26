#include "distance_transform_2d.h"
#include "textures.h"
#include "globals.h"
#include "Array3D.h"

// Accelerated version

// 1. create a texture of size tex_size/2 with the same seed points  (create a shader
//  for this
//

//----------------------------------------------------------------------
DistanceTransformAcc::DistanceTransformAcc(Globals* g, int tex_size) 
{
	this->g = g;
	this->tex_size = tex_size;
	vbo = new VBO<POINT4, POINT4>();
	setupTextures();
	printf("log2(512)= %d\n", (int) log2(512));
	printf("log2(511)= %d\n", (int) log2(511));
}
//----------------------------------------------------------------------
DistanceTransformAcc::~DistanceTransformAcc()
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

#define PRINT(msg, ping, tex) \
	glFinish(); \
	printf("---- %s ---------------\n", msg); \
	ping->print(0,0,tex.getWidth(), tex.getHeight())

void DistanceTransformAcc::run()
{
// set i to pps.size();
// while (;;) {
// 	  draw seeds into level i
//    construct voronoi in level i
//    exit if i == 0
//    i = i - 1;
// }

	int nbsec = 3;

// First texture
	int i = nb_pingpong_levels - 1;
	i = 1;
	PingPong* ppo = pps[i];
	cur_ping_pong = ppo;
	TexOGL& tex = ppo->getTexture();
	//printf("*** w/h= %d, %d\n", tex.getWidth(), tex.getHeight());
	drawSeeds(*ppo); // seems ok
	//PRINT("DRAWSEEDS: ", ppo, tex); //exit(0);
	//SWAPBUF(ppo, "after drawSeeds"); 
	int nbl =  log2(ppo->getTexture().getWidth()+.001);  // square?
	//printf("nbl= %d\n", nbl); exit(0);
	drawVoronoi(ppo, nbl);
	//PRINT("DRAWVORONOI: ", ppo, tex); //exit(0);
	//SWAPBUF(ppo, "after drawVoronoi");

	while (1) {
		i = i - 1; // i=0: finest texture
		PingPong* ppn = pps[i];
		cur_ping_pong = ppn;
		// draw pps[i] into pps[i-1];
		ppn->begin();
		ppn->drawTexture(ppo->getTexture()); // NOT WORKING!!!
		//drawSeedsNoBegin(*ppn, ppo->getTexture()); // including textures
		ppn->end();
		ppn->undoSwap();
		drawSeeds(*ppn); // including textures
		//drawSeedsNoBegin(*ppn, ppo->getTexture()); // including textures
		//ppn->end();
		//PRINT("DRAWSEEDS: ", ppn, ppn->getTexture()); //exit(0);
		//ppn->undoSwap();
		//SWAPBUF(ppn, "after drawTexture"); 
		//drawSeeds(*ppn);
		//SWAPBUF(ppn, "after drawSeeds"); 
		int nbl =  2; // should be 1
		drawVoronoi(ppn, nbl);
		//SWAPBUF(ppn, "after drawVoronoi"); 
		if (i == 0) {
			cur_ping_pong = ppn;
		//PRINT("DRAWVORONOI: ", ppn, ppn->getTexture()); exit(0);
			ppn->toBackBuffer();
			//printf("last iteration\n"); fflush(stdout); glutSwapBuffers(); sleep(50);
			return;
		}
		ppo = ppn;
	}
}
//----------------------------------------------------------------------
void DistanceTransformAcc::drawVoronoi(PingPong* pph, TexOGL& tex)
{
	stepLength = 1;
	for (int i=0; i < 1; i++) {
		updateTexture(pph, tex);
		stepLength *= 2;
	}
	return;

	for (int i=0; i < 10; i++) {
		if (stepLength < 1) stepLength= 1;
		updateTexture(pph, tex);
		stepLength /= 2;
	}
}
//----------------------------------------------------------------------
void DistanceTransformAcc::drawVoronoi(PingPong* pph, int steps)
{
	stepLength= pph->getTexture().getHeight() / 2;
	//stepLength= tex_size/2.;

	int nbl = steps+1;
	stepLength = (int) pow(2., steps-1);
	//printf("nbl= %d, stepLength= %d\n", nbl, stepLength);

	for (int i=0; i < nbl; i++) {
		updateTexture(pph);
		stepLength /= 2;
		if (stepLength < 1) stepLength = 1;
	}
	//printf("end drawVoronoi\n");
	return;
}
//----------------------------------------------------------------------
void DistanceTransformAcc::updateTexture(PingPong* ping, TexOGL& tex)
{
/// tex: texture to bind to
	ping->begin();
	glEnable(tex.getTarget());
	tex.bind();
	CG::Program& pg = g->enableShader(g->voronoi_gpu_rect_acc);

	pg.set_param1("stepLength", (float) stepLength);
	pg.set_param1("tex_size", (float) tex.getWidth());
	pg.set_tex("texture", tex, 0);

	int sz = tex.getWidth(); 
	if (tex.getHeight() != sz) {
		printf("updateTexture: texture should be square for now\n");
		exit(0);
	}
// sz is 512

	glBegin(GL_QUADS);
		glTexCoord2d(0., 0.);
		glVertex2f(0., 0.);
		glTexCoord2d(sz, 0.);
		glVertex2f(1., 0.);
		glTexCoord2d(sz, sz);
		glVertex2f(1., 1.);
		glTexCoord2d(0, sz);
		glVertex2f(0., 1.);
	glEnd();

	glDisable(tex.getTarget());
	glUseProgram(0);
	ping->end();
}
//----------------------------------------------------------------------
void DistanceTransformAcc::updateTexture(PingPong* ping)
{
	TexOGL& tex = ping->getTexture();

	ping->begin();
	glEnable(tex.getTarget());
	tex.bind();
	CG::Program& pg = g->enableShader(g->voronoi_gpu_rect_acc);

	//printf("step: %f\n", (float) stepLength);
	pg.set_param1("stepLength", (float) stepLength);
	pg.set_param1("tex_size", (float) tex.getWidth());
	pg.set_tex("texture", tex, 0);

	int sz = tex.getWidth(); 
	if (tex.getHeight() != sz) {
		printf("updateTexture: texture should be square for now\n");
		exit(0);
	}

	glBegin(GL_QUADS);
		glTexCoord2d(0., 0.);
		glVertex2f(0., 0.);
		glTexCoord2d(sz, 0.);
		glVertex2f(1., 0.);
		glTexCoord2d(sz, sz);
		glVertex2f(1., 1.);
		glTexCoord2d(0, sz);
		glVertex2f(0., 1.);
	glEnd();

	glDisable(tex.getTarget());
	glUseProgram(0);
	ping->end();
}
//----------------------------------------------------------------------
void DistanceTransformAcc::computeSeeds(int nb)
{
	Seed seed;
	POINT4 pt;

#if 0
	seed.x = 0.02; seed.y = 0.02; seed.z = 0.0; seed.w = 1.0;
	pt.x = seed.x; pt.y = seed.y; pt.z = seed.z; pt.w = seed.w;
	seeds.push_back(seed); seed_pts.push_back(pt); seed_col.push_back(pt);

	seed.x = 0.5; seed.y = 0.5; seed.z = 0.0; seed.w = 1.0;
	pt.x = seed.x; pt.y = seed.y; pt.z = seed.z; pt.w = seed.w;
	seeds.push_back(seed); seed_pts.push_back(pt); seed_col.push_back(pt);

	seed.x = 0.9; seed.y = 0.9; seed.z = 0.0; seed.w = 1.0;
	pt.x = seed.x; pt.y = seed.y; pt.z = seed.z; pt.w = seed.w;
	seeds.push_back(seed); seed_pts.push_back(pt); seed_col.push_back(pt);



	vbo->create(&seed_pts, &seed_col);
	return;
#endif

	#if 0
	for (int i=0; i < 128; i++) {
		seed.x = i / 128.;
		seed.y = 1. / 128.;
		seeds.push_back(seed);
	}

	return;
	#endif

	srand(12); // seed for random number generator
	float_seeds.reserve(nb*4);
	printf("float_seeds size: %d\n", float_seeds.size());

	for (int i=0; i < nb; i++) {
		//seed.x = u.rand_float();
		//seed.y = u.rand_float();
		seed.x = rand() / (double) RAND_MAX;
		seed.y = rand() / (double) RAND_MAX;
		seed.z = 0.0;
		//seed.w = 1.; // (float) i; (does not work if > 1 because of saturation issues)
		seed.w = (float) i; // (does not work if > 1 because of saturation issues)
		seeds.push_back(seed);

		float_seeds[     i] = seed.x;
		float_seeds[  nb+i] = seed.y;
		float_seeds[2*nb+i] = seed.z;
		float_seeds[3*nb+i] = seed.w;

		pt.x = seed.x;
		pt.y = seed.y;
		pt.z = seed.z;
		pt.w = seed.w;
		seed_pts.push_back(pt);
		seed_col.push_back(pt);
		//printf("***seed %d: %f, %f\n", i, seed.x, seed.y, seed.z, seed.w);
		if (i < 10) printf("***seed %d: %f, %f\n", i, 1024*seed.x, 1024*seed.y, seed.z, seed.w);
	}
	printf("*** ......\n");

	vbo->create(&seed_pts, &seed_col);
}
//----------------------------------------------------------------------
//void DistanceTransformAcc::setupPingPong(int tex_size)
//{
//}
//----------------------------------------------------------------------
void DistanceTransformAcc::drawSeedsNoBegin(PingPong& ping, TexOGL& tex)
{
// Somehow, the background color is white!!

	glUseProgram(0);
	glDisable(ping.getTexture().getTarget());
	glDisable(GL_TEXTURE_2D); // should not be required
	//ping.begin();

	CG::Program& pg = g->enableShader(g->draw_seed);
	pg.set_param1("szx", (float) tex.getWidth());
	pg.set_param1("szy", (float) tex.getHeight());

	//glClearColor(1.,0.,0.,1.); // WORKED
    //glClear(GL_COLOR_BUFFER_BIT);

	// Should not be required!
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0., tex.getWidth(), 0., tex.getHeight());
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	tex.bind();
	glEnable(tex.getTarget());
	int szx = tex.getWidth();
	int szy = tex.getHeight();

	glBegin(GL_QUADS);
	  glTexCoord2f(0., 0.);
	  glVertex2f(0., 0.);

	  glTexCoord2f(szx, 0.);
	  glVertex2f(1., 0.);

	  glTexCoord2f(szx, szy);
	  glVertex2f(1., 1.);

	  glTexCoord2f(0, szy);
	  glVertex2f(0., 1.);
	glEnd();

	glDisable(tex.getTarget());


	//glColor3f(1.,1.,1.); // HOW CAN THIS INFLUENCE THE ENTIRE TEXTURE????
	vbo->draw(GL_POINTS, seed_pts.size());
	//printf("nb seeds: %d\n", seed_pts.size()); exit(0);

	// should not be required
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glUseProgram(0);
	//ping.end();
}
//----------------------------------------------------------------------
void DistanceTransformAcc::drawSeeds(PingPong& ping)
{
// Somehow, the background color is white!!

	glUseProgram(0);
	TexOGL& tex = ping.getTexture();
	glDisable(ping.getTexture().getTarget());
	glDisable(GL_TEXTURE_2D); // should not be required
	ping.begin();

	CG::Program& pg = g->enableShader(g->draw_seed);
	pg.set_param1("szx", (float) tex.getWidth());
	pg.set_param1("szy", (float) tex.getHeight());


	//glClearColor(1.,0.,0.,1.); // WORKED
    //glClear(GL_COLOR_BUFFER_BIT);

	// Should not be required!
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0., tex.getWidth(), 0., tex.getHeight());
	//printf("w,h= %d, %d\n", tex.getWidth(), tex.getHeight()); exit(0);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	vbo->draw(GL_POINTS, seed_pts.size());

	// should not be required
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glUseProgram(0);
	ping.end();
}
//----------------------------------------------------------------------
void DistanceTransformAcc::setupTextures()
{
	// add seeds
	float dx = 1./(tex_size);
	float dy = 1./(tex_size);


	// I get one more domain than the number of seeds. Correct?
	int nb_seeds = 5000;
	computeSeeds(nb_seeds);

	Textures tx(tex_size);
	//GLuint internal_format = GL_RGBA;
	GLuint internal_format = FLOAT_BUFFER;
	GLenum format 		= GL_RGBA;
	GLenum data_type 	= GL_FLOAT;
	//GLenum target = GL_TEXTURE_2D;
	GLenum target = TARGET;
	tx.setTarget(target);
	tx.setFormat(internal_format, format, data_type);

	TexOGL* tex = tx.createEmpty();
	pp = new PingPong(tex);

	// cannot reuse tex since only pointers are manipulated. There are no copies.
	Textures txh(tex_size/2); // for multigrid approach
	//internal_format = GL_RGBA; // really want float_buffer
	txh.setFormat(internal_format, format, data_type);
	txh.setTarget(target);
	TexOGL* texh1 = txh.createOneColor(100.,100.,100.);
	TexOGL* texh2 = txh.createOneColor(100.,100.,100.);
	pph = new PingPong(texh1, texh2);

	pp->point();
	pph->point();

	// Create hierarchy of pingpong textures
	int sz = tex_size;
	int nbl = 0;

	while (sz >= 2) {
		pps.push_back(PingPongFactory(sz));
		sz /= 2;
		nbl++;
	}

	nb_pingpong_levels = nbl;

	printf("nbLevels: %d\n", nb_pingpong_levels);
	printf("pps size = %d\n", pps.size());

	pp->printInfo("pp: ");
	pph->printInfo("pph: ");
}
//----------------------------------------------------------------------
PingPong* DistanceTransformAcc::PingPongFactory(int tex_size)
{
	Textures tx(tex_size);
	//GLuint internal_format = GL_RGBA;
	GLuint internal_format = FLOAT_BUFFER;
	GLenum format 		= GL_RGBA;
	GLenum data_type 	= GL_FLOAT;
	//GLenum target = GL_TEXTURE_2D;
	GLenum target = TARGET;
	tx.setTarget(target);
	tx.setFormat(internal_format, format, data_type);

	// cannot reuse tex since only pointers are manipulated. There are no copies.
	//internal_format = GL_RGBA; // really want float_buffer
	// Next two lines ONLY work for FLOAT textures
	//TexOGL* tex1 = tx.createOneColor(0.5,0.5,0.5);
	//TexOGL* tex2 = tx.createOneColor(0.5,0.5,0.5);
	TexOGL* tex1 = tx.createOneColor(100.,100.,100.);
	TexOGL* tex2 = tx.createOneColor(100.,100.,100.);

	PingPong* pp = new PingPong(tex1, tex2);
	pp->point();
	pp->printInfo("pps: ");

	return pp;
}
//----------------------------------------------------------------------
