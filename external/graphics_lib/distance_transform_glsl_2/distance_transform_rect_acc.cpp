#include "distance_transform_acc.h"
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
	vbo = new VBO<POINT3, POINT3>();
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

	 
void DistanceTransformAcc::run()
{
// set i to pps.size();
// while (;;) {
// 	  draw seeds into level i
//    construct voronoi in level i
//    exit if i == 0
//    i = i - 1;
// }

	int nbsec = 0;

// First texture
	int i = nb_pingpong_levels - 1;
	//i = 2;
	PingPong* ppo = pps[i];
	drawSeeds(*ppo); // seems ok
	//SWAPBUF(ppo, "after drawSeeds"); 
	drawVoronoi(ppo);
	//SWAPBUF(ppo, "after drawVoronoi");

	while (1) {
		i = i - 1; // i=0: finest texture
		PingPong* ppn = pps[i];
		// draw pps[i] into pps[i-1];
		ppn->begin();
		//ppn->drawTexture(ppo->getTexture()); // NOT WORKING!!!
		drawSeedsNoBegin(*ppn, ppo->getTexture()); // including textures
		//drawSeedsNoBegin(*ppn, ppo->getBuffer()); // SHOULD NOT BE CORRECT
		ppn->end();
		//ppn->undoSwap();
		//SWAPBUF(ppn, "after drawTexture"); 
		//drawSeeds(*ppn);
		//SWAPBUF(ppn, "after drawSeeds"); 
		drawVoronoi(ppn);
		//SWAPBUF(ppn, "after drawVoronoi"); 
		if (i == 0) {
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
void DistanceTransformAcc::drawVoronoi(PingPong* pph)
{
	stepLength= pph->getTexture().getHeight() / 2;
	//stepLength= tex_size/2.;
	stepLength= 2;
	int nbl = log2(pph->getTexture().getWidth()) + 1;
	for (int i=0; i < 2; i++) {
		updateTexture(pph);
		stepLength /= 2;
		if (stepLength < 1) stepLength = 1;
	}
	//printf("end drawVoronoi\n");
	return;
}
//----------------------------------------------------------------------
void DistanceTransformAcc::resetTextureWithSeeds() { }
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
	POINT3 pt;

	#if 0
	for (int i=0; i < 128; i++) {
		seed.x = i / 128.;
		seed.y = 1. / 128.;
		seeds.push_back(seed);
	}

	return;
	#endif

	srand(12); // seed for random number generator

	for (int i=0; i < nb; i++) {
		//seed.x = u.rand_float();
		//seed.y = u.rand_float();
		seed.x = rand() / (double) RAND_MAX;
		seed.y = rand() / (double) RAND_MAX;
		seeds.push_back(seed);

		pt.x = seed.x;
		pt.y = seed.y;
		pt.z = 0.0;
		seed_pts.push_back(pt);
		seed_col.push_back(pt);
		printf("seed %d: %f, %f\n", i, seed.x, seed.y);
	}

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

	//glClearColor(1.,0.,0.,1.); // WORKED
    //glClear(GL_COLOR_BUFFER_BIT);

	// Should not be required!
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0., 1., 0., 1.);
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

	// should not be required
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	//ping.end();
}
//----------------------------------------------------------------------
void DistanceTransformAcc::drawSeeds(PingPong& ping)
{
// Somehow, the background color is white!!

	glUseProgram(0);
	glDisable(ping.getTexture().getTarget());
	glDisable(GL_TEXTURE_2D); // should not be required
	ping.begin();

	//glClearColor(1.,0.,0.,1.); // WORKED
    //glClear(GL_COLOR_BUFFER_BIT);

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
void DistanceTransformAcc::setupTextures()
{
	// add seeds
	Array3D a(4, tex_size, tex_size);
	float dx = 1./(tex_size);
	float dy = 1./(tex_size);


	for (int j=0; j < tex_size; j++) {
	for (int i=0; i < tex_size; i++) {
		a(0,i,j) = 1000.;
		a(1,i,j) = 1000.;
		a(2,i,j) = 0.;
		a(3,i,j) = 1.;
	}}

	// I get one more domain than the number of seeds. Correct?
	int nb_seeds = 20;
	computeSeeds(nb_seeds);

	for (int i=0; i < nb_seeds; i++) {
		int x = tex_size * u.rand_float();
		int y = tex_size * u.rand_float();
		a(0, x, y) =  seeds[i].x;
		a(1, x, y) =  seeds[i].y;
		a(2, x, y) = 0.0;
		a(3, x, y) = 1.0;
	}

	Textures tx(tex_size);
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
