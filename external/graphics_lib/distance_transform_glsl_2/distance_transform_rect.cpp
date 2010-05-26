#include "distance_transform.h"
#include "textures.h"
#include "globals.h"
#include "Array3D.h"

//----------------------------------------------------------------------
DistanceTransform::DistanceTransform(Globals* g, int tex_size) 
{
	this->g = g;
	this->tex_size = tex_size;
	setupTextures();
}
//----------------------------------------------------------------------
DistanceTransform::~DistanceTransform()
{	
	delete pp;
}
//----------------------------------------------------------------------
void DistanceTransform::run()
{
	stepLength = tex_size / 2;

	for (int i=0; i < 9; i++) {
		if (stepLength < 1) stepLength = 1;
		updateTexture();
		stepLength /= 2;
	}

	pp->toBackBuffer();
}
//----------------------------------------------------------------------
void DistanceTransform::updateTexture()
{
	TexOGL& tex = pp->getTexture();

	pp->begin();
	glEnable(tex.getTarget());
	tex.bind();
	CG::Program& pg = g->enableShader(g->voronoi_gpu_rect);

	//u.checkError("update");

	printf("step: %f\n", (float) stepLength);
	pg.set_param1("stepLength", (float) stepLength);
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
	pp->end();
}
//----------------------------------------------------------------------
void DistanceTransform::setupTextures()
{
	// add seeds
	Array3D a(4, tex_size, tex_size);
	float dx = 1./(tex_size);
	float dy = 1./(tex_size);

	//a.setTo(1000.);

	for (int j=0; j < tex_size; j++) {
	for (int i=0; i < tex_size; i++) {
		a(0,i,j) = 1000.;
		a(1,i,j) = 1000.;
		a(2,i,j) = 0.;
		a(3,i,j) = 1.;
	}}

	// I get one more domain than the number of seeds. Correct?
	int nb_seeds = 200;

	#if 1
	for (int i=0; i < nb_seeds; i++) {
		int x = tex_size * u.rand_float();
		int y = tex_size * u.rand_float();
		printf("x,y= %d, %d\n", x, y);
		a(0, x, y) =  x * dx;    // seed coordinate [0,1]
		a(1, x, y) =  y * dy;	   // seed coordinate [0,1]
		a(2, x, y) = 0.0;          // distance
		a(3, x, y) = 1.0;          // not used
	}
	#endif

	Textures tx(tex_size);
	//GLuint internal_format = GL_RGBA;
	GLuint internal_format = FLOAT_BUFFER;
	GLenum format 		= GL_RGBA;
	GLenum data_type 	= GL_FLOAT;
	//GLenum target = GL_TEXTURE_2D;
	GLenum target = TARGET;
	tx.setTarget(target);
	tx.setFormat(internal_format, format, data_type);

	//TexOGL* tex = tx.createBWNoise();
	TexOGL* tex = tx.userDefined(a);
	pp = new PingPong(tex);

	TexOGL& t1 = pp->getTexture();
	TexOGL& t2 = pp->getBuffer();
	t1.point();
	t2.point();

	printf("pp info: %s\n", pp->info().c_str());
}
//----------------------------------------------------------------------
