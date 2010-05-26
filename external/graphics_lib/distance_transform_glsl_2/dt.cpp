#include "distance_transform.h"
#include "textures.h"
#include "globals.h"
#include "Array3D.h"

//----------------------------------------------------------------------
DistanceTransform::DistanceTransform(Globals* g, int tex_size) : AbstractLic(g, tex_size)
{
	this->g = g;
	this->tex_size = tex_size;
	setupTextures();
	setupFBOs();
	curBuf = 0;
}
//----------------------------------------------------------------------
DistanceTransform::~DistanceTransform()
{}
//----------------------------------------------------------------------
TexOGL* DistanceTransform::getFinalTexture()
{
	return pos_tx[curTex];
}
//----------------------------------------------------------------------
void DistanceTransform::run()
{
	stepLength = tex_size / 2;
	fbo_pos->Bind();


	for (int i=0; i < 1; i++) {
		if (stepLength < 1) stepLength = 1;
		curBuf = 1 - curBuf;
		curTex = 1 - curBuf;
		glDrawBuffer(fboBuf[curBuf]); // write to curBuf, read from texture 1-curBuf
		updateTexture();
		stepLength /= 2;
	}
	FramebufferObject::Disable();
	curTex = curBuf;
}
//----------------------------------------------------------------------
void DistanceTransform::updateTexture()
{
	glEnable(GL_TEXTURE_2D);
	CG::GL& gl = *g->gl;
	CG::Program& pg = g->enableShader(g->voronoi_gpu);

	float step = (float) stepLength / tex_size;
	pg.set_param1("stepLength", step);
	pg.set_tex("texture", *pos_tx[1-curBuf], 0);

	// clearing should not be required
    //glClearColor(0.,0.,0.,0.);
    //glClear(GL_COLOR_BUFFER_BIT);

	u.draw_quad_multi(*pos_tx[curBuf]);
	glUseProgram(0);
}
//----------------------------------------------------------------------
void DistanceTransform::setupFBOs()
{
    fbo_pos = new FramebufferObject();
    setupFbo(*fbo_pos, *pos_tx[0], *pos_tx[1]);
}
//----------------------------------------------------------------------
void DistanceTransform::setupTextures()
{
	// add seeds
	Array3D a(4, tex_size, tex_size);
	float dx = 1./(tex_size);
	float dy = 1./(tex_size);

	a.setTo(1.0);

	#if 1
	for (int j=0; j < tex_size; j++) {
	for (int i=0; i < tex_size; i++) {
		a(0, i, j) = 1.;
		a(1, i, j) = 1.;
		a(2, i, j) = 1.;
		a(3, i, j) = 1.;
	}}
	#endif

	#if 0
	for (int i=0; i < 500; i++) {
		int x = tex_size * u.rand_float();
		int y = tex_size * u.rand_float();
		a(0, x, y) = x * dx;
		a(1, x, y) = y * dy;
		a(2, x, y) = 0.0;
		a(3, x, y) = 1.0;
		printf("i=%d, x,y= %f, %f\n", i, x*dx, y*dy);
	}
	#endif

	Textures tx(tex_size);
	GLuint internal_format = GL_RGBA;
	//GLuint internal_format = FLOAT_BUFFER;
	GLenum format 		= GL_RGBA;
	GLenum data_type 	= GL_FLOAT;
	GLenum target = GL_TEXTURE_2D;
	tx.setTarget(target);
	tx.setFormat(internal_format, format, data_type);
	//GLuint target = GL_TEXTURE_RECTANGLE_NV;

	pos_tx[0] = tx.userDefined(a);
	pos_tx[1] = tx.userDefined(a);
	pos_tx[0]->point();
	pos_tx[1]->point();
}
//----------------------------------------------------------------------
