// must initialize glew to make shaders work (DO NOT KNOW WHY)

#include "glincludes.h"

#include <stdio.h>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include "globals.h"
#include "timingGE.h"
#include "tex_ogl.h"
#include "textures.h"
#include "ping_pong.h"
#include "distance_transform_3d.h"
#include "utils.h"
#include "clock.h"
#include "timege.h"

//DistanceTransform* dtransf;
DistanceTransform3D* dtransf;
utils u;

using namespace std;
using namespace GE;

Globals g;

Timings ts;
Timer t1(ts, "frame");

//Clock clock1("gordon");
//Clock clock2("sleep");

GE::Time clock1("gordon");
GE::Time clock2("sleep");

void help();


//======================================================================
// this method should be in gl_state.cpp (must get rid of global reference)
//======================================================================
void init_shaders()
{
	g.gl->setupShaderProgram("passthrough",  &g.p_passthrough, CG::GL::FRAG_SHADER); // frag only
	g.gl->setupShaderProgram("voronoi_gpu_3d",  &g.voronoi_gpu_3d, CG::GL::BOTH_SHADERS);
	g.gl->setupShaderProgram("voronoi_gpu_3d_full",  &g.voronoi_gpu_3d_full, CG::GL::BOTH_SHADERS);
	g.gl->setupShaderProgram("draw_seed",  &g.draw_seed, CG::GL::BOTH_SHADERS);
	g.gl->setupShaderProgram("draw_seed_final",  &g.draw_seed_final, CG::GL::BOTH_SHADERS);
	g.gl->setupShaderProgram("color_voronoi",  &g.color_voronoi, CG::GL::FRAG_SHADER);
}
//----------------------------------------------------------------------

// Write into a texture some image. Copy this image into a second texture
// (use a pingpong, initialize to previous texture: (use it as first texture
// of pingpong). Copy this texture into the second texture, and write to 
// the second texture. Draw the first texture to the first half of the screen, 
// and the second texture to the second half of the screen. 
// FBO: cur_tex: I'll call the frontmost texture (the one that was just written to). 
// The second texture is the one that WILL be written to. 
// How to write a texture to the screen? 
//----------------------------------------------------------------------
void display(void)
{
	static int count=0; 
	//t1.start();

	count++;

	// use fbos, then draw to back buffer
	//dtransf->run();
	clock1.start();
	dtransf->runFull();

	//glFlush();
	//glFinish();
	glutSwapBuffers();
	clock1.end();


	#if 1
	//t1.end();
	if (count % 10 == 0) {
		printf("count= %d\n", count);
		//ts.dumpTimings();
		clock1.print();
		clock1.reset();
		//t1.reset();
	}
	#endif

	//t1.reset();
}
//----------------------------------------------------------------------
void reshape(int w, int h)
{
	printf("w,h= %d, %d\n", w, h);
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double near = -1.;
    double far = 1.;
	double scale = 4;

    //glOrtho(-1.,1.,-1.,1.,-1.,1.);
    glOrtho(0.,1.,0.,1.,0.,1.);
    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
//----------------------------------------------------------------------
void initializeEnvironment()
{
    glClearColor(0.,0.,0.,0.);
    glShadeModel(GL_SMOOTH);

#if 0
    glBlendFunc (GL_SRC_ALPHA, GL_ONE);
#endif

    GLfloat light_ambient[] = {0.2,0.2,0.2,0.9};
    GLfloat light_diffuse[] = {0.5,0.5,0.5,1.0};
    GLfloat light_specular[] = {.5,.5,.5, 1.};
    GLfloat light_specular_power[] = {15.,15.,15.,0.};

    GLfloat light_position[] = {0.,0.,1.,0.};
    GLfloat light_position1[] = {0.,0.,-1.,0.};

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

    glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT1, GL_POSITION, light_position1);
    glLightfv(GL_LIGHT1, GL_SPECULAR, light_specular);

    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    //glColorMaterial(GL_FRONT, GL_AMBIENT); // (might not exist)

	// Materials are NOT not having any effect
    GLfloat ambientMaterial[4] = { 1.0, 0.7, 1.0, 1.0 };
    //GLfloat diffuseMaterial[4] = { 0.7, 0.7, 0.7, 1.0 };
    GLfloat diffuseMaterial[4] = { 0.0, 0.0, 0.0, 1.0 };
    glMaterialfv(GL_FRONT, GL_AMBIENT, ambientMaterial);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseMaterial);

    //glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);  // do not draw circles facing way from the viewer
	//glFrontFace(GL_CW);
	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
}
//----------------------------------------------------------------------
void setupObjects(int sz3d)
{
	dtransf = new DistanceTransform3D(&g, sz3d);
}
//----------------------------------------------------------------------
static void keyboard(unsigned char key, int x, int y)
{
#if 1
    switch (key) {
      case 'q':
	  case 27:
		exit(0);
		break;
	}
#endif
}

//----------------------------------------------------------------------
void help()
{
}
//----------------------------------------------------------------------
void idle()
{
	glutPostRedisplay();
}
//----------------------------------------------------------------------
void glutInitialize()
{
	int winx = 1024; // window size
	int winy = 1024;

    //glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    //glutInitWindowSize(tex_size, tex_size);
    glutInitWindowSize(winx, winy);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Quadrics");
    glutKeyboardFunc(keyboard);
	glViewport(0,0, winx, winy);
    glutIdleFunc(idle);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

	g.setWinSize(winx, winy);
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
	//int sz3d = 256; // 4096 x 4096 textures: does not work too much
	int sz3d = 8;
	//int sz3d = 64;

    glutInit(&argc, argv);
    glutInitialize();
	//initializeEnvironment();

  	glewInit();
	setupObjects(sz3d);
	init_shaders();
	g.disableShaders();  // disable shaders

    glutMainLoop();

    return 0;
}
//----------------------------------------------------------------------
