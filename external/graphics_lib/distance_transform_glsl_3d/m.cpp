// must initialize glew to make shaders work (DO NOT KNOW WHY)

#include "glincludes.h"

#include <stdio.h>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include "Vec3.h"
#include "ArrayT.h"
#include "gl_state.h"
#include "globals.h"
#include "timingGE.h"
#include "tex_ogl.h"
#include "textures.h"
#include "Array3D.h"
#include "utils.h"
#include "framebufferObject.h"
#include "distance_transform.h"
#include "gl_state.h"

using namespace std;


DistanceTransform* dtransf;
utils u;
TexOGL* quad1_tex;
TexOGL* pos_tx[2]; // to hold data related to Voronoi mesh
GLuint quad1;
FramebufferObject* fbo_pos;
//AbstractLic* alic;


Timings ts;
Timer t1(ts, "frame");

#define FRAND_MAX  (1./(RAND_MAX));

Globals g;  // may be required as a global

void help();


// create a list of circles, put them in a display list
// and draw them. Probably not the most efficient way. 
// Resolution of the circle is fixed. For now, draw polygons (16 pts
// per circle. Not very smart. Best to use texture splats in 3D. For 
// later. 


//----------------------------------------------------------------------
// should go into more general library
void disableAllShaders()
{
	glUseProgram(0);
}
//----------------------------------------------------------------------
void checkError(char* msg)   //   also sin superquadric
{
#if 1
	printf("glerror: %s,  %s\n", msg, gluErrorString(glGetError())); // error
#endif
}
//----------------------------------------------------------------------
CG::Program& setupShaderProgram(char* name, GLuint *shader_id, int which=0)
{
	CG::GL& gl = *g.gl;
	printf("g.gl= %d\n", g.gl);
	*shader_id = gl.setupShaderProgram();
	printf("shader_id= %d\n", *shader_id);
	CG::Program* pg = gl.getShader(*shader_id);
	printf("pg= %d\n", pg);

	char frag_name[32];
	char vert_name[32];

	sprintf(frag_name, "%s.frag", name);
	sprintf(vert_name, "%s.vert", name);

	printf("frag_name= %s\n", frag_name);
	printf("vert_name= %s\n", vert_name);

	switch (which) {
	case 0:
		pg->addFragmentShader(frag_name);
	checkError("error 3.2");
		pg->addVertexShader(vert_name);
	checkError("error 3.1");
		printf("vert_name= %s\n", vert_name);
		break;
	case 1:
		pg->addVertexShader(vert_name);
	checkError("error 4");
		break;
	case 2:
		pg->addFragmentShader(frag_name);
	checkError("error 5");
		break;
	}

	pg->link();
}
//----------------------------------------------------------------------
void init_shaders()
{
	CG::GL& gl = *g.gl;
	CG::Program *pg;

	// add multiple shaders to a particular program
	#if 0
	g.p_bench11_rect = gl.setupShaderProgram();
	pg = gl.getShader(g.p_bench11_rect);
	pg->addFragmentShader("bench11_rect.frag");
	pg->addFragmentShader("boundaries_age_rect.frag");
	pg->link();
	#endif

	// add vertex and fragment shaders
	setupShaderProgram("voronoi_gpu",  &g.voronoi_gpu, 0);
}
//----------------------------------------------------------------------
void display(void)
{
	static int count=0; 

	t1.start();

	count++;
	if (count % 10 == 0) printf("count= %d\n", count);

	#if 0
	glBegin(GL_POLYGON);
	glColor3f(1.,1.,0.);
	glVertex2f(0.,0.);
	glVertex2f(1.,0.);
	glVertex2f(1.,1.);
	glVertex2f(0.,1.);
	glVertex2f(0.,0.);
	glEnd();
	#endif

	dtransf->run();
	TexOGL& final = *dtransf->getFinalTexture();
	// draw this texture to the screen
	glUseProgram(0);

    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT);
	u.draw_quad_multi(final);

	//glFlush();
	glFinish();
	glutSwapBuffers();


	t1.end();
	if (count % 100 == 0) {
		ts.dumpTimings();
		t1.reset();
	}

	//sleep(2);
}
//----------------------------------------------------------------------
void reshape(int w, int h)
{
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double near = -1.;
    double far = 1.;
	double scale = 1;
    if (w <= h) {
        glOrtho(-scale,scale,-scale*h/w, scale*h/w, near, far);
    } else {
        glOrtho(-scale*w/h,scale*w/h, -scale, scale, near, far);
    }
    glMatrixMode(GL_MODELVIEW);
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

    //glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

	// Materials are NOT not having any effect
    GLfloat ambientMaterial[4] = { 1.0, 0.7, 1.0, 1.0 };
    GLfloat diffuseMaterial[4] = { 0.7, 0.7, 0.7, 1.0 };
    glMaterialfv(GL_FRONT, GL_AMBIENT, ambientMaterial);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseMaterial);

    //glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);  // do not draw circles facing way from the viewer
	//glFrontFace(GL_CW);

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
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    //glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    //glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(512, 512);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Quadrics");
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitialize();
	//initializeEnvironment();

  	glewInit();

	init_shaders(); // define and compile program shaders
	disableAllShaders();  // disable shaders


	int tex_size = 512;
	dtransf = new DistanceTransform(&g, tex_size);

    glutMainLoop();

    return 0;
}
//----------------------------------------------------------------------
