#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <glincludes.h>

//#include <vector_functions.h> // cuda

#include "stopwatch_linux.h"
#include "distance_transform_2d.h"
#include "centroids.h"

#include "utils.h"
#include "globals.h"
#include "timege.h"

int window_width  = 512;
int window_height = 512;

GE::Time clock1("gordon");
GE::Time clock2("sleep");
GE::Time clock3("histogram");


// used to handle templatized functions in .cu files
#include "map.h"

// Voronoi 3D generator
DistanceTransformAcc* dtransf;
Centroids* centroids;
Globals g;

//======================================================================
// this method should be in gl_state.cpp (must get rid of global reference)
//======================================================================
void init_shaders()
{
	g.gl->setupShaderProgram("voronoi_gpu_rect_acc",  &g.voronoi_gpu_rect_acc, CG::GL::BOTH_SHADERS);
	g.gl->setupShaderProgram("draw_seed",  &g.draw_seed, CG::GL::BOTH_SHADERS);
}
//----------------------------------------------------------------------
void testPingPong_a()
{ }
//----------------------------------------------------------------------
void checkError(char* msg)
{
#if 1
	printf("glerror: %s,  %s\n", msg, gluErrorString(glGetError())); // error
#endif
}
//----------------------------------------------------------------------
void generateVoronoiRasterization()
{
	dtransf->run();     // Voronoi generator
}
//----------------------------------------------------------------------
void postprocessCentroids()
{
	int szx, szy;
	centroids->getDims(&szx, &szy);
	centroids->computeOnCPU(szx, szy);
	centroids->simulateGPUonCPU(szx, szy);

	std::vector<Seed>& seeds = dtransf->getSeeds();
	centroids->printComputationCPU();
	centroids->printComputationGPUonCPU();
	centroids->printComputationGPU();
	centroids->printSeeds(seeds);
	centroids->printError(seeds);
}
//----------------------------------------------------------------------
void display()
{
	static int count = 0;
	clock1.start();

	#if 0
	generateVoronoiRasterization();
	centroids->computeOnGPU();
	postprocessCentroids();
	#endif
	
	#if 1
		printf("******* count= %d *************\n", count);
		generateVoronoiRasterization();
		std::vector<Seed>& seeds = dtransf->getSeeds();
		float4* bins = centroids->computeOnGPU();
		//if (count % 2 == 0) postprocessCentroids();
		postprocessCentroids();
		for (unsigned int i=0; i < seeds.size(); i++) {
			seeds[i].x = bins[i].x;
			seeds[i].y = bins[i].y;
			seeds[i].z = 0.;
			seeds[i].w = (float) i;
		}
	#endif

	glutSwapBuffers();
	clock1.end();

	if (count % 10 == 0)  {
		printf("count= %d\n", count);
		clock1.print();
		clock1.reset();
	}

	count++;

	if (count > 10) {
		exit(0);
	}

	glutPostRedisplay();
}
//----------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////
CUTBoolean initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor( 0.0, 0.0, 0.0, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
    gluOrtho2D(0.,1.,0.,1.);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}
//----------------------------------------------------------------------
void setupObjects(int sz2d)
{
	dtransf = new DistanceTransformAcc(&g, sz2d);
	centroids = new Centroids();
	centroids->setDistanceTransform(*dtransf);
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
	cudaInit();

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "Cuda GL interop");

    // initialize GL and glew
    if( CUTFalse == initGL()) {
        return 0;
    }

    glutDisplayFunc(display);

	//int tex_size = 512; // cannot run FBO size 4096^2
	//int tex_size = 2048; // has problems on my mac
	int tex_size = 1024; // cannot run FBO size 4096^2

	// hook up Voronoi generator
	setupObjects(tex_size);
	init_shaders();
	g.disableShaders();  // disable shaders

	glutMainLoop();
}
//----------------------------------------------------------------------
