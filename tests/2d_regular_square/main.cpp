#include <math.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <float.h>
#include "square_patch.h"
using namespace std;

// GLOBAL VARIABLES

//----------------------------------------------------------------------
int main (int argc, char** argv)
{
	int numU = 11; 
	int numV = 5; 
	
	double maxU = 1.0;
	double maxV = 1.0;
	
	double du = maxU / (double)(numU - 1); 
	double dv = maxV / (double)(numV - 1);
	
	//cout << "du: " << du << "\tdv: " <<  dv << endl; 
	
	// Given normal, guess principal axes 
	// WARNING!! Princ_axes are guessed. This may not sample x, y, z the direction you want. 
	// See comments on SquarePatch::guessP1() for details.
	Vec3 normal(1.0f, 0.0f, 0.0f);
	ParametricPatch* ep1 = new SquarePatch(0.,maxU,0.,maxV,10,10, 0.,0.,1., 0., 0., 1.);

	// Given principal axes we can calculate normal and guarantee orientation
	// and sampling along axes
	// X-Z Plane ( Normal=<0,1,0>=princ_axis_1.cross(princ_axis_2); )
	Vec3 princ_axis_1(1.0f,0.0f,0.0f); 
	Vec3 princ_axis_2(0.0f,0.0f,1.0f);

	//ParametricPatch* ep2 = new SquarePatch(0.,maxU,0.,maxV,10,10, 1.,1.,0.);
	
	
	// Test the patch
	// NOTE: in order to reach our upper bound we must add machine epsilon or
	// 		the machine will find cases where i=1.0000001, maxU=1 (i not <= maxU)
	
	// NOTE: sampling on two endcaps will not be the same as the loop unless du == dv
	printf("a = [");
	for (double u = -1.; u <= maxU + DBL_EPSILON; u+=du) {
		for (double v = -1.; v <= maxV + DBL_EPSILON /*, v <= u*/; v+=dv) {
			//printf("f(%f, %f) = (%f, %f, %f)\n", u, v, ep->x(u,v), ep->y(u,v), ep->z(u,v));
			printf("%f, %f, %f;", ep1->x(u,v), ep1->y(u,v), ep1->z(u,v));
		}
	}	
	
	Vec3 pt(0., 1., -1.); 
	
	fprintf(stderr,"BELOW: %d\n", ((SquarePatch*)ep1)->isBelow(pt));
	fprintf(stderr,"INSIDE: %d\n", ((SquarePatch*)ep1)->isInShadow(pt));
	
	printf("%f, %f, %f", pt.x(), pt.y(), pt.z());
	printf("];\n");

	if (argc > 1) {
		return EXIT_FAILURE; 	// FAIL TEST
	} 

	return EXIT_SUCCESS; 		// PASS TEST
}
//----------------------------------------------------------------------
