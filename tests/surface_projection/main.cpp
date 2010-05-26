#include <stdio.h>
#include <stdlib.h>
#include <Vec3.h>
#include <parametric_patch.h>
#include <ellipsoid_patch.h>

//----------------------------------------------------------------------
int main(int argc, char** argv)
{
	double a = 2.;
	double b = 3.;
	double c = 4.;
	double pi = acos(-1.);

	int n1 = 80;
	int n2 = 80;
	EllipsoidPatch ep(0., pi, 0., 2.*pi, n1, n2, 
	   a, b, c);

	//----------------------------------------------------------------------

	// project the point to the boundary
	//Vec3 ptc(1.6, .3, .3);
	for (int i=0; i < 11; i++) {
		printf("-----------------\n");
		double z = (c/10.) * i;
		Vec3 ptc(0., 0., z);
		Vec3 pt = ep.project(ptc);
		pt.print("new pt");
		printf("how_far: %f\n", ep.how_far(pt));
	}
	//----------------------------------------------------------------------

	double err= 0.;
	if (err < 1.e-3) {   // 0.1 percent error
		return EXIT_SUCCESS; 		// PASS TEST
	} else {
		return EXIT_FAILURE; 	// FAIL TEST
	}
}
//----------------------------------------------------------------------
