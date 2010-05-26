#include <math.h>
#include <stdlib.h>
#include <vector>
#include <ellipsoid_patch.h>
#include <parametric_patch.h>
#include <octree.h>

using namespace std;

// GLOBAL VARIABLES

//----------------------------------------------------------------------
int main (int argc, char** argv)
{
	// domain dimensions
	double pi = acos(-1.);
	double a = 2.;
	double b = 3.;
	double c = 4.;

	int n1 = 40;
	int n2 = 40;
	ParametricPatch* ep = new EllipsoidPatch(0., pi, 0., 2.*pi, n1, n2, a, b, c);

	Octree oct(ep);

	if (argc > 1) {
		return EXIT_FAILURE; 	// FAIL TEST
	} 

	return EXIT_SUCCESS; 		// PASS TEST
}
//----------------------------------------------------------------------
