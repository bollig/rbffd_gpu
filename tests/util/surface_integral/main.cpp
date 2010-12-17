#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/geom/ellipsoid_patch.h"


int main()
{
	printf("Surface of a sphere\n");
	double radius = 5.;
	double pi = acos(-1.);


	int n1 = 4000;
	int n2 = 4000;
	EllipsoidPatch ep(0., 2.*pi, 0., pi, n1, n2, radius, radius, radius);
    double surf_int = ep.surfaceIntegral();

	double exact_surface = 4.*pi*powf(radius,2.);
	printf("\texact_surface= %f\n", exact_surface);
	printf("\tcalculated surface (sphere): %f\n", surf_int);

	if (fabs(surf_int - exact_surface) < 1.e-5) {
		return EXIT_SUCCESS;
	} else {
		return EXIT_FAILURE;
	}
	exit(0);

#if 0
    if (argc > 1) {
        return EXIT_FAILURE;    // FAIL TEST
    }

    return EXIT_SUCCESS;        // PASS TEST
#endif
}
