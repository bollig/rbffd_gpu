#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "cube.h"
#include "Vec3.h"
using namespace std;

// To see the results of this test run it under OSX:
//  $> ./3d_regular_cube.x | pbcopy
// This will copy the stdout to the clipboard. Paste the results into matlab and
// then run the matlab command:
//  EDU>> plot3(a(:,1), a(:,2), a(:,3), '*'); grid on; xlabel('x'); ylabel('y'); zlabel('z')
// This will plot the output points and you can see the projection is working

//----------------------------------------------------------------------

int main(int argc, char** argv) {
    int numX = 11;
    int numY = 5;
    int numZ = 8;

    double maxX = 1.;
    double minX = 0.;
    double maxY = 2.;
    double minY = 0.;
    double maxZ = 3.;
    double minZ = 0.;

    // Unit cube
    CubeGeometry* cube = new CubeGeometry(minX, maxX, minY, maxY, minZ, maxZ, numX, numY, numZ);

    fprintf(stderr, "Surface Area: %f\n", cube->surfaceIntegral());
    fprintf(stderr, "Volume: %f\n", cube->volumeIntegral());

    Vec3** samples = cube->getSamples();
    
    fprintf(stdout, "a = [");

    // Test a full volume
    for (int i = 0; i < numX; i++) {
        for (int j = 0; j < numY; j++) {
            for (int k = 0; k < numZ; k++) {
                Vec3 pt = *(samples[i*numY*numZ + j*numZ + k]);
                fprintf(stdout, "%f, %f, %f;", pt.x(), pt.y(), pt.z());
            }
        }
    }


    // Test project a single point
    Vec3 pt(0.1, 0.1, 0.1);
    Vec3 proj = cube->project(pt);
    fprintf(stdout, "%f, %f, %f;", proj.x(), proj.y(), proj.z());
    fprintf(stdout, "%f, %f, %f;", pt.x(), pt.y(), pt.z());

    fprintf(stdout, "];");

    delete cube;
    
    if (argc > 1) {
        return EXIT_FAILURE; // FAIL TEST
    }

    return EXIT_SUCCESS; // PASS TEST
}
//----------------------------------------------------------------------
