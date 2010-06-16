#include <stdlib.h>
#include "constrained_cvt.h"
#include <vector>

using namespace std;

void test2DSquareCCVT(int N);
void test3DCubeCCVT(int N);
#if 0
void test2DCircle(int N);
void test3DSphere(int N);
void test2DEllipse(int N);
void test3DEllipsoid(int N);
void testNestedSpheres(int N);
#endif

int main(int argc, char** argv) {
int N = 1000;

test2DSquareCCVT(N);
//test3DCubeCCVT(N);
#if 0
test2DCircle(N); 
test3DSphere(N);
test2DEllipse(N);
test3DEllipsoid(N); 
testNestedSpheres(N); 
#endif

    return EXIT_SUCCESS;
}


void test2DSquareCCVT(int N) {
    int DIM_NUM = 2;
   
    // Number of sample pts generated at a time
    int batch = 1000;
    // Include comments in the output file?
    bool comment = true;

    // Discrete energy divided by number of sample pts
    double energy;

    // Output filename
    char file_out_name[80] = "cvt_2D_square.txt";

    // How points are initialized:
    //    -1, 'RANDOM', using C++ RANDOM function;
    //     0, 'UNIFORM', using a simple uniform RNG;
    //     1, 'HALTON', from a Halton sequence;
    //     2, 'GRID', points from a grid;
    //     3, 'USER', call "user" routine;
    //     4, points are already initialized on input.
    int init = -1;

    // String should correspond to strings for variable [int init;]
    char init_string[80] = "RANDOM";

    // L2 norm of difference between iteration output
    double it_diff;

    // number of iterations with a fixed set of samples
    int it_fixed = 1;

    // maximum number of iterations
    int it_max = 1000;

    // number of iterations taken (output by cvt3d, input to cvt_write)
    int it_num = 0;

    // How sampling is done
    //    -1, 'RANDOM', using C++ RANDOM function;
    //     0, 'UNIFORM', using a simple uniform RNG;
    //     1, 'HALTON', from a Halton sequence;
    //     2, 'GRID', points from a grid;
    //     3, 'USER', call "user" routine.
    int sample = 0;

    // String should correspond to sample type variable [int sample;]
    char sample_string[80] = "UNIFORM";

    // number of sample points
    int sample_num = 10000;

    // original random number seed
    int seed_init = 123456789;

    // current random number seed
    int seed = seed_init;
    // generator points
    double r[DIM_NUM * N];

    Density* rho = new Density();

    CVT* cvt = new ConstrainedCVT();

    cvt->setNbPts(N);
    cvt->setDensity(rho);

    // Generate the CVT
    cvt->cvt(DIM_NUM, N, batch, init, sample, sample_num, it_max, it_fixed,
            &seed, r, &it_num, &it_diff, &energy);

    cvt->cvt_write(DIM_NUM, N, batch, seed_init, seed, init_string,
            it_max, it_fixed, it_num, it_diff, energy, sample_string, sample_num, r,
            file_out_name, comment);
}
//----------------------------------------------------------------------

void test3DCubeCCVT(int N) {
    int DIM_NUM = 3;

    // Number of sample pts generated at a time
    int batch = 1000;
    // Include comments in the output file?
    bool comment = true;

    // Discrete energy divided by number of sample pts
    double energy;

    // Output filename
    char file_out_name[80] = "cvt_3D_cube.txt";

    // How points are initialized:
    //    -1, 'RANDOM', using C++ RANDOM function;
    //     0, 'UNIFORM', using a simple uniform RNG;
    //     1, 'HALTON', from a Halton sequence;
    //     2, 'GRID', points from a grid;
    //     3, 'USER', call "user" routine;
    //     4, points are already initialized on input.
    int init = -1;

    // String should correspond to strings for variable [int init;]
    char init_string[80] = "RANDOM";

    // L2 norm of difference between iteration output
    double it_diff;

    // number of iterations with a fixed set of samples
    int it_fixed = 1;

    // maximum number of iterations
    int it_max = 1000;

    // number of iterations taken (output by cvt3d, input to cvt_write)
    int it_num = 0;

    // How sampling is done
    //    -1, 'RANDOM', using C++ RANDOM function;
    //     0, 'UNIFORM', using a simple uniform RNG;
    //     1, 'HALTON', from a Halton sequence;
    //     2, 'GRID', points from a grid;
    //     3, 'USER', call "user" routine.
    int sample = 0;

    // String should correspond to sample type variable [int sample;]
    char sample_string[80] = "UNIFORM";

    // number of sample points
    int sample_num = 10000;

    // original random number seed
    int seed_init = 123456789;

    // current random number seed
    int seed = seed_init;
    // generator points
    double r[DIM_NUM * N];

    Density* rho = new Density();

    CVT* cvt = new ConstrainedCVT();

    cvt->setNbPts(N);
    cvt->setDensity(rho);

    // Generate the CVT
    cvt->cvt(DIM_NUM, N, batch, init, sample, sample_num, it_max, it_fixed,
            &seed, r, &it_num, &it_diff, &energy);

    cvt->cvt_write(DIM_NUM, N, batch, seed_init, seed, init_string,
            it_max, it_fixed, it_num, it_diff, energy, sample_string, sample_num, r,
            file_out_name, comment);
}
