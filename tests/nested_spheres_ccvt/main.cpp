#include <stdlib.h>
#include "nested_sphere_cvt.h"
#include <vector>

using namespace std;

#define N_OUTER_BND 1000
#define N_INNER_BND 1000
#define N_INTERIOR 30

int main(int argc, char** argv) {

    int N_TOT = N_INNER_BND + N_OUTER_BND + N_INTERIOR;

    int DIM_NUM = 3;

    // Include comments in the output file?
    bool comment = false;

    // Discrete energy divided by number of sample pts
    double energy;

    // Output filename
    char file_out_name[80] = "cvt_nested_spheres.txt";

    // L2 norm of difference between iteration output
    double it_diff;

    // maximum number of iterations
    int it_max = 500;

    // number of iterations taken (output by cvt3d, input to cvt_write)
    int it_num_boundary = 0;
    int it_num_interior = 0;

    // number of sample points
    int sample_num = 300000;

    // generator points
    double r[DIM_NUM * N_TOT];

    Density* rho = new Density();

    NestedSphereCVT* cvt = new NestedSphereCVT(N_INNER_BND, N_OUTER_BND, N_INTERIOR);

    //    cvt->SetDensity(rho);

    // Generate the CVT
    //cvt->cvt(N, batch, init, sample, sample_num, it_max, it_fixed, &seed, r, &it_num, &it_diff, &energy);
    cvt->cvt(&r[0], &it_num_boundary, &it_num_interior, &it_diff, &energy, it_max);


    //    cvt->cvt_write(DIM_NUM, N_TOT, batch, seed_init, seed, init_string,
    //          it_max, it_fixed, it_num, it_diff, energy, sample_string, sample_num, r,
    //        file_out_name, comment);
}
//----------------------------------------------------------------------