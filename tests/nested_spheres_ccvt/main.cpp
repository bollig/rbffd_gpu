#include <stdlib.h>

#include "nested_sphere_cvt.h"

using namespace std;

#define NB_INNER_BND 1000
#define NB_OUTER_BND 1000
#define NB_INTERIOR 5000
#define NB_SAMPLES 10000
#define DIM_NUM 2


int main(int argc, char** argv) {
    
    int N_TOT = NB_INNER_BND + NB_OUTER_BND + NB_INTERIOR;

    // Discrete energy divided by number of sample pts
    double energy;

    // Output filename
    char file_out_name[80] = "cvt_nested_spheres.txt";

    // L2 norm of difference between iteration output
    double it_diff;

    // maximum number of iterations
    int it_max_bnd = 100;    // Boundary
    int it_max_int = 500;   // Interior

    // number of iterations taken (output by cvt3d, input to cvt_write)
    int it_num_boundary = 0;
    int it_num_interior = 0;

    int sample_num = NB_SAMPLES;

    // generator points
    double r[DIM_NUM * N_TOT];

    Density* rho = new Density();



    NestedSphereCVT* cvt = new NestedSphereCVT(NB_INNER_BND, NB_OUTER_BND, NB_INTERIOR, DIM_NUM);

    //    cvt->SetDensity(rho);

    // Generate the CVT
    //cvt->cvt(N, batch, init, sample, sample_num, it_max, it_fixed, &seed, r, &it_num, &it_diff, &energy);
    cvt->cvt(&r[0], &it_num_boundary, &it_num_interior, &it_diff, &energy, it_max_bnd, it_max_int, sample_num);


    delete(cvt);
    //    cvt->cvt_write(DIM_NUM, N_TOT, batch, seed_init, seed, init_string,
    //          it_max, it_fixed, it_num, it_diff, energy, sample_string, sample_num, r,
    //        file_out_name, comment);
}
//----------------------------------------------------------------------