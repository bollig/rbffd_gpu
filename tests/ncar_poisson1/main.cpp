#include <stdlib.h>

// INTERESTING: the poisson include must come first. Otherwise I get an
// error in the constant definitions for MPI. I wonder if its because
// nested_sphere_cvt.h accidentally overrides one of the defines for MPI
#include "ncar_poisson1.h"
#include "nested_sphere_cvt.h"
#include "cvt.h"

using namespace std;

#define NB_INNER_BND 20
#define NB_OUTER_BND 20
#define NB_INTERIOR 50
#define NB_SAMPLES 10000
#define DIM_NUM 3

int main(int argc, char** argv) {
    
    int N_TOT = NB_INNER_BND + NB_OUTER_BND + NB_INTERIOR;

    // Discrete energy divided by number of sample pts
    double energy;

    // L2 norm of difference between iteration output
    double it_diff;

    // maximum number of iterations
    int it_max_bnd = 60;    // Boundary
    int it_max_int = 100;   // Interior

    // number of iterations taken (output by cvt3d, input to cvt_write)
    int it_num_boundary = 0;
    int it_num_interior = 0;
    int it_num =0;      // Total number of iterations taken.

    int sample_num = NB_SAMPLES;

    // generator points
    double r[DIM_NUM * N_TOT];

    CVT* cvt = new NestedSphereCVT("nested_spheres", NB_INNER_BND, NB_OUTER_BND, NB_INTERIOR, it_max_bnd, it_max_int, DIM_NUM);
    //    cvt->SetDensity(rho);
    // Generate the CVT
    //cvt->cvt(N, batch, init, sample, sample_num, it_max, it_fixed, &seed, r, &it_num, &it_diff, &energy);
    //cvt->cvt(&r[0], &it_num_boundary, &it_num_interior, &it_diff, &energy, it_max_bnd, it_max_int, sample_num);
    int load_errors = cvt->cvt_load(-1);

    if (load_errors) { // File does not exist
        cvt->cvt(&it_num, &it_diff, &energy, &r[0]);
    }


    delete(cvt);
    //    cvt->cvt_write(DIM_NUM, N_TOT, batch, seed_init, seed, init_string,
    //          it_max, it_fixed, it_num, it_diff, energy, sample_string, sample_num, r,
    //        file_out_name, comment);
}
//----------------------------------------------------------------------