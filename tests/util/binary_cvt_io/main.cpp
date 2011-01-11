#include <stdio.h>
#include <stdlib.h>
#include "grids/cvt/nested_sphere_cvt.h"
#include <vector>

using namespace std;

#define NB_INNER_BND 50000
#define NB_OUTER_BND 50000
#define NB_INTERIOR 10
#define NB_SAMPLES 100
#define DIM_NUM 2

int main(int argc, char** argv) {

    double tol = 1e-6;
    int comment = 1; // 1=output comments in cvt files; 0=only BINARY/ASCII + data
    int VERBOSE = 1;
    int N_TOT = NB_INNER_BND + NB_OUTER_BND + NB_INTERIOR;

    // Discrete energy divided by number of sample pts
    double energy;

    // Output filename
    char file_out_name[80] = "cvt_nested_spheres.txt";
    char bin_file_out_name[80] = "cvt_nested_spheres.bin";

    // L2 norm of difference between iteration output
    double it_diff;

    // maximum number of iterations
    int it_max_bnd = 1; // Boundary
    int it_max_int = 1; // Interior

    // number of iterations taken (output by cvt3d, input to cvt_write)
    int it_num_boundary = 0;
    int it_num_interior = 0;

    int sample_num = NB_SAMPLES;

    // generator points
    double r[DIM_NUM * N_TOT];
    double rin_ascii[DIM_NUM * N_TOT];
    double rin_binary[DIM_NUM * N_TOT];

    //Density* rho = new Density();


	printf("Constructing CVT Class\n");

    NestedSphereCVT* cvt = new NestedSphereCVT("binary_io_test", NB_INNER_BND, NB_OUTER_BND, NB_INTERIOR, DIM_NUM);

	printf("Generating CVT\n");

    //    cvt->SetDensity(rho);

    // Generate the CVT
    // Regular CVT call: 
    //cvt->cvt(N, batch, init, sample, sample_num, it_max, it_fixed, &seed, r, &it_num, &it_diff, &energy);
    // Simplified CVT call
    cvt->cvt(&r[0], &it_num_boundary, &it_num_interior, &it_diff, &energy, it_max_bnd, it_max_int, sample_num);

	printf("CVT generated. Writing to disk\n");

    cvt->cvt_write(DIM_NUM, N_TOT, 0, 0, 0, "RANDOM",
            it_max_bnd, 1, it_num_boundary, it_diff, energy, "RANDOM", sample_num, &r[0],
            file_out_name, comment);
    cvt->cvt_write_binary(DIM_NUM, N_TOT, 0, 0, 0, "RANDOM",
            it_max_bnd, 1, it_num_boundary, it_diff, energy, "RANDOM", sample_num, &r[0],
            bin_file_out_name, comment);

    printf("TESTING ASCII DATA: \n");
    cvt->data_read(file_out_name, DIM_NUM, N_TOT, &rin_ascii[0]);
    for (int i = 0; i < N_TOT; i++) {
        if (VERBOSE) {
            printf("r[%d]=%f, rin[%d]=%f : ", i, r[i], i, rin_ascii[i]);
        }
        double diff = fabs(r[i] - rin_ascii[i]);
        if (diff < tol) {
            if (VERBOSE) {
                printf("PASS\n");
            }
        } else {
            if (VERBOSE) {
                printf("FAIL %e\n", diff);
            }
            return EXIT_FAILURE;
        }
    }

    printf("TESTING BINARY DATA: \n");
    cvt->data_read(bin_file_out_name, DIM_NUM, N_TOT, &rin_binary[0]);
    for (int i = 0; i < N_TOT; i++) {
        if (VERBOSE) {
            printf("r[%d]=%f, rin[%d]=%f : ", i, r[i], i, rin_binary[i]);
        }
        double diff = fabs(r[i] - rin_binary[i]);
        if (diff < tol) {
            if (VERBOSE) {
                printf("PASS\n");
            }
        } else {
            if (VERBOSE) {
                printf("FAIL %e\n", diff);
            }
            return EXIT_FAILURE;
        }
    }

    printf("SUCCESS! ALL VALUES READ FROM FILES \"%s\" and \"%s\" ARE WITHIN %e OF THE ORIGINAL GENERATORS\n", file_out_name, bin_file_out_name, tol);
    return EXIT_SUCCESS;
}
//----------------------------------------------------------------------
