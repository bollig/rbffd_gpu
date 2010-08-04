#include <stdlib.h>

// INTERESTING: the poisson include must come first. Otherwise I get an
// error in the constant definitions for MPI. I wonder if its because
// nested_sphere_cvt.h accidentally overrides one of the defines for MPI
#include "ncar_poisson1.h"
#include "ncar_poisson1_cusp.h"
#include "ncar_poisson1_cl.h"
#include "grid.h"
#include "nested_sphere_cvt.h"
#include "cvt.h"
#include "gpu.h"
#include "derivative.h"
#include "derivative_tests.h"
#include "exact_solution.h"
#include "exact_ncar_poisson1.h"
#include "exact_ncar_poisson2.h"
#include "communicator.h"
#include "parse_command_line.h"
using namespace std;


#if 0
// The ultimate benchmark. Proves that KDTree
// is good for acceleration in CVT so long as
// we have enough samples to amortize the cost
// of reconstructing the tree each iteration.
#define NB_INNER_BND 1000
#define NB_OUTER_BND 2000
#define NB_INTERIOR 8000
#define NB_SAMPLES 8000000
#define DIM_NUM 3
#define STENCIL_SIZE 50
#else
#if 0
// 6K nodes (Match roughly with Joe's nodeset)
#define NB_INNER_BND 120
#define NB_OUTER_BND 240
#define NB_INTERIOR 5640
#define NB_SAMPLES 80000
#define DIM_NUM 2
#define STENCIL_SIZE 50
#else
#if 0
// Gordon's tests
#define NB_INNER_BND 25
#define NB_OUTER_BND 45
#define NB_INTERIOR 700
#define NB_SAMPLES 80000
#define DIM_NUM 2
#define STENCIL_SIZE 60
#else
#if 1
// 3K nodes (Match roughly with Joe's nodeset)
#define NB_INNER_BND 94
#define NB_OUTER_BND 181
#define NB_INTERIOR 2725
#define NB_SAMPLES 80000
#define DIM_NUM 2
#define STENCIL_SIZE 60
#else
#if 0
// Simple test case to show some convergence
#define NB_INNER_BND 100
#define NB_OUTER_BND 200
#define NB_INTERIOR 800
#define NB_SAMPLES 80000
#define DIM_NUM 2
#define STENCIL_SIZE 50
#else
#if 1
// Simple 3D test case
#define NB_INNER_BND 200
#define NB_OUTER_BND 400
#define NB_INTERIOR 2400
#define NB_SAMPLES 80000
#define DIM_NUM 3
#define STENCIL_SIZE 50
#else
// Basic case to prove code runs
#define NB_INNER_BND 10
#define NB_OUTER_BND 10
#define NB_INTERIOR 10
#define NB_SAMPLES 80000
#define DIM_NUM 2
#define STENCIL_SIZE 5
#endif
#endif
#endif 
#endif
#endif
#endif

int main(int argc, char** argv) {
    
    Communicator* comm_unit = new Communicator(argc, argv);

    parseCommandLineArgs(argc, argv, comm_unit->getRank());

    int N_TOT = NB_INNER_BND + NB_OUTER_BND + NB_INTERIOR;

    // Discrete energy divided by number of sample pts
    double energy;

    // L2 norm of difference between iteration output
    double it_diff;

    // maximum number of iterations
    int it_max_bnd = 60;    // Boundary
    int it_max_int = 100;   // Interior
    int stencil_size = STENCIL_SIZE;

    // number of iterations taken (output by cvt3d, input to cvt_write)
    int it_num_boundary = 0;
    int it_num_interior = 0;
    int it_num =0;      // Total number of iterations taken.

    int sample_num = NB_SAMPLES;

    // generator points
    //double r[DIM_NUM * N_TOT];

    CVT* cvt = new NestedSphereCVT("nested_spheres", NB_INNER_BND, NB_OUTER_BND, NB_INTERIOR, sample_num, it_max_bnd, it_max_int, DIM_NUM);
    //    cvt->SetDensity(rho);
    // Generate the CVT
    int load_errors = cvt->cvt_load(-1);
    if (load_errors) { // File does not exist
        //cvt->cvt(N, batch, init, sample, sample_num, it_max, it_fixed, &seed, r, &it_num, &it_diff, &energy);
        //cvt->cvt(&r[0], &it_num_boundary, &it_num_interior, &it_diff, &energy, it_max_bnd, it_max_int, sample_num);
        cvt->cvt(&it_num, &it_diff, &energy);
    }
    double* generators = cvt->getGenerators();

    // TODO: run this in parallel:

    Grid* grid = new Grid(DIM_NUM);
    // Compute stencils given a set of generators
    grid->computeStencils(generators, stencil_size, NB_INNER_BND + NB_OUTER_BND, N_TOT, cvt->getKDTree());
    //grid->avgStencilRadius();

    GPU* subdomain = new GPU(-1.,1.,-1.,1.,-1.,1.,0.,comm_unit->getRank(),comm_unit->getSize());      // TODO: get these extents from the cvt class (add constructor to GPU)

    // Clean this up. Have GPU class fill data on constructor. Pass Grid class to constructor.
    // Remove need for extents in constructor.
    subdomain->fillLocalData(grid->getRbfCenters(), grid->getStencil(), grid->getBoundary(), grid->getAvgDist()); // Forms sets (Q,O,R) and l2g/g2l maps
    subdomain->fillVarData(grid->getRbfCenters()); // Sets function values in U

    // Verbosely print the memberships of all nodes within the subdomain
    //subdomain->printCenterMemberships(subdomain->G, "G");

    // 0: 2D problem; 1: 3D problem
#if DIM_NUM == 3
    ExactSolution* exact_poisson = new ExactNCARPoisson1();
#else
    ExactSolution* exact_poisson = new ExactNCARPoisson2();
#endif

    // Clean this up. Have the Poisson class construct Derivative internally.
    Derivative* der = new Derivative(subdomain->G_centers, subdomain->Q_stencils, subdomain->global_boundary_nodes.size());
    cout << "SET EPSILON = 8" << endl;
    der->setEpsilon(6.0);

    DerivativeTests* der_test = new DerivativeTests();
    //der_test->testAllFunctions(*der, *grid);

    NCARPoisson1* poisson = new NCARPoisson1_CL(exact_poisson, subdomain, der, 0, DIM_NUM);

    poisson->initialConditions();
    poisson->solve(comm_unit);

	
    delete(poisson);
    delete(subdomain);
    delete(grid);
    delete(cvt);
 
    cout.flush();
 
    exit(EXIT_SUCCESS);
    //    cvt->cvt_write(DIM_NUM, N_TOT, batch, seed_init, seed, init_string,
    //          it_max, it_fixed, it_num, it_diff, energy, sample_string, sample_num, r,
    //        file_out_name, comment);
}
//----------------------------------------------------------------------
