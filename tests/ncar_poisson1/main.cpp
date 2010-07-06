#include <stdlib.h>

// INTERESTING: the poisson include must come first. Otherwise I get an
// error in the constant definitions for MPI. I wonder if its because
// nested_sphere_cvt.h accidentally overrides one of the defines for MPI
#include "ncar_poisson1_explicit.h"
#include "nested_sphere_cvt.h"
#include "cvt.h"
#include "grid.h"
#include "gpu.h"
#include "derivative.h"
#include "exact_solution.h"
#include "exact_ncar_poisson1.h"
#include "communicator.h"

using namespace std;

#define NB_INNER_BND 4000
#define NB_OUTER_BND 4000
#define NB_INTERIOR 20000
#define NB_SAMPLES 200000
#define DIM_NUM 3

int main(int argc, char** argv) {
    
    Communicator* comm_unit = new Communicator(argc, argv);

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
    //double r[DIM_NUM * N_TOT];

    CVT* cvt = new NestedSphereCVT("nested_spheres", NB_INNER_BND, NB_OUTER_BND, NB_INTERIOR, sample_num, it_max_bnd, it_max_int, DIM_NUM);
    //    cvt->SetDensity(rho);
    // Generate the CVT
    int load_errors = cvt->cvt_load(-1);
    if (load_errors) { // File does not exist
        //cvt->cvt(N, batch, init, sample, sample_num, it_max, it_fixed, &seed, r, &it_num, &it_diff, &energy);
        //cvt->cvt(&r[0], &it_num_boundary, &it_num_interior, &it_diff, &energy, it_max_bnd, it_max_int, sample_num);
        cvt->cvt(&it_num, &it_diff, &energy);
        exit(0);
    }
    double* generators = cvt->getGenerators();
    int stencil_size = 20;

    // TODO: run this in parallel:

    Grid* grid = new Grid(DIM_NUM);
    // Compute stencils given a set of generators
    grid->computeStencils(generators, stencil_size, NB_INNER_BND + NB_OUTER_BND, N_TOT);

    GPU* subdomain = new GPU(-1.,1.,-1.,1.,-1.,1.,0.,comm_unit->getRank(),comm_unit->getSize());      // TODO: get these extents from the cvt class (add constructor to GPU)

    // Clean this up. Have GPU class fill data on constructor. Pass Grid class to constructor.
    // Remove need for extents in constructor.
    subdomain->fillLocalData(grid->getRbfCenters(), grid->getStencil(), grid->getBoundary(), grid->getAvgDist()); // Forms sets (Q,O,R) and l2g/g2l maps
    subdomain->fillVarData(grid->getRbfCenters()); // Sets function values in U

    // Verbosely print the memberships of all nodes within the subdomain
    subdomain->printCenterMemberships(subdomain->G, "G");

    ExactSolution* exact_poisson = new ExactNCARPoisson1();

    // Clean this up. Have the Poisson class construct Derivative internally.
    Derivative* der = new Derivative(subdomain->G_centers, subdomain->Q_stencils, subdomain->global_boundary_nodes.size());

    NCARPoisson1Explicit* poisson = new NCARPoisson1Explicit(exact_poisson, subdomain, der, 0);

    poisson->initialConditions();
    poisson->solve(comm_unit);

    delete(subdomain);
    delete(grid);
    delete(cvt);
    //    cvt->cvt_write(DIM_NUM, N_TOT, batch, seed_init, seed, init_string,
    //          it_max, it_fixed, it_num, it_diff, energy, sample_string, sample_num, r,
    //        file_out_name, comment);
}
//----------------------------------------------------------------------
