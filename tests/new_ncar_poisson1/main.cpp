#include <stdlib.h>

// INTERESTING: the poisson include must come first. Otherwise I get an
// error in the constant definitions for MPI. I wonder if its because
// nested_sphere_cvt.h accidentally overrides one of the defines for MPI
//#include "ncar_poisson1.h"
//#include "ncar_poisson1_cusp.h"
//#include "ncar_poisson1_cl.h"
#include "nonuniform_poisson1_cl.h"
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
#include "projectsettings.h"

using namespace std;

int main(int argc, char** argv) {

    Communicator* comm_unit = new Communicator(argc, argv);

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit);

    // Discrete energy divided by number of sample pts
    double energy;

    // L2 norm of difference between iteration output
    double it_diff;

    // number of iterations taken (output by cvt3d, input to cvt_write)
    int it_num =0;      // Total number of iterations taken.

    int dim = settings->GetSettingAs<int>("DIMENSION");

    // Generate the CVT if the file doesnt already exist
    CVT* cvt = new NestedSphereCVT(settings);
    int load_errors = cvt->cvt_load(-1);
    if (load_errors) { // File does not exist
        cvt->cvt(&it_num, &it_diff, &energy);
    }

    // TODO: run this in parallel:
    double* generators = cvt->getGenerators();
    Grid* grid = new Grid(settings);
    // Compute stencils given a set of generators
    grid->computeStencils(generators, cvt->getKDTree());

    GPU* subdomain = new GPU(-1.,1.,-1.,1.,-1.,1.,0.,comm_unit->getRank(),comm_unit->getSize());      // TODO: get these extents from the cvt class (add constructor to GPU)

    // Clean this up. Have GPU class fill data on constructor. Pass Grid class to constructor.
    // Remove need for extents in constructor.
    subdomain->fillLocalData(grid->getRbfCenters(), grid->getStencil(), grid->getBoundary(), grid->getAvgDist()); // Forms sets (Q,O,R) and l2g/g2l maps
    subdomain->fillVarData(grid->getRbfCenters()); // Sets function values in U

    // Verbosely print the memberships of all nodes within the subdomain
    //subdomain->printCenterMemberships(subdomain->G, "G");

    // 0: 2D problem; 1: 3D problem
    ExactSolution* exact_poisson;
    if (dim == 3) {
        exact_poisson = new ExactNCARPoisson1();        // 3D problem is not verified yet
    } else {
        exact_poisson = new ExactNCARPoisson2();        // 2D problem works with uniform diffusion
    }

    // Clean this up. Have the Poisson class construct Derivative internally.
    Derivative* der = new Derivative(subdomain->G_centers, subdomain->Q_stencils, subdomain->global_boundary_nodes.size());
    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);

    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS")) {
        DerivativeTests* der_test = new DerivativeTests();
        der_test->testAllFunctions(*der, *grid);
    }

    NCARPoisson1* poisson = new NonUniformPoisson1_CL(settings, exact_poisson, subdomain, der, 0, dim);

    poisson->initialConditions();
    poisson->solve(comm_unit);


    delete(poisson);
    delete(subdomain);
    delete(grid);
    delete(cvt);
    delete(settings);

    cout.flush();

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
