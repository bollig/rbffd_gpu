#include <stdlib.h>

#include "utils/conf/projectsettings.h"

#include "grids/cvt/nested_sphere_cvt.h"

#include "utils/comm/communicator.h"

#include "pdes/elliptic/nonuniform_poisson1_cl.cpp"
#include "exact_solutions/exact_ncar_poisson2.h"

#include "rbffd/rbffd_cl.h"
#include "rbffd/rbffd.h"
#include "rbffd/derivative_tests.h"

using namespace std;

int main(int argc, char** argv) {

    ProjectSettings* settings = new ProjectSettings(argc, argv);


    int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required); 
    int nb_interior = settings->GetSettingAs<int>("NB_INTERIOR", ProjectSettings::required); 
    int nb_inner_boundary = settings->GetSettingAs<int>("NB_INNER_BOUNDARY", ProjectSettings::required); 
    int nb_outer_boundary = settings->GetSettingAs<int>("NB_OUTER_BOUNDARY", ProjectSettings::required); 
    int nb_boundary = nb_inner_boundary + nb_outer_boundary; 

    if (dim > 3) {
        cout << "ERROR! Dim > 3 Not supported!" << endl;
        exit(EXIT_FAILURE); 
    }

    double inner_r = settings->GetSettingAs<double>("INNER_RADIUS", ProjectSettings::optional, "0.5"); 
    double outer_r = settings->GetSettingAs<double>("OUTER_RADIUS", ProjectSettings::optional, "1.0"); 

    int ns_nx = settings->GetSettingAs<int>("NS_NB_X", ProjectSettings::optional, "10"); 
    int ns_ny = settings->GetSettingAs<int>("NS_NB_Y", ProjectSettings::optional, "10");
    int ns_nz = settings->GetSettingAs<int>("NS_NB_Z", ProjectSettings::optional, "10");
        
    double minX = settings->GetSettingAs<double>("MIN_X", ProjectSettings::optional, "-1."); 	
    double maxX = settings->GetSettingAs<double>("MAX_X", ProjectSettings::optional, "1."); 	
    double minY = settings->GetSettingAs<double>("MIN_Y", ProjectSettings::optional, "-1."); 	
    double maxY = settings->GetSettingAs<double>("MAX_Y", ProjectSettings::optional, "1."); 	
    double minZ = settings->GetSettingAs<double>("MIN_Z", ProjectSettings::optional, "-1."); 	
    double maxZ = settings->GetSettingAs<double>("MAX_Z", ProjectSettings::optional, "1."); 

    double debug = settings->GetSettingAs<int>("DEBUG", ProjectSettings::optional, "0"); 


    // 0 = Dirichlet, 1 = neumann, 2 = robin
    int boundary_condition = settings->GetSettingAs<int>("BOUNDARY_CONDITION", ProjectSettings::optional, "0"); 
    // 0 = discrete rhs, 1 = exact (test discrete compat condition)
    int use_discrete_rhs = settings->GetSettingAs<int>("USE_DISCRETE_RHS", ProjectSettings::optional, "0"); 
    // 0 = assume non-uniform diffusion, 1 = assume uniform 
    int use_uniform_diffusion = settings->GetSettingAs<int>("USE_UNIFORM_DIFFUSION", ProjectSettings::optional, "1"); 
    int run_derivative_tests = settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS", ProjectSettings::optional, "1");

    int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 
    
    
    int nb_samples = settings->GetSettingAs<int>("NB_CVT_SAMPLES", ProjectSettings::required); 
    int it_max_interior = settings->GetSettingAs<int>("NB_CVT_ITERATIONS", ProjectSettings::required); 
    // Generate a CVT with nx*ny*nz nodes, in 1, 2 or 3D with 0 locked boundary nodes, 
    // 20000 samples per iteration for 30 iterations
    NestedSphereCVT* grid = new NestedSphereCVT(nb_interior, nb_inner_boundary, nb_outer_boundary, dim, 0, nb_samples, it_max_interior); 
    grid->setExtents(minX, maxX, minY, maxY, minZ, maxZ);
    grid->setInnerRadius(inner_r); 
    grid->setOuterRadius(outer_r); 
    grid->setDebug(debug);
    grid->setMaxStencilSize(stencil_size);
    grid->setNSHashDims(ns_nx, ns_ny, ns_nz);
    if(grid->loadFromFile())
    {
        printf("************** Generating new Grid **************\n"); 
        grid->setSortBoundaryNodes(true); 
        //            tm["grid"]->start(); 
        grid->generate();
        //          tm["grid"]->stop(); 
        std::cout << "Generating stencils\n";
        //        tm["stencils"]->start(); 
        grid->generateStencils(Grid::ST_HASH);   // nearest stencil_size 
        //grid->generateStencils(Grid::ST_BRUTE_FORCE);   // nearest stencil_size 
        //      tm["stencils"]->stop();
        grid->writeToFile(); 
    }

    // 0: 2D problem; 1: 3D problem
    ExactSolution* exact_poisson;
    if (dim == 3) {
        std::cout << "ERROR! 3D not verified yet! exiting..." << std::endl;
        exit(EXIT_FAILURE);
        //     exact_poisson = new ExactNCARPoisson1();        // 3D problem is not verified yet
    } else {
        exact_poisson = new ExactNCARPoisson2();        // 2D problem works with uniform diffusion
    }

    bool use_gpu = true;
    RBFFD* der;
    if (use_gpu) {
        der = new RBFFD_CL(grid, dim); 
    } else {
        der = new RBFFD(grid, dim); 
    }


   // Enable variable epsilon. Not verified to be perfected in the derivative calculation.
   // But it has improved the heat equation already
    int use_var_eps = settings->GetSettingAs<int>("USE_VAR_EPSILON", ProjectSettings::optional, "0");
    if (use_var_eps) {
        double alpha = settings->GetSettingAs<double>("VAR_EPSILON_ALPHA", ProjectSettings::optional, "1.0"); 
        double beta = settings->GetSettingAs<double>("VAR_EPSILON_BETA", ProjectSettings::optional, "1.0"); 
        der->setVariableEpsilon(alpha, beta); 
    } else {
        double epsilon = settings->GetSettingAs<double>("EPSILON", ProjectSettings::required);
        der->setEpsilon(epsilon);
    }

    der->computeAllWeightsForAllStencils();

    if (run_derivative_tests) {
        DerivativeTests* der_test = new DerivativeTests(der, grid, true);
        if (use_gpu) {
            // Applies weights on both GPU and CPU and compares results for the first 10 stencils
            der_test->compareGPUandCPUDerivs(10);
        }
        // Test approximations to derivatives of functions f(x,y,z) = 0, x, y, xy, etc. etc.
        der_test->testAllFunctions();
        // For now we can only test eigenvalues on an MPI size of 1 (we could distribute with Par-Eiegen solver)
        if (settings->GetSettingAs<int>("DERIVATIVE_EIGENVALUE_TEST", ProjectSettings::optional, "0")) 
        {
            // FIXME: why does this happen? Perhaps because X Y and Z are unidirectional? 
            // Test X and 4 eigenvalues are > 0
            // Test Y and 30 are > 0
            // Test Z and 36 are > 0
            // NOTE: the 0 here implies we compute the eigenvalues but do not run the iterations of the random perturbation test
            der_test->testEigen(RBFFD::LAPL, 0);
        }
    }
 

    NCARPoisson1* poisson = new NonUniformPoisson1_CL(exact_poisson, grid, der, 0, dim);
    poisson->setBoundaryCondition(boundary_condition); 
    poisson->setUseDiscreteRHS(use_discrete_rhs); 
    poisson->setUseUniformDiffusivity(use_uniform_diffusion);

    poisson->initialConditions();
    poisson->solve();

    delete(poisson);
//    delete(der);
    delete(grid);
    delete(settings);
#if 0
    Grid* grid2 = new Grid(); 
    grid2->loadFromFile("initial_grid.ascii"); 
    grid2->writeToFile("final_grid.ascii");

    cout.flush();
#endif 
    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
