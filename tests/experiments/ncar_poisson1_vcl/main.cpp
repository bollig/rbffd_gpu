#include <stdlib.h>

#include "utils/conf/projectsettings.h"

#include "grids/cvt/nested_sphere_cvt.h"

#include "utils/comm/communicator.h"

#include "pdes/elliptic/nonuniform_poisson1_cl.cpp"
#include "exact_solutions/exact_ncar_poisson2.h"

#include "rbffd/derivative.h"
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
    if(grid->loadFromFile())
    {
        printf("************** Generating new Grid **************\n"); 
        grid->setSortBoundaryNodes(true); 
        //            tm["grid"]->start(); 
        grid->generate();
        //          tm["grid"]->stop(); 
        std::cout << "Generating stencils\n";
        //        tm["stencils"]->start(); 
        grid->generateStencils(Grid::ST_BRUTE_FORCE);   // nearest stencil_size 
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

    Derivative* der = new Derivative(grid->getNodeList(), grid->getStencils(), grid->getBoundaryIndicesSize(), dim);

   // Enable variable epsilon. Not verified to be perfected in the derivative calculation.
   // But it has improved the heat equation already
    int use_var_eps = settings->GetSettingAs<int>("USE_VAR_EPSILON", ProjectSettings::optional, "0");
    if (use_var_eps) {
        double alpha = settings->GetSettingAs<double>("VAR_EPSILON_ALPHA", ProjectSettings::optional, "1.0"); 
        double beta = settings->GetSettingAs<double>("VAR_EPSILON_BETA", ProjectSettings::optional, "1.0"); 
        der->setVariableEpsilon(grid->getStencilRadii(), alpha, beta); 
    } else {
        double epsilon = settings->GetSettingAs<double>("EPSILON", ProjectSettings::required);
        der->setEpsilon(epsilon);
    }

    if (run_derivative_tests) {
        DerivativeTests* der_test = new DerivativeTests();
        der_test->testAllFunctions(*der, *grid);
//        delete(der_test);
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
