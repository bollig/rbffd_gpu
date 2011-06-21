#include <stdlib.h>

#include "pdes/parabolic/heat_pde.h"

#include "grids/regulargrid.h"

#include "rbffd/rbffd_cl.h"

#include "exact_solutions/exact_regulargrid.h"

#include "utils/comm/communicator.h"

using namespace std;

int main(int argc, char** argv) {

    Communicator* comm_unit = new Communicator(argc, argv);

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit->getRank());

	
    int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required); 
    int nx = settings->GetSettingAs<int>("NB_X", ProjectSettings::required); 
    int ny = 1; 
    int nz = 1; 
    if (dim > 1) {
    	ny = settings->GetSettingAs<int>("NB_Y", ProjectSettings::required); 
    }
    if (dim > 2) {
	nz = settings->GetSettingAs<int> ("NB_Z", ProjectSettings::required); 
    } 
    if (dim > 3) {
	cout << "ERROR! Dim > 3 Not supported!" << endl;
	exit(EXIT_FAILURE); 
    }

    double minX = settings->GetSettingAs<double>("MIN_X", ProjectSettings::optional, "-1."); 	
    double maxX = settings->GetSettingAs<double>("MAX_X", ProjectSettings::optional, "1."); 	
    double minY = settings->GetSettingAs<double>("MIN_Y", ProjectSettings::optional, "-1."); 	
    double maxY = settings->GetSettingAs<double>("MAX_Y", ProjectSettings::optional, "1."); 	
    double minZ = settings->GetSettingAs<double>("MIN_Z", ProjectSettings::optional, "-1."); 	
    double maxZ = settings->GetSettingAs<double>("MAX_Z", ProjectSettings::optional, "1."); 

    double stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

    int use_gpu = settings->GetSettingAs<int>("USE_GPU", ProjectSettings::optional, "1"); 

    Grid* grid; 
 
    if (dim == 1) {
	    grid = new RegularGrid(nx, 1, minX, maxX, 0., 0.); 
    } else if (dim == 2) {
	    grid = new RegularGrid(nx, ny, minX, maxX, minY, maxY); 
    } else if (dim == 3) {
	    grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
    } else {
	    cout << "ERROR! Dim > 3 Not Supported!" << endl;
    }

    grid->setSortBoundaryNodes(true); 
    grid->generate();
    grid->generateStencils(stencil_size, Grid::ST_BRUTE_FORCE);   // nearest nb_points
    grid->writeToFile(); 

    // 0: 2D problem; 1: 3D problem
    ExactSolution* exact_heat_regulargrid = new ExactRegularGrid(dim, 1.0, 1.0);

    RBFFD* der;
    if (use_gpu) {
        der = new RBFFD_CL(grid, dim); 
    } else {
        der = new RBFFD(grid, dim); 
    }


    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);

    printf("start computing weights\n");
    vector<StencilType>& stencil = grid->getStencils();
    vector<NodeType>& rbf_centers = grid->getNodeList();
    der->computeAllWeightsForAllStencils();
    cout << "end computing weights" << endl;

    vector<double> u(rbf_centers.size(),1.);
    cout << "start computing derivative (on CPU)" << endl;
	    

    vector<double> xderiv_cpu(rbf_centers.size());	
    vector<double> xderiv_gpu(rbf_centers.size());	
    vector<double> yderiv_cpu(rbf_centers.size());	
    vector<double> yderiv_gpu(rbf_centers.size());	
    vector<double> zderiv_cpu(rbf_centers.size());	
    vector<double> zderiv_gpu(rbf_centers.size());	
    vector<double> lderiv_cpu(rbf_centers.size());	
    vector<double> lderiv_gpu(rbf_centers.size());	

    // Verify that the CPU works
    // NOTE: we pass booleans at the end of the param list to indicate that
    // the function "u" is new (true) or same as previous calls (false). This
    // helps avoid overhead of passing "u" to the GPU.
    der->RBFFD::applyWeightsForDeriv(RBFFD::X, u, xderiv_cpu, true);
    der->RBFFD::applyWeightsForDeriv(RBFFD::Y, u, yderiv_cpu, false);
    der->RBFFD::applyWeightsForDeriv(RBFFD::Z, u, zderiv_cpu, false);
    der->RBFFD::applyWeightsForDeriv(RBFFD::LAPL, u, lderiv_cpu, false);

    der->applyWeightsForDeriv(RBFFD::X, u, xderiv_gpu, true);
    der->applyWeightsForDeriv(RBFFD::Y, u, yderiv_gpu, false);
    der->applyWeightsForDeriv(RBFFD::Z, u, zderiv_gpu, false);
    der->applyWeightsForDeriv(RBFFD::LAPL, u, lderiv_gpu, false);

    for (int i = 0; i < rbf_centers.size(); i++) {
//        std::cout << "cpu_x_deriv[" << i << "] - gpu_x_deriv[" << i << "] = " << xderiv_cpu[i] - xderiv_gpu[i] << std::endl;
        if ( (xderiv_gpu[i] - xderiv_cpu[i] > 1e-5) 
        || (yderiv_gpu[i] - yderiv_cpu[i] > 1e-5) 
        || (zderiv_gpu[i] - zderiv_cpu[i] > 1e-5) 
        || (lderiv_gpu[i] - lderiv_cpu[i] > 1e-5) )
        {
            std::cout << "WARNING! SINGLE PRECISION GPU COULD NOT CALCULATE DERIVATIVE WELL ENOUGH!\n";
	    std::cout << "Test failed on " << i << std::endl;
	    std::cout << "X: " << xderiv_gpu[i] - xderiv_cpu[i] << std:: endl; 
	    std::cout << "Y: " << yderiv_gpu[i] - yderiv_cpu[i] << std:: endl; 
	    std::cout << "Z: " << zderiv_gpu[i] - zderiv_cpu[i] << std:: endl; 
	    std::cout << "LAPL: " << lderiv_gpu[i] - lderiv_cpu[i] << std:: endl; 
            exit(EXIT_FAILURE); 
        }
    }
    std::cout << "CONGRATS! ALL DERIVATIVES WERE CALCULATED THE SAME IN OPENCL AND ON THE CPU\n";
       // (WITH AN AVERAGE ERROR OF:" << avg_error << std::endl;

   // der->applyWeightsForDeriv(RBFFD::Y, u, yderiv);
   // der->applyWeightsForDeriv(RBFFD::LAPL, u, lapl_deriv);


#if 0
    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS")) {
        RBFFDTests* der_test = new DerivativeTests();
        der_test->testAllFunctions(*der, *grid);
    }
#endif 


//    delete(subdomain);
    delete(grid);
    delete(settings);

    cout.flush();

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
