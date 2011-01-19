#include <stdlib.h>

#include "pdes/parabolic/new_heat.h"

#include "grids/regulargrid.h"
#include "grids/stencil_generator.h"

//#include "grids/domain_decomposition/domain.h"
#include "rbffd/derivative_cl.h"
//#include "rbffd/new_derivative_tests.h"

#include "exact_solutions/exact_regulargrid.h"

#include "utils/comm/communicator.h"

using namespace std;

int main(int argc, char** argv) {

    Communicator* comm_unit = new Communicator(argc, argv);

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit);

	
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
    grid->generateStencils(new StencilGenerator(stencil_size));   // nearest nb_points
    grid->writeToFile(); 
#if 0
    // Compute stencils given a set of generators
    // TODO: get these extents from the cvt class (add constructor to GPU)
    Domain* subdomain = new Domain(-1.,1.,-1.,1.,-1.,1.,0.,comm_unit->getRank(),comm_unit->getSize());    

    // Clean this up. Have GPU class fill data on constructor. Pass Grid class to constructor.
    // Remove need for extents in constructor.
    // Forms sets (Q,O,R) and l2g/g2l maps
    subdomain->fillLocalData(grid->getNodeList(), grid->getStencils(), 
                             grid->getBoundaryIndices(), grid->getStencilRadii());    
    subdomain->fillVarData(grid->getNodeList()); // Sets function values in U

    // Verbosely print the memberships of all nodes within the subdomain
    //subdomain->printCenterMemberships(subdomain->G, "G");
#endif

    // 0: 2D problem; 1: 3D problem
    ExactSolution* exact_heat_regulargrid = new ExactRegularGrid(1.0, 1.0);

#if 0
    // Clean this up. Have the Poisson class construct Derivative internally.
    Derivative* der = new DerivativeCL(subdomain->G_centers, subdomain->Q_stencils, subdomain->global_boundary_nodes.size(), dim, comm_unit->getRank());
#endif 

    Derivative* der = new DerivativeCL(grid->getNodeList(), grid->getStencils(), grid->getBoundaryIndices().size(), dim, comm_unit->getRank()); 

    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);

    printf("start computing weights\n");
    vector<StencilType>& stencil = grid->getStencils();
    vector<NodeType>& rbf_centers = grid->getNodeList();
    for (int irbf=0; irbf < rbf_centers.size(); irbf++) {
		der->computeWeights(rbf_centers, stencil[irbf], irbf);
	}
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
    der->computeDerivCPU(Derivative::X, u, xderiv_cpu);
    der->computeDeriv(Derivative::X, u, xderiv_gpu);

    der->computeDerivCPU(Derivative::Y, u, yderiv_cpu);
    der->computeDeriv(Derivative::Y, u, yderiv_gpu);
    
    der->computeDerivCPU(Derivative::Z, u, zderiv_cpu);
    der->computeDeriv(Derivative::Z, u, zderiv_gpu);

    der->computeDerivCPU(Derivative::LAPL, u, lderiv_cpu);
    der->computeDeriv(Derivative::LAPL, u, lderiv_gpu);

    for (int i = 0; i < rbf_centers.size(); i++) {
//        std::cout << "cpu_x_deriv[" << i << "] - gpu_x_deriv[" << i << "] = " << xderiv_cpu[i] - xderiv_gpu[i] << std::endl;
        if ( (xderiv_gpu[i] - xderiv_cpu[i] > 1e-5) 
        || (yderiv_gpu[i] - yderiv_cpu[i] > 1e-5) 
        || (zderiv_gpu[i] - zderiv_cpu[i] > 1e-5) 
        || (lderiv_gpu[i] - lderiv_cpu[i] > 1e-5) )
        {
            std::cout << "WARNING! SINGLE PRECISION GPU COULD NOT CALCULATE DERIVATIVE WELL ENOUGH!\n"; 
            exit(EXIT_FAILURE); 
        }
    }
    std::cout << "CONGRATS! ALL DERIVATIVES WERE CALCULATED THE SAME IN OPENCL AND ON THE CPU\n";
       // (WITH AN AVERAGE ERROR OF:" << avg_error << std::endl;

   // der->computeDeriv(Derivative::Y, u, yderiv);
   // der->computeDeriv(Derivative::LAPL, u, lapl_deriv);


#if 0
    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS")) {
        DerivativeTests* der_test = new DerivativeTests();
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
