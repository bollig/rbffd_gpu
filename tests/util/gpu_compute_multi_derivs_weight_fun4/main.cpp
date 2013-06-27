#include <stdlib.h>

#include "pdes/parabolic/heat_pde.h"

#include "grids/regulargrid.h"

//#include "rbffd/rbffd_cl.h"
#include "rbffd/rbffd_multi_weight_fun4_cl.h"
#include "rbffd/rbffd_cl.h"

#include "exact_solutions/exact_regulargrid.h"

#include "utils/comm/communicator.h"
#include "timer_eb.h"


using namespace std;

int main(int argc, char** argv) {

    EB::TimerList tm; 
    tm["main_total"] = new EB::Timer("[main] Total Time");
    tm["total"] = new EB::Timer("[main] Remaining time");
    tm["rbffd"] = new EB::Timer("[main] RBFFD constructor");
    tm["destructor"] = new EB::Timer("[main] Destructors");
    tm["stencils"] = new EB::Timer("[main] Stencil computation");
    tm["cpu_tests"] = new EB::Timer("[main] CPU tests");
    tm["gpu_tests"] = new EB::Timer("[main] GPU tests");
    tm["compute_weights"] = new EB::Timer("[main] Stencil weights");
    tm["sort+grid"] = new EB::Timer("[main] Sort + Grid generation");
	tm["solution_check"] = new EB::Timer("[main] Solution check");

    tm["main_total"]->start();
	tm["total"]->start();

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

	// FIX: PROGRAM TO DEAL WITH SINGLE WEIGHT 
	//

    double minX = settings->GetSettingAs<double>("MIN_X", ProjectSettings::optional, "-1."); 	
    double maxX = settings->GetSettingAs<double>("MAX_X", ProjectSettings::optional, "1."); 	
    double minY = settings->GetSettingAs<double>("MIN_Y", ProjectSettings::optional, "-1."); 	
    double maxY = settings->GetSettingAs<double>("MAX_Y", ProjectSettings::optional, "1."); 	
    double minZ = settings->GetSettingAs<double>("MIN_Z", ProjectSettings::optional, "-1."); 	
    double maxZ = settings->GetSettingAs<double>("MAX_Z", ProjectSettings::optional, "1."); 

    double stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

    int use_gpu = settings->GetSettingAs<int>("USE_GPU", ProjectSettings::optional, "1"); 

    Grid* grid = NULL; 
 
    if (dim == 1) {
	    grid = new RegularGrid(nx, 1, minX, maxX, 0., 0.); 
    } else if (dim == 2) {
	    grid = new RegularGrid(nx, ny, minX, maxX, minY, maxY); 
    } else if (dim == 3) {
	    grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
    } else {
	    cout << "ERROR! Dim > 3 Not Supported!" << endl;
    }
	tm["total"]->end();

    tm["sort+grid"]->start();
    grid->setSortBoundaryNodes(true); 
    grid->generate();
    tm["sort+grid"]->end();


    tm["stencils"]->start();
    //grid->generateStencils(stencil_size, Grid::ST_BRUTE_FORCE);   // nearest nb_points
    //grid->generateStencils(stencil_size, Grid::ST_HASH);   // nearest nb_points
    //grid->generateStencils(stencil_size, Grid::ST_KDTREE);   // nearest nb_points
    //grid->generateStencils(stencil_size, Grid::ST_COMPACT);   // nearest nb_points
    grid->generateStencils(stencil_size, Grid::ST_RANDOM);   // nearest nb_points
    tm["stencils"]->end();
    //grid->writeToFile(); 


    // 0: 2D problem; 1: 3D problem
    //ExactSolution* exact_heat_regulargrid = new ExactRegularGrid(dim, 1.0, 1.0);

	tm["rbffd"]->start();
    RBFFD* der;

    if (use_gpu) {
        der = new RBFFD_MULTI_WEIGHT_FUN4_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
        //der = new RBFFD_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    } else {
        der = new RBFFD(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    }
	tm["rbffd"]->end();

	// weights are all in one large array (for all derivatives)

    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);

    printf("start computing weights\n");
    //vector<StencilType>& stencil = grid->getStencils();
	tm["total"]->start();
    vector<NodeType>& rbf_centers = grid->getNodeList();
	tm["total"]->end();
    //der->computeAllWeightsForAllStencils();

    tm["compute_weights"]->start();
    der->computeAllWeightsForAllStencilsEmpty();
    tm["compute_weights"]->end();
    cout << "end computing weights" << endl;

	// I will be trying to handle 4 solution vectors stored in u
    vector<double> u(4*rbf_centers.size(),1.);
    cout << "start computing derivative (on CPU)" << endl;
	    

    vector<double> xderiv_cpu(rbf_centers.size());	
    vector<double> xderiv_gpu(rbf_centers.size());	
    vector<double> yderiv_cpu(rbf_centers.size());	
    vector<double> yderiv_gpu(rbf_centers.size());	
    vector<double> zderiv_cpu(rbf_centers.size());	
    vector<double> zderiv_gpu(rbf_centers.size());	
    vector<double> lderiv_cpu(rbf_centers.size());	
    vector<double> lderiv_gpu(rbf_centers.size());	


	// weights should be one large array
	// 

    // Verify that the CPU works
    // NOTE: we pass booleans at the end of the param list to indicate that
    // the function "u" is new (true) or same as previous calls (false). This
    // helps avoid overhead of passing "u" to the GPU.
    tm["cpu_tests"]->start();
    der->RBFFD::applyWeightsForDeriv(RBFFD::X, u, xderiv_cpu, true);
    der->RBFFD::applyWeightsForDeriv(RBFFD::Y, u, yderiv_cpu, false); // originally false
    der->RBFFD::applyWeightsForDeriv(RBFFD::Z, u, zderiv_cpu, false); // orig false
    der->RBFFD::applyWeightsForDeriv(RBFFD::LAPL, u, lderiv_cpu, false); // orig false
    tm["cpu_tests"]->end();

    der->applyWeightsForDeriv(u, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true); // do not time
    tm["gpu_tests"]->start();
    der->applyWeightsForDeriv(u, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true);
    tm["gpu_tests"]->end();
    //der->applyWeightsForDeriv(RBFFD::Y, u, yderiv_gpu, false); // orig false
    //der->applyWeightsForDeriv(RBFFD::Z, u, zderiv_gpu, false); // orig: false
    //der->applyWeightsForDeriv(RBFFD::LAPL, u, lderiv_gpu, false); // orig: false
//

	tm["solution_check"]->start();
    double max_diff = 0.; 
    for (size_t i = 0; i < rbf_centers.size(); i++) {
	double xdiff = fabs(xderiv_gpu[i] - xderiv_cpu[i]); 
	double ydiff = fabs(yderiv_gpu[i] - yderiv_cpu[i]);
	double zdiff = fabs(zderiv_gpu[i] - zderiv_cpu[i]);
	double ldiff = fabs(lderiv_gpu[i] - lderiv_cpu[i]);

	if (xdiff > max_diff) { max_diff = xdiff; }
	if (ydiff > max_diff) { max_diff = ydiff; }
	if (zdiff > max_diff) { max_diff = zdiff; }
	if (ldiff > max_diff) { max_diff = ldiff; }

//        std::cout << "cpu_x_deriv[" << i << "] - gpu_x_deriv[" << i << "] = " << xderiv_cpu[i] - xderiv_gpu[i] << std::endl;
        if (( xdiff > 1e-5) 
        || ( ydiff > 1e-5) 
        || ( zdiff > 1e-5) 
        || ( ldiff > 1e-5))
        {
            std::cout << "WARNING! SINGLE PRECISION GPU COULD NOT CALCULATE DERIVATIVE WELL ENOUGH!\n";
	    	std::cout << "Test failed on " << i << std::endl;
	    	std::cout << "X: " << xderiv_gpu[i] - xderiv_cpu[i] << std:: endl; 
	    	std::cout << "X: " << xderiv_gpu[i] << ", " <<  xderiv_cpu[i] << std:: endl; 
	    	std::cout << "Y: " << yderiv_gpu[i] - yderiv_cpu[i] << std:: endl; 
	    	std::cout << "Y: " << yderiv_gpu[i] << ", " <<  yderiv_cpu[i] << std:: endl; 
	    	std::cout << "Z: " << zderiv_gpu[i] - zderiv_cpu[i] << std:: endl; 
	    	std::cout << "Z: " << zderiv_gpu[i] << ", " <<  zderiv_cpu[i] << std:: endl; 
	    	std::cout << "LAPL: " << lderiv_gpu[i] - lderiv_cpu[i] << std:: endl; 
			der->printAllTimings();
			tm.printAll(stdout, 80);
            exit(EXIT_FAILURE); 
        }
    }
    std::cout << "Max difference between weights: " << max_diff << std::endl;
    std::cout << "CONGRATS! ALL DERIVATIVES WERE CALCULATED THE SAME IN OPENCL AND ON THE CPU\n";
	tm["solution_check"]->end();
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

    tm["destructor"]->start();
	delete(der);
    delete(grid);
    delete(settings);
    cout.flush();
    tm["destructor"]->end();

    tm["main_total"]->end();
	tm.printAll(stdout, 60);

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
