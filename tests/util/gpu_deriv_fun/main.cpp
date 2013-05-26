#include <stdlib.h>

#include "pdes/parabolic/heat_pde.h"

#include "grids/regulargrid.h"

#include "rbffd/rbffd_cl.h"
#include "rbffd/fun_cl.h"

#include "exact_solutions/exact_regulargrid.h"

#include "utils/comm/communicator.h"
#include "timer_eb.h"

vector<double> u_cpu, xderiv_cpu, yderiv_cpu, zderiv_cpu, lderiv_cpu; 

Grid* grid;
int dim;
int stencil_size;
int use_gpu;
EB::TimerList tm; 
ProjectSettings* settings;
FUN_CL* der;

using namespace std;

//----------------------------------------------------------------------
void initializeArrays(int size)
{
    u_cpu.resize(4*size);
    xderiv_cpu.resize(size);
    yderiv_cpu.resize(size);
    zderiv_cpu.resize(size);
    lderiv_cpu.resize(size);
}
//----------------------------------------------------------------------
//void check_cpu(u, xderiv_cpu, yderiv_cpu, zderiv_cpu, lderiv_cpu);
void check_cpu()
{
    // Verify that the CPU works
    // NOTE: we pass booleans at the end of the param list to indicate that
    // the function "u" is new (true) or same as previous calls (false). This
    // helps avoid overhead of passing "u" to the GPU.
	#if 0
    tm["cpu_tests"]->start();
    der->RBFFD::applyWeightsForDeriv(RBFFD::X, u, xderiv_cpu, true);
    der->RBFFD::applyWeightsForDeriv(RBFFD::Y, u, yderiv_cpu, false); // originally false
    der->RBFFD::applyWeightsForDeriv(RBFFD::Z, u, zderiv_cpu, false); // orig false
    der->RBFFD::applyWeightsForDeriv(RBFFD::LAPL, u, lderiv_cpu, false); // orig false
    tm["cpu_tests"]->end();
	#endif

}
//----------------------------------------------------------------------
void setupTimers(EB::TimerList& tm) {
    tm["main_total"] 		= new EB::Timer("[main] Total Time");
    tm["total"] 			= new EB::Timer("[main] Remaining time");
    tm["rbffd"] 			= new EB::Timer("[main] RBFFD constructor");
    tm["destructor"] 		= new EB::Timer("[main] Destructors");
    tm["stencils"] 			= new EB::Timer("[main] Stencil computation");
    tm["cpu_tests"] 		= new EB::Timer("[main] CPU tests");
    tm["gpu_tests"] 		= new EB::Timer("[main] GPU tests");
    tm["compute_weights"] 	= new EB::Timer("[main] Stencil weights");
    tm["sort+grid"] 		= new EB::Timer("[main] Sort + Grid generation");
	tm["solution_check"] 	= new EB::Timer("[main] Solution check");
	tm.printAll(stdout, 60);
}
//----------------------------------------------------------------------
void createGrid()
{
	tm["total"]->start();

    int dim = 3;
    int nx = settings->GetSettingAs<int>("NB_X", ProjectSettings::required); 
    int ny = settings->GetSettingAs<int>("NB_Y", ProjectSettings::required); 
	int nz = settings->GetSettingAs<int>("NB_Z", ProjectSettings::required); 

	// FIX: PROGRAM TO DEAL WITH SINGLE WEIGHT 

    double minX = settings->GetSettingAs<double>("MIN_X", ProjectSettings::optional, "-1."); 	
    double maxX = settings->GetSettingAs<double>("MAX_X", ProjectSettings::optional, "1."); 	
    double minY = settings->GetSettingAs<double>("MIN_Y", ProjectSettings::optional, "-1."); 	
    double maxY = settings->GetSettingAs<double>("MAX_Y", ProjectSettings::optional, "1."); 	
    double minZ = settings->GetSettingAs<double>("MIN_Z", ProjectSettings::optional, "-1."); 	
    double maxZ = settings->GetSettingAs<double>("MAX_Z", ProjectSettings::optional, "1."); 

    stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 
    use_gpu = settings->GetSettingAs<int>("USE_GPU", ProjectSettings::optional, "1"); 

	grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
	printf("ny,ny,nz= %d, %d, %d\n", nx, ny, nz);
	printf("min/max= %f, %f, ,%f %f, %f, %f\n", minX, maxX, minY, maxY, minZ, maxZ);
	tm["total"]->end();

    tm["sort+grid"]->start();
    grid->setSortBoundaryNodes(true); 
    grid->generate();
    tm["sort+grid"]->end();

    tm["stencils"]->start();
	Grid::st_generator_t stencil_type = Grid::ST_COMPACT;
	//Grid::st_generator_t stencil_type = Grid::ST_RANDOM;
    grid->generateStencils(stencil_size, stencil_type);   // nearest nb_points
    tm["stencils"]->end();

}
//----------------------------------------------------------------------

int main(int argc, char** argv)
{
	setupTimers(tm);
	tm.printAll(stdout, 60);

    tm["main_total"]->start();

    Communicator* comm_unit = new Communicator(argc, argv);
    settings = new ProjectSettings(argc, argv, comm_unit->getRank());

	createGrid();
	initializeArrays(grid->getNodeList().size());


    // 0: 2D problem; 1: 3D problem
    //ExactSolution* exact_heat_regulargrid = new ExactRegularGrid(dim, 1.0, 1.0);

	tm["rbffd"]->start();
    //RBFFD* der;

    if (use_gpu) {
        der = new FUN_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
        //der = new RBFFD_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    } else {
        //der = new RBFFD(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    }
	tm["rbffd"]->end();

    tm["compute_weights"]->start();
    der->computeAllWeightsForAllStencilsEmpty();
    tm["compute_weights"]->end();
    cout << "end computing weights" << endl;

	// weights are all in one large array (for all derivatives)

    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);

    //printf("start computing weights\n");
    //vector<StencilType>& stencil = grid->getStencils();
	//tm["total"]->start();
    //vector<NodeType>& rbf_centers = grid->getNodeList();
	//tm["total"]->end();
    //der->computeAllWeightsForAllStencils();


	// I will be trying to handle 4 solution vectors stored in u
    cout << "start computing derivative (on CPU)" << endl;
	    

	SuperBuffer<double> u_gpu(u_cpu); // not used yet
	SuperBuffer<double> xderiv_gpu(xderiv_cpu); // not used yet
	SuperBuffer<double> yderiv_gpu(yderiv_cpu);
	SuperBuffer<double> zderiv_gpu(zderiv_cpu);
	SuperBuffer<double> lderiv_gpu(lderiv_cpu);

	 u_gpu.copyToDevice();
	 xderiv_gpu.copyToDevice();
	 yderiv_gpu.copyToDevice();
	 zderiv_gpu.copyToDevice();
	 lderiv_gpu.copyToDevice();

	// weights should be one large array

	vector<double>& uu = *u_gpu.host;
	vector<double>& xd = *xderiv_gpu.host;
	vector<double>& yd = *yderiv_gpu.host;
	vector<double>& zd = *zderiv_gpu.host;
	vector<double>& ld = *lderiv_gpu.host;

	//check_cpu(u, xderiv_cpu, yderiv_cpu, zderiv_cpu, lderiv_cpu);
	check_cpu();

	// compute on and retrieve weights from GPU
    //der->applyWeightsForDeriv(u, *xderiv_gpu.host, *yderiv_gpu.host, *zderiv_gpu.host, *lderiv_gpu.host, true); // do not time
    //der->applyWeightsForDeriv(u, &(*xderiv_gpu.host)[0], &(*yderiv_gpu.host)[0], &(*zderiv_gpu.host)[0], &(*lderiv_gpu.host)[0], true); // do not time


	printf("xd = %f\n", xd[10]);
    //der->applyWeightsForDeriv(u, xd, yd, zd, ld, true); // do not time
	// Not in in RBBF (knows nothing about SuperBuffer). Must redesign
    der->calcDerivs(u_gpu, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true); // do not time
    tm["gpu_tests"]->start();
    //der->applyWeightsForDeriv(u, xd, yd, zd, ld, true); // do not time
    der->calcDerivs(u_gpu, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true); // do not time
    tm["gpu_tests"]->end();

	#if 0
	tm["solution_check"]->start();
    double max_diff = 0.; 
    for (size_t i = 0; i < rbf_centers.size(); i++) {
	double xdiff = fabs(xd[i] - xderiv_cpu[i]); 
	double ydiff = fabs(yd[i] - yderiv_cpu[i]);
	double zdiff = fabs(zd[i] - zderiv_cpu[i]);
	double ldiff = fabs(ld[i] - lderiv_cpu[i]);

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
   #endif


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
