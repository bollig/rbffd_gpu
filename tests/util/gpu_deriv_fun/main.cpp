#include <stdlib.h>

#include "pdes/parabolic/heat_pde.h"

#include "grids/regulargrid.h"

#include "rbffd/rbffd_cl.h"
#include "rbffd/fun_cl.h"
#include "utils/random.h"

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
RBFFD* der_cpu;
FUN_CL* der;
//RBFFD_CL* der;

using namespace std;

//----------------------------------------------------------------------
void initializeArrays()
{
	int size = grid->getNodeList().size();
    u_cpu.resize(4*size); 
    xderiv_cpu.resize(4*size);
    yderiv_cpu.resize(4*size);
    zderiv_cpu.resize(4*size);
    lderiv_cpu.resize(4*size);

	for (int i=0; i < u_cpu.size(); i++) {
		u_cpu[i] = randf(-1.,1.);
	}
}

//----------------------------------------------------------------------
void computeOnCPU()
{
    der_cpu = new RBFFD(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    der_cpu->computeAllWeightsForAllStencilsEmpty();

    // Verify that the CPU works
    // NOTE: we pass booleans at the end of the param list to indicate that
    // the function "u" is new (true) or same as previous calls (false). This
    // helps avoid overhead of passing "u" to the GPU.
	#if 1
    tm["cpu_tests"]->start();
	// u_cpu stores 4 functions. 
	// each derivative array stores 4 x-derivatives, one per function
	printf("size of u_cpu: %d\n", u_cpu.size());
	printf("size of xderiv_cpu: %d\n", xderiv_cpu.size());
    der_cpu->computeDeriv(RBFFD::X, u_cpu, xderiv_cpu, true);
    der_cpu->computeDeriv(RBFFD::Y, u_cpu, yderiv_cpu, false); // originally false
    der_cpu->computeDeriv(RBFFD::Z, u_cpu, zderiv_cpu, false); // orig false
    der_cpu->computeDeriv(RBFFD::LAPL, u_cpu, lderiv_cpu, false); // orig false

	for (int i=0; i < 10; i++) {
		printf("(CPU) u(%d)= %f, derx(%d)= %f\n", i, u_cpu[i], i, xderiv_cpu[i]);
	}
	tm["cpu_tests"]->end();
	#endif

}
//----------------------------------------------------------------------
void checkDerivativeAccuracy() {
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

#endif

#if 0
    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS")) {
        RBFFDTests* der_test = new DerivativeTests();
        der_test->testAllFunctions(*der, *grid);
    }
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
void setupDerivativeWeights()
{
	tm["rbffd"]->start();
    if (use_gpu) {
        der = new FUN_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
        //der = new RBFFD_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    } else {
        //der = new RBFFD(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    }
	tm["rbffd"]->end();

    tm["compute_weights"]->start();
    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);
	printf("*** epsilon= %f\n", epsilon);

	// weights are all in one large array (for all derivatives)
    der->computeAllWeightsForAllStencilsEmpty();
    tm["compute_weights"]->end();
    cout << "end computing weights" << endl;

}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void cleanup()
{
    tm["destructor"]->start();
	delete(der);
	printf("after delete der\n");
	delete(der_cpu);
	printf("after delete der_cpu\n");
    delete(grid);  // **** ERROR I BELIEVE (WHY?)
	printf("after delete grid\n");
    delete(settings);
	printf("after delete settings\n");
    cout.flush();
    tm["destructor"]->end();
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
	initializeArrays();
	setupDerivativeWeights();

	printf("dim = %d, before new RBFFD\n");


	// **** NEED A COPY CONSTRUCTOR or an operator=
	// Ideally, I should be able to work with RBFFD only, with no knowledge of OpenCL. 

	// The Superbuffer makes this difficult
	RBFFD_CL::SuperBuffer<double> u_gpu(u_cpu, "u_cpu"); // not used yet

	// Do not overwrite xderiv_cpu, so allocate new space on host (to compare against CPU results)
	RBFFD_CL::SuperBuffer<double> xderiv_gpu(xderiv_cpu.size(), "xderiv_cpu"); // not used yet
	RBFFD_CL::SuperBuffer<double> yderiv_gpu(yderiv_cpu.size(), "yderiv_cpu");
	RBFFD_CL::SuperBuffer<double> zderiv_gpu(zderiv_cpu.size(), "zderiv_cpu");
	RBFFD_CL::SuperBuffer<double> lderiv_gpu(lderiv_cpu.size(), "lderiv_cpu");

	u_gpu.copyToDevice();

	// Not in in RBBF (knows nothing about SuperBuffer). Must redesign
    der->computeDerivs(u_gpu, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true); 

    tm["gpu_tests"]->start();  // skip first call
    der->computeDerivs(u_gpu, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true); 
    tm["gpu_tests"]->end();

	for (int i=0; i < 10; i++) {
		printf("GPU bef) xder(i) = %f\n", xderiv_gpu[i]);
	}

	xderiv_gpu.copyToHost();
	yderiv_gpu.copyToHost();
	zderiv_gpu.copyToHost();
	lderiv_gpu.copyToHost();

	u_gpu.copyToHost();

	for (int i=0; i < 10; i++) {
		printf("u_gpu[%d] = %f\n", i, u_gpu[i]);
	}

	for (int i=0; i < 10; i++) {
		printf("(GPU aft) xder(i) = %f\n", xderiv_gpu[i]);
	}


	printf("\n***** ComputeOnCPU *****\n");
	computeOnCPU();
	printf("***** exit ComputeOnCPU *****\n\n");
	checkDerivativeAccuracy();

	printf("before cleanup\n");
	cleanup();
	printf("after cleanup\n");
    tm["main_total"]->end();
	tm.printAll(stdout, 60);

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
//
//stencils on GPU are zero. WHY? 
