#include "utils/comm/communicator.h"
#include <stdlib.h>

#include "pdes/parabolic/heat_pde.h"

#include "grids/regulargrid.h"

#include "rbffd/rbffd_cl.h"
#include "rbffd/fun_cl.h"
#include "utils/random.h"
#include "utils/norms.h"

#include "exact_solutions/exact_regulargrid.h"

#include "timer_eb.h"

vector<double> u_cpu, xderiv_cpu, yderiv_cpu, zderiv_cpu, lderiv_cpu; 
RBFFD_CL::SuperBuffer<double> u_gpu;
RBFFD_CL::SuperBuffer<double> xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu;

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
void setupTimers(EB::TimerList& tm) {
    tm["main_total"] 		= new EB::Timer("[main] Total Time");
    tm["total"] 			= new EB::Timer("[main] Remaining time");
    tm["rbffd"] 			= new EB::Timer("[main] RBFFD constructor");
    tm["destructor"] 		= new EB::Timer("[main] Destructors");
    tm["stencils"] 			= new EB::Timer("[main] Stencil computation");
    tm["cpu_tests"] 		= new EB::Timer("[main] CPU tests");
    tm["gpu_tests"] 		= new EB::Timer("[main] GPU tests");
    tm["compute_weights"] 	= new EB::Timer("[main] Stencil weights");
    tm["deriv_accuracy"] 	= new EB::Timer("[main] Derivative Accuracy");
    tm["sort+grid"] 		= new EB::Timer("[main] Sort + Grid generation");
	tm["solution_check"] 	= new EB::Timer("[main] Solution check");
	tm.printAll(stdout, 60);
}
//----------------------------------------------------------------------
void initializeArrays()
{
	int size = grid->getNodeList().size();
    u_cpu.resize(4*size); 
    xderiv_cpu.resize(4*size);
    yderiv_cpu.resize(4*size);
    zderiv_cpu.resize(4*size);
    lderiv_cpu.resize(4*size);

	// d/dx(u) = u (given my weight definitions)
	// Output: dudx, dudy, dudz, dudl
	// All derivatives are equal if weight matrix is identity. 
	for (int i=0; i < u_cpu.size(); i+=4) {
		double rnd = randf(-1.,1.);
		if (i < 20) printf("rnd= %f\n", rnd);
		u_cpu[i] =      rnd;
		u_cpu[i+1] = 2.*rnd;
		u_cpu[i+2] = 3.*rnd;
		u_cpu[i+3] = 4.*rnd;
	}
}
//----------------------------------------------------------------------
void computeOnGPU()
{
	// Do not overwrite xderiv_cpu, so allocate new space on host (to compare against CPU results)
    tm["gpu_tests"]->start();
	xderiv_gpu = RBFFD_CL::SuperBuffer<double>(xderiv_cpu.size(), "xderiv_cpu"); 
	yderiv_gpu = RBFFD_CL::SuperBuffer<double>(yderiv_cpu.size(), "yderiv_cpu");
	zderiv_gpu = RBFFD_CL::SuperBuffer<double>(zderiv_cpu.size(), "zderiv_cpu");
	lderiv_gpu = RBFFD_CL::SuperBuffer<double>(lderiv_cpu.size(), "lderiv_cpu");

	u_gpu = RBFFD_CL::SuperBuffer<double>(u_cpu, "u_cpu"); 
	u_gpu.copyToDevice();
	for (int i=0; i < 20; i++) {
		printf("[main] u_gpu[%d]= %f\n", i, u_gpu[i]);
	}

	// Not in in RBBF (knows nothing about SuperBuffer). Must redesign
    der->computeDerivs(u_gpu, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true); 
    der->computeDerivs(u_gpu, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true); 

	//for (int i=0; i < 10; i++) {
		//printf("GPU bef) xder(i) = %f\n", i, xderiv_gpu[i]);
	//}

	xderiv_gpu.copyToHost();
	yderiv_gpu.copyToHost();
	zderiv_gpu.copyToHost();
	lderiv_gpu.copyToHost();

	u_gpu.copyToHost();
    tm["gpu_tests"]->end();
}
//----------------------------------------------------------------------
void computeOnCPU()
{
	printf("\n***** ComputeOnCPU *****\n");
    tm["cpu_tests"]->start();
    der_cpu = new RBFFD(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    der_cpu->computeAllWeightsForAllStencilsEmpty();

    // Verify that the CPU works
    // NOTE: we pass booleans at the end of the param list to indicate that
    // the function "u" is new (true) or same as previous calls (false). This
    // helps avoid overhead of passing "u" to the GPU.
	#if 1
	// u_cpu stores 4 functions. 
	// each derivative array stores 4 x-derivatives, one per function
	printf("size of u_cpu: %d\n", u_cpu.size());
	printf("size of xderiv_cpu: %d\n", xderiv_cpu.size());

	int nb_nodes = u_cpu.size() / 4;
    der_cpu->computeDeriv(RBFFD::X, u_cpu, xderiv_cpu, true);
    der_cpu->computeDeriv(RBFFD::X, &u_cpu[1*nb_nodes], &xderiv_cpu[1*nb_nodes], false); // originally false
    der_cpu->computeDeriv(RBFFD::X, &u_cpu[2*nb_nodes], &xderiv_cpu[2*nb_nodes], false); // originally false
    der_cpu->computeDeriv(RBFFD::X, &u_cpu[3*nb_nodes], &xderiv_cpu[3*nb_nodes], false); // originally false

    der_cpu->computeDeriv(RBFFD::Y,  u_cpu, 		    yderiv_cpu, true);
    der_cpu->computeDeriv(RBFFD::Y, &u_cpu[1*nb_nodes], &yderiv_cpu[1*nb_nodes], false); // originally false
    der_cpu->computeDeriv(RBFFD::Y, &u_cpu[2*nb_nodes], &yderiv_cpu[2*nb_nodes], false); // originally false
    der_cpu->computeDeriv(RBFFD::Y, &u_cpu[3*nb_nodes], &yderiv_cpu[3*nb_nodes], false); // originally false

    der_cpu->computeDeriv(RBFFD::Z,  u_cpu,              zderiv_cpu, true);
    der_cpu->computeDeriv(RBFFD::Z, &u_cpu[1*nb_nodes], &zderiv_cpu[1*nb_nodes], false); // originally false
    der_cpu->computeDeriv(RBFFD::Z, &u_cpu[2*nb_nodes], &zderiv_cpu[2*nb_nodes], false); // originally false
    der_cpu->computeDeriv(RBFFD::Z, &u_cpu[3*nb_nodes], &zderiv_cpu[3*nb_nodes], false); // originally false

    der_cpu->computeDeriv(RBFFD::LAPL,  u_cpu, 			    lderiv_cpu, true);
    der_cpu->computeDeriv(RBFFD::LAPL, &u_cpu[1*nb_nodes], &lderiv_cpu[1*nb_nodes], false); // originally false
    der_cpu->computeDeriv(RBFFD::LAPL, &u_cpu[2*nb_nodes], &lderiv_cpu[2*nb_nodes], false); // originally false
    der_cpu->computeDeriv(RBFFD::LAPL, &u_cpu[3*nb_nodes], &lderiv_cpu[3*nb_nodes], false); // originally false

	tm["cpu_tests"]->end();
	#endif
	printf("***** exit ComputeOnCPU *****\n\n");
}
//----------------------------------------------------------------------
void checkDerivativeAccuracy()
{
    tm["deriv_accuracy"]->start();
	double xnorm = linfnorm(*xderiv_gpu.host, xderiv_cpu);
	double ynorm = linfnorm(*yderiv_gpu.host, yderiv_cpu);
	double znorm = linfnorm(*zderiv_gpu.host, zderiv_cpu);
	double lnorm = linfnorm(*lderiv_gpu.host, lderiv_cpu);
	printf("x/y/z/l derivative error norms: %f, %f, %f, %f\n", 
	         xnorm, ynorm, znorm, lnorm);
	double eps = 1.e-5;
	if (xnorm > eps || ynorm > eps || znorm > eps || lnorm > eps) {
		printf("CPU and GPU derivative do not match to within %f\n", eps);
	}

	printf("***\n***Derivatives with errors larger than 1.e-5 ***\n");

	for (int i=0; i < xderiv_gpu.hostSize(); i++) {
		if (abs(xderiv_gpu[i] - xderiv_cpu[i]) > 1.e-5) {
			printf("(GPU aft) xder[%d]=%f, yder[%d]=%f, zder[%d]=%f, lder[%d]=%f\n", 
			   i, xderiv_gpu[i], i, yderiv_gpu[i], 
			   i, zderiv_gpu[i], i, lderiv_gpu[i]); 
			printf("(CPU aft) xder[%d]=%f, yder[%d]=%f, zder[%d]=%f, lder[%d]=%f\n", 
			   i, xderiv_cpu[i], i, yderiv_cpu[i], 
			   i, zderiv_cpu[i], i, lderiv_cpu[i]); 
		}
	}
    tm["deriv_accuracy"]->end();
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
    tm["compute_weights"]->start();
    if (use_gpu) {
        der = new FUN_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
		// Must be called before setKernelType
    	der->computeAllWeightsForAllStencilsEmpty(); 
		printf("before setKernelType\n");
		//der->setKernelType(FUN_CL::FUN_KERNEL); // necessary
		printf("after setKernelType\n");
		der->setKernelType(FUN_CL::FUN_INV_KERNEL);
        //der = new RBFFD_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    } else {
		printf("Routine meant to test GPU only\n");
		exit(0);
    }

    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);
	printf("*** epsilon= %f\n", epsilon);

	// weights are all in one large array (for all derivatives)
    tm["compute_weights"]->end();

}
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

	computeOnCPU(); // must be called before GPU
	computeOnGPU();
	checkDerivativeAccuracy();

    tm["main_total"]->end();
	tm.printAll(stdout, 60);

	cleanup();


    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
//
//stencils on GPU are zero. WHY? 
