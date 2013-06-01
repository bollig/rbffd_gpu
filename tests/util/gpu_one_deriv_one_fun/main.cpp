#include <stdlib.h>

#include "pdes/parabolic/heat_pde.h"

#include "grids/regulargrid.h"

#include "rbffd/rbffd_cl.h"
#include "rbffd/fun_cl.h"
#include "utils/random.h"
#include "utils/norms.h"

#include "exact_solutions/exact_regulargrid.h"

#include "utils/comm/communicator.h"
#include "timer_eb.h"

vector<double> u_cpu;
RBFFD_CL::SuperBuffer<double> u_gpu;

vector<double> xderiv_cpu;
vector<double> yderiv_cpu;
vector<double> zderiv_cpu;
vector<double> lderiv_cpu;

RBFFD_CL::SuperBuffer<double> xderiv_gpu;
RBFFD_CL::SuperBuffer<double> yderiv_gpu;
RBFFD_CL::SuperBuffer<double> zderiv_gpu;
RBFFD_CL::SuperBuffer<double> lderiv_gpu;

RBFFD_CL::SuperBuffer<double> deriv4_gpu;



// Sames types as in rbffd/fun_cl.h
enum KernelType {FUN_KERNEL, FUN_INV_KERNEL, FUN_DERIV4_KERNEL,
	FUN1_DERIV4_WEIGHT4,
	FUN1_DERIV1_WEIGHT4};
KernelType kernel_type;

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
	// Redundant initializations since we have tons of memory
	int size = grid->getNodeList().size();
    u_cpu.resize(size); 
    xderiv_cpu.resize(size);
    yderiv_cpu.resize(size);
    zderiv_cpu.resize(size);
    lderiv_cpu.resize(size);

	// d/dx(u) = u (given my weight definitions)
	// Output: dudx, dudy, dudz, dudl
	// All derivatives are equal if weight matrix is identity. 
	for (int i=0; i < u_cpu.size(); i++) {
		double rnd = randf(-1.,1.);
		if (i < 20) printf("rnd= %f\n", rnd);
		u_cpu[i] =      rnd;
	}
}
//----------------------------------------------------------------------
void computeOnGPU4()
{
	//printf("Enter computerOnGPU4\n");
	// Do not overwrite xderiv_cpu, so allocate new space on host (to compare against CPU results)
    tm["gpu_tests"]->start();

	u_gpu = RBFFD_CL::SuperBuffer<double>(u_cpu, "u_cpu"); 
	u_gpu.copyToDevice();

	for (int i=0; i < 5; i++) {
		printf("[main] u_gpu[%d]= %f\n", i, u_gpu[i]);
	}

	der->convertWeights();

	// Not in in RBBF (knows nothing about SuperBuffer). Must redesign
	switch (kernel_type) {
	case FUN1_DERIV4_WEIGHT4:
		deriv4_gpu = RBFFD_CL::SuperBuffer<double>(4*xderiv_cpu.size(), "deriv4_gpu"); 
    	der->computeDerivs(u_gpu, deriv4_gpu, true); 
		deriv4_gpu.copyToHost();
		break;
	case FUN1_DERIV1_WEIGHT4:
		xderiv_gpu = RBFFD_CL::SuperBuffer<double>(xderiv_cpu.size(), "xderiv_gpu"); 
		yderiv_gpu = RBFFD_CL::SuperBuffer<double>(yderiv_cpu.size(), "yderiv_gpu"); 
		zderiv_gpu = RBFFD_CL::SuperBuffer<double>(zderiv_cpu.size(), "zderiv_gpu"); 
		lderiv_gpu = RBFFD_CL::SuperBuffer<double>(lderiv_cpu.size(), "lderiv_gpu"); 

    	der->computeDerivs(u_gpu, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true); 
    	der->computeDerivs(u_gpu, xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu, true); 

		xderiv_gpu.copyToHost();
		yderiv_gpu.copyToHost();
		zderiv_gpu.copyToHost();
		lderiv_gpu.copyToHost();
		break;
	}

	u_gpu.copyToHost();

    tm["gpu_tests"]->end();
}
//----------------------------------------------------------------------
void computeOnGPU()
{
	//printf("Enter computerOnGPU\n");
	// Do not overwrite xderiv_cpu, so allocate new space on host (to compare against CPU results)
    tm["gpu_tests"]->start();
	xderiv_gpu = RBFFD_CL::SuperBuffer<double>(xderiv_cpu.size(), "xderiv_cpu"); 

	u_gpu = RBFFD_CL::SuperBuffer<double>(u_cpu, "u_cpu"); 
	u_gpu.copyToDevice();
	//for (int i=0; i < 20; i++) {
		//printf("[main] u_gpu[%d]= %f\n", i, u_gpu[i]);
	//}

	der->convertWeights();
	// Not in in RBBF (knows nothing about SuperBuffer). Must redesign
    der->computeDerivs(u_gpu, xderiv_gpu, true); 
    der->computeDerivs(u_gpu, xderiv_gpu, true); 

	//for (int i=0; i < 10; i++) {
		//printf("GPU bef) xder(i) = %f\n", i, xderiv_gpu[i]);
	//}

	xderiv_gpu.copyToHost();
	//u_gpu.copyToHost();
    tm["gpu_tests"]->end();
}
//----------------------------------------------------------------------
void computeOnCPU4()
{
    tm["cpu_tests"]->start();
    der_cpu = new RBFFD(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 

    der_cpu->computeAllWeightsForAllStencilsEmpty();
	// Weights must already be computed

    der_cpu->computeDeriv(RBFFD::X,    u_cpu, xderiv_cpu, true);
    der_cpu->computeDeriv(RBFFD::Y,    u_cpu, yderiv_cpu, true);
    der_cpu->computeDeriv(RBFFD::Z,    u_cpu, zderiv_cpu, true);
    der_cpu->computeDeriv(RBFFD::LAPL, u_cpu, lderiv_cpu, true);
	tm["cpu_tests"]->end();

	for (int i=0; i < 20; i++) {
		printf("(%d), CPU4: u,dudx= %f, %f, %f, %f\n", i, u_cpu[i], xderiv_cpu[i], yderiv_cpu[i], zderiv_cpu[i], lderiv_cpu[i]);
	}
}
//----------------------------------------------------------------------
void computeOnCPU()
{
//	printf("\n***** ComputeOnCPU *****\n");
    tm["cpu_tests"]->start();

    der_cpu = new RBFFD(RBFFD::X, grid, dim); 
    der_cpu->computeAllWeightsForAllStencilsEmpty();
	// Weights must already be computed

    // Verify that the CPU works
    // NOTE: we pass booleans at the end of the param list to indicate that
    // the function "u" is new (true) or same as previous calls (false). This
    // helps avoid overhead of passing "u" to the GPU.
	#if 1
	// u_cpu stores a single function. 
	//printf("size of u_cpu: %d\n", u_cpu.size());
	//printf("size of xderiv_cpu: %d\n", xderiv_cpu.size());

    der_cpu->computeDeriv(RBFFD::X, u_cpu, xderiv_cpu, true);

	tm["cpu_tests"]->end();
	#endif
	//printf("***** exit ComputeOnCPU *****\n\n");
}
//----------------------------------------------------------------------
void checkDerivativeAccuracy()
{
    tm["deriv_accuracy"]->start();
	double xnorm = linfnorm(*xderiv_gpu.host, xderiv_cpu);
	printf("*********************************************\n");
	printf("**       INF derivative error norm: %f    ***\n", xnorm);
	printf("*********************************************\n");
	double eps = 1.e-5;
	if (xnorm > eps) {
		printf("CPU and GPU derivative do not match to within %f\n", eps);
	}

	//printf("***\n***Derivatives with errors larger than 1.e-5 ***\n");

	#if 0
	for (int i=0; i < 50; i++) {
		printf("(CPU/GPU der) %f, %f\n", i, xderiv_cpu[i], xderiv_gpu[i]); 
	}
	#endif

	#if 0
	for (int i=0; i < xderiv_gpu.hostSize(); i++) {
		if (i > 20) {
			printf("too many to print ...\n"); 
			break;
		}
		if (abs(xderiv_gpu[i] - xderiv_cpu[i]) > 1.e-5) {
			printf("(GPU aft) xder[%d]=%f\n", i, xderiv_gpu[i]); 
			printf("(CPU aft) xder[%d]=%f\n", i, xderiv_cpu[i]);
		}
	}
	#endif
    tm["deriv_accuracy"]->end();
}
//----------------------------------------------------------------------
// Array of Structures to Structure of Arrays
void AoS2SoA(std::vector<double>& xderiv_cpu, std::vector<double>& yderiv_cpu, 
			 std::vector<double>& zderiv_cpu, std::vector<double>& lderiv_cpu, 
			 std::vector<double>& deriv4_cpu)
{
	deriv4_cpu.resize(4*xderiv_cpu.size());

	for (int j=0, i=0; i < xderiv_cpu.size(); i++) {
		deriv4_cpu[j++] = xderiv_cpu[i];
		deriv4_cpu[j++] = yderiv_cpu[i];
		deriv4_cpu[j++] = zderiv_cpu[i];
		deriv4_cpu[j++] = lderiv_cpu[i];
	}
}
//----------------------------------------------------------------------
void checkDerivativeAccuracy4()
{
    tm["deriv_accuracy"]->start();
	// deriv_gpu: (ux,uy,uz,ul)_1, (ux,uy,uz,ul)_2
	// Array of Structures to Structure of Arrays
	vector<double> deriv4_cpu;
	double xnorm;

	switch (kernel_type) {
	case FUN1_DERIV4_WEIGHT4:
		AoS2SoA(xderiv_cpu, yderiv_cpu, zderiv_cpu, lderiv_cpu, deriv4_cpu);
		xnorm = linfnorm(*deriv4_gpu.host, deriv4_cpu);
		break;
	case FUN1_DERIV1_WEIGHT4:
		xnorm = linfnorm(*xderiv_gpu.host, xderiv_cpu);
		double ynorm = linfnorm(*xderiv_gpu.host, xderiv_cpu);
		double znorm = linfnorm(*xderiv_gpu.host, xderiv_cpu);
		double lnorm = linfnorm(*xderiv_gpu.host, xderiv_cpu);
		printf("x,y,z,l norms: %f, %f, %f, %f\n", xnorm, ynorm, znorm, lnorm);
		xnorm = (xnorm > ynorm) ? xnorm : ynorm;
		xnorm = (xnorm > znorm) ? xnorm : znorm;
		xnorm = (xnorm > lnorm) ? xnorm : lnorm;
		break;
	}

	printf("*********************************************\n");
	printf("INF derivative error norm: %f\n", xnorm);
	printf("*********************************************\n");
	double eps = 1.e-5;
	if (xnorm > eps) {
		printf("CPU and GPU derivative do not match to within %f\n", eps);
	}

	//printf("***\n***Derivatives with errors larger than 1.e-5 ***\n");

	#if 0
	for (int i=0; i < 20; i++) {
		printf("(CPU/GPU der) %f, %f\n", i, xderiv_cpu[i], deriv4_gpu[4*i]); 
		printf("(CPU/GPU der) %f, %f\n", i, yderiv_cpu[i], deriv4_gpu[4*i+1]); 
		printf("(CPU/GPU der) %f, %f\n", i, zderiv_cpu[i], deriv4_gpu[4*i+2]); 
		printf("(CPU/GPU der) %f, %f\n", i, lderiv_cpu[i], deriv4_gpu[4*i+3]); 
	}
	#endif

	#if 0
	for (int i=0; i < xderiv_cpu.size(); i++) {
		if (i > 20) {
			printf("too many to print ...\n"); 
			break;
		}
		if (abs(deriv4_gpu[4*i] - xderiv_cpu[i]) > 1.e-) {
			printf("(GPU aft) xder[%d]=%f\n", i, deriv4_gpu[4*i]); 
			printf("(CPU aft) xder[%d]=%f\n", i, xderiv_cpu[i]);
		}
	}
	#endif
	printf("----\n");
	switch (kernel_type) {
	case FUN1_DERIV4_WEIGHT4:
		for (int i=0; i < 20; i++) {
			printf("(%d) GPU: x,y,z,l= %f, %f, %f, %f\n", i, deriv4_gpu[4*i], deriv4_gpu[4*i+1], deriv4_gpu[4*i+2], deriv4_gpu[4*i+3]);
		}
		break;
	case FUN1_DERIV1_WEIGHT4:
		for (int i=0; i < 20; i++) {
			printf("(%d) GPU: x,y,z,l= %f, %f, %f, %f\n", i, xderiv_gpu[i], yderiv_gpu[i], zderiv_gpu[i], lderiv_gpu[i]);
		}
		break;
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
	//Grid::st_generator_t stencil_type = Grid::ST_COMPACT;
	Grid::st_generator_t stencil_type = Grid::ST_RANDOM;
    grid->generateStencils(stencil_size, stencil_type);   // nearest nb_points
    tm["stencils"]->end();

}
//----------------------------------------------------------------------
void setupDerivativeWeights()
{
	// Might need more options for setKernelType
	//
    tm["compute_weights"]->start();
    if (use_gpu) {
		switch (kernel_type) {
		case FUN_KERNEL:
			//printf("\n\nbefore new FUN_CL\n");
        	der = new FUN_CL(RBFFD::X, grid, dim); 
			//printf("  ** enter setKernelType\n");
			der->setKernelType(FUN_CL::FUN_KERNEL); // necessary
			//printf("  ** exited setKernelType\n");
			break;
		case FUN_INV_KERNEL:
        	der = new FUN_CL(RBFFD::X, grid, dim); 
			der->setKernelType(FUN_CL::FUN_INV_KERNEL);
			break;
		case FUN1_DERIV1_WEIGHT4:
        	der = new FUN_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
			der->setKernelType(FUN_CL::FUN1_DERIV1_WEIGHT4);
			break;
		case FUN1_DERIV4_WEIGHT4:
        	der = new FUN_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
			der->setKernelType(FUN_CL::FUN1_DERIV4_WEIGHT4);
			break;
		}
		//printf("before computeAllWeights\n");
    	der->computeAllWeightsForAllStencilsEmpty(); 
		//printf("after computeAllWeights\n");
    } else {
		//printf("Routine meant to test GPU only\n");
		exit(0);
    }

    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);
	//printf("*** epsilon= %f\n", epsilon);

	//printf("*** exit setupDerivativeWeights\n");

	// weights are all in one large array (for all derivatives)
    tm["compute_weights"]->end();

}
//----------------------------------------------------------------------
void cleanup()
{
    tm["destructor"]->start();
	delete(der);
	//printf("after delete der\n");
	delete(der_cpu);
	//printf("after delete der_cpu\n");
    delete(grid);  // **** ERROR I BELIEVE (WHY?)
	//printf("after delete grid\n");
    delete(settings);
	//printf("after delete settings\n");
    cout.flush();
    tm["destructor"]->end();
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
	setupTimers(tm);
	tm.printAll(stdout, 60);

	//kernel_type = FUN_KERNEL;
	//kernel_type = FUN1_DERIV4_WEIGHT4;
	kernel_type = FUN1_DERIV1_WEIGHT4;

    tm["main_total"]->start();

    Communicator* comm_unit = new Communicator(argc, argv);
    settings = new ProjectSettings(argc, argv, comm_unit->getRank());

	//printf("******   creeatGrid *****************\n");
	createGrid();
	//printf("******   initialize arrays *****************\n");
	initializeArrays();
	//printf("*******  setupDerivativeWeights *************\n");
	setupDerivativeWeights();
	//printf("*******  exit setupDerivativeWeights *************\n");

	switch (kernel_type) {
	case FUN_KERNEL:
	case FUN_INV_KERNEL:
		//printf("**** Compute on CPU ****\n");
		computeOnCPU(); // must be called before GPU
		//printf("**** Compute on GPU ****\n");
		computeOnGPU();
		checkDerivativeAccuracy();
		break;
	case FUN1_DERIV4_WEIGHT4:
	case FUN1_DERIV1_WEIGHT4:
		//printf("**** Compute on CPU4 ****\n");
		computeOnCPU4(); // must be called before GPU
		//printf("**** Compute on GPU4 ****\n");
		computeOnGPU4();
		checkDerivativeAccuracy4();
		break;
	}


    tm["main_total"]->end();
	tm.printAll(stdout, 60);

	cleanup();

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
//
//stencils on GPU are zero. WHY? 
