#include <stdlib.h>
#include <math.h>
#include "fun_cl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT


using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
FUN_CL::FUN_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank)
: RBFFD_CL(typesToCompute, grid, dim_num, rank)
{
	nb_nodes = grid_ref.getNodeListSize();
	std::vector<StencilType>& stencil_map = grid_ref.getStencils();
	nodes_per_stencil = stencil_map[0].size(); // assumed constant
    nb_stencils = stencil_map.size();
}
//----------------------------------------------------------------------
void FUN_CL::setKernelType(KernelType kernel_type_)
{
	kernel_type = kernel_type_;

	switch (kernel_type) {
	case FUN_KERNEL:
    	//loadKernel("computeDerivMultiWeightFunKernel", "derivative_kernels.cl");
    	loadKernel("computeDerivWeight1Fun1Kernel", "derivative_kernels.cl");
		break;
	case FUN_DERIV4_KERNEL:
    	loadKernel("computeDeriv4Weight1Fun1Kernel", "derivative_kernels.cl");
		break;
	case FUN_INV_KERNEL:
    	loadKernel("computeDerivWeight1Fun1InvKernel", "derivative_kernels.cl");
		break;
 	case FUN1_DERIV4_WEIGHT4:
    	loadKernel("computeDeriv4Weight4Fun1Kernel", "derivative_kernels.cl");
		break;
 	case FUN1_DERIV1_WEIGHT4:
    	loadKernel("computeDeriv1Weight4Fun1Kernel", "derivative_kernels.cl");
		break;
 	case FUN4_DERIV4_WEIGHT4:
    	loadKernel("computeDeriv4Weight4Fun4Kernel", "derivative_kernels.cl");
		break;
 	case FUN4_DERIV4_WEIGHT4_INV:
    	loadKernel("computeDeriv4Weight4Fun4InvKernel", "derivative_kernels.cl");
		break;
	}

	// Derivative weights must have been computed by now. 
	//printf("*** before allocateGPUMem ***\n");
    this->allocateGPUMem();
	//printf("*** after allocateGPUMem ***\n");
}
//----------------------------------------------------------------------
void FUN_CL::allocateGPUMem()
{
	//RBFFD_CL::allocateGPUMem();

	//printf("entering allocateGPUMem in fun_cl.cpp\n");
    unsigned int float_size = useDouble? sizeof(double) : sizeof(float);

	//printf("nb_stencils= %d\n", nb_stencils);
	//printf("nb_nodes= %d\n", nb_nodes);
	//printf("nodes_per_stencil= %d\n", nodes_per_stencil);

	//printf("**** nb_nodes= %d\n", nb_nodes);
	//printf("nodes_per_stencil= %d\n", nodes_per_stencil);
	//sup_stencils = SuperBuffer<int>(nb_nodes*nodes_per_stencil);
	//sup_all_weights = SuperBuffer<double>(nb_nodes*nodes_per_stencil*4);
	// Must replace with actual values at some point

	sup_stencils.create(nb_nodes*nodes_per_stencil);
	// Save space for x,y,z,l derivative (even if I compute only one, for code 
	// simplification
	sup_all_weights.create(nb_nodes*nodes_per_stencil*4);


	printf("====================================================\n");
	printf("**** INFO on SUP_STENCILS and SUP_ALL_WEIGHT ***S\n");
	printf("sup_stencils, host = %d pts\n", sup_stencils.hostSize());
	printf("sup_stencils,  dev = %d pts\n", sup_stencils.devSize());
	printf("sup_all_weights, host = %d pts\n", sup_all_weights.hostSize());
	printf("sup_all_weights,  dev = %d pts\n", sup_all_weights.devSize());
}
//----------------------------------------------------------------------
void FUN_CL::convertWeights()
{
	bool nbnode_nbsten_type;
	bool is_padded = false;

	switch (kernel_type) {
	case FUN_KERNEL:
	case FUN_DERIV4_KERNEL:
	case FUN1_DERIV4_WEIGHT4:
	case FUN1_DERIV1_WEIGHT4:
	case FUN4_DERIV4_WEIGHT4:
		//printf("FUN_KERNEL\n");
		nbnode_nbsten_type = true;
		// Weights must have been computed before converting
		convertWeightToContiguous(*sup_all_weights.host, *sup_stencils.host, nodes_per_stencil, is_padded, nbnode_nbsten_type);
		//printf("after convertWeight\n");
		break;
	case FUN_INV_KERNEL:
	case FUN4_DERIV4_WEIGHT4_INV:
		//printf("FUN_INV_KERNEL\n");
		nbnode_nbsten_type = false;
		convertWeightToContiguous(*sup_all_weights.host, *sup_stencils.host, nodes_per_stencil, is_padded, nbnode_nbsten_type);
		//printf("after convertWeight\n");
		break;
	}
}
//----------------------------------------------------------------------
void FUN_CL::computeDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, 
			SuperBuffer<double>& deriv_y, SuperBuffer<double>& deriv_z, SuperBuffer<double>& deriv_l, bool isChangedU)
{
	if (kernel_type  == -1) {
		printf("Kernel not defined\n");
		exit(0);
	}

    int nb_stencils = nb_nodes;
    int stencil_size = grid_ref.getMaxStencilSize();

	printf("computeDerivs/applyWeightsFoDerivDouble using SuperBuffer arguments\n");
	//if (isChangedU) u.copyToDevice();
	
	// transform to 1D arrays
	// nbnode_nbsten_type == true : weights[rbf_nodes][stencil_nodes][der_type]
	// nbnode_nbsten_type == false: weights[stencil_nodes][rbf_nodes][der_type]

	sup_stencils.copyToDevice();
	sup_all_weights.copyToDevice();

	#if 0
	for (int i=0; i < 20; i++) {
		printf("sup_stencils[%d] = %d\n", i, sup_stencils[i]);
	}
	#endif

	#if 0
	// Seem to be correct
	for (int i=0; i < sup_all_weights.hostSize(); i++) {
		printf("sup_all_weights[%d,%d] = %f\n", i, i%stencil_size, sup_all_weights[i]);
	}
	exit(0);
	#endif

	#if 0
	for (int i=0; i < 20; i++) {
		printf("[beforeCopyToHost] u.dev[%d] = %f\n", i, (*u.host)[i]);
	}
	u.copyToHost();
	for (int i=0; i < 20; i++) {
		printf("[copyToHost] u.dev[%d] = %f\n", i, (*u.host)[i]);
	}
	#endif

	//printf("fun_cl::computeDerivs\n"); exit(0);

    err = queue.finish(); // added by GE
    tm["applyWeights"]->start();

	//printf("*** nb_stencils= %d\n", nb_stencils);
	//printf("*** stencil_size= %d\n", stencil_size);

    try {
        int i = 0;
        kernel.setArg(i++, sup_stencils.dev); //gpu_stencils);
        kernel.setArg(i++, sup_all_weights.dev); //gpu_all_weights);
        kernel.setArg(i++, u.dev);              // 4 functions
        kernel.setArg(i++, deriv_x.dev);        
        kernel.setArg(i++, deriv_y.dev);       
        kernel.setArg(i++, deriv_z.dev);      
        kernel.setArg(i++, deriv_l.dev);     

        //FIXME: we want to pass a unsigned int for maximum array lengths, but OpenCL does not allow
        //unsigned int arguments at this time
        kernel.setArg(i++, sizeof(unsigned int), &nb_stencils);               // const
        kernel.setArg(i++, sizeof(unsigned int), &stencil_size);            // const
        //kernel.setArg(i++, sizeof(unsigned int), &stencil_padded_size);            // const
        std::cout << "Set " << i << " kernel args\n";
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
    }

#if 0
    // User specifies work group size
    int items_per_group = 32;
    int nn = nb_stencils % items_per_group;
    int nd = nb_stencils / items_per_group;
    int tot_items = (nn != 0) ? (nd+1)*items_per_group : nb_stencils; 
    std::cout << "** nb_stencils: " << nb_stencils << std::endl;
    std::cout << "** nb_stencils % items_per_group: " << nn << std::endl;
    std::cout << "** nb_stencils / items_per_group: " << nd << std::endl;
    std::cout << "** total number items: " << tot_items << std::endl;
    std::cout << "** items per group: " << items_per_group << std::endl;
    enqueueKernel(kernel, cl::NDRange(tot_items), cl::NDRange(items_per_group), true);
#else
	printf("nb_stencils= %d\n", nb_stencils);
	enqueueKernel(kernel, cl::NDRange(nb_stencils), cl::NullRange, true);
#endif

    tm["applyWeights"]->end();
}
//----------------------------------------------------------------------
void FUN_CL::computeDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, bool isChangedU)
{
	if (kernel_type  == -1) {
		printf("Kernel not defined\n");
		exit(0);
	}

	printf("1D: computeDerivs/applyWeightsFoDerivDouble using SuperBuffer arguments\n");
	//if (isChangedU) u.copyToDevice();
	
	// transform to 1D arrays
	// nbnode_nbsten_type == true : weights[rbf_nodes][stencil_nodes][der_type]
	// nbnode_nbsten_type == false: weights[stencil_nodes][rbf_nodes][der_type]

	sup_stencils.copyToDevice();
	sup_all_weights.copyToDevice();


	#if 0
	for (int i=0; i < sup_stencils.hostSize(); i++) {
		printf("(%d) sup_stencils[%d] = %d\n", i%8, i, sup_stencils[i]);
	}

	for (int i=0; i < sup_all_weights.hostSize(); i++) {
		printf("(%d) sup_all_weights[%d] = %f\n", i%8, i, sup_all_weights[i]);
	}
	#endif

	#if 0
	for (int i=0; i < 20; i++) {
		printf("[beforeCopyToHost] u.dev[%d] = %f\n", i, (*u.host)[i]);
	}
	u.copyToHost();
	for (int i=0; i < 20; i++) {
		printf("[copyToHost] u.dev[%d] = %f\n", i, (*u.host)[i]);
	}
	#endif

    err = queue.finish(); // added by GE
    tm["applyWeights"]->start();
    int nb_stencils = nb_nodes;
    int stencil_size = grid_ref.getMaxStencilSize();

	//printf("*** nb_stencils= %d\n", nb_stencils);
	//printf("*** stencil_size= %d\n", stencil_size);
	//printf("*** deriv_x_size= %d\n", deriv_x.hostSize());

	//for (int i=0; i < 50; i++) {
		//printf("w[%d] = %f\n", i, sup_all_weights[i]);
		//printf("u[%d] = %f\n", i, u[i]);
	//}
	//exit(0);

    try {
        int i = 0;
        kernel.setArg(i++, sup_stencils.dev); //gpu_stencils);
        kernel.setArg(i++, sup_all_weights.dev); //gpu_all_weights);
        kernel.setArg(i++, u.dev);              // 4 functions
        kernel.setArg(i++, deriv_x.dev);        
        //FIXME: we want to pass a unsigned int for maximum array lengths, but OpenCL does not allow
        //unsigned int arguments at this time
        kernel.setArg(i++, sizeof(int), &nb_stencils);               // const
        kernel.setArg(i++, sizeof(int), &stencil_size);            // const
        //kernel.setArg(i++, sizeof(unsigned int), &stencil_padded_size);            // const
        std::cout << "Set " << i << " kernel args\n";
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
    }

#if 0
    // User specifies work group size
    int items_per_group = 32;
    int nn = nb_stencils % items_per_group;
    int nd = nb_stencils / items_per_group;
    int tot_items = (nn != 0) ? (nd+1)*items_per_group : nb_stencils; 
    std::cout << "** nb_stencils: " << nb_stencils << std::endl;
    std::cout << "** nb_stencils % items_per_group: " << nn << std::endl;
    std::cout << "** nb_stencils / items_per_group: " << nd << std::endl;
    std::cout << "** total number items: " << tot_items << std::endl;
    std::cout << "** items per group: " << items_per_group << std::endl;
    enqueueKernel(kernel, cl::NDRange(tot_items), cl::NDRange(items_per_group), true);
#else
	// 16 RBF nodes are allocated per work-item, but the size of the workgroup is 
	// determined by the computer. Therefore, there are a total nb_stencils/16 
	// work-items per work-group. 

	int tot_items = getNextMultipleOf(nb_stencils, 64);
	switch (kernel_type) {
		case FUN_DERIV4_KERNEL:
			tot_items /= 4;
			tot_items = getNextMultipleOf(nb_stencils, 4);
			break;
		case FUN1_DERIV4_WEIGHT4:
			break;
	}
	//int tot_items = nb_stencils;
    //int items_per_group = 16;
    //int nn = nb_stencils % items_per_group;
    //int nd = nb_stencils / items_per_group;
    //int tot_items = (nn != 0) ? (nd+1)*items_per_group : nb_stencils; 
	//printf("nb_stencils= %d\n", nb_stencils);
	enqueueKernel(kernel, cl::NDRange(tot_items), cl::NullRange, true);
#endif

    tm["applyWeights"]->end();
}
//----------------------------------------------------------------------
