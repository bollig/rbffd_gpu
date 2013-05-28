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

    this->loadKernel("computeDerivMultiWeightFunKernel", "derivative_kernels.cl");
    this->allocateGPUMem();
}
//----------------------------------------------------------------------
void FUN_CL::allocateGPUMem()
{
	//RBFFD_CL::allocateGPUMem();

	printf("entering allocateGPUMem in fun_cl.cpp\n");
    unsigned int float_size = useDouble? sizeof(double) : sizeof(float);

	printf("nb_stencils= %d\n", nb_stencils);
	printf("nb_nodes= %d\n", nb_nodes);
	printf("nodes_per_stencil= %d\n", nodes_per_stencil);

	printf("**** nb_nodes= %d\n", nb_nodes);
	printf("nodes_per_stencil= %d\n", nodes_per_stencil);
	//sup_stencils = SuperBuffer<int>(nb_nodes*nodes_per_stencil);
	//sup_all_weights = SuperBuffer<double>(nb_nodes*nodes_per_stencil*4);
	// Must replace with actual values at some point

	sup_stencils.create(nb_nodes*nodes_per_stencil);
	sup_all_weights.create(nb_nodes*nodes_per_stencil*4);

	#if 0
	// assumes stencils of constant size, no padding
	int count = 0;
	printf("count= %d\n", count);
	std::vector<double>& h = *sup_all_weights.host;
	printf("h size: %d\n", h.size());
	for (int k=0; k < NUM_DERIVATIVE_TYPES; k++) {
		DerType dt = getDerType((DerTypeIndx)k); 
//        std::vector<double*> weights[NUM_DERIVATIVE_TYPES]; 
		printf("weights[%d] size = %d\n", k, weights[k].size());
		printf("weights[1] size = %d\n", weights[1].size());
		printf("weights[2] size = %d\n", weights[2].size());
		printf("weights[3] size = %d\n", weights[3].size());
		printf("weights[4] size = %d\n", weights[4].size());
		printf("weights[5] size = %d\n", weights[5].size());
		printf("weights[6] size = %d\n", weights[6].size());
//		printf("weights[0][0] size = %d\n", weights[0][0].size());
		if (!isSelected(dt)) continue;
		int which = getDerType(k);
		for (int i=0; i < nb_nodes; i++) {
			for (int j=0; j < nodes_per_stencil; j++) {
				printf("count= %d\n", count);
				printf("which= %d, i,j= %d, %d\n", which, i, j);
				printf("k= %d\n", k);
				printf("weights= %f\n", weights[k][i][j]);
				h[count++] = weights[k][i][j];
			}
		}
	}
	#endif


	printf("====================================================\n");
	printf("**** INFO on SUP_STENCILS and SUP_ALL_WEIGHT ***S\n");
	printf("sup_stencils, host = %d pts\n", sup_stencils.hostSize());
	printf("sup_stencils,  dev = %d pts\n", sup_stencils.devSize());
	printf("sup_all_weights, host = %d pts\n", sup_all_weights.hostSize());
	printf("sup_all_weights,  dev = %d pts\n", sup_all_weights.devSize());
}
//----------------------------------------------------------------------
void FUN_CL::computeDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, 
			SuperBuffer<double>& deriv_y, SuperBuffer<double>& deriv_z, SuperBuffer<double>& deriv_l, bool isChangedU)
{
	printf("computeDerivs/applyWeightsFoDerivDouble using SuperBuffer arguments\n");
	//if (isChangedU) u.copyToDevice();
	
	// transform to 1D arrays
	bool is_padded = false;
	// nbnode_nbsten_type == true : weights[rbf_nodes][stencil_nodes][der_type]
	// nbnode_nbsten_type == false: weights[stencil_nodes][rbf_nodes][der_type]
	bool nbnode_nbsten_type = true;
	convertWeightToContiguous(*sup_all_weights.host, *sup_stencils.host, nodes_per_stencil, is_padded, nbnode_nbsten_type);

	sup_stencils.copyToDevice();
	sup_all_weights.copyToDevice();

	for (int i=0; i < 20; i++) {
		printf("sup_stencils[%d] = %d\n", i, sup_stencils[i]);
	}

	for (int i=0; i < 20; i++) {
		printf("sup_all_weights[%d] = %f\n", i, sup_all_weights[i]);
	}

	for (int i=0; i < 20; i++) {
		printf("[beforeCopyToHost] u.dev[%d] = %f\n", i, (*u.host)[i]);
	}
	u.copyToHost();
	for (int i=0; i < 20; i++) {
		printf("[copyToHost] u.dev[%d] = %f\n", i, (*u.host)[i]);
	}

	//printf("fun_cl::computeDerivs\n"); exit(0);

    err = queue.finish(); // added by GE
    tm["applyWeights"]->start();
    unsigned int nb_stencils = nb_nodes;
    unsigned int stencil_size = grid_ref.getMaxStencilSize();

	printf("*** nb_stencils= %d\n", nb_stencils);
	printf("*** stencil_size= %d\n", stencil_size);

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

#if 1
    // User specifies work group size
    int items_per_group = 16;
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
	enqueueKernel:kernel, cl::NDRange(nb_stencils), cl::NullRange, true);
#endif

    tm["applyWeights"]->end();
}
//----------------------------------------------------------------------
