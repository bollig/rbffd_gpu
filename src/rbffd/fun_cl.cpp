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
    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
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

	sup_stencils = SuperBuffer<int>(nb_nodes);
	sup_all_weights = SuperBuffer<double>(nb_nodes*nb_stencils*4);
}
//----------------------------------------------------------------------
void FUN_CL::calcDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, 
			SuperBuffer<double>& deriv_y, SuperBuffer<double>& deriv_z, SuperBuffer<double>& deriv_l, bool isChangedU)
{
	printf("applyWeightsFoDerivDouble using SuperBuffer arguments\n");
	if (isChangedU) u.copyToDevice();
	sup_all_weights.copyToDevice();

    err = queue.finish(); // added by GE
    tm["applyWeights"]->start();
    unsigned int nb_stencils = nb_nodes;

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
        unsigned int stencil_size = grid_ref.getMaxStencilSize();
        kernel.setArg(i++, sizeof(unsigned int), &stencil_size);            // const
        //kernel.setArg(i++, sizeof(unsigned int), &stencil_padded_size);            // const
        std::cout << "Set " << i << " kernel args\n";
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

	enqueueKernel(kernel, cl::NDRange(nb_stencils), cl::NullRange, true);
    tm["applyWeights"]->end();
}
//----------------------------------------------------------------------
