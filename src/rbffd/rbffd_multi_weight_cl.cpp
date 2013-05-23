#include <stdlib.h>
#include <math.h>
#include "rbffd_multi_weight_cl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT


using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
RBFFD_MULTI_WEIGHT_CL::RBFFD_MULTI_WEIGHT_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank)
   : RBFFD_CL(typesToCompute, grid, dim_num, rank)
{
    this->loadKernel("computeDerivMultiWeightKernel", "derivative_kernels.cl");
    this->allocateGPUMem();
    //this->updateStencilsOnGPU(false);
    this->updateStencilsOnGPU(true); //GE
    std::cout << "Done copying stencils to GPU\n";

    //this->updateNodesOnGPU(false);
    this->updateNodesOnGPU(true);
    std::cout << "Done copying nodes to GPU\n";
}
//----------------------------------------------------------------------
void RBFFD_MULTI_WEIGHT_CL::allocateGPUMem()
{
	RBFFD_CL::allocateGPUMem();

    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
    unsigned int float_size = useDouble? sizeof(double) : sizeof(float);
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int nb_stencils = stencil_map.size();

	// CHECK that stencils were computed (NOT DONE)

	all_weights_bytes = nb_nodes * stencil_map[0].size() * float_size * 4;
	std::cout << "allocateGPU, stencil size: " << stencil_map[0].size() << std::endl;
	gpu_all_weights = cl::Buffer(context, CL_MEM_READ_WRITE, all_weights_bytes, NULL, &err);
	bytesAllocated += all_weights_bytes;

    function_mem_bytes = nb_nodes * float_size * 4;
    gpu_function = cl::Buffer(context, CL_MEM_READ_ONLY, function_mem_bytes, NULL, &err);
	bytesAllocated += function_mem_bytes;

    deriv_mem_bytes = nb_stencils * float_size;
    gpu_deriv_x_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    gpu_deriv_y_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    gpu_deriv_z_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    gpu_deriv_l_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    bytesAllocated += deriv_mem_bytes*4; // 4 derivatives 

    gpu_nodes = cl::Buffer(context, CL_MEM_READ_ONLY, nodes_mem_bytes, NULL, &err);
    bytesAllocated += nodes_mem_bytes;

    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
}
//----------------------------------------------------------------------
void RBFFD_MULTI_WEIGHT_CL::applyWeightsForDerivDouble(unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv_x, double* deriv_y, double* deriv_z, double* deriv_l, bool isChangedU)
{
	cout << "****** enter of applyWeightsForDerivativeDouble ******\n";

    if (isChangedU) {
        this->updateFunctionOnGPU(start_indx, 4*nb_stencils, u, false);
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(true);

    err = queue.finish(); // added by GE
    tm["applyWeights"]->start();

	double* all_weights = new double[nb_stencils*4]; 

    try {
        int i = 0;
        kernel.setArg(i++, gpu_stencils);
        kernel.setArg(i++, gpu_all_weights);
        //kernel.setArg(i++, this->getGPUWeights(RBFFD::X));
        //kernel.setArg(i++, this->getGPUWeights(RBFFD::Y));
        //kernel.setArg(i++, this->getGPUWeights(RBFFD::Z));
        //kernel.setArg(i++, this->getGPUWeights(RBFFD::LAPL));
        kernel.setArg(i++, gpu_function);                 // COPY_IN // need double4
        kernel.setArg(i++, gpu_deriv_x_out);           // COPY_OUT
        kernel.setArg(i++, gpu_deriv_y_out);           // COPY_OUT
        kernel.setArg(i++, gpu_deriv_z_out);           // COPY_OUT
        kernel.setArg(i++, gpu_deriv_l_out);           // COPY_OUT
        //FIXME: we want to pass a unsigned int for maximum array lengths, but OpenCL does not allow
        //unsigned int arguments at this time
        unsigned int nb_stencils = grid_ref.getStencilsSize();
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


    if (nb_stencils *sizeof(double) != deriv_mem_bytes) { 
        std::cout << "npts*sizeof(double) [" << nb_stencils*sizeof(double) << "] != deriv_mem_bytes [" << deriv_mem_bytes << "]" << std::endl;
        exit(EXIT_FAILURE);
    }

	// not required
	//copyResultsToHost(deriv_x, deriv_y, deriv_z, deriv_l);
}
//----------------------------------------------------------------------
void RBFFD_MULTI_WEIGHT_CL::updateWeightsDouble(bool forceFinish)
{
// simply create a large array of zeros.
//

	std::cout << "GE enter updateWeightsDouble\n";
    if (weightsModified) {

        tm["sendWeights"]->start();
        unsigned int weights_mem_size = gpu_stencil_size * sizeof(double);

        std::cout << "[RBFFD_MULTI_WEIGHT_CL] Writing weights to GPU memory\n";

        unsigned int nb_stencils = grid_ref.getStencilsSize();
		unsigned int  tot_elements = gpu_stencil_size * nb_stencils * 4;

		double* all_weights = new double [tot_elements];
		for (int i=0; i < tot_elements; i++) {
			all_weights[i] = 0.0;
		}

        err = queue.enqueueWriteBuffer(gpu_all_weights, CL_TRUE, 0, tot_elements*sizeof(int), &(all_weights[0]), NULL, &event);
        if (forceFinish) {
            queue.finish();
		}


        if ((nb_stencils * stencil_padded_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*stencil_padded_size != gpu_stencil_size" << std::endl;
            exit(EXIT_FAILURE);
        }

        tm["sendWeights"]->end();

        weightsModified = false;

    } else {
        //        std::cout << "No need to update gpu_weights" << std::endl;
    }
}
//----------------------------------------------------------------------
