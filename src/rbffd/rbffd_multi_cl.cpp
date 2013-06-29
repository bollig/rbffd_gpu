#include "rbffd_multi_cl.h"
#include <stdlib.h>
#include <math.h>
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT


using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
RBFFD_MULTI_CL::RBFFD_MULTI_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank)
   : RBFFD_CL(typesToCompute, grid, dim_num, rank)
{
    this->loadKernel("computeDerivMultiKernel", "derivative_kernels.cl");
    this->allocateGPUMem();
    //this->updateStencilsOnGPU(false);
    this->updateStencilsOnGPU(true); //GE
    std::cout << "Done copying stencils to GPU\n";

    //this->updateNodesOnGPU(false);
    this->updateNodesOnGPU(true);
    std::cout << "Done copying nodes to GPU\n";
}
//----------------------------------------------------------------------
void RBFFD_MULTI_CL::allocateGPUMem()
{
	RBFFD_CL::allocateGPUMem();

    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
    unsigned int float_size = useDouble? sizeof(double) : sizeof(float);
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int nb_stencils = stencil_map.size();

    function_mem_bytes = nb_nodes * float_size * 4;
    gpu_function = cl::Buffer(context, CL_MEM_READ_ONLY, function_mem_bytes, NULL, &err);

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
void RBFFD_MULTI_CL::applyWeightsForDerivDouble(unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv_x, double* deriv_y, double* deriv_z, double* deriv_l, bool isChangedU)
{
	cout << "****** enter of applyWeightsForDerivativeDouble ******\n";

    if (isChangedU) {
        this->updateFunctionOnGPU(start_indx, 4*nb_stencils, u, false);
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(false);

    err = queue.finish(); // added by GE
    tm["applyWeights"]->start();

    try {
        int i = 0;
        kernel.setArg(i++, gpu_stencils);
        kernel.setArg(i++, this->getGPUWeights(RBFFD::X));
        kernel.setArg(i++, this->getGPUWeights(RBFFD::Y));
        kernel.setArg(i++, this->getGPUWeights(RBFFD::Z));
        kernel.setArg(i++, this->getGPUWeights(RBFFD::LAPL));
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

	copyArrayToHost<double>(gpu_deriv_x_out, &deriv_x[0]); // did not work
	copyArrayToHost(gpu_deriv_y_out, &deriv_y[0]);
	copyArrayToHost(gpu_deriv_z_out, &deriv_z[0]);
	copyArrayToHost(gpu_deriv_l_out, &deriv_l[0]);
}
//----------------------------------------------------------------------
#if 0
void RBFFD_MULTI_CL::copyResultsToHost(double* deriv_x, double* deriv_y, double* deriv_z, double* deriv_l)
{
// deriv_mem_bytes is sets by various subclass, and could change. But this function will always work, 
// as long as the number of derivative arrays does not change. Even that could be changed through loops
// and pointers. 
//
    err = queue.enqueueReadBuffer(gpu_deriv_x_out, CL_TRUE, 0, deriv_mem_bytes, &deriv_x[0], NULL, &event);
    err = queue.enqueueReadBuffer(gpu_deriv_y_out, CL_TRUE, 0, deriv_mem_bytes, &deriv_y[0], NULL, &event);
    err = queue.enqueueReadBuffer(gpu_deriv_z_out, CL_TRUE, 0, deriv_mem_bytes, &deriv_z[0], NULL, &event);
    err = queue.enqueueReadBuffer(gpu_deriv_l_out, CL_TRUE, 0, deriv_mem_bytes, &deriv_l[0], NULL, &event);
    //    queue.flush();

    if (err != CL_SUCCESS) {
        std::cerr << " enequeue ERROR: " << err << std::endl;
    }

    err = queue.finish();

    if (err != CL_SUCCESS) {
        std::cerr << "Event::wait() failed (" << err << ")\n";
        std::cout << "Failed to finish queue" << std::endl;
    } else {
        //        std::cout << "CL program finished!" << std::endl;
    }
}
#endif
//----------------------------------------------------------------------
