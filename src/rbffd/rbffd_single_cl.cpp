#include <stdlib.h>
#include <math.h>
#include "rbffd_single_cl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT


using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
    RBFFD_SINGLE_CL::RBFFD_SINGLE_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank)
: RBFFD_CL(typesToCompute, grid, dim_num, rank)
{
    this->loadKernel("computeDerivKernel", "derivative_kernels.cl");
    this->allocateGPUMem();
    //this->updateStencilsOnGPU(false);
    this->updateStencilsOnGPU(true); //GE
    std::cout << "Done copying stencils to GPU\n";

    //this->updateNodesOnGPU(false);
    this->updateNodesOnGPU(true);
    std::cout << "Done copying nodes to GPU\n";
}
//----------------------------------------------------------------------
void RBFFD_SINGLE_CL::allocateGPUMem()
{
	RBFFD_CL::allocateGPUMem();

    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
    unsigned int float_size = useDouble? sizeof(double) : sizeof(float);
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int nb_stencils = stencil_map.size();

	std::cout << "*** allocateGPUMemory ***\n";
	std::cout << "** float_size= " << float_size << std::endl;
	std::cout << "** nb_nodes= " << nb_nodes << std::endl;
    function_mem_bytes = nb_nodes * float_size;
	std::cout << "** function_mem_bytes= " << function_mem_bytes << std::endl;
    gpu_function = cl::Buffer(context, CL_MEM_READ_ONLY, function_mem_bytes, NULL, &err);

    deriv_mem_bytes = nb_stencils * float_size;
    gpu_deriv_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    bytesAllocated += deriv_mem_bytes; 

    gpu_nodes = cl::Buffer(context, CL_MEM_READ_ONLY, nodes_mem_bytes, NULL, &err);
    bytesAllocated += nodes_mem_bytes;

    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
}
//----------------------------------------------------------------------
//
void RBFFD_SINGLE_CL::applyWeightsForDerivDouble(DerType which, unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU)
{
    //TODO: FIX case when start_indx != 0
    //std::cout << "EVAN HERE\n";

    //cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;
    if (isChangedU) {
        //this->updateFunctionOnGPU(start_indx, nb_stencils, u, false);
        this->updateFunctionOnGPU(start_indx, nb_stencils, u, true);  //GE
    }


    this->updateWeightsOnGPU(true); //GE

    err = queue.finish(); // added by GE

	// strictly speaking, I should not include settings the kernel arguments, but that time should be insignificant. 
	// One should not take into account the time for read/write of data to the GPU, since in a good code, 
	// one would minimize read/writing to the GPU. 
    tm["applyWeights"]->start();

    try {
        int i = 0;
        kernel.setArg(i++, gpu_stencils);
        kernel.setArg(i++, this->getGPUWeights(which));
        kernel.setArg(i++, gpu_function);                 // COPY_IN
        kernel.setArg(i++, gpu_deriv_out);           // COPY_OUT
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

	copyArrayToHost(gpu_deriv_out, &deriv[0]);
	//copyResultsToHost(deriv);
}
//----------------------------------------------------------------------
#if 0
void RBFFD_SINGLE_CL::copyResultsToHost(double* deriv)
{
    err = queue.enqueueReadBuffer(gpu_deriv_out, CL_TRUE, 0, deriv_mem_bytes, &deriv[0], NULL, &event);
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
