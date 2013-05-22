#include <stdlib.h>
#include <math.h>
#include "rbffd_multi_cl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT


using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
RBFFD_MULTI_CL::RBFFD_MULTI_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank)
   : RBFFD_CL(typesToCompute, grid, dim_num, rank)
    //deleteCPUWeightsBuffer(false)
    //deleteCPUNodesBuffer(false),
    //deleteCPUStencilsBuffer(false),
    //useDouble(true),
// Gordon: changing alignWeights to true will break gpu_compute_derivs; 
    //alignWeights(false), alignMultiple(32)
{

	//GE: added as a means to avoid deleting that which is not allocated
	for (int i=0; i < NUM_DERIVATIVE_TYPES; i++) {
		cpu_weights_d[i] = 0;
	}

    this->setupTimers();
    //this->loadKernel();
    this->loadKernel("computeDerivMultiKernel", "cl_kernels/derivative_kernels.cl");
    this->allocateGPUMem();
    //this->updateStencilsOnGPU(false);
    this->updateStencilsOnGPU(true); //GE
    std::cout << "Done copying stencils\n";

    //this->updateNodesOnGPU(false);
    this->updateNodesOnGPU(true);
    std::cout << "Done copying nodes\n";
}


//----------------------------------------------------------------------
//
void RBFFD_MULTI_CL::setupTimers() {
    tm["loadAttach"] = new Timer("m [RBFFD_MULTI_CL] Load and Attach Kernel");
    tm["construct"] = new Timer("m [RBFFD_MULTI_CL] RBFFD_MULTI_CL (constructor)");
    tm["computeDerivs"] = new Timer("m [RBFFD_MULTI_CL] computeRBFFD_MULTI_s (compute derivatives using OpenCL");
    tm["sendWeights"] = new Timer("m [RBFFD_MULTI_CL]   (send stencil weights to GPU)");
    tm["applyWeights"] = new Timer("m [RBFFD_MULTI_CL] Evaluate single derivative by applying weights to function", 1);
}


//----------------------------------------------------------------------
//
#if 0
void RBFFD_MULTI_CL::loadKernel() {
    tm["construct"]->start();

    cout << "Inside RBFFD_MULTI_CL constructor" << endl;

    tm["loadAttach"]->start();

    // Split the kernels by __kernel keyword and do not discards the keyword.
    // TODO: support extensions declared for kernels (this class can add FP64
    // support)
    // std::vector<std::string> separated_kernels = this->split(kernel_source, "__kernel", true);

    std::string kernel_name = "computeDerivMultiKernel";

    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false;
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        useDouble = false;
    }

    std::string my_source;
    if(useDouble) {
#define FLOAT double
#include "cl_kernels/multi_derivative_kernel.cl"
        my_source = kernel_source;
#undef FLOAT
    }else {
#define FLOAT float
#include "cl_kernels/multi_derivative_kernel.cl"
        my_source = kernel_source;
#undef FLOAT
    }

    std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n";
    std::cout << "Loading program source: multi_derivative_kernel.cl\n";
    this->loadProgram(my_source, useDouble);

    try{
        std::cout << "Loading kernel \""<< kernel_name << "\" with double precision = " << useDouble << "\n";
        kernel = cl::Kernel(program, kernel_name.c_str(), &err);
        std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel RBFFD_MULTI] ERROR: %s(%d)\n", er.what(), er.err());
    }
    tm["loadAttach"]->end();
}
#endif

//----------------------------------------------
void RBFFD_MULTI_CL::allocateGPUMem() {

    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int nb_stencils = stencil_map.size();

    cout << "Allocating GPU memory for stencils, solution, weights and derivative" << endl;

    unsigned int max_stencil_size = grid_ref.getMaxStencilSize();
    if (alignWeights) {
        stencil_padded_size =  this->getNextMultiple(max_stencil_size);
        std::cout << "STENCIL ALIGNED TO SIZE: " << stencil_padded_size << std::endl;
    } else {
        stencil_padded_size = max_stencil_size;
#if 0
        // No need to assume we're going to have non-uniform stencil sizes. If we do, we'll pad them all to be
        // the max_stencil_size.
       gpu_stencil_size = 0;
        for (unsigned int i = 0; i < stencil_map.size(); i++) {
            gpu_stencil_size += stencil_map[i].size();
        }
#endif
    }

    gpu_stencil_size = stencil_padded_size * stencil_map.size();

    unsigned int float_size;
    if (useDouble) {
        float_size = sizeof(double);
    } else {
        float_size = sizeof(float);
    }
    std::cout << "FLOAT_SIZE=" << float_size << std::endl;;

    stencil_mem_bytes = gpu_stencil_size * sizeof(unsigned int);
    function_mem_bytes = nb_nodes * float_size * 4; // u,v,w,p
    weights_mem_bytes = gpu_stencil_size * float_size;
    deriv_mem_bytes = nb_stencils * float_size * 4; // four variables per derivative

    nodes_mem_bytes = nb_nodes * sizeof(double4);

    std::cout << "Allocating GPU memory\n";

    unsigned int bytesAllocated = 0;

    // Two input arrays:
    // 	This one is allocated once on GPU and reused until our nodes move or we change the stencil size
    gpu_stencils = cl::Buffer(context, CL_MEM_READ_WRITE, stencil_mem_bytes, NULL, &err);
    bytesAllocated += stencil_mem_bytes;

    gpu_function = cl::Buffer(context, CL_MEM_READ_ONLY, function_mem_bytes, NULL, &err);

	// GE: Hardcoded for now, testing routine for efficient derivative calculation
	computedTypes = RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL; 
    int iterator = computedTypes;
    int which = 0;
    int type_i       = 0;
    // Iterate until we get all 0s. This allows SOME shortcutting.
    while (iterator) {
        if (computedTypes & getDerType(which)) {
            gpu_weights[which] = cl::Buffer(context, CL_MEM_READ_ONLY, weights_mem_bytes, NULL, &err);
            bytesAllocated += weights_mem_bytes;
            type_i+=1;
        }
        else {
            // HACK: my gpu kernels take ALL weights on gpu as parameters. This allows me to put only one value for "empty" weight types
            // minimal memory consumption. It works but its a band-aid
            gpu_weights[which] = cl::Buffer(context, CL_MEM_READ_ONLY, 1*float_size, NULL, &err);
            bytesAllocated += 1*float_size; //added by GE
        }
        iterator >>= 1;
        which += 1;
    }

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
void RBFFD_MULTI_CL::updateNodesOnGPU(bool forceFinish) {
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    cpu_nodes = new double4[nb_nodes];

    for (unsigned int i = 0; i < nb_nodes; i++) {
        cpu_nodes[i] = double4( nodes[i].x(),
                nodes[i].y(),
                nodes[i].z(),
                0.0 );
    }
    err = queue.enqueueWriteBuffer(gpu_nodes, CL_TRUE, 0, nodes_mem_bytes, &cpu_nodes[0], NULL, &event);
    //        queue.flush();
    if (forceFinish) {
        queue.finish();
		// BUG? deleting weights not yet allocated? 
		// Or add check in clearCPUWeights: should have null pointer if nothing allocated. 
        //GE this->clearCPUWeights();
		std::cout << "GE: after clearCPUWeights() 3\n";
		// GE: what is this variable for? 
        deleteCPUNodesBuffer= false;
    } else {
        deleteCPUNodesBuffer = true;
    }
}

//----------------------------------------------------------------------
//
void RBFFD_MULTI_CL::updateStencilsOnGPU(bool forceFinish) {
    std::cout << "UPDATING STENCILS\n";
    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
    unsigned int nb_stencils = stencil_map.size();
    // TODO: queue a WriteBuffer for each stencil and pass the &vector[0] ptr. CPU should handle that
    //      better than a loop, and we dont have to allocate memory then
    // FIXME: Store stencils as linear unsigned int array, not int** or vecctor<int>.
    // FIXME: If we align memory to the nearest XXX we could improve access patterns.
    //  (figure out the stencil size to target)
    //  NOTE: its not essential since we only put stencils on GPU one time.
    cpu_stencils = new unsigned int[gpu_stencil_size];
    for (unsigned int i = 0; i < nb_stencils; i++) {
        unsigned int j;
        for (j = 0; j < stencil_map[i].size(); j++) {
            unsigned int indx = i*stencil_padded_size+j;
            cpu_stencils[indx] = stencil_map[i][j];
            //std::cout << cpu_stencils[indx] << "   ";
        }
        // Buffer remainder of stencils with the stencil center (so we can
        // break on GPU when center ID is duplicated. This also prevents random
        // values from showing up beyond our stencil index.
        // NOTE: another failsafe is to have the weights set to 0 beyond this
        // point so we dont add any values from outside indices
        for (; j < stencil_padded_size; j++) {
            unsigned int indx = i*stencil_padded_size+j;
            cpu_stencils[indx] = stencil_map[i][0];
            //std::cout << cpu_stencils[indx] << "   ";
        }
        //std::cout << endl;
    }

    //    std::cout << "Writing GPU Stencils buffer: (bytes)" << stencil_mem_bytes << std::endl;
    err = queue.enqueueWriteBuffer(gpu_stencils, CL_TRUE, 0, stencil_mem_bytes, &cpu_stencils[0], NULL, &event);
    //    queue.flush();
    if (forceFinish) {
        queue.finish();
        this->clearCPUStencils();
        deleteCPUStencilsBuffer = false;
    } else {
        deleteCPUStencilsBuffer = true;
    }

}

void RBFFD_MULTI_CL::clearCPUNodes() {
    delete [] cpu_nodes;
}


void RBFFD_MULTI_CL::clearCPUStencils() {
    delete [] cpu_stencils;
}



void RBFFD_MULTI_CL::clearCPUWeights() {
    std::cout << "DELETING CPU WEIGHTS\n";
    // Clear out buffer. No need to keep it since this should only happen once
    // NOTE: make sure we delete only the single or double precision cpu side buffers;
    int iterator = computedTypes;
    int which = 0;
    int type_i       = 0;
    // Iterate until we get all 0s. This allows SOME shortcutting.
    while (iterator) {
        if (computedTypes & getDerType(which)) {
            if (useDouble) {
    			//std::cout << "GE  useDouble, which= " << which << "\n";
                delete [] cpu_weights_d[which];
				if (cpu_weights_d[which]) cpu_weights_d[which] = 0; //GE added
            } else {
    			//std::cout << "GE  useSingle\n";
                delete [] cpu_weights_f[which];
				if (cpu_weights_f[which]) cpu_weights_f[which] = 0; //GE added
				//cpu_weights_f[which] = 0; //GE added
            }
            type_i+=1;
        }
        iterator >>= 1;
        which += 1;
    }
	std::cout << "exit clearCPUWeights\n";
}

//----------------------------------------------------------------------
//
// Update the stencil weights on the GPU using double precision:
// 1) Get the correct weights for the DerType
// 2) send weights to GPU
// 3) send new u to GPU
// 4) call kernel to inner prod weights and u writing to deriv
// 5) get deriv from GPU
void RBFFD_MULTI_CL::updateWeightsDouble(bool forceFinish) {

	//std::cout << "GE enter updateWeightsDouble\n";
    if (weightsModified) {

        tm["sendWeights"]->start();
        unsigned int weights_mem_size = gpu_stencil_size * sizeof(double);

        std::cout << "Writing weights to GPU memory\n";

        unsigned int nb_stencils = grid_ref.getStencilsSize();

        if ((nb_stencils * stencil_padded_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*stencil_padded_size != gpu_stencil_size" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Copy the std::vector<std::vector<..> > into a contiguous memory space
        // FIXME: inside grid_interface we could allocate contig mem and avoid this cost
        // FIXME: copy more than just the 4 types of weights
        int iterator = computedTypes;
		//std::cout << "GE updateWeightsDouble, computedTypes= " << computedTypes << std::endl;
//std::cout << "EVAN PADDED: " << stencil_padded_size << " of " << grid_ref.getStencilSize(0) << std::endl;
        int which = 0;
        int type_i = 0;
        // Iterate until we get all 0s. This allows SOME shortcutting.
        while (iterator) {
            if (computedTypes & getDerType(which)) {
                cpu_weights_d[which] = new double[gpu_stencil_size];
				//std::cout << "GE allocted cpu_weights, which= " << which << std::endl;
				printf("updateWeightsDouble, COMPUTED type: which= %d\n", which);
                for (unsigned int i = 0; i < nb_stencils; i++) {
                    unsigned int stencil_size = grid_ref.getStencilSize(i);
                    unsigned int j = 0;
                    for (j = 0; j < stencil_size; j++) {
                        unsigned int indx = i*stencil_padded_size + j;
                        cpu_weights_d[which][indx] = (double)weights[which][i][j];
                        //  std::cout << cpu_weights[which][indx] << "   ";
                    }
                    // Pad end of the stencil with 0's so our linear combination
                    // excludes whatever function values are found at the end of
                    // the stencil (i.e., we can include extra terms in summation
                    // without added effect
                    for (; j < stencil_padded_size; j++) {
                        unsigned int indx = i*stencil_padded_size + j;
                        cpu_weights_d[which][indx] = (double)0.;
                        //  std::cout << cpu_weights[which][indx] << "   ";
                    }
                    //std::cout << std::endl;
                }
                //std::cout << std::endl;
                // Send to GPU
                err = queue.enqueueWriteBuffer(gpu_weights[which], CL_TRUE, 0, weights_mem_size, &(cpu_weights_d[which][0]), NULL, &event);
//std::cout << "Wrote weight buffer to gpu: " << which << "," << weights_mem_size << " bytes\n";
                //            queue.flush();

                type_i+=1;
            }
            iterator >>= 1;
            which += 1;
        }

        if (forceFinish) {
            queue.finish();

            this->clearCPUWeights();
			std::cout << "GE: after clearCPUWeights() 1\n";
            deleteCPUWeightsBuffer = false;
        } else {
            deleteCPUWeightsBuffer = true;
        }

        tm["sendWeights"]->end();

        weightsModified = false;

    } else {
        //        std::cout << "No need to update gpu_weights" << std::endl;
    }
}

//-----------------
void RBFFD_MULTI_CL::updateWeightsSingle(bool forceFinish) {
	printf("updateWeightsSingle not implemented\n");
    exit(EXIT_FAILURE);
}



//----------------------------------------------------------------------
//
void RBFFD_MULTI_CL::updateFunctionDouble(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish) {

    cout << "Sending " << nb_vals << " solution updates to GPU: (bytes)" << function_mem_bytes << endl;


    // There is a bug conditional is true
    if (function_mem_bytes != 4*nb_vals*sizeof(double)) { // four variables interleaved
        std::cout << "function_mem_bytes != nb_nodes*sizeof(double)" << std::endl;
        exit(EXIT_FAILURE);
    } else {
	std::cout << "Updating solution: " << function_mem_bytes << " bytes \n";
	std::cout << start_indx << ", " << nb_vals << "\n";
    }
    // TODO: mask off fields not update
    err = queue.enqueueWriteBuffer(gpu_function, CL_TRUE, start_indx*sizeof(double), function_mem_bytes, &u[start_indx], NULL, &event);
    //    queue.flush();

    if (forceFinish) {
        queue.finish();
    }
}

//----------------------------------------------------------------------
//
void RBFFD_MULTI_CL::updateFunctionSingle(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish) {
	printf("updateFunctionSingle not implemented\n");
    exit(EXIT_FAILURE);
}


//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//

void RBFFD_MULTI_CL::applyWeightsForDerivDouble(unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv_x, double* deriv_y, double* deriv_z, double* deriv_l, bool isChangedU)
{
    //TODO: FIX case when start_indx != 0
    //std::cout << "EVAN HERE\n";
	cout << "****** enter of applyWeightsForDerivativeDouble ******\n";

    //cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;

    if (isChangedU) {
        this->updateFunctionOnGPU(start_indx, nb_stencils, u, false);
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(false);

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
		#if 0
        kernel.setArg(i++, gpu_stencils);
        kernel.setArg(i++, this->getGPUWeights(which));
        kernel.setArg(i++, gpu_function);                 // COPY_IN
        kernel.setArg(i++, gpu_deriv_out);           // COPY_OUT
		#endif
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


    err = queue.enqueueNDRangeKernel(kernel, /* offset */ cl::NullRange,
            /* GLOBAL (work-groups in the grid)  */   cl::NDRange(nb_stencils),
            /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);
    //    queue.flush();
    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (4*nb_stencils *sizeof(double) != deriv_mem_bytes) { // four variables
        std::cout << "npts*sizeof(double) [" << nb_stencils*sizeof(double) << "] != deriv_mem_bytes [" << deriv_mem_bytes << "]" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = queue.finish(); // added by GE (for more accurate timings, although in real code, one would only use finish()
	                      // GE   as far back into the code as possible. 
    tm["applyWeights"]->end();

    // Pull the computed derivative back to the CPU
    //err = queue.enqueueReadBuffer(gpu_deriv_out, CL_TRUE, 0, deriv_mem_bytes, &deriv[0], NULL, &event);
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
    //tm["applyWeights"]->end(); // original location set by Evan
	cout << "****** enter of applyWeightsForDerivativeDouble ******\n";
}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//
//
#if 0
void RBFFD_MULTI_CL::applyWeightsForDerivSingle(DerType which, unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU)
{
	printf("applyWeightsForDerivSingle not implemented\n");
    exit(EXIT_FAILURE);
}
#endif
//----------------------------------------------------------------------








// TODO:
//      1) Move writeBuffers to separate routine for stencils and weights.
//      2) Send solution to GPU in separate routine, then have routine for UPDATES only (we dont want to copy entire solution each step).
//      3) Timers
//      4) Offload more work (for example the timestep update in PDE is a vector-plus-scalar-vector operation.
//      5) computeDeriv should operate on memory owned by the PDE. The GPU pointer to the solution should be passed in.
//      6) The PDE class should manage the vec+scal*vec operation to update the solution on GPU.
//          (inherit original PDE class with CL specific one; both can use RBFFD_MULTI_CL, but if its a CL
//          class it should be able to pass in GPU mem pointer via advanced API).

