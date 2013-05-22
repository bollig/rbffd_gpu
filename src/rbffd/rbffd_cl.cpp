#include <stdlib.h>
#include <math.h>
#include "rbffd_cl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT


using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
    RBFFD_CL::RBFFD_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank)
: RBFFD(typesToCompute, grid, dim_num, rank), CLBaseClass(rank),
    deleteCPUWeightsBuffer(false),
    deleteCPUNodesBuffer(false),
    deleteCPUStencilsBuffer(false),
    useDouble(true),
// Gordon: changing alignWeights to true will break gpu_compute_derivs; 
    alignWeights(false), alignMultiple(32)
{

	//GE: added as a means to avoid deleting that which is not allocated
	for (int i=0; i < NUM_DERIVATIVE_TYPES; i++) {
		cpu_weights_d[i] = 0;
	}

    this->setupTimers();
    //this->allocateGPUMem();
}

//----------------------------------------------------------------------
//
void RBFFD_CL::setupTimers() {
    tm["loadAttach"] = new Timer("[RBFFD_CL] Load and Attach Kernel");
    tm["construct"] = new Timer("[RBFFD_CL] RBFFD_CL (constructor)");
    tm["computeDerivs"] = new Timer("[RBFFD_CL] computeRBFFD_s (compute derivatives using OpenCL");
    tm["sendWeights"] = new Timer("[RBFFD_CL]   (send stencil weights to GPU)");
    tm["applyWeights"] = new Timer("[RBFFD_CL] Evaluate single derivative by applying weights to function", 1);
}


//----------------------------------------------------------------------
void RBFFD_CL::loadKernel(const std::string& kernel_name, const std::string& kernel_source_file)
{
    tm["loadAttach"]->start();

    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false;
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        useDouble = false;
    }

	cout << "kernel_name= " << kernel_name << endl;
	cout << "kernel_source_file = " << kernel_source_file << endl;

    // The true here specifies we search throught the dir specified by environment variable CL_KERNELS
    std::string my_source = this->loadFileContents(kernel_source_file.c_str(), true);

    std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n";
	std::cout  << my_source  << std::endl;
    this->loadProgram(my_source, useDouble);
	std::cout << "after load Program \n";

    try{
        std::cout << "Loading kernel \""<< kernel_name << "\" with double precision = " << useDouble << "\n";
        kernel = cl::Kernel(program, kernel_name.c_str(), &err);
        std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
    }

    tm["loadAttach"]->end();
}
//----------------------------------------------------------------------
//
#if 0
void RBFFD_CL::loadKernel()
{
    tm["loadAttach"]->start();

    // Split the kernels by __kernel keyword and do not discards the keyword.
    // TODO: support extensions declared for kernels (this class can add FP64
    // support)
    // std::vector<std::string> separated_kernels = this->split(kernel_source, "__kernel", true);

    std::string kernel_name = "computeDerivKernel";
	cout << "kernel_name= " << kernel_name << endl;

    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false;
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        useDouble = false;
    }

    std::string my_source;
    if(useDouble) {
#define FLOAT double
#include "cl_kernels/derivative_kernels.cl"
        my_source = kernel_source;
#undef FLOAT
    }else {
#define FLOAT float
#include "cl_kernels/derivative_kernels.cl"
        my_source = kernel_source;
#undef FLOAT
    }

    std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n";
    std::cout << "Loading program source: derivative_kernels.cl\n";
    this->loadProgram(my_source, useDouble);

    try{
        std::cout << "Loading kernel \""<< kernel_name << "\" with double precision = " << useDouble << "\n";
        kernel = cl::Kernel(program, kernel_name.c_str(), &err);
        std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel RBFFD] ERROR: %s(%d)\n", er.what(), er.err());
    }
    tm["loadAttach"]->end();
}
#endif

//----------------------------------------------
void RBFFD_CL::allocateGPUMem() {

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
    weights_mem_bytes = gpu_stencil_size * float_size;

    nodes_mem_bytes = nb_nodes * sizeof(double4);

    std::cout << "Allocating GPU memory\n";

    bytesAllocated = 0;

    // Two input arrays:
    // 	This one is allocated once on GPU and reused until our nodes move or we change the stencil size
    gpu_stencils = cl::Buffer(context, CL_MEM_READ_WRITE, stencil_mem_bytes, NULL, &err);
    bytesAllocated += stencil_mem_bytes;


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

	#if 0
    gpu_deriv_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    bytesAllocated += deriv_mem_bytes;

    gpu_nodes = cl::Buffer(context, CL_MEM_READ_ONLY, nodes_mem_bytes, NULL, &err);
    bytesAllocated += nodes_mem_bytes;
	#endif

    //std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
}

//----------------------------------------------------------------------
void RBFFD_CL::updateNodesOnGPU(bool forceFinish) {
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
void RBFFD_CL::updateStencilsOnGPU(bool forceFinish) {
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

void RBFFD_CL::clearCPUNodes() {
    delete [] cpu_nodes;
}


void RBFFD_CL::clearCPUStencils() {
    delete [] cpu_stencils;
}



void RBFFD_CL::clearCPUWeights() {
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
void RBFFD_CL::updateWeightsDouble(bool forceFinish) {

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

void RBFFD_CL::updateWeightsSingle(bool forceFinish) {

		//std::cout << "GE enter updateWeightsSingle\n";
    if (weightsModified) {

        tm["sendWeights"]->start();
        unsigned int weights_mem_size = gpu_stencil_size * sizeof(float);

        std::cout << "Writing weights to GPU memory\n";

        unsigned int nb_stencils = grid_ref.getStencilsSize();

        if ((nb_stencils * stencil_padded_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*stencil_padded_size != gpu_stencil_size" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Copy the std::vector<std::vector<..> > into a contiguous memory space
        // FIXME: inside grid_interface we could allocate contig mem and avoid this cost
        // FIXME: we only pass four weights to the GPU

        int iterator = computedTypes;
		//std::cout << "GE updateWeightsSingle, computedTypes= " << computedTypes << std::endl;
        int which   = 0;
        int type_i  = 0;
        // Iterate until we get all 0s. This allows SOME shortcutting.
        while (iterator) {
            if (computedTypes & getDerType(which)) {
                cpu_weights_f[which] = new float[gpu_stencil_size];
                for (unsigned int i = 0; i < nb_stencils; i++) {
                    unsigned int stencil_size = grid_ref.getStencilSize(i);
                    unsigned int j = 0;
                    for (j = 0; j < stencil_size; j++) {
                        unsigned int indx = i*stencil_padded_size + j;
                        cpu_weights_f[which][indx] = (float)weights[which][i][j];
                        //  std::cout << cpu_weights[which][indx] << "   ";
                    }
                    // Pad end of the stencil with 0's so our linear combination
                    // excludes whatever function values are found at the end of
                    // the stencil (i.e., we can include extra terms in summation
                    // without added effect
                    for (; j < stencil_padded_size; j++) {
                        unsigned int indx = i*stencil_padded_size + j;
                        cpu_weights_f[which][indx] = (float)0.f;
                        //  std::cout << cpu_weights[which][indx] << "   ";
                    }
                    //std::cout << std::endl;
                }
                //std::cout << std::endl;
                // Send to GPU
                err = queue.enqueueWriteBuffer(gpu_weights[which], CL_TRUE, 0, weights_mem_size, &(cpu_weights_f[which][0]), NULL, &event);
                //            queue.flush();
                type_i +=1;
            }
            iterator >>= 1;
            which+= 1;
        }

        if (forceFinish) {
            queue.finish();

            this->clearCPUWeights();
			std::cout << "GE: after clearCPUWeights() 2 \n";
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



//----------------------------------------------------------------------
//
void RBFFD_CL::updateFunctionDouble(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish) {

    //    cout << "Sending " << nb_nodes << " solution updates to GPU: (bytes)" << function_mem_bytes << endl;


    if (function_mem_bytes != nb_vals*sizeof(double)) {
        std::cout << "function_mem_bytes != nb_nodes*sizeof(double)" << std::endl;
		std::cout << "nb_vals= " << nb_vals << "\n";
		std::cout << "function_mem_bytes= " << function_mem_bytes << "\n";
		std::cout << "nb_vals= " << nb_vals << "\n";
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
void RBFFD_CL::updateFunctionSingle(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish) {

    //    cout << "Sending " << nb_nodes << " solution updates to GPU: (bytes)" << function_mem_bytes << endl;

    // TODO: mask off fields not update
    // update the GPU's view of our solution
    float* cpu_u = new float[nb_vals];
    for (unsigned int i = 0; i < nb_vals; i++) {
        cpu_u[i] = (float)u[start_indx + i];
        //  std::cout << cpu_u[i] << "  ";
    }

    // There is a bug fi this works
    if (function_mem_bytes != nb_vals*sizeof(float)) {
        std::cout << "function_mem_bytes != nb_vals*sizeof(float)" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = queue.enqueueWriteBuffer(gpu_function, CL_TRUE, start_indx*sizeof(float), function_mem_bytes, &cpu_u[0], NULL, &event);
    //err = queue.enqueueWriteBuffer(gpu_function, CL_TRUE, start_indx*sizeof(double), function_mem_bytes, &cpu_u[0], NULL, &event);

    //    if (forceFinish) {
    queue.finish();
    //  }
    delete [] cpu_u;
}


//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//
//
void RBFFD_CL::applyWeightsForDerivDouble(DerType which, unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU)
{
	std::cout << "**** RBFFD_CL::applyWeightsForDerivDouble\n";
    //TODO: FIX case when start_indx != 0
    //std::cout << "EVAN HERE\n";

    //cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;
    if (isChangedU) {
        //this->updateFunctionOnGPU(start_indx, nb_stencils, u, false);
        this->updateFunctionOnGPU(start_indx, nb_stencils, u, true);  //GE
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    //this->updateWeightsOnGPU(false);

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

    err = queue.finish(); // added by GE (for more accurate timings, although in real code, one would only use finish()
	                      // GE   as far back into the code as possible. 
    tm["applyWeights"]->end();

    if (nb_stencils *sizeof(double) != deriv_mem_bytes) {
        std::cout << "npts*sizeof(double) [" << nb_stencils*sizeof(double) << "] != deriv_mem_bytes [" << deriv_mem_bytes << "]" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Pull the computed derivative back to the CPU
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
//----------------------------------------------------------------------


//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//
//
void RBFFD_CL::applyWeightsForDerivSingle(DerType which, unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU)
{
    //cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;

    if (isChangedU) {
        this->updateFunctionOnGPU(start_indx, nb_stencils, u, false);
        //this->updateFunctionOnGPU(start_indx, nb_stencils, u, true); //GE
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(false);
    //this->updateWeightsOnGPU(true); //GE

    tm["applyWeights"]->start();

    try {
        int i = 0;
        kernel.setArg(i++, gpu_stencils);
        kernel.setArg(i++, this->getGPUWeights(which));
        kernel.setArg(i++, gpu_function);                 // COPY_IN
        kernel.setArg(i++, gpu_deriv_out);           // COPY_OUT
        //FIXME: we want to pass a size_t for maximum array lengths, but OpenCL does not allow
        //size_t arguments at this time
        unsigned int nb_stencils = grid_ref.getStencilsSize();
        kernel.setArg(i++, sizeof(unsigned int), &nb_stencils);               // const
        unsigned int stencil_size = grid_ref.getMaxStencilSize();
        kernel.setArg(i++, sizeof(unsigned int), &stencil_size);            // const
        //kernel.setArg(i++, sizeof(unsigned int), &stencil_padded_size);            // const

    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

	enqueueKernel(kernel, cl::NDRange(nb_stencils), cl::NullRange, true);


    tm["applyWeights"]->end();

    float* deriv_temp = new float[nb_stencils];

    if (nb_stencils *sizeof(float) != deriv_mem_bytes) {
        std::cout << "npts*sizeof(float) [" << nb_stencils*sizeof(float) << "] != deriv_mem_bytes [" << deriv_mem_bytes << "]" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Pull the computed derivative back to the CPU
    err = queue.enqueueReadBuffer(gpu_deriv_out, CL_TRUE, 0, deriv_mem_bytes, &deriv_temp[0], NULL, &event);
    if (err != CL_SUCCESS) {
        std::cerr << " enequeue ERROR: " << err << std::endl;
    }

    err = queue.finish();

    //    std::cout << "WARNING! derivatives are only computed in single precision.\n";

    for (unsigned int i = 0; i < nb_stencils; i++) {

#if 0
        std::cout << "deriv[" << i << "] = " << deriv_temp[i] << std::endl;

        std::vector<StencilType>& stencil_map = grid_ref.getStencils();
        double sum = 0.f;
        for (int j = 0; j < grid_ref.getMaxStencilSize(); j++) {
            sum += weights[which][i][j] * u[stencil_map[i][j]];
        }
        std::cout << "sum should be: " << sum << "\tDifference: " << deriv_temp[i] - sum << std::endl;
#endif
        deriv[i] = (double)deriv_temp[i];
    }

    if (err != CL_SUCCESS) {
        std::cerr << "Event::wait() failed (" << err << ")\n";
        std::cout << "Failed to finish queue" << std::endl;
    } else {
        //        std::cout << "CL program finished!" << std::endl;
    }

    delete [] deriv_temp;

}
//----------------------------------------------------------------------
void RBFFD_CL::enqueueKernel(const cl::Kernel& kernel, const cl::NDRange& tot_work_items, const cl::NDRange& items_per_workgroup, bool is_finish)
{
    err = queue.enqueueNDRangeKernel(kernel, /* offset */ cl::NullRange,
            tot_work_items, items_per_workgroup, NULL, &event);

    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
        exit(EXIT_FAILURE);
    }

	if (is_finish) {
    	err = queue.finish();
        //queue.flush();
	}
}
//----------------------------------------------------------------------







// TODO:
//      1) Move writeBuffers to separate routine for stencils and weights.
//      2) Send solution to GPU in separate routine, then have routine for UPDATES only (we dont want to copy entire solution each step).
//      3) Timers
//      4) Offload more work (for example the timestep update in PDE is a vector-plus-scalar-vector operation.
//      5) computeDeriv should operate on memory owned by the PDE. The GPU pointer to the solution should be passed in.
//      6) The PDE class should manage the vec+scal*vec operation to update the solution on GPU.
//          (inherit original PDE class with CL specific one; both can use RBFFD_CL, but if its a CL
//          class it should be able to pass in GPU mem pointer via advanced API).

