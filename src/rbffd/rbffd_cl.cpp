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
    useDouble(true)
{
            this->setupTimers();
            this->loadKernel();
            this->allocateGPUMem();
            this->updateStencilsOnGPU(false);
            std::cout << "Done copying stencils\n"; 
            
            this->updateNodesOnGPU(false);
            std::cout << "Done copying nodes\n"; 
}


//----------------------------------------------------------------------
//
void RBFFD_CL::setupTimers() {
    tm["loadAttach"] = new Timer("[RBFFD_CL] Load and Attach Kernel"); 
    tm["construct"] = new Timer("[RBFFD_CL] RBFFD_CL (constructor)"); 
    tm["computeDerivs"] = new Timer("[RBFFD_CL] computeRBFFD_s (compute derivatives using OpenCL"); 
    tm["sendWeights"] = new Timer("[RBFFD_CL]   (send stencil weights to GPU)"); 
    tm["applyWeights"] = new Timer("[RBFFD_CL] Evaluate single derivative by applying weights to function");
}


//----------------------------------------------------------------------
//
void RBFFD_CL::loadKernel() {
    tm["construct"]->start(); 

    cout << "Inside RBFFD_CL constructor" << endl;

    tm["loadAttach"]->start(); 

    // Split the kernels by __kernel keyword and do not discards the keyword.
    // TODO: support extensions declared for kernels (this class can add FP64
    // support)
   // std::vector<std::string> separated_kernels = this->split(kernel_source, "__kernel", true);

    std::string kernel_name = "computeDerivKernel"; 

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

    //std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n";
    std::cout << "Loading program source: derivative_kernels.cl\n";
    this->loadProgram(my_source, useDouble); 

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

void RBFFD_CL::allocateGPUMem() {

    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int nb_stencils = stencil_map.size();

    cout << "Allocating GPU memory for stencils, solution, weights and derivative" << endl;

    gpu_stencil_size = 0; 

    for (unsigned int i = 0; i < stencil_map.size(); i++) {
        gpu_stencil_size += stencil_map[i].size(); 
    }

    unsigned int float_size; 
    if (useDouble) {
        float_size = sizeof(double); 
    } else {
        float_size = sizeof(float);
    }
    std::cout << "FLOAT_SIZE=" << float_size << std::endl;;

    stencil_mem_bytes = gpu_stencil_size * sizeof(unsigned int); 
    function_mem_bytes = nb_nodes * float_size; 
    weights_mem_bytes = gpu_stencil_size * float_size; 
    deriv_mem_bytes = nb_stencils * float_size;

    nodes_mem_bytes = nb_nodes * sizeof(double4);

    std::cout << "Allocating GPU memory\n"; 

    unsigned int bytesAllocated = 0;

    // Two input arrays: 
    // 	This one is allocated once on GPU and reused until our nodes move or we change the stencil size
    gpu_stencils = cl::Buffer(context, CL_MEM_READ_WRITE, stencil_mem_bytes, NULL, &err);
    bytesAllocated += stencil_mem_bytes; 

    gpu_function = cl::Buffer(context, CL_MEM_READ_ONLY, function_mem_bytes, NULL, &err);

    for (int which = 0; which < NUM_DERIV_TYPES; which++) {
        gpu_weights[which] = cl::Buffer(context, CL_MEM_READ_ONLY, weights_mem_bytes, NULL, &err); 
        bytesAllocated += weights_mem_bytes; 
        gpu_deriv_out[which] = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
        bytesAllocated += deriv_mem_bytes; 
    }

    gpu_nodes = cl::Buffer(context, CL_MEM_READ_ONLY, nodes_mem_bytes, NULL, &err);

    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
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
                this->clearCPUWeights();
                deleteCPUNodesBuffer= false;
        } else {
                deleteCPUNodesBuffer = true;
        }
}

//----------------------------------------------------------------------
//
void RBFFD_CL::updateStencilsOnGPU(bool forceFinish) {
    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
    unsigned int nb_stencils = stencil_map.size();
    // TODO: queue a WriteBuffer for each stencil and pass the &vector[0] ptr. CPU should handle that
    //      better than a loop, and we dont have to allocate memory then
    // FIXME: Store stencils as linear unsigned int array, not int** or vecctor<int>.  
    // FIXME: If we align memory to the nearest XXX we could improve access patterns.
    //  (figure out the stencil size to target)
    //  NOTE: its not essential since we only put stencils on GPU one time.
    cpu_stencils = new unsigned int[gpu_stencil_size];  
    unsigned int max_stencil_size = grid_ref.getMaxStencilSize();
    for (unsigned int i = 0; i < nb_stencils; i++) {
        unsigned int j; 
        for (j = 0; j < stencil_map[i].size(); j++) {
            unsigned int indx = i*max_stencil_size+j; 
            cpu_stencils[indx] = stencil_map[i][j];
            //std::cout << cpu_stencils[indx] << "   ";
        }
        // Buffer remainder of stencils with the stencil center (so we can
        // break on GPU when center ID is duplicated. This also prevents random
        // values from showing up beyond our stencil index.
        // NOTE: another failsafe is to have the weights set to 0 beyond this
        // point so we dont add any values from outside indices
        for (; j < max_stencil_size; j++) {
            unsigned int indx = i*max_stencil_size+j; 
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
        this->clearCPUWeights();
        deleteCPUStencilsBuffer = false;
    } else { 
        deleteCPUStencilsBuffer = true;
    }

}

void RBFFD_CL::clearCPUNodes() {
        delete [] cpu_nodes;
}


void RBFFD_CL::clearCPUStencils() {
    // Clear out buffer. No need to keep it since this should only happen once
    // NOTE: make sure we delete only the single or double precision cpu side buffers;
    for (unsigned int which = 0; which < NUM_DERIV_TYPES; which++) {
        delete [] cpu_stencils;
    }
}



void RBFFD_CL::clearCPUWeights() {
    // Clear out buffer. No need to keep it since this should only happen once
    // NOTE: make sure we delete only the single or double precision cpu side buffers;
    // FIXME: change the 4 here to NUM_DERIV_TYPES (need to compute all deriv
    // types and save them to disk, then load them from file. 
    if (useDouble) {
        for (unsigned int which = 0; which < NUM_DERIV_TYPES; which++) {
            delete [] cpu_weights_d[which];
        }
    } else {

        for (unsigned int which = 0; which < NUM_DERIV_TYPES; which++) {
            delete [] cpu_weights_f[which];
        }
    }
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

    if (weightsModified) {

        tm["sendWeights"]->start();
        unsigned int weights_mem_size = gpu_stencil_size * sizeof(double);  
        
        std::cout << "Writing weights to GPU memory\n"; 

        unsigned int max_stencil_size = grid_ref.getMaxStencilSize();
        unsigned int nb_stencils = grid_ref.getStencilsSize();

        if ((nb_stencils * max_stencil_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*max_stencil_size != gpu_stencil_size" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Copy the std::vector<std::vector<..> > into a contiguous memory space
        // FIXME: inside grid_interface we could allocate contig mem and avoid this cost 
        // FIXME: copy more than just the 4 types of weights
        for (unsigned int which = 0; which < NUM_DERIV_TYPES; which++) {
            cpu_weights_d[which] = new double[gpu_stencil_size]; 
            for (unsigned int i = 0; i < nb_stencils; i++) {
                unsigned int stencil_size = grid_ref.getStencilSize(i); 
                unsigned int j = 0; 
                for (j = 0; j < stencil_size; j++) {
                    unsigned int indx = i*stencil_size + j; 
                    cpu_weights_d[which][indx] = (double)weights[which][i][j]; 
                    //  std::cout << cpu_weights[which][indx] << "   ";
                }
                // Pad end of the stencil with 0's so our linear combination
                // excludes whatever function values are found at the end of
                // the stencil (i.e., we can include extra terms in summation
                // without added effect
                for (; j < max_stencil_size; j++) {
                    unsigned int indx = i*stencil_size + j; 
                    cpu_weights_d[which][indx] = (double)0.;
                    //  std::cout << cpu_weights[which][indx] << "   ";
                }
                //std::cout << std::endl;
            }
            //std::cout << std::endl;
            // Send to GPU
            err = queue.enqueueWriteBuffer(gpu_weights[which], CL_TRUE, 0, weights_mem_size, &(cpu_weights_d[which][0]), NULL, &event); 
//            queue.flush();
        }

        if (forceFinish) {
            queue.finish(); 

            this->clearCPUWeights();
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

    if (weightsModified) {

        tm["sendWeights"]->start();
        unsigned int weights_mem_size = gpu_stencil_size * sizeof(float);  
        
        std::cout << "Writing weights to GPU memory\n"; 

        unsigned int max_stencil_size = grid_ref.getMaxStencilSize();
        unsigned int nb_stencils = grid_ref.getStencilsSize();

        if ((nb_stencils * max_stencil_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*max_stencil_size != gpu_stencil_size" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Copy the std::vector<std::vector<..> > into a contiguous memory space
        // FIXME: inside grid_interface we could allocate contig mem and avoid this cost 
        // FIXME: we only pass four weights to the GPU
        for (unsigned int which = 0; which < NUM_DERIV_TYPES; which++) {
            cpu_weights_f[which] = new float[gpu_stencil_size]; 
            for (unsigned int i = 0; i < nb_stencils; i++) {
                unsigned int stencil_size = grid_ref.getStencilSize(i); 
                unsigned int j = 0; 
                for (j = 0; j < stencil_size; j++) {
                    unsigned int indx = i*stencil_size + j; 
                    cpu_weights_f[which][indx] = (float)weights[which][i][j]; 
                    //  std::cout << cpu_weights[which][indx] << "   ";
                }
                // Pad end of the stencil with 0's so our linear combination
                // excludes whatever function values are found at the end of
                // the stencil (i.e., we can include extra terms in summation
                // without added effect
                for (; j < max_stencil_size; j++) {
                    unsigned int indx = i*stencil_size + j; 
                    cpu_weights_f[which][indx] = (float)0.f;
                    //  std::cout << cpu_weights[which][indx] << "   ";
                }
                //std::cout << std::endl;
            }
            //std::cout << std::endl;
            // Send to GPU
            err = queue.enqueueWriteBuffer(gpu_weights[which], CL_TRUE, 0, weights_mem_size, &(cpu_weights_f[which][0]), NULL, &event); 
//            queue.flush();
        }

        if (forceFinish) {
            queue.finish(); 

            this->clearCPUWeights();
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
void RBFFD_CL::updateFunctionDouble(unsigned int nb_nodes, double* u, bool forceFinish) {

    //    cout << "Sending " << nb_nodes << " solution updates to GPU: (bytes)" << function_mem_bytes << endl;


    // There is a bug fi this works
    if (function_mem_bytes != nb_nodes*sizeof(double)) {
        std::cout << "function_mem_bytes != nb_nodes*sizeof(double)" << std::endl;
        exit(EXIT_FAILURE);
    }

    // TODO: mask off fields not update
    err = queue.enqueueWriteBuffer(gpu_function, CL_TRUE, 0, function_mem_bytes, &u[0], NULL, &event);
//    queue.flush();

    if (forceFinish) {
        queue.finish(); 
    }
}

//----------------------------------------------------------------------
//
void RBFFD_CL::updateFunctionSingle(unsigned int nb_nodes, double* u, bool forceFinish) {

    //    cout << "Sending " << nb_nodes << " solution updates to GPU: (bytes)" << function_mem_bytes << endl;

    // TODO: mask off fields not update
    // update the GPU's view of our solution 
    float* cpu_u = new float[nb_nodes]; 
    for (int i = 0; i < nb_nodes; i++) {
        cpu_u[i] = (float)u[i]; 
        //  std::cout << cpu_u[i] << "  "; 
    }

    // There is a bug fi this works
    if (function_mem_bytes != nb_nodes*sizeof(float)) {
        std::cout << "function_mem_bytes != nb_nodes*sizeof(float)" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = queue.enqueueWriteBuffer(gpu_function, CL_TRUE, 0, function_mem_bytes, &cpu_u[0], NULL, &event);

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
void RBFFD_CL::applyWeightsForDerivDouble(DerTypeID which, unsigned int nb_nodes, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU)
{
    //cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;
    tm["applyWeights"]->start(); 

    if (isChangedU) {
        this->updateFunctionOnGPU(nb_nodes, u, false); 
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(false);

    try {
            int i = 0;
        kernel.setArg(i++, gpu_stencils);
        kernel.setArg(i++, gpu_weights[which]);
        kernel.setArg(i++, gpu_function);                 // COPY_IN
        kernel.setArg(i++, gpu_deriv_out[which]);           // COPY_OUT
        //FIXME: we want to pass a unsigned int for maximum array lengths, but OpenCL does not allow
        //unsigned int arguments at this time
        unsigned int nb_stencils = grid_ref.getStencilsSize(); 
        kernel.setArg(i++, sizeof(unsigned int), &nb_stencils);               // const
        unsigned int stencil_size = grid_ref.getMaxStencilSize(); 
        kernel.setArg(i++, sizeof(unsigned int), &stencil_size);            // const
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

    if (nb_stencils *sizeof(double) != deriv_mem_bytes) {
        std::cout << "npts*sizeof(double) [" << nb_stencils*sizeof(double) << "] != deriv_mem_bytes [" << deriv_mem_bytes << "]" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Pull the computed derivative back to the CPU
    err = queue.enqueueReadBuffer(gpu_deriv_out[which], CL_TRUE, 0, deriv_mem_bytes, &deriv[0], NULL, &event);
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
    tm["applyWeights"]->end();
}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//	
//
void RBFFD_CL::applyWeightsForDerivSingle(DerTypeID which, unsigned int nb_nodes, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU)
{
    //cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;
    tm["applyWeights"]->start(); 

    if (isChangedU) {
        this->updateFunctionOnGPU(nb_nodes, u, false); 
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(false);

    try {
        int i = 0;
        kernel.setArg(i++, gpu_stencils);
        kernel.setArg(i++, gpu_weights[which]);
        kernel.setArg(i++, gpu_function);                 // COPY_IN
        kernel.setArg(i++, gpu_deriv_out[which]);           // COPY_OUT
        //FIXME: we want to pass a size_t for maximum array lengths, but OpenCL does not allow
        //size_t arguments at this time
        unsigned int nb_stencils = grid_ref.getStencilsSize(); 
        kernel.setArg(i++, sizeof(unsigned int), &nb_stencils);               // const
        unsigned int stencil_size = grid_ref.getMaxStencilSize(); 
        kernel.setArg(i++, sizeof(unsigned int), &stencil_size);            // const

    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    err = queue.enqueueNDRangeKernel(kernel, /* offset */ cl::NullRange, 
            /* GLOBAL (work-groups in the grid)  */   cl::NDRange(nb_stencils), 
            /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);

    err = queue.finish();
    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
        exit(EXIT_FAILURE);
    }

    //  err = queue.finish();
    float* deriv_temp = new float[nb_stencils]; 

    if (nb_stencils *sizeof(float) != deriv_mem_bytes) {
        std::cout << "npts*sizeof(float) [" << nb_stencils*sizeof(float) << "] != deriv_mem_bytes [" << deriv_mem_bytes << "]" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Pull the computed derivative back to the CPU
    err = queue.enqueueReadBuffer(gpu_deriv_out[which], CL_TRUE, 0, deriv_mem_bytes, &deriv_temp[0], NULL, &event);
    if (err != CL_SUCCESS) {
        std::cerr << " enequeue ERROR: " << err << std::endl; 
    }


    err = queue.finish();

//    std::cout << "WARNING! derivatives are only computed in single precision.\n"; 

    for (int i = 0; i < nb_stencils; i++) {

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

    tm["applyWeights"]->end();
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

