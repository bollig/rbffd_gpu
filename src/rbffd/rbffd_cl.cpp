#include <stdlib.h>
#include <math.h>
#include "rbffd_cl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT

using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
    RBFFD_CL::RBFFD_CL(Domain* grid, int dim_num, int rank)
: RBFFD(grid, dim_num, rank), CLBaseClass(rank)	
{
    this->setupTimers(); 
    this->loadKernel(); 
    this->allocateGPUMem();
    this->updateStencils(true); 
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

    // Defines std::string kernel_source: 
#include "cl_kernels/derivative_kernels.cl"

    // Split the kernels by __kernel keyword and do not discards the keyword.
    // TODO: support extensions declared for kernels (this class can add FP64
    // support)
    std::vector<std::string> separated_kernels = this->split(kernel_source, "__kernel", true);


    std::string kernel_name = "computeDerivKernelDOUBLE"; 
    bool useDouble = true; 
    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false; 
    }
    if ((sizeof(FLOAT) == sizeof(float)) 
            || !useDouble) {
        kernel_name = "computeDerivKernelFLOAT"; 

        // Since double is disabled we need to try and find ONLY the specified
        // kernel so we dont try to build kernels that would error out because
        // they contain the keyword double
        std::vector<std::string>::iterator k; 
        for (k = separated_kernels.begin(); k!=separated_kernels.end(); k++) {
            if ((*k).find(kernel_name) != std::string::npos) {
                this->loadProgram(*k, useDouble);
            }
        }

    } else {
        this->loadProgram(kernel_source, useDouble); 
    }

    //    std::cout << "This is my kernel source: ...\n" << kernel_source << "\n...END\n"; 

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
    size_t nb_nodes = grid_ref.getNodeListSize();
    size_t nb_stencils = stencil_map.size();

    cout << "Allocating GPU memory for stencils, solution, weights and derivative" << endl;

    gpu_stencil_size = 0; 

    for (size_t i = 0; i < stencil_map.size(); i++) {
        gpu_stencil_size += stencil_map[i].size(); 
    }

    stencil_mem_bytes = gpu_stencil_size * sizeof(int); 
    function_mem_bytes = nb_nodes * sizeof(FLOAT); 
    weights_mem_bytes = gpu_stencil_size * sizeof(FLOAT); 
    deriv_mem_bytes = nb_stencils * sizeof(FLOAT); 

    std::cout << "Allocating GPU memory\n"; 

    size_t bytesAllocated = 0;

    // Two input arrays: 
    // 	This one is allocated once on GPU and reused until our nodes move or we change the stencil size
    gpu_stencils = cl::Buffer(context, CL_MEM_READ_WRITE, stencil_mem_bytes, NULL, &err);
    bytesAllocated += stencil_mem_bytes; 
    
    gpu_function = cl::Buffer(context, CL_MEM_READ_WRITE, function_mem_bytes, NULL, &err);

    for (int i = 0; i < NUM_DERIV_TYPES; i++) {
        gpu_weights[i] = cl::Buffer(context, CL_MEM_READ_ONLY, weights_mem_bytes, NULL, &err); 
        bytesAllocated += weights_mem_bytes; 
        gpu_deriv_out[i] = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
        bytesAllocated += deriv_mem_bytes; 
    }    
    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
}

//----------------------------------------------------------------------
//
void RBFFD_CL::updateStencils(bool forceFinish) {
    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
    size_t nb_stencils = stencil_map.size();
    // TODO: queue a WriteBuffer for each stencil and pass the &vector[0] ptr. CPU should handle that
    //      better than a loop, and we dont have to allocate memory then
    int* cpu_stencils = new int[gpu_stencil_size];  
    size_t max_stencil_size = grid_ref.getMaxStencilSize();
    for (int i = 0; i < nb_stencils; i++) {
        size_t j; 
        for (j = 0; j < stencil_map[i].size(); j++) {
            size_t indx = i*max_stencil_size+j; 
            cpu_stencils[indx] = stencil_map[i][j];
        }
        // Buffer remainder of stencils with the stencil center (so we can
        // break on GPU when center ID is duplicated
        for (; j < max_stencil_size; j++) {
            size_t indx = i*max_stencil_size+j; 
            cpu_stencils[indx] = stencil_map[i][0];
        }
    }

    std::cout << "Writing GPU Stencils buffer: (bytes)" << stencil_mem_bytes << std::endl;
    err = queue.enqueueWriteBuffer(gpu_stencils, CL_TRUE, 0, stencil_mem_bytes, &cpu_stencils[0], NULL, &event);
    if (forceFinish) {
        queue.finish();
    }
    //delete(cpu_stencils);
}



//----------------------------------------------------------------------
//
// Update the stencil weights on the GPU: 
// 1) Get the correct weights for the DerType
// 2) send weights to GPU
// 3) send new u to GPU
// 4) call kernel to inner prod weights and u writing to deriv
// 5) get deriv from GPU
void RBFFD_CL::updateWeights(bool forceFinish) {

    if (weightsModified) {

        tm["sendWeights"]->start();
        int weights_mem_size = gpu_stencil_size * sizeof(FLOAT);  
        std::cout << "Writing weights to GPU memory\n"; 

        FLOAT* cpu_weights[NUM_DERIV_TYPES]; 

        size_t max_stencil_size = grid_ref.getMaxStencilSize();
        size_t nb_stencils = grid_ref.getStencilsSize();

        if ((nb_stencils * max_stencil_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            exit(EXIT_FAILURE);
        }
        
        // Copy the std::vector<std::vector<..> > into a contiguous memory space
        // FIXME: inside grid_interface we could allocate contig mem and avoid this cost 
        for (size_t which = 0; which < NUM_DERIV_TYPES; which++) {
            cpu_weights[which] = new FLOAT[gpu_stencil_size]; 
            for (size_t i = 0; i < nb_stencils; i++) {

                size_t stencil_size = grid_ref.getStencilSize(i); 
                size_t j = 0; 
                for (j = 0; j < stencil_size; j++) {
                    size_t indx = i*stencil_size + j; 
                    cpu_weights[which][indx] = (FLOAT)weights[which][i][j]; 
                }
                // Pad end of the stencil with 0's so our linear combination
                // excludes whatever function values are found at the end of
                // the stencil (i.e., we can include extra terms in summation
                // without added effect
                for (; j < max_stencil_size; j++) {
                    size_t indx = i*stencil_size + j; 
                    cpu_weights[which][indx] = (FLOAT)0.;
                }

            }

            // Send to GPU
            err = queue.enqueueWriteBuffer(gpu_weights[which], CL_TRUE, 0, weights_mem_size, cpu_weights[which], NULL, &event); 
        }

        if (forceFinish) {
            queue.finish(); 
        }

        // Clear out buffer. No need to keep it since this should only happen once
        for (int which = 0; which < NUM_DERIV_TYPES; which++) {
            delete [] cpu_weights[which];
        }

        tm["sendWeights"]->end();

        weightsModified = false;

    } else {
        std::cout << "No need to update gpu_weights" << std::endl;
    }
}

//----------------------------------------------------------------------
//
void RBFFD_CL::updateFunction(size_t nb_nodes, double* u, bool forceFinish) {

    cout << "Sending " << nb_nodes << " solution updates to GPU: (bytes)" << function_mem_bytes << endl;
    
    // update the GPU's view of our solution 
    FLOAT* cpu_u = new FLOAT[nb_nodes]; 
    for (int i = 0; i < nb_nodes; i++) {
        cpu_u[i] = u[i]; 
    }

    // There is a bug fi this works
    if (function_mem_bytes != nb_nodes*sizeof(FLOAT)) {
        exit(EXIT_FAILURE);
    }

    err = queue.enqueueWriteBuffer(gpu_function, CL_TRUE, 0, function_mem_bytes, cpu_u, NULL, &event);

    if (forceFinish) {
        queue.finish(); 
    }
}

//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//	
//
void RBFFD_CL::applyWeightsForDeriv(DerType which, int npts, double* u, double* deriv, bool isChangedU)
{
    tm["applyWeights"]->start(); 

    if (isChangedU) {
        this->updateFunction(npts, u, true); 
    }

    // Will only update if necessary
    this->updateWeights(true);

    cout << "COMPUTING DERIVATIVE (ON GPU): " << which << std::endl;

    try {
        kernel.setArg(0, gpu_stencils); 
        kernel.setArg(0, gpu_weights[which]); 
        kernel.setArg(2, gpu_function);                 // COPY_IN
        kernel.setArg(3, gpu_deriv_out[which]);           // COPY_OUT 
        //FIXME: we want to pass a size_t for maximum array lengths, but OpenCL does not allow
        //size_t arguments at this time
        int nb_stencils = (int)grid_ref.getStencilsSize(); 
        kernel.setArg(4, sizeof(int), &nb_stencils);               // const 
        int stencil_size = (int)grid_ref.getMaxStencilSize(); 
        kernel.setArg(5, sizeof(int), &stencil_size);            // const
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    err = queue.enqueueNDRangeKernel(kernel, /* offset */ cl::NullRange, 
            /* GLOBAL (work-groups in the grid)  */   cl::NDRange(npts), 
            /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);

    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
        exit(EXIT_FAILURE);
    }

  //  err = queue.finish();
    FLOAT* deriv_temp = new FLOAT[npts]; 
    // Pull the computed derivative back to the CPU
    err = queue.enqueueReadBuffer(gpu_deriv_out[which], CL_TRUE, 0, deriv_mem_bytes, deriv_temp, NULL, &event);
    if (err != CL_SUCCESS) {
        std::cerr << " enequeue ERROR: " << err << std::endl; 
    }
   

    err = queue.finish();

    if (sizeof(FLOAT) == sizeof(float)) {
        std::cout << "WARNING! derivatives are only computed in single precision.\n"; 
    }

    for (int i = 0; i < npts; i++) {

#if 1
        std::cout << "deriv[" << i << "] = " << deriv_temp[i] << std::endl;

        std::vector<StencilType>& stencil_map = grid_ref.getStencils();
        FLOAT sum = 0.f;
        for (int j = 0; j < grid_ref.getMaxStencilSize(); j++) {
            sum += (FLOAT)weights[which][i][j] * u[stencil_map[i][j]]; 
        }
        std::cout << "sum should be: " << sum << "\tDifference: " << deriv_temp[i] - sum << std::endl;
#endif 
        deriv[i] = deriv_temp[i]; 
    }

    if (err != CL_SUCCESS) {
        std::cerr << "Event::wait() failed (" << err << ")\n";
        std::cout << "Failed to finish queue" << std::endl;
    } else {
        std::cout << "CL program finished!" << std::endl;
    }
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

