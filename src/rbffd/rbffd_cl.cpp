#include <stdlib.h>
#include <math.h>
#include "rbffd_cl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT

using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
    RBFFD_CL::RBFFD_CL(Grid* grid, int dim_num, int rank)
: RBFFD(grid, dim_num, rank), CLBaseClass(rank), 
    useDouble(true)
{
    this->setupTimers(); 
    this->loadKernel(); 
    this->allocateGPUMem();
    this->updateStencils(false); 
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

    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false; 
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        kernel_name = "computeDerivKernelFLOAT"; 

        useDouble = false;
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

    //   std::cout << "This is my kernel source: ...\n" << kernel_source << "\n...END\n"; 

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

    unsigned int float_size; 
    if (useDouble) {
        float_size = sizeof(double); 
    } else {
        float_size = sizeof(float);
    }
    std::cout << "FLOAT_SIZE=" << float_size << std::endl;;

    stencil_mem_bytes = gpu_stencil_size * sizeof(int); 
    function_mem_bytes = nb_nodes * float_size; 
    weights_mem_bytes = gpu_stencil_size * float_size; 
    deriv_mem_bytes = nb_stencils * float_size; 

    std::cout << "Allocating GPU memory\n"; 

    size_t bytesAllocated = 0;

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
    for (size_t i = 0; i < nb_stencils; i++) {
        size_t j; 
        for (j = 0; j < stencil_map[i].size(); j++) {
            size_t indx = i*max_stencil_size+j; 
            cpu_stencils[indx] = stencil_map[i][j];
            //std::cout << cpu_stencils[indx] << "   ";
        }
        // Buffer remainder of stencils with the stencil center (so we can
        // break on GPU when center ID is duplicated
        for (; j < max_stencil_size; j++) {
            size_t indx = i*max_stencil_size+j; 
            cpu_stencils[indx] = stencil_map[i][0];
            //std::cout << cpu_stencils[indx] << "   ";
        }
        //std::cout << endl;
    }

    //    std::cout << "Writing GPU Stencils buffer: (bytes)" << stencil_mem_bytes << std::endl;
    err = queue.enqueueWriteBuffer(gpu_stencils, CL_TRUE, 0, stencil_mem_bytes, &cpu_stencils[0], NULL, &event);
    if (forceFinish) {
        queue.finish();
    }
    //delete(cpu_stencils);
}



void RBFFD_CL::clearCPUWeights() {
    // Clear out buffer. No need to keep it since this should only happen once
    // NOTE: make sure we delete only the single or double precision cpu side buffers;
    if (useDouble) {
        for (size_t which = 0; which < NUM_DERIV_TYPES; which++) {
            delete [] cpu_weights_d[which];
        }
    } else {

        for (size_t which = 0; which < NUM_DERIV_TYPES; which++) {
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
        int weights_mem_size = gpu_stencil_size * sizeof(double);  
        //        std::cout << "Writing weights to GPU memory\n"; 

        size_t max_stencil_size = grid_ref.getMaxStencilSize();
        size_t nb_stencils = grid_ref.getStencilsSize();

        if ((nb_stencils * max_stencil_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*max_stencil_size != gpu_stencil_size" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Copy the std::vector<std::vector<..> > into a contiguous memory space
        // FIXME: inside grid_interface we could allocate contig mem and avoid this cost 
        for (size_t which = 0; which < NUM_DERIV_TYPES; which++) {
            cpu_weights_d[which] = new double[gpu_stencil_size]; 
            for (size_t i = 0; i < nb_stencils; i++) {
                size_t stencil_size = grid_ref.getStencilSize(i); 
                size_t j = 0; 
                for (j = 0; j < stencil_size; j++) {
                    size_t indx = i*stencil_size + j; 
                    cpu_weights_d[which][indx] = (double)weights[which][i][j]; 
                    //  std::cout << cpu_weights[which][indx] << "   ";
                }
                // Pad end of the stencil with 0's so our linear combination
                // excludes whatever function values are found at the end of
                // the stencil (i.e., we can include extra terms in summation
                // without added effect
                for (; j < max_stencil_size; j++) {
                    size_t indx = i*stencil_size + j; 
                    cpu_weights_d[which][indx] = (double)0.;
                    //  std::cout << cpu_weights[which][indx] << "   ";
                }
                //std::cout << std::endl;
            }
            //std::cout << std::endl;
            // Send to GPU
            err = queue.enqueueWriteBuffer(gpu_weights[which], CL_TRUE, 0, weights_mem_size, &(cpu_weights_d[which][0]), NULL, &event); 
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
        int weights_mem_size = gpu_stencil_size * sizeof(float);  
        //        std::cout << "Writing weights to GPU memory\n"; 

        size_t max_stencil_size = grid_ref.getMaxStencilSize();
        size_t nb_stencils = grid_ref.getStencilsSize();

        if ((nb_stencils * max_stencil_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*max_stencil_size != gpu_stencil_size" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Copy the std::vector<std::vector<..> > into a contiguous memory space
        // FIXME: inside grid_interface we could allocate contig mem and avoid this cost 
        for (size_t which = 0; which < NUM_DERIV_TYPES; which++) {
            cpu_weights_f[which] = new float[gpu_stencil_size]; 
            for (size_t i = 0; i < nb_stencils; i++) {
                size_t stencil_size = grid_ref.getStencilSize(i); 
                size_t j = 0; 
                for (j = 0; j < stencil_size; j++) {
                    size_t indx = i*stencil_size + j; 
                    cpu_weights_f[which][indx] = (float)weights[which][i][j]; 
                    //  std::cout << cpu_weights[which][indx] << "   ";
                }
                // Pad end of the stencil with 0's so our linear combination
                // excludes whatever function values are found at the end of
                // the stencil (i.e., we can include extra terms in summation
                // without added effect
                for (; j < max_stencil_size; j++) {
                    size_t indx = i*stencil_size + j; 
                    cpu_weights_f[which][indx] = (float)0.f;
                    //  std::cout << cpu_weights[which][indx] << "   ";
                }
                //std::cout << std::endl;
            }
            //std::cout << std::endl;
            // Send to GPU
            err = queue.enqueueWriteBuffer(gpu_weights[which], CL_TRUE, 0, weights_mem_size, &(cpu_weights_f[which][0]), NULL, &event); 
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
void RBFFD_CL::updateFunctionDouble(size_t nb_nodes, double* u, bool forceFinish) {

    //    cout << "Sending " << nb_nodes << " solution updates to GPU: (bytes)" << function_mem_bytes << endl;


    // There is a bug fi this works
    if (function_mem_bytes != nb_nodes*sizeof(double)) {
        std::cout << "function_mem_bytes != nb_nodes*sizeof(double)" << std::endl;
        exit(EXIT_FAILURE);
    }

    // TODO: mask off fields not update
    err = queue.enqueueWriteBuffer(gpu_function, CL_TRUE, 0, function_mem_bytes, &u[0], NULL, &event);

    if (forceFinish) {
        queue.finish(); 
    }
}

//----------------------------------------------------------------------
//
void RBFFD_CL::updateFunctionSingle(size_t nb_nodes, double* u, bool forceFinish) {

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

    if (forceFinish) {
        queue.finish(); 
    }
}


//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//	
//
void RBFFD_CL::applyWeightsForDerivDouble(DerType which, size_t nb_nodes, size_t nb_stencils, double* u, double* deriv, bool isChangedU)
{
    cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;
    tm["applyWeights"]->start(); 

    if (isChangedU) {
        this->updateFunction(nb_nodes, u, false); 
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeights(false);

    try {
        kernel.setArg(0, gpu_stencils); 
        kernel.setArg(1, gpu_weights[which]); 
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
            /* GLOBAL (work-groups in the grid)  */   cl::NDRange(nb_stencils), 
            /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);

    err = queue.finish();
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
void RBFFD_CL::applyWeightsForDerivSingle(DerType which, size_t nb_nodes, size_t nb_stencils, double* u, double* deriv, bool isChangedU)
{
    cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;
    tm["applyWeights"]->start(); 

    if (isChangedU) {
        this->updateFunction(nb_nodes, u, false); 
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeights(false);

    try {
        kernel.setArg(0, gpu_stencils); 
        kernel.setArg(1, gpu_weights[which]); 
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

    std::cout << "WARNING! derivatives are only computed in single precision.\n"; 

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

