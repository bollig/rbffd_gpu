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
    this->updateStencils(); 
}


//----------------------------------------------------------------------
//
void RBFFD_CL::setupTimers() {
    tm["loadAttach"] = new Timer("[RBFFD_CL] Load and Attach Kernel"); 
    tm["construct"] = new Timer("[RBFFD_CL] RBFFD_CL (constructor)"); 
    tm["computeDerivs"] = new Timer("[RBFFD_CL] computeRBFFD_s (compute derivatives using OpenCL"); 
    tm["sendWeights"] = new Timer("[RBFFD_CL]   (send stencil weights to GPU)"); 
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
    // TODO: queue a WriteBuffer for each stencil and pass the &vector[0] ptr. CPU should handle that
    //      better than a loop, and we dont have to allocate memory then
    size_t* cpu_stencils = new size_t[gpu_stencil_size];  
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
        //std::cout << std::endl;
    }

    size_t stencil_mem_bytes = gpu_stencil_size * sizeof(size_t); 
    size_t solution_mem_bytes = nb_nodes * sizeof(FLOAT); 
    size_t weights_mem_bytes = gpu_stencil_size * sizeof(FLOAT); 
    size_t deriv_mem_bytes = nb_stencils * sizeof(FLOAT); 


    std::cout << "Allocating GPU memory\n"; 

    size_t bytesAllocated = 0;

    // Two input arrays: 
    // 	This one is allocated once on GPU and reused until our nodes move or we change the stencil size
    gpu_stencils = cl::Buffer(context, CL_MEM_READ_WRITE, stencil_mem_bytes, NULL, &err);
    bytesAllocated += stencil_mem_bytes; 

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
void RBFFD_CL::updateStencils() {
    std::cout << "Writing to GPU memory\n"; 
    // Copy our knowns into GPU memory: stencil indices, stencil weights (done in computeRBFFD_s)
    err = queue.enqueueWriteBuffer(gpu_stencils, CL_TRUE, 0, stencil_mem_size, &cpu_stencils[0], NULL, &event);
    queue.finish(); 
    //delete(cpu_stencils);
    std::cout << "DONE ALLOCATING MEMORY" << std::endl;
    tm["construct"]->end(); 
}



//----------------------------------------------------------------------
// Update the stencil weights on the GPU: 
// 1) Get the correct weights for the DerType
// 2) send weights to GPU
// 3) send new u to GPU
// 4) call kernel to inner prod weights and u writing to deriv
// 5) get deriv from GPU
void RBFFD_CL::updateWeights() {
    static int COMPUTED_WEIGHTS_ON_GPU=0; 

    if (!COMPUTED_WEIGHTS_ON_GPU) {
        tm["sendWeights"]->start();
        int weights_mem_size = gpu_stencil_size * sizeof(FLOAT);  
        std::cout << "Writing x_weights to GPU memory\n"; 

        FLOAT* cpu_x_weights = new FLOAT[gpu_stencil_size]; 
        FLOAT* cpu_y_weights = new FLOAT[gpu_stencil_size]; 
        FLOAT* cpu_z_weights = new FLOAT[gpu_stencil_size]; 
        FLOAT* cpu_lapl_weights = new FLOAT[gpu_stencil_size]; 

        int stencil_size = stencil[0].size();
        if (stencil.size() * stencil_size != gpu_stencil_size) {
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < stencil.size(); i++) {
            for (int j = 0; j < stencil_size; j++) {
                int indx = i*stencil_size +j; 
                cpu_x_weights[indx] = (FLOAT)x_weights[i][j]; 
                cpu_y_weights[indx] = (FLOAT)y_weights[i][j]; 
                cpu_z_weights[indx] = (FLOAT)z_weights[i][j]; 
                cpu_lapl_weights[indx] = (FLOAT)lapl_weights[i][j]; 
            }
        }

        err = queue.enqueueWriteBuffer(gpu_x_deriv_weights, CL_TRUE, 0, weights_mem_size, cpu_x_weights, NULL, &event); 
        err = queue.enqueueWriteBuffer(gpu_y_deriv_weights, CL_TRUE, 0, weights_mem_size, cpu_y_weights, NULL, &event); 
        err = queue.enqueueWriteBuffer(gpu_z_deriv_weights, CL_TRUE, 0, weights_mem_size, cpu_z_weights, NULL, &event); 
        err = queue.enqueueWriteBuffer(gpu_laplacian_weights, CL_TRUE, 0, weights_mem_size, cpu_lapl_weights, NULL, &event); 
        queue.finish(); 

        COMPUTED_WEIGHTS_ON_GPU = 1; 
        tm["sendWeights"]->end();
    }
}

void RBFFD_CL::updateFunction() {
    cout << "Sending " << rbf_centers.size() << " solution updates to GPU" << endl;
    // update the GPU's view of our solution 
    FLOAT* cpu_u = new FLOAT[rbf_centers.size()]; 
    for (int i = 0; i < rbf_centers.size(); i++) {
        cpu_u[i] = u[i]; 
    }
    err = queue.enqueueWriteBuffer(gpu_solution, CL_TRUE, 0, rbf_centers.size()*sizeof(FLOAT), cpu_u, NULL, &event);
    queue.finish(); 
}
//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//	
//
void RBFFD_CL::applyWeightsForDeriv(DerType which, int npts, double* u, double* deriv)
{
    tm["applyWeights"]->start(); 

    cout << "COMPUTING DERIVATIVE (ON GPU): ";

    try {
        kernel.setArg(0, gpu_stencils); 

        switch(which) {
            case X:
                kernel.setArg(1, gpu_x_deriv_weights); 
                cout << "X" << endl;
                break;

            case Y:
                kernel.setArg(1, gpu_y_deriv_weights); 
                cout << "Y" << endl;
                break;

            case Z:
                kernel.setArg(1, gpu_z_deriv_weights); 
                cout << "Z" << endl;
                break;

            case LAPL:
                kernel.setArg(1, gpu_laplacian_weights); 
                cout << "LAPL" << endl;
                break;

            default:
                cout << "???" << endl;
                printf("Wrong derivative choice\n");
                printf("Should not happen\n");
                exit(EXIT_FAILURE);
        }

        kernel.setArg(2, gpu_solution);                 // COPY_IN
        kernel.setArg(3, gpu_derivative_out);           // COPY_OUT 
        int nb_stencils_ = stencil.size(); 
        kernel.setArg(4, sizeof(int), &nb_stencils_);               // const 
        int stencil_size_ = stencil[0].size(); 
        kernel.setArg(5, sizeof(int), &stencil_size_);            // const
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }


    std::cout<<"Running CL program for " << npts << " stencils\n";
    err = queue.enqueueNDRangeKernel(kernel, /* offset */ cl::NullRange, 
            /* GLOBAL (work-groups in the grid)  */   cl::NDRange(npts), 
            /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);

    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = queue.finish();
    FLOAT* deriv_temp = new FLOAT[npts]; 
    // Pull the computed derivative back to the CPU
    err = queue.enqueueReadBuffer(gpu_derivative_out, CL_TRUE, 0, npts*sizeof(FLOAT), deriv_temp, NULL, &event);

    err = queue.finish();

    if (sizeof(FLOAT) == sizeof(float)) {
        std::cout << "WARNING! derivatives are only computed in single precision.\n"; 
    }

    for (int i = 0; i < npts; i++) {

#if 0
        std::cout << "deriv[" << i << "] = " << deriv_temp[i] << std::endl;

        FLOAT sum = 0.f;
        for (int j = 0; j < stencil[0].size(); j++) {
            sum += (FLOAT)x_weights[i][j]; 
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

