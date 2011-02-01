#include <stdlib.h>
#include <math.h>
#include "derivative_cl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT

using namespace EB;
using namespace std;

void DerivativeCL::setupTimers() {
    tm["loadAttach"] = new Timer("[DerivativeCL] Load and Attach Kernel"); 
	tm["construct"] = new Timer("[DerivativeCL] DerivativeCL (constructor)"); 
	tm["computeDerivs"] = new Timer("[DerivativeCL] computeDerivatives (compute derivatives using OpenCL"); 
	tm["sendWeights"] = new Timer("[DerivativeCL]   (send stencil weights to GPU)"); 
}


//----------------------------------------------------------------------
    DerivativeCL::DerivativeCL(vector<NodeType>& rbf_centers_, vector<StencilType>& stencil_, int nb_bnd_, int dimensions, int rank)
: Derivative(rbf_centers_, stencil_, nb_bnd_, dimensions), CLBaseClass(rank)	
{
	this->setupTimers(); 
    tm["construct"]->start(); 

    cout << "Inside DerivativeCL constructor" << endl;

    tm["loadAttach"]->start(); 

    #include "cl_kernels/derivative_kernels.cl"
//    std::cout << "USING PROGRAM SOURCE: \n====\n" << kernel_source << "\n=====\n"; 
    this->loadProgram(kernel_source);

    //    std::cout << "This is my kernel source: ...\n" << kernel_source << "\n...END\n"; 

    try{
        if (sizeof(FLOAT) == sizeof(float)) {
            std::cout << "Loading single precision kernel\n"; 
            kernel = cl::Kernel(program, "computeDerivKernelFLOAT", &err);
        } else {
            std::cout << "Loading double precision kernel\n"; 
            kernel = cl::Kernel(program, "computeDerivKernelDOUBLE", &err);
        }
        std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
    }
    tm["loadAttach"]->end(); 

    cout << "Allocating GPU memory for stencils, solution, weights and derivative" << endl;

    total_num_stencil_elements = 0; 

    for (int i = 0; i < stencil_.size(); i++) {
        total_num_stencil_elements += stencil_[i].size(); 
    }
    // TODO: queue a WriteBuffer for each stencil and pass the &vector[0] ptr. CPU should handle that
    //      better than a loop, and we dont have to allocate memory then
    int* cpu_stencils = new int[total_num_stencil_elements];  
    int stencil_size = stencil_[0].size();
    for (int i = 0; i < stencil_.size(); i++) {
        for (int j = 0; j < stencil_size; j++) {
            int indx = i*stencil_size+j; 
            cpu_stencils[indx] = stencil_[i][j];
            // cout << "[" << stencil_[i][j] << "] "; 
        }
        //std::cout << std::endl;
    }

    int stencil_mem_size = total_num_stencil_elements * sizeof(int); 
    int solution_mem_size = rbf_centers_.size() * sizeof(FLOAT); 
    int weights_mem_size = total_num_stencil_elements * sizeof(FLOAT); 
    int deriv_mem_size = stencil_.size() * sizeof(FLOAT); 


    std::cout << "Allocating GPU memory\n"; 

    // Two input arrays: 
    // 	This one is allocated once on GPU and reused until our nodes move or we change the stencil size
    gpu_stencils = cl::Buffer(context, CL_MEM_READ_WRITE, stencil_mem_size, NULL, &err);
    //	This one is updated each iteration with the new solution for the previous timestep
    gpu_solution = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_size, NULL, &err);

    gpu_x_deriv_weights = cl::Buffer(context, CL_MEM_READ_ONLY, weights_mem_size, NULL, &err); 
    gpu_y_deriv_weights = cl::Buffer(context, CL_MEM_READ_ONLY, weights_mem_size, NULL, &err); 
    gpu_z_deriv_weights = cl::Buffer(context, CL_MEM_READ_ONLY, weights_mem_size, NULL, &err); 
    gpu_laplacian_weights = cl::Buffer(context, CL_MEM_READ_ONLY, weights_mem_size, NULL, &err); 

    // One output array: 
    // 	This is our derivative used to update the solution for the current timestep
    gpu_derivative_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_size, NULL, &err);

    std::cout << "Writing to GPU memory\n"; 
    // Copy our knowns into GPU memory: stencil indices, stencil weights (done in computeDerivatives)
    err = queue.enqueueWriteBuffer(gpu_stencils, CL_TRUE, 0, stencil_mem_size, &cpu_stencils[0], NULL, &event);
    queue.finish(); 
    //delete(cpu_stencils);
    std::cout << "DONE ALLOCATING MEMORY" << std::endl;
    tm["construct"]->end(); 
}

DerivativeCL::~DerivativeCL() {
    // No need to free buffers because theyre not pointers. They get freed automatically
}

//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//	
//
void DerivativeCL::computeDerivatives(DerType which, double* u, double* deriv, int npts)
{
    tm["computeDerivs"]->start(); 
    // 1) Get the correct weights for the DerType
    // 2) send weights to GPU
    // 3) send new u to GPU
    // 4) call kernel to inner prod weights and u writing to deriv
    // 5) get deriv from GPU
    static int COMPUTED_WEIGHTS_ON_GPU=0; 

    if (!COMPUTED_WEIGHTS_ON_GPU) {
	    tm["sendWeights"]->start();
        int weights_mem_size = total_num_stencil_elements * sizeof(FLOAT);  
        std::cout << "Writing x_weights to GPU memory\n"; 

        FLOAT* cpu_x_weights = new FLOAT[total_num_stencil_elements]; 
        FLOAT* cpu_y_weights = new FLOAT[total_num_stencil_elements]; 
        FLOAT* cpu_z_weights = new FLOAT[total_num_stencil_elements]; 
        FLOAT* cpu_lapl_weights = new FLOAT[total_num_stencil_elements]; 

        int stencil_size = stencil[0].size();
        if (stencil.size() * stencil_size != total_num_stencil_elements) {
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


    cout << "Sending " << rbf_centers.size() << " solution updates to GPU" << endl;
    // update the GPU's view of our solution 
    FLOAT* cpu_u = new FLOAT[rbf_centers.size()]; 
    for (int i = 0; i < rbf_centers.size(); i++) {
        cpu_u[i] = u[i]; 
    }
    err = queue.enqueueWriteBuffer(gpu_solution, CL_TRUE, 0, rbf_centers.size()*sizeof(FLOAT), cpu_u, NULL, &event);
    queue.finish(); 


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
    tm["computeDerivs"]->end();
}
//----------------------------------------------------------------------


// TODO: 
//      1) Move writeBuffers to separate routine for stencils and weights.
//      2) Send solution to GPU in separate routine, then have routine for UPDATES only (we dont want to copy entire solution each step). 
//      3) Timers
//      4) Offload more work (for example the timestep update in PDE is a vector-plus-scalar-vector operation.
//      5) computeDeriv should operate on memory owned by the PDE. The GPU pointer to the solution should be passed in.
//      6) The PDE class should manage the vec+scal*vec operation to update the solution on GPU. 
//          (inherit original PDE class with CL specific one; both can use DerivativeCL, but if its a CL 
//          class it should be able to pass in GPU mem pointer via advanced API).  

