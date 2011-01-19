#include <stdlib.h>
#include <math.h>
#include "derivative_cl.h"

using namespace std;

//----------------------------------------------------------------------
    DerivativeCL::DerivativeCL(vector<NodeType>& rbf_centers_, vector<StencilType>& stencil_, int nb_bnd_, int dimensions, int rank)
: Derivative(rbf_centers_, stencil_, nb_bnd_, dimensions), CLBaseClass(rank)	
{
    cout << "Inside DerivativeCL constructor" << endl;

    #include "cl_kernels/derivative_kernels.cl"
    this->loadProgram(kernel_source);

//    std::cout << "This is my kernel source: ...\n" << kernel_source << "\n...END\n"; 

    try{
        kernel = cl::Kernel(program, "computeDerivKernel", &err);

        std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
    }

    cout << "Allocating GPU memory for stencils, solution, weights and derivative" << endl;

    total_num_stencil_elements = 0; 

    for (int i = 0; i < stencil_.size(); i++) {
        total_num_stencil_elements += stencil_[i].size(); 
    }
    // TODO: queue a WriteBuffer for each stencil and pass the &vector[0] ptr. CPU should handle that
    //      better than a loop, and we dont have to allocate memory then
    int* cpu_stencils = new int[total_num_stencil_elements];  
    for (int i = 0; i < stencil_.size(); i++) {
        for (int j = 0; j < stencil_[i].size(); j++) {
            cpu_stencils[i*stencil_.size()+j] = stencil_[i][j];
            cout << "[" << stencil_[i][j] << "] "; 
        }
        std::cout << std::endl;
    }

    int stencil_mem_size = total_num_stencil_elements * sizeof(int); 
    int solution_mem_size = rbf_centers_.size() * sizeof(double); 
    int weights_mem_size = total_num_stencil_elements * sizeof(double); 
    int deriv_mem_size = rbf_centers_.size() * sizeof(double); 


    std::cout << "Allocating GPU memory\n"; 

    // Two input arrays: 
    // 	This one is allocated once on GPU and reused until our nodes move or we change the stencil size
    gpu_stencils = cl::Buffer(context, CL_MEM_READ_ONLY, stencil_mem_size, NULL, &err);
    //	This one is updated each iteration with the new solution for the previous timestep
    gpu_solution = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_size, NULL, &err);

    gpu_x_deriv_weights = cl::Buffer(context, CL_MEM_READ_ONLY, weights_mem_size, NULL, &err); 

    // One output array: 
    // 	This is our derivative used to update the solution for the current timestep
    gpu_derivative_out = cl::Buffer(context, CL_MEM_WRITE_ONLY, deriv_mem_size, NULL, &err);

    std::cout << "Writing to GPU memory\n"; 
    // Copy our knowns into GPU memory: stencil indices, stencil weights (done in computeDerivatives)
    err = queue.enqueueWriteBuffer(gpu_stencils, CL_TRUE, 0, stencil_mem_size, cpu_stencils, NULL, &event);
    queue.finish(); 
    //delete(cpu_stencils);
    std::cout << "DONE ALLOCATING MEMORY" << std::endl;
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

    // 1) Get the correct weights for the DerType
    // 2) send weights to GPU
    // 3) send new u to GPU
    // 4) call kernel to inner prod weights and u writing to deriv
    // 5) get deriv from GPU
    static int COMPUTED_WEIGHTS_ON_GPU=0; 

    if (!COMPUTED_WEIGHTS_ON_GPU) {
        int weights_mem_size = total_num_stencil_elements * sizeof(double);  
        std::cout << "Writing x_weights to GPU memory\n"; 
    
        float* cpu_weights = new float[total_num_stencil_elements]; 
        for (int i = 0; i < npts; i++) {
            for (int j = 0; j < stencil[i].size(); j++) {
                cpu_weights[i*stencil[0].size() + j] = x_weights[i][j]; //(j > 0) ? 0 : i * stencil[0].size() + j; //x_weights[i][j];
            }
        }

        err = queue.enqueueWriteBuffer(gpu_x_deriv_weights, CL_TRUE, 0, weights_mem_size, cpu_weights, NULL, &event); 
        queue.finish(); 

        COMPUTED_WEIGHTS_ON_GPU = 1; 
    }

    cout << "COMPUTING DERIVATIVE (ON GPU): ";
    vector<double*>* weights_p;

    switch(which) {
        case X:
            weights_p = &x_weights;
            //printf("weights_p= %d\n", weights_p);
            //exit(0);
            cout << "X" << endl;
            break;

        case Y:
            weights_p = &y_weights;
            cout << "Y" << endl;
            break;

        case Z:
            //vector<mat>& weights = z_weights;
            weights_p = &z_weights;
            cout << "Z" << endl;
            break;

        case LAPL:
            weights_p = &lapl_weights;
            cout << "LAPL" << endl;
            break;

        default:
            cout << "???" << endl;
            printf("Wrong derivative choice\n");
            printf("Should not happen\n");
            exit(EXIT_FAILURE);
    }

    vector<double*>& weights = *weights_p;

    cout << "Sending " << rbf_centers.size() << " solution updates to GPU" << endl;
    // update the GPU's view of our solution 
    err = queue.enqueueWriteBuffer(gpu_solution, CL_TRUE, 0, npts*sizeof(float), u, NULL, &event);
    queue.finish(); 

    cout << "Setting kernel arguments\n"; 
    try {
        kernel.setArg(0, gpu_stencils); 
        kernel.setArg(1, gpu_x_deriv_weights); 
        kernel.setArg(2, gpu_solution);                 // COPY_IN
        kernel.setArg(3, gpu_derivative_out);           // COPY_OUT 
        kernel.setArg(4, stencil.size());               // const 
        kernel.setArg(5, stencil[0].size());            // const
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
    float* deriv_temp = new float[npts]; 
    // Pull the computed derivative back to the CPU
    err = queue.enqueueReadBuffer(gpu_derivative_out, CL_TRUE, 0, npts*sizeof(float), deriv_temp, NULL, &event);

    err = queue.finish();

    std::cout << "WARNING! derivatives are only computed in single precision.\n"; 
    for (int i = 0; i < npts; i++) {
        std::cout << "deriv[" << i << "] = " << deriv_temp[i] << std::endl;
        deriv[i] = deriv_temp[i]; 
    }

    if (err != CL_SUCCESS) {
        std::cerr << "Event::wait() failed (" << err << ")\n";
        std::cout << "Failed to finish queue" << std::endl;
    } else {
        std::cout << "CL program finished!" << std::endl;
    }
}
//----------------------------------------------------------------------
