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

    int total_num_stencil_elements = 0; 
    for (int i = 0; i < stencil_.size(); i++) {
        total_num_stencil_elements += stencil_[i].size(); 
    }

    int stencil_mem_size = total_num_stencil_elements * sizeof(int); 
    int solution_mem_size = rbf_centers_.size() * sizeof(double); 
    int weights_mem_size = total_num_stencil_elements * sizeof(double); 
    int deriv_mem_size = rbf_centers_.size() * sizeof(double); 


    // Two input arrays: 
    // 	This one is allocated once on GPU and reused until our nodes move or we change the stencil size
    gpu_stencils = cl::Buffer(context, CL_MEM_READ_ONLY, stencil_mem_size, NULL, &err);
    //	This one is updated each iteration with the new solution for the previous timestep
    gpu_solution = cl::Buffer(context, CL_MEM_READ_ONLY, solution_mem_size, NULL, &err);

    gpu_x_deriv_weights = cl::Buffer(context, CL_MEM_READ_ONLY, weights_mem_size, NULL, &err); 

    // One output array: 
    // 	This is our derivative used to update the solution for the current timestep
    gpu_derivative_out = cl::Buffer(context, CL_MEM_WRITE_ONLY, deriv_mem_size, NULL, &err);

    // Copy our knowns into GPU memory: stencil indices, stencil weights
    err = queue.enqueueWriteBuffer(gpu_stencils, CL_TRUE, 0, stencil_mem_size, &stencil_[0], NULL, &event);
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

#if 0

    double der;

    for (int i=0; i < stencil.size(); i++) {
        double* w = weights[i];
        vector<int>& st = stencil[i];
        der = 0.0;
        int n = st.size();
        for (int j=0; j < n; j++) {
            der += w[j] * u[st[j]]; 
        }
        deriv[i] = der;
    }
#endif 


    std::cout<<"Running CL program\n";
    err = queue.enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(4, 4), cl::NDRange(2, 2)
            );

    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = queue.finish();
    if (err != CL_SUCCESS) {
        std::cerr << "Event::wait() failed (" << err << ")\n";
        std::cout << "Failed to finish queue" << std::endl;
    } else {
        std::cout << "CL program finished!" << std::endl;
    }
}
//----------------------------------------------------------------------
