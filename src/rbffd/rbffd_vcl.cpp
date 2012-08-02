#include <stdlib.h>
#include <math.h>
#include "rbffd_vcl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT


using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
    RBFFD_VCL::RBFFD_VCL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank)
: RBFFD(typesToCompute, grid, dim_num, rank), 
    useDouble(true), alignWeights(true), alignMultiple(32)
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
void RBFFD_VCL::setupTimers() {
    tm["loadAttach"] = new Timer("[RBFFD_VCL] Load and Attach Kernel"); 
    tm["construct"] = new Timer("[RBFFD_VCL] RBFFD_VCL (constructor)"); 
    tm["computeDerivs"] = new Timer("[RBFFD_VCL] computeRBFFD_s (compute derivatives using OpenCL"); 
    tm["sendWeights"] = new Timer("[RBFFD_VCL]   (send stencil weights to GPU)"); 
    tm["applyWeights"] = new Timer("[RBFFD_VCL] Evaluate single derivative by applying weights to function");
}


//----------------------------------------------------------------------
//
void RBFFD_VCL::loadKernel() {
    tm["construct"]->start(); 

    cout << "Inside RBFFD_VCL constructor" << endl;

    tm["loadAttach"]->start(); 

    // Split the kernels by __kernel keyword and do not discards the keyword.
    // TODO: support extensions declared for kernels (this class can add FP64
    // support)
    // std::vector<std::string> separated_kernels = this->split(kernel_source, "__kernel", true);

#if 0
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
        printf("[AttachKernel RBFFD] ERROR: %s(%d)\n", er.what(), er.err());
    }
#endif 
    // take double as default
    useDouble =true;
    tm["loadAttach"]->end(); 
}

void RBFFD_VCL::allocateGPUMem() {

#if 0
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
        // No need to assume we're goign to have non-uniform stencil sizes. If we do, we'll pad them all to be 
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
        }
        iterator >>= 1; 
        which += 1;
    }

    gpu_deriv_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    bytesAllocated += deriv_mem_bytes; 

    gpu_nodes = cl::Buffer(context, CL_MEM_READ_ONLY, nodes_mem_bytes, NULL, &err);
    bytesAllocated += nodes_mem_bytes;
    
#endif  
    unsigned int bytesAllocated = 0;

    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int N = grid_ref.getStencilsSize(); 
    gpu_stencil_size = N;
    unsigned int n = grid_ref.getMaxStencilSize(); 
    unsigned int NNZ = n*N;
    unsigned int nrows = N;
    unsigned int ncols = nb_nodes;
 
    std::cout << "Allocating GPU memory\n"; 

    int iterator = computedTypes; 
    int which = 0;
    int type_i       = 0;     
    // Iterate until we get all 0s. This allows SOME shortcutting.
    while (iterator) {
        if (computedTypes & getDerType(which)) {
            gpu_weights[which] = new VCL_MAT_t(nrows, ncols, NNZ);
            bytesAllocated += weights_mem_bytes; 
            type_i+=1; 
        }
        else {
            // HACK: my gpu kernels take ALL weights on gpu as parameters. This allows me to put only one value for "empty" weight types
            // minimal memory consumption. It works but its a band-aid
            gpu_weights[which] = NULL;
        }
        iterator >>= 1; 
        which += 1;
    }

    weights_mem_bytes = N * n * sizeof(double); 
    deriv_mem_bytes =  1 * nrows * sizeof(double);
    nodes_mem_bytes =  3 * nrows * sizeof(double);

    gpu_deriv_out = new VCL_VEC_t(nrows);  
    bytesAllocated += deriv_mem_bytes; 

    gpu_nodes = new VCL_VEC4_t(nb_nodes, 4);

    bytesAllocated += nodes_mem_bytes;

    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
}

//----------------------------------------------------------------------
void RBFFD_VCL::updateNodesOnGPU(bool forceFinish) {
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    cpu_nodes = new UBLAS_VEC4_t(nb_nodes, 4);

    for (unsigned int i = 0; i < nb_nodes; i++) {
        cpu_nodes->insert_element(i,0, nodes[i].x());
        cpu_nodes->insert_element(i,1, nodes[i].y());
        cpu_nodes->insert_element(i,2, nodes[i].z());
        // Assume 0 on 4th element
    }

    viennacl::copy(*(cpu_nodes),*(gpu_nodes)); 

#if 0
    //        queue.flush();
    if (forceFinish) {
        queue.finish();
        this->clearCPUWeights();
        deleteCPUNodesBuffer= false;
    } else {
        deleteCPUNodesBuffer = true;
    }
#endif 
    // TODO: make sure VCL flushes queue
}

void RBFFD_VCL::clearCPUNodes() {
    delete [] cpu_nodes;
}


void RBFFD_VCL::clearCPUWeights() {
    // Clear out buffer. No need to keep it since this should only happen once
    // NOTE: make sure we delete only the single or double precision cpu side buffers;
    int iterator = computedTypes; 
    int which = 0;
    int type_i       = 0;     
    // Iterate until we get all 0s. This allows SOME shortcutting.
    while (iterator) {
        if (computedTypes & getDerType(which)) {
            delete [] cpu_weights_d[which];
            type_i+=1; 
        }
        iterator >>= 1; 
        which += 1;
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
void RBFFD_VCL::updateWeightsDouble(bool forceFinish) {

    if (weightsModified) {

        tm["sendWeights"]->start();
        unsigned int weights_mem_size = gpu_stencil_size * sizeof(double);  

        std::cout << "Writing weights to GPU memory\n"; 

        unsigned int nb_stencils = grid_ref.getStencilsSize();
        unsigned int nb_nodes = grid_ref.getNodeListSize();
        unsigned int n = grid_ref.getMaxStencilSize();

        if ((nb_stencils * stencil_padded_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*stencil_padded_size != gpu_stencil_size" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Copy the std::vector<std::vector<..> > into a contiguous memory space
        // FIXME: inside grid_interface we could allocate contig mem and avoid this cost 
        // FIXME: copy more than just the 4 types of weights
        int iterator = computedTypes; 
        int which = 0;
        int type_i = 0;     
        // Iterate until we get all 0s. This allows SOME shortcutting.
        while (iterator) {
            if (computedTypes & getDerType(which)) {
                cpu_weights_d[which] = new UBLAS_MAT_t(nb_stencils, nb_nodes, nb_stencils*n );  


                // Weights should be in csr format
                for (unsigned int i = nb_stencils; i < nb_stencils; i++) {
                    StencilType& sten = grid_ref.getStencil(i); 

                    // Ublas assembles csr fast with an accumulator
                    for (unsigned int j = 0; j < sten.size(); j++) {
                        (*(cpu_weights_d[which]))(i, sten[j]) = weights[which][i][j]; 
                    }
                }

                viennacl::copy(*(gpu_weights[which]), *(cpu_weights_d[which]));

                type_i+=1; 
            }
            iterator >>= 1; 
            which += 1;
        }

        if (forceFinish) {
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
void RBFFD_VCL::updateFunctionDouble(unsigned int nb_nodes, double* u, bool forceFinish) {

    //    cout << "Sending " << nb_nodes << " solution updates to GPU: (bytes)" << function_mem_bytes << endl;


    // There is a bug fi this works
    if (function_mem_bytes != nb_nodes*sizeof(double)) {
        std::cout << "function_mem_bytes != nb_nodes*sizeof(double)" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<double> cpu_func(u, u + nb_nodes);

    viennacl::copy(cpu_func, *gpu_function);
}

//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//	
//
void RBFFD_VCL::applyWeightsForDerivDouble(DerType which, unsigned int nb_nodes, unsigned int nb_stencils, double* u, double* deriv, bool isChangedU)
{
    //cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;
    tm["applyWeights"]->start(); 

    if (isChangedU) {
        this->updateFunctionOnGPU(nb_nodes, u, false); 
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(false);


    // Apply DM as product
//TODO    *gpu_deriv_out = viennacl::prod(*(gpu_weights[which]), *gpu_function);  
    // Pull the computed derivative back to the CPU
    //viennacl::copy(*deriv, *gpu_deriv_out);

    tm["applyWeights"]->end();
}
//----------------------------------------------------------------------


