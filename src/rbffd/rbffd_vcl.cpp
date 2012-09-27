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
    tm["applyWeights_cpu"] = new Timer("[RBFFD_VCL] Test derivative of CPU Vector");
    tm["applyWeights_gpu"] = new Timer("[RBFFD_VCL] Evaluate derivative on GPU");
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
    unsigned int bytesAllocated = 0;

    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int N = grid_ref.getStencilsSize();
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

    gpu_nnz = NNZ;

    function_mem_bytes = ncols * sizeof(double);
    gpu_function = new VCL_VEC_t(ncols);
    bytesAllocated += function_mem_bytes;

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
    std::cout << "CLEAR CPU WEIGHTS CALLED\n";
    // Clear out buffer. No need to keep it since this should only happen once
    // NOTE: make sure we delete only the single or double precision cpu side buffers;
    int iterator = computedTypes;
    int which = 0;
    int type_i       = 0;
    // Iterate until we get all 0s. This allows SOME shortcutting.
    while (iterator) {
        if (computedTypes & getDerType(which)) {
            delete(cpu_weights_d[which]);
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
        unsigned int weights_mem_size = gpu_nnz * sizeof(double);

        std::cout << "[RBFFD_VCL] Writing weights to GPU memory\n";

        unsigned int nb_stencils = grid_ref.getStencilsSize();
        unsigned int nb_nodes = grid_ref.getNodeListSize();
        unsigned int n = grid_ref.getMaxStencilSize();

        std::cout << "NS: " << nb_stencils << ", n: " << n << ", nnz: " << gpu_nnz << std::endl;
        if ((nb_stencils * n) != gpu_nnz) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*n != gpu_nnz" << std::endl;
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
                std::cout << "Allocating CPU_WEIGHTS_D["<< which << "]\n";
                cpu_weights_d[which] = new UBLAS_MAT_t(nb_stencils, nb_nodes, nb_stencils*n );

                // Weights should be in csr format
                for (unsigned int i = 0; i < nb_stencils; i++) {
                    StencilType& sten = grid_ref.getStencil(i);

                    // Ublas assembles csr fast with an accumulator
                    for (unsigned int j = 0; j < sten.size(); j++) {
                        (*(cpu_weights_d[which]))(i, sten[j]) = weights[which][i][j];
                    }
                }

                viennacl::copy(*(cpu_weights_d[which]), *(gpu_weights[which]));

                std::cout << "COPIED WEIGHT " << derTypeStr[which] << std::endl;

                //viennacl::io::write_matrix_market_file(*(cpu_weights_d[which]), derTypeStr[which] + "_weights.mtx");
                type_i+=1;
            }
            iterator >>= 1;
            which += 1;
        }

        if (forceFinish) {
            std::cout << "CLEARING OUT THE WEIGHTS\n";
            this->clearCPUWeights();
            deleteCPUWeightsBuffer = false;
        } else {
            deleteCPUWeightsBuffer = true;
        }

        tm["sendWeights"]->end();

        weightsModified = false;

    } else {
                std::cout << "No need to update gpu_weights" << std::endl;
    }
    std::cout << "DONE\n";
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
    //     cout << "GPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;
    tm["applyWeights_cpu"]->start();

    if (isChangedU) {
        this->updateFunctionOnGPU(nb_nodes, u, false);
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(false);

    int which_indx = getDerTypeIndx(which);

    // Apply DM as product
    *gpu_deriv_out = viennacl::linalg::prod(*(gpu_weights[which_indx]), *gpu_function);

    viennacl::copy(gpu_deriv_out->begin(),gpu_deriv_out->end(), deriv);

    tm["applyWeights_cpu"]->end();
}
//----------------------------------------------------------------------

void RBFFD_VCL::applyWeightsForDeriv(DerType which, VCL_VEC_t& u, VCL_VEC_t& deriv, bool isChangedU) {
    std::cout << "[RBFFD_VCL] Warning! Using GPU to apply weights, but NOT advance timestep\n";

    tm["applyWeights_gpu"]->start();
    unsigned int nb_stencils = grid_ref.getStencilsSize();
    if (deriv.size() != nb_stencils) {
        std::cout << "ERROR deriv size != nb_stencils\n";
        exit(-1);
    }
    //deriv.resize(nb_stencils);

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(false);

    int which_indx = getDerTypeIndx(which);

    // Apply DM as product
    deriv = viennacl::linalg::prod(*(gpu_weights[which_indx]), u);

    tm["applyWeights_gpu"]->end();
}


