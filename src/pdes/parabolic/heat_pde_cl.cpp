#include "heat_pde_cl.h"

#include "rbffd/rbffd_cl.h"

//----------------------------------------------------------------------

void HeatPDE_CL::setupTimers()
{
   tm["advance_gpu"] = new EB::Timer("Advance the PDE one step on the GPU") ;
   tm["loadAttach"] = new EB::Timer("Load the GPU Kernels for HeatPDE");
}

//----------------------------------------------------------------------

void HeatPDE_CL::assemble() 
{
    if (!weightsPrecomputed) {
        der_ref_gpu.computeAllWeightsForAllStencils();
    }
    // This will avoid multiple writes to GPU if they latest version is already in place
    // FIXME: allow this to finish later
    der_ref_gpu.updateWeightsOnGPU(false);
}

//----------------------------------------------------------------------

void HeatPDE_CL::advance(TimeScheme which, double delta_t) {
    tm["advance_gpu"]->start(); 
    switch (which) 
    {
        case EULER: 
            advanceFirstOrderEuler(delta_t); 
            break; 
#if 0
        case MIDPOINT: 
            advanceSecondOrderMidpoint(delta_t);
            break;  
        case RK4: 
            advanceRungeKutta4(delta_t); 
            break;
#endif 
        default: 
            std::cout << "[TimeDependentPDE] Invalid TimeScheme specified. Bailing...\n";
            exit(EXIT_FAILURE); 
            break; 
    };
    cur_time += delta_t; 
    tm["advance_gpu"]->stop(); 
}

//----------------------------------------------------------------------
//
void HeatPDE_CL::advanceFirstOrderEuler(double delta_t) {

    size_t nb_stencils = grid_ref.getStencilsSize(); 
    size_t nb_nodes = grid_ref.getNodeListSize(); 
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    std::vector<double> original_solution = this->U_G; 
    std::vector<double>& s = this->U_G; 
    std::vector<SolutionType> feval1(nb_stencils);  

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

    // TODO: update GPU solution with set R updates

    // Launch kernel
    this->launchEulerKernel( ); 

    // TODO: copy set R from GPU solution to CPU solution

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time); 

    // synchronize();
    this->sendrecvUpdates(s, "U_G");
}

void HeatPDE_CL::launchEulerKernel() {
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
}


//----------------------------------------------------------------------
//
void HeatPDE_CL::loadKernels() {
    tm["loadAttach"]->start(); 

    this->loadEulerKernel(); 

    tm["loadAttach"]->stop(); 
}


void HeatPDE_CL::loadEulerKernel() {


    std::string kernel_name = "advanceFirstOrderEuler"; 

    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false;
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        useDouble = false;
    } 

    std::string my_source; 
    if(useDouble) {
#define FLOAT double 
#include "cl_kernels/euler_heat.cl"
        my_source = kernel_source;
#undef FLOAT
    }else {
#define FLOAT float
#include "cl_kernels/euler_heat.cl"
        my_source = kernel_source;
#undef FLOAT
    }

    //std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n"; 
    this->loadProgram(my_source, useDouble); 

    try{
        std::cout << "Loading kernel \""<< kernel_name << "\" with double precision = " << useDouble << "\n"; 
        kernel = cl::Kernel(program, kernel_name.c_str(), &err);
        std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
    }
}


//----------------------------------------------------------------------
//
void HeatPDE_CL::allocateGPUMem() {
#if 0
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
#endif 
}

//----------------------------------------------------------------------
//

