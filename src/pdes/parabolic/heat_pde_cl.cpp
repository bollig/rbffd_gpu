#include "heat_pde_cl.h"

#include "rbffd/rbffd_cl.h"

//----------------------------------------------------------------------

void HeatPDE_CL::setupTimers()
{
   tm["advance_gpu"] = new EB::Timer("Advance the PDE one step on the GPU") ;
   tm["loadAttach"] = new EB::Timer("Load the GPU Kernels for HeatPDE");
}

void HeatPDE_CL::fillInitialConditions(ExactSolution* exact) {
    // Fill U_G with initial conditions
    this->HeatPDE::fillInitialConditions(exact);

    size_t nb_nodes = grid_ref.G.size();
    size_t solution_mem_bytes = nb_nodes*this->getFloatSize(); 
   
    if (useDouble) {
        // Fill GPU mem with initial solution 
        err = queue.enqueueWriteBuffer(gpu_solution[INDX_IN], CL_TRUE, 0, solution_mem_bytes, &U_G[0], NULL, &event);
    } else {
        float* U_G_f = new float[nb_nodes];
        for (size_t i = 0; i < nb_nodes; i++) {
            U_G_f[i] = (float)U_G[i];
        }
        err = queue.enqueueWriteBuffer(gpu_solution[INDX_IN], CL_TRUE, 0, solution_mem_bytes, &U_G_f[0], NULL, &event);
    }
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
// FIXME: this is a single precision version
void HeatPDE_CL::advanceFirstOrderEuler(double delta_t) {

    size_t nb_nodes = grid_ref.getNodeListSize();
    size_t set_G_size = grid_ref.G.size();
    size_t set_Q_size = grid_ref.Q.size() - 5; 
    size_t set_R_size = grid_ref.R.size() + 5;
    size_t set_O_size = grid_ref.R.size() + 3;
    size_t float_size = this->getFloatSize();
    size_t set_R_bytes = set_R_size * float_size;
    size_t solution_mem_bytes = set_G_size*float_size; 
    // OUR SOLUTION IS ARRANGED IN THIS FASHION: 
    //  { Q\B D O R } where B = union(O, D) and Q = union(Q\B D O)
    size_t offset_to_set_R = set_Q_size;
    size_t offset_to_set_O = (set_Q_size - set_O_size);
    size_t set_O_bytes = set_O_size * float_size;

    // backup the current solution so we can perform intermediate steps
    std::vector<float> s(nb_nodes,0.); //= this->U_G; 
    std::vector<float> test_output(nb_nodes,0.);

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

    queue.finish();

    if (set_R_size > 0) {

    // Update CPU mem with R;
    std::set<int>::iterator it; 
    for (it = grid_ref.R.begin(); it != grid_ref.R.end(); it++) {
       s[*it] = *it; 
    }

    // Synchronize just the R part on GPU (CL_FALSE here indicates we dont block on write
    err = queue.enqueueWriteBuffer(gpu_solution[INDX_IN], CL_FALSE, offset_to_set_R * float_size, set_R_bytes, &U_G[0], NULL, &event);

    // readback all of gpu buffer to see 0s and R vals
    //err = queue.enqueueReadBuffer(gpu_solution[INDX_IN], CL_TRUE, 0, solution_mem_bytes, &test_output[0], NULL, &event);

#if  1
    // Launch kernel
    this->launchEulerKernel( ); 

    // TODO: copy set R from GPU solution to CPU solution

    // Pull the computed derivative back to the CPU
    // enqueueReadBuffer(buffer, blocking, offset, size, host_data, event queue to finish prior, returned_event_for_status_or_queue)
    err = queue.enqueueReadBuffer(gpu_solution[INDX_OUT], CL_FALSE, offset_to_set_O * float_size, set_O_bytes, &test_output[offset_to_set_O], NULL, &event);
    //err = queue.enqueueReadBuffer(gpu_solution[INDX_IN], CL_TRUE, 0, solution_mem_bytes, &test_output[0], NULL, &event);

    queue.finish();
    // reset boundary solution
   // this->enforceBoundaryConditions(s, cur_time); 

    // synchronize();
  //  this->sendrecvUpdates(s, "U_G");

    queue.finish();
    for (int i = 0; i < nb_nodes; i++) {
        std::cout << "u[" << i << "] = " << test_output[i] << std::endl;
    }
    }
    exit(EXIT_FAILURE);

    swap(INDX_IN, INDX_OUT);
#endif 
}

void HeatPDE_CL::launchEulerKernel() {

    int nb_stencils = (int)grid_ref.getStencilsSize(); 
    int stencil_size = (int)grid_ref.getMaxStencilSize(); 
    int nb_nodes = (int)grid_ref.getNodeListSize(); 

    try {
        kernel.setArg(0, der_ref_gpu.getGPUStencils()); 
    } catch (cl::Error er) {
        printf("[setKernelArg*] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
    try {
        kernel.setArg(1, der_ref_gpu.getGPUWeights(RBFFD::LAPL)); 
        kernel.setArg(2, this->gpu_solution[INDX_IN]);                 // COPY_IN / COPY_OUT
        kernel.setArg(3, sizeof(int), &nb_stencils);               // const 
        kernel.setArg(4, sizeof(int), &nb_nodes);                  // const 
        kernel.setArg(5, sizeof(int), &stencil_size);            // const
        kernel.setArg(6, this->gpu_solution[INDX_OUT]);                 // COPY_IN / COPY_OUT
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
#if 1
    size_t nb_nodes = grid_ref.getNodeListSize();
    size_t nb_stencils = grid_ref.getStencilsSize();

    cout << "Allocating GPU memory for HeatPDE\n";

    size_t set_G_size = grid_ref.G.size();
    size_t solution_mem_bytes = set_G_size = set_G_size * this->getFloatSize();

    size_t bytesAllocated = 0;

    gpu_solution[INDX_IN] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 
    gpu_solution[INDX_OUT] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 

    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
#endif 
}

//----------------------------------------------------------------------
//

