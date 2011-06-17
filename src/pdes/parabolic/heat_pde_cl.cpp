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

    this->sendrecvUpdates(U_G, "U_G");

    size_t nb_nodes = grid_ref.G.size();
    size_t solution_mem_bytes = nb_nodes*this->getFloatSize(); 
   
    std::vector<double> diffusivity(nb_nodes, 0.);

    //FIXME: we're assuming float type on diffusivity. IF we need double, we'll
    //have to move this down.
    this->fillDiffusion(diffusivity, U_G, 0., nb_nodes);

    std::cout << "[HeatPDE_CL] Writing initial conditions to GPU\n"; 
    if (useDouble) {

        // Fill GPU mem with initial solution 
        err = queue.enqueueWriteBuffer(gpu_solution[INDX_IN], CL_TRUE, 0, solution_mem_bytes, &U_G[0], NULL, &event);
    
        err = queue.enqueueWriteBuffer(gpu_diffusivity, CL_TRUE, 0, solution_mem_bytes, &diffusivity[0], NULL, &event);

        queue.finish();
    } else {

        float* U_G_f = new float[nb_nodes];
        float* diffusivity_f = new float[nb_nodes];
        for (size_t i = 0; i < nb_nodes; i++) {
            U_G_f[i] = (float)U_G[i];
            diffusivity_f[i] = (float)diffusivity[i];
        }
        err = queue.enqueueWriteBuffer(gpu_solution[INDX_IN], CL_TRUE, 0, solution_mem_bytes, &U_G_f[0], NULL, &event);

        err = queue.enqueueWriteBuffer(gpu_diffusivity, CL_TRUE, 0, solution_mem_bytes, &diffusivity_f[0], NULL, &event);
        queue.finish();

        delete [] U_G_f; 
        delete [] diffusivity_f; 

    }
   
    // FIXME: change all size_t to int. Or unsigned int. Size_t is not supported by GPU.
    size_t nb_bnd = grid_ref.getBoundaryIndicesSize();
    int* bindices = new int[nb_bnd]; 
    // Would be unnec. if indices were in int instead of size_t
    for (size_t i =0; i < nb_bnd; i++) {
        bindices[i] = grid_ref.getBoundaryIndex(i);
    }
    err = queue.enqueueWriteBuffer(gpu_boundary_indices, CL_TRUE, 0, nb_bnd*sizeof(int), &bindices[0], NULL, &event);

    std::cout << "[HeatPDE_CL] Done\n"; 
}

// Handle the boundary conditions however we want to. 
// NOTE: we must update the solution on the GPU too. 
void HeatPDE_CL::enforceBoundaryConditions(std::vector<SolutionType>& u_t, double t)
{
//    this->HeatPDE::enforceBoundaryConditions(u_t, t);
    
    int nb_stencils = (int)grid_ref.getStencilsSize(); 
    int stencil_size = (int)grid_ref.getMaxStencilSize(); 
    int nb_bnd = (int)grid_ref.getBoundaryIndicesSize();
    int nb_nodes = (int)grid_ref.getNodeListSize(); 
    float cur_time_f = (float) cur_time;
    
    try {
        bc_kernel.setArg(0, this->gpu_solution[INDX_OUT]);                 // COPY_IN  / COPY OUT
        bc_kernel.setArg(1, this->gpu_boundary_indices);                 // COPY_IN 
        bc_kernel.setArg(2, sizeof(int), &nb_bnd);               // const 
        bc_kernel.setArg(3, sizeof(float), &cur_time_f);
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    err = queue.enqueueNDRangeKernel(bc_kernel, /* offset */ cl::NullRange, 
            /* GLOBAL (work-groups in the grid)  */   cl::NDRange(max(nb_bnd, 32)), 
            /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);

    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
        exit(EXIT_FAILURE);
    }

    //queue.finish();
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

void HeatPDE_CL::syncSetRSingle() {
    size_t nb_nodes = grid_ref.getNodeListSize();
    size_t set_G_size = grid_ref.G.size();
    size_t set_Q_size = grid_ref.Q.size(); 
    size_t set_R_size = grid_ref.R.size();

    size_t float_size = this->getFloatSize();

    // OUR SOLUTION IS ARRANGED IN THIS FASHION: 
    //  { Q\B D O R } where B = union(O, D) and Q = union(Q\B D O)
    size_t offset_to_set_R = set_Q_size;

    size_t solution_mem_bytes = set_G_size*float_size; 
    size_t set_R_bytes = set_R_size * float_size;

    // backup the current solution so we can perform intermediate steps
    std::vector<float> r_update_f(set_R_size,-1.); //= this->U_G; 

    if (set_R_size > 0) {

        // Update CPU mem with R; 
        // NOTE: This is a single precision kernel call so we need to convert
        // the U_G to single precision
        for (int i = 0 ; i < set_R_size; i++) {
            r_update_f[i] = (float)U_G[offset_to_set_R + i]; 
        }

        // Synchronize just the R part on GPU (CL_FALSE here indicates we dont block on write
        // NOTE: offset parameter to enqueueWriteBuffer is ONLY for the GPU side offset. The CPU offset needs to be managed directly on the CPU pointer: &U_G[offset_cpu]
       err = queue.enqueueWriteBuffer(gpu_solution[INDX_IN], CL_TRUE, offset_to_set_R * float_size, set_R_bytes, &r_update_f[0], NULL, &event);
       
    }
}


void HeatPDE_CL::syncSetRDouble() {
    size_t nb_nodes = grid_ref.getNodeListSize();
    size_t set_G_size = grid_ref.G.size();
    size_t set_Q_size = grid_ref.Q.size(); 
    size_t set_R_size = grid_ref.R.size();

    size_t float_size = this->getFloatSize();

    // OUR SOLUTION IS ARRANGED IN THIS FASHION: 
    //  { Q\B D O R } where B = union(O, D) and Q = union(Q\B D O)
    size_t offset_to_set_R = set_Q_size;

    size_t solution_mem_bytes = set_G_size*float_size; 
    size_t set_R_bytes = set_R_size * float_size;

    if (set_R_size > 0) {

        // Synchronize just the R part on GPU (CL_FALSE here indicates we dont
        // block on write NOTE: offset parameter to enqueueWriteBuffer is ONLY
        // for the GPU side offset. The CPU offset needs to be managed directly
        // on the CPU pointer: &U_G[offset_cpu]
       err = queue.enqueueWriteBuffer(gpu_solution[INDX_IN], CL_TRUE, offset_to_set_R * float_size, set_R_bytes, &U_G[offset_to_set_R], NULL, &event);
       
    }
}

void HeatPDE_CL::syncSetOSingle() {
    size_t nb_nodes = grid_ref.getNodeListSize();
    size_t set_G_size = grid_ref.G.size();
    size_t set_Q_size = grid_ref.Q.size(); 
    size_t set_O_size = grid_ref.O.size();

    size_t float_size = this->getFloatSize();

    // OUR SOLUTION IS ARRANGED IN THIS FASHION: 
    //  { Q\B D O R } where B = union(O, D) and Q = union(Q\B D O)
    size_t offset_to_set_O = (set_Q_size - set_O_size);

    size_t solution_mem_bytes = set_G_size*float_size; 
    size_t set_O_bytes = set_O_size * float_size;

    // backup the current solution so we can perform intermediate steps
    std::vector<float> o_update_f(set_O_size,1.);


    if (set_O_size > 0) {
        // Pull only information required for neighboring domains back to the CPU 
        err = queue.enqueueReadBuffer(gpu_solution[INDX_OUT], CL_TRUE, offset_to_set_O * float_size, set_O_bytes, &o_update_f[0], NULL, &event);

       // Probably dont need this if we want to overlap comm and comp. 
       queue.finish();

        // NOTE: this is only required because we're calling a single precision
        // kernel 
        for (size_t i = 0; i < set_O_size; i++) {
        //    std::cout << "output u[" << i << "(global: " << grid_ref.l2g(offset_to_set_O+i) << ")] = " << U_G[offset_to_set_O + i] << "\t" << o_update_f[i] << std::endl;
            U_G[offset_to_set_O + i] = (double) o_update_f[i];
        }
    }
}


void HeatPDE_CL::syncSetODouble() {
    size_t nb_nodes = grid_ref.getNodeListSize();
    size_t set_G_size = grid_ref.G.size();
    size_t set_Q_size = grid_ref.Q.size(); 
    size_t set_O_size = grid_ref.O.size();

    size_t float_size = this->getFloatSize();

    // OUR SOLUTION IS ARRANGED IN THIS FASHION: 
    //  { Q\B D O R } where B = union(O, D) and Q = union(Q\B D O)
    size_t offset_to_set_O = (set_Q_size - set_O_size);

    size_t solution_mem_bytes = set_G_size*float_size; 
    size_t set_O_bytes = set_O_size * float_size;

    // backup the current solution so we can perform intermediate steps
    std::vector<float> o_update_f(set_O_size,1.);


    if (set_O_size > 0) {
        // Pull only information required for neighboring domains back to the CPU 
        err = queue.enqueueReadBuffer(gpu_solution[INDX_OUT], CL_TRUE, offset_to_set_O * float_size, set_O_bytes, &U_G[offset_to_set_O], NULL, &event);

    }
}


//----------------------------------------------------------------------
// FIXME: this is a single precision version
void HeatPDE_CL::advanceFirstOrderEuler(double delta_t) {

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

    if (useDouble) {
        this->syncSetRDouble();
    } else {
        this->syncSetRSingle(); 
    }

    // Launch kernel
    this->launchEulerKernel( delta_t ); 

    // reset boundary solution on INDX_OUT
    this->enforceBoundaryConditions(U_G, cur_time); 

    if (useDouble) {
        this->syncSetODouble();
    } else {
        this->syncSetOSingle(); 
    }

    queue.finish();

    this->syncCPUtoGPU(); 
         
#if 0
    for (int i = 0; i < nb_nodes; i++) {
        std::cout << "u[" << i << "] = " << U_G[i] << std::endl;
    }
#endif 

    // synchronize();
    this->sendrecvUpdates(U_G, "U_G");

    //exit(EXIT_FAILURE);

    swap(INDX_IN, INDX_OUT);
}

void HeatPDE_CL::launchEulerKernel( double dt ) {

    int nb_stencils = (int)grid_ref.getStencilsSize(); 
    int stencil_size = (int)grid_ref.getMaxStencilSize(); 
    int nb_nodes = (int)grid_ref.getNodeListSize(); 
    float dt_f = (float) dt;
    float cur_time_f = (float) cur_time;

    try {
        euler_kernel.setArg(0, der_ref_gpu.getGPUStencils()); 
        euler_kernel.setArg(1, der_ref_gpu.getGPUWeights(RBFFD::LAPL)); 
        euler_kernel.setArg(2, der_ref_gpu.getGPUWeights(RBFFD::X)); 
        euler_kernel.setArg(3, der_ref_gpu.getGPUWeights(RBFFD::Y)); 
        euler_kernel.setArg(4, der_ref_gpu.getGPUWeights(RBFFD::Z)); 
        euler_kernel.setArg(5, this->gpu_solution[INDX_IN]);                 // COPY_IN / COPY_OUT
        euler_kernel.setArg(6, this->gpu_diffusivity);                 // COPY_IN 
        euler_kernel.setArg(7, sizeof(int), &nb_stencils);               // const 
        euler_kernel.setArg(8, sizeof(int), &nb_nodes);                  // const 
        euler_kernel.setArg(9, sizeof(int), &stencil_size);            // const
        euler_kernel.setArg(10, sizeof(float), &dt_f);            // const
        euler_kernel.setArg(11, sizeof(float), &cur_time_f);            // const
        euler_kernel.setArg(12, this->gpu_solution[INDX_OUT]);                 // COPY_IN / COPY_OUT
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    err = queue.enqueueNDRangeKernel(euler_kernel, /* offset */ cl::NullRange, 
            /* GLOBAL (work-groups in the grid)  */   cl::NDRange(nb_stencils), 
            /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);

//    err = queue.finish();
    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void HeatPDE_CL::syncCPUtoGPU() {
    size_t nb_nodes = grid_ref.getNodeListSize();
    size_t solution_mem_bytes = nb_nodes * this->getFloatSize();

    if (useDouble) {
        err = queue.enqueueReadBuffer(gpu_solution[INDX_OUT], CL_TRUE, 0, solution_mem_bytes, &U_G[0], NULL, &event);
    } else {
        float* U_G_f = new float[nb_nodes]; 
        err = queue.enqueueReadBuffer(gpu_solution[INDX_OUT], CL_TRUE, 0, solution_mem_bytes, &U_G_f[0], NULL, &event);

        for (size_t i = 0; i < nb_nodes; i++) {
#if 0
            double diff = fabs( U_G[i] - U_G_f[i] ); 
            if (diff > 1e-4) {
                std::cout << "GPUvsCPU diff[" << i << "]: " << diff << std::endl;
            }
#endif 
            U_G[i] = (double)U_G_f[i]; 
        }
        delete [] U_G_f; 
    }

}

//----------------------------------------------------------------------
//
void HeatPDE_CL::loadKernels(std::string& local_sources) {
    tm["loadAttach"]->start(); 

    this->loadBCKernel(local_sources);
    this->loadEulerKernel(local_sources); 

    tm["loadAttach"]->stop(); 
}

void HeatPDE_CL::loadBCKernel(std::string& local_sources) {

    std::string kernel_name = "enforceBoundaryConditions"; 

    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false;
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        useDouble = false;
    } 

    std::string my_source = local_sources; 
    if(useDouble) {
#define FLOAT double 
#include "cl_kernels/boundary_conditions.cl"
        my_source.append(kernel_source);
#undef FLOAT
    }else {
#define FLOAT float
#include "cl_kernels/boundary_conditions.cl"
        my_source.append(kernel_source);
#undef FLOAT
    }

    //std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n"; 
    
    // FIXME: this overwrites the previous program source. Is this a problem? 
    this->loadProgram(my_source, useDouble); 

    try{
        std::cout << "Loading kernel \""<< kernel_name << "\" with double precision = " << useDouble << "\n"; 
        bc_kernel = cl::Kernel(program, kernel_name.c_str(), &err);
        std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
    }
}



void HeatPDE_CL::loadEulerKernel(std::string& local_sources) {


    std::string kernel_name = "advanceFirstOrderEuler"; 

    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false;
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        useDouble = false;
    } 

    std::string my_source = local_sources; 
    if(useDouble) {
#define FLOAT double 
#include "cl_kernels/euler_heat.cl"
        my_source.append(kernel_source);
#undef FLOAT
    }else {
#define FLOAT float
#include "cl_kernels/euler_heat.cl"
        my_source.append(kernel_source);
#undef FLOAT
    }

    //std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n"; 
    this->loadProgram(my_source, useDouble); 

    try{
        std::cout << "Loading kernel \""<< kernel_name << "\" with double precision = " << useDouble << "\n"; 
        euler_kernel = cl::Kernel(program, kernel_name.c_str(), &err);
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
    size_t nb_bnd = grid_ref.getBoundaryIndicesSize();

    cout << "Allocating GPU memory for HeatPDE\n";

    size_t solution_mem_bytes = nb_nodes * this->getFloatSize();

    size_t bytesAllocated = 0;

    gpu_solution[INDX_IN] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 
    gpu_solution[INDX_OUT] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 
    
    gpu_diffusivity = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);

    gpu_boundary_indices = cl::Buffer(context, CL_MEM_READ_ONLY, nb_bnd * sizeof(int), NULL, &err);

    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
#endif 
}

//----------------------------------------------------------------------
//

