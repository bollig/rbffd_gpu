
#include "time_dependent_pde_cl.h"

//----------------------------------------------------------------------

void TimeDependentPDE_CL::setupTimers()
{
   tm["advance_gpu"] = new EB::Timer("Advance the PDE one step on the GPU") ;
   tm["loadAttach"] = new EB::Timer("Load the GPU Kernels for TimeDependentPDE_CL");
}

//----------------------------------------------------------------------

#if 0
void TimeDependentPDE_CL::fillInitialConditions(ExactSolution* exact) {
    // Fill U_G with initial conditions
    this->TimeDependentPDE::fillInitialConditions(exact);

    this->sendrecvUpdates(U_G, "U_G");

    unsigned int nb_nodes = grid_ref.G.size();
    unsigned int solution_mem_bytes = nb_nodes*this->getFloatSize(); 
   
    std::vector<double> diffusivity(nb_nodes, 0.);

    //FIXME: we're assuming float type on diffusivity. IF we need double, we'll
    //have to move this down.
    this->fillDiffusion(diffusivity, U_G, 0., nb_nodes);

    std::cout << "[TimeDependentPDE_CL] Writing initial conditions to GPU\n"; 
    if (useDouble) {

        // Fill GPU mem with initial solution 
        err = queue.enqueueWriteBuffer(gpu_solution[INDX_IN], CL_FALSE, 0, solution_mem_bytes, &U_G[0], NULL, &event);
    
        err = queue.enqueueWriteBuffer(gpu_diffusivity, CL_FALSE, 0, solution_mem_bytes, &diffusivity[0], NULL, &event);

        queue.finish();
    } else {

        float* U_G_f = new float[nb_nodes];
        float* diffusivity_f = new float[nb_nodes];
        for (unsigned int i = 0; i < nb_nodes; i++) {
            U_G_f[i] = (float)U_G[i];
            diffusivity_f[i] = (float)diffusivity[i];
        }
        err = queue.enqueueWriteBuffer(gpu_solution[INDX_IN], CL_FALSE, 0, solution_mem_bytes, &U_G_f[0], NULL, &event);

        err = queue.enqueueWriteBuffer(gpu_diffusivity, CL_FALSE, 0, solution_mem_bytes, &diffusivity_f[0], NULL, &event);
        queue.finish();

        delete [] U_G_f; 
        delete [] diffusivity_f; 

    }
   
    // FIXME: change all unsigned int to int. Or unsigned int. Size_t is not supported by GPU.
    std::vector<unsigned int>& bindices = grid_ref.getBoundaryIndices();
    unsigned int nb_bnd = bindices.size();
    err = queue.enqueueWriteBuffer(gpu_boundary_indices, CL_FALSE, 0, nb_bnd*sizeof(unsigned int), &bindices[0], NULL, &event);

    std::cout << "[TimeDependentPDE_CL] Done\n"; 
}
#endif 


//----------------------------------------------------------------------

void TimeDependentPDE_CL::assemble() 
{
    if (!weightsPrecomputed) {
        der_ref_gpu.computeAllWeightsForAllStencils();
    }
    // This will avoid multiple writes to GPU if the latest version is already in place
    // FIXME: allow this to finish later
    der_ref_gpu.updateWeightsOnGPU(false);
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::syncSetRSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int set_G_size = grid_ref.G.size();
    unsigned int set_Q_size = grid_ref.Q.size(); 
    unsigned int set_R_size = grid_ref.R.size();

    unsigned int float_size = this->getFloatSize();

    // OUR SOLUTION IS ARRANGED IN THIS FASHION: 
    //  { Q\B D O R } where B = union(O, D) and Q = union(Q\B D O)
    unsigned int offset_to_set_R = set_Q_size;

    unsigned int solution_mem_bytes = set_G_size*float_size; 
    unsigned int set_R_bytes = set_R_size * float_size;

    // backup the current solution so we can perform intermediate steps
    std::vector<float> r_update_f(set_R_size,-1.); 

    if (set_R_size > 0) {

        // Update CPU mem with R; 
        // NOTE: This is a single precision kernel call so we need to convert
        // the U_G to single precision
        for (int i = 0 ; i < set_R_size; i++) {
            r_update_f[i] = (float)vec[offset_to_set_R + i]; 
        }

        // Synchronize just the R part on GPU (CL_FALSE here indicates we dont block on write
        // NOTE: offset parameter to enqueueWriteBuffer is ONLY for the GPU side offset. The CPU offset needs to be managed directly on the CPU pointer: &U_G[offset_cpu]
       err = queue.enqueueWriteBuffer(gpu_vec, CL_FALSE, offset_to_set_R * float_size, set_R_bytes, &r_update_f[0], NULL, &event);
       
    }
}

//----------------------------------------------------------------------

// General routine to copy the set R indices vec up to gpu_vec
void TimeDependentPDE_CL::syncSetRDouble(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int set_G_size = grid_ref.G.size();
    unsigned int set_Q_size = grid_ref.Q.size(); 
    unsigned int set_R_size = grid_ref.R.size();

    unsigned int float_size = this->getFloatSize();

    // OUR SOLUTION IS ARRANGED IN THIS FASHION: 
    //  { Q\B D O R } where B = union(O, D) and Q = union(Q\B D O)
    unsigned int offset_to_set_R = set_Q_size;

    unsigned int solution_mem_bytes = set_G_size*float_size; 
    unsigned int set_R_bytes = set_R_size * float_size;

    if (set_R_size > 0) {

        // Synchronize just the R part on GPU (CL_FALSE here indicates we dont
        // block on write NOTE: offset parameter to enqueueWriteBuffer is ONLY
        // for the GPU side offset. The CPU offset needs to be managed directly
        // on the CPU pointer: &U_G[offset_cpu]
       err = queue.enqueueWriteBuffer(gpu_vec, CL_FALSE, offset_to_set_R * float_size, set_R_bytes, &vec[offset_to_set_R], NULL, &event);
       
    }
}

//----------------------------------------------------------------------

// General routine to copy the set O indices from gpu_vec down to vec
void TimeDependentPDE_CL::syncSetOSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int set_G_size = grid_ref.G.size();
    unsigned int set_Q_size = grid_ref.Q.size(); 
    unsigned int set_O_size = grid_ref.O.size();

    unsigned int float_size = this->getFloatSize();

    // OUR SOLUTION IS ARRANGED IN THIS FASHION: 
    //  { Q\B D O R } where B = union(O, D) and Q = union(Q\B D O)
    unsigned int offset_to_set_O = (set_Q_size - set_O_size);

    unsigned int solution_mem_bytes = set_G_size*float_size; 
    unsigned int set_O_bytes = set_O_size * float_size;

    // backup the current solution so we can perform intermediate steps
    std::vector<float> o_update_f(set_O_size,1.);


    if (set_O_size > 0) {
        // Pull only information required for neighboring domains back to the CPU 
        err = queue.enqueueReadBuffer(gpu_vec, CL_FALSE, offset_to_set_O * float_size, set_O_bytes, &o_update_f[0], NULL, &event);

       // Probably dont need this if we want to overlap comm and comp. 
       queue.finish();

        // NOTE: this is only required because we're calling a single precision
        // kernel 
        for (unsigned int i = 0; i < set_O_size; i++) {
        //    std::cout << "output u[" << i << "(global: " << grid_ref.l2g(offset_to_set_O+i) << ")] = " << U_G[offset_to_set_O + i] << "\t" << o_update_f[i] << std::endl;
            vec[offset_to_set_O + i] = (double) o_update_f[i];
        }
    }
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::syncSetODouble(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int set_G_size = grid_ref.G.size();
    unsigned int set_Q_size = grid_ref.Q.size(); 
    unsigned int set_O_size = grid_ref.O.size();

    unsigned int float_size = this->getFloatSize();

    // OUR SOLUTION IS ARRANGED IN THIS FASHION: 
    //  { Q\B D O R } where B = union(O, D) and Q = union(Q\B D O)
    unsigned int offset_to_set_O = (set_Q_size - set_O_size);

    unsigned int solution_mem_bytes = set_G_size*float_size; 
    unsigned int set_O_bytes = set_O_size * float_size;

    // backup the current solution so we can perform intermediate steps
    std::vector<float> o_update_f(set_O_size,1.);


    if (set_O_size > 0) {
        // Pull only information required for neighboring domains back to the CPU 
        err = queue.enqueueReadBuffer(gpu_vec, CL_FALSE, offset_to_set_O * float_size, set_O_bytes, &vec[offset_to_set_O], NULL, &event);

    }
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::syncCPUtoGPU() {
    std::cout << "SYNC CPU to GPU: " << INDX_IN << std::endl;
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int solution_mem_bytes = nb_nodes * this->getFloatSize();

    if (useDouble) {
        err = queue.enqueueReadBuffer(gpu_solution[INDX_IN], CL_FALSE, 0, solution_mem_bytes, &U_G[0], NULL, &event);
    } else {
        float* U_G_f = new float[nb_nodes]; 
        err = queue.enqueueReadBuffer(gpu_solution[INDX_IN], CL_FALSE, 0, solution_mem_bytes, &U_G_f[0], NULL, &event);

        queue.finish();
        for (unsigned int i = 0; i < nb_nodes; i++) {
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

void TimeDependentPDE_CL::advance(TimeScheme which, double delta_t) {
    tm["advance_gpu"]->start(); 
    switch (which) 
    {
#if 0
        case EULER: 
            advanceFirstOrderEuler(delta_t); 
            break; 

        case MIDPOINT: 
            advanceSecondOrderMidpoint(delta_t);
            break;  
#endif 
        case RK4: 
            advanceRungeKutta4(delta_t); 
            break;

        default: 
            std::cout << "[TimeDependentPDE_CL] Invalid TimeScheme specified. Bailing...\n";
            exit(EXIT_FAILURE); 
            break; 
    };
    cur_time += delta_t; 
    tm["advance_gpu"]->stop(); 
}

//----------------------------------------------------------------------

// FIXME: this is a single precision version
void TimeDependentPDE_CL::advanceRK4(double delta_t) {

#if EVAN_UPDATE_THESE
    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

    //-------- Overlap beweeen these: ------------
    // NOTE: syncSet*** ONLY copies between CPU and GPU. It does not synchronize across CPUs.
    // Use sendrecvUpdates to perform an interproc comm.
    if (useDouble) {
        this->syncSetRDouble(this->U_G, gpu_solution[INDX_IN]);
    } else {
        this->syncSetRSingle(this->U_G, gpu_solution[INDX_IN]); 
    }

    // Launch kernel
    //  params: timestep, vec_for_deriv_calc, vec_for_sum_rhs, vec_for_sum_lhs
    //  In other words: s2 = s1 + dt * d(s0)/dt; 
    //
    //  Euler: 
    //      s1 = s0 + dt * d(s0)/dt
    //
    //  Midpoint: 
    //      s1 = s0 + 0.5 dt * d(s0)/dt
    //      s2 = s0 + dt * d(s1)/dt
    //
    //  RK4: 
    //  
    // K1 t_n = cur_time + 0*dt
    // S1 = s0 + 0.5dt * (k1 + f)
    // params: dt on solve, dt on advance, input solve, output solve, input advance, output advance
    this->launchRK4_K_Kernel( 0.f, 0.5*delta_t, this->gpu_solution[INDX_IN], this->gpu_feval[0], this->gpu_solution[INDX_IN], this->gpu_solution[INDX_INTERMEDIATE_1] ); 
    
    // Enforce boundary using GPU, but specify we want to use the intermediate buffer
    this->enforceBoundaryConditions(U_G, this->gpu_solution[INDX_INTERMEDIATE_1], cur_time+0.5*delta_t); 
    
    // Since our syncSet****(..) routines ONLY sync the sets at the tail end of
    // the solution (i.e., sets O and R), 
    // we'll just re-use U_G as scratch space. So long as we dont copy U_G to
    // the GPU calling syncSet*** on our INDX_OUT will overwrite any
    // intermediate values stored there temporarily
    // If we want to match the GPU we
    // should do: syncCPUtoGPU()
    if (useDouble) {
        this->syncSetODouble(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]);
    } else {
        this->syncSetOSingle(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]); 
    }

    // Should send intermediate steps by copying down from GPU, sending, then
    // copying back up to GPU
    this->sendrecvUpdates(this->U_G, "intermediate_U_G");
   
    if (useDouble) {
        this->syncSetRDouble(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]);
    } else {
        this->syncSetRSingle(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]); 
    }

    // K2 t_n = cur_time + 0.5*dt
    // S2 = s0 + 0.5dt * (k2 + f)
    this->launchRK4_K_Kernel( 0.5f*delta_t, 0.5*delta_t, this->gpu_solution[INDX_INTERMEDIATE_1], this->gpu_feval[1], this->gpu_solution[INDX_IN], this->gpu_solution[INDX_INTERMEDIATE_1] ); 
   
    // Enforce boundary using GPU, but specify we want to use the intermediate buffer
    this->enforceBoundaryConditions(U_G, this->gpu_solution[INDX_INTERMEDIATE_1], cur_time+0.5*delta_t); 
    
    // Since our syncSet****(..) routines ONLY sync the sets at the tail end of
    // the solution (i.e., sets O and R), 
    // we'll just re-use U_G as scratch space. So long as we dont copy U_G to
    // the GPU, calling syncSet*** on our INDX_OUT will overwrite any
    // intermediate values stored there temporarily
    // If we want to match the GPU we
    // should do: syncCPUtoGPU()
    if (useDouble) {
        this->syncSetODouble(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]);
    } else {
        this->syncSetOSingle(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]); 
    }

    // Should send intermediate steps by copying down from GPU, sending, then
    // copying back up to GPU
    this->sendrecvUpdates(this->U_G, "intermediate_U_G");
   
    if (useDouble) {
        this->syncSetRDouble(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]);
    } else {
        this->syncSetRSingle(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]); 
    }


    // K3 t_n = cur_time + 0.5*dt
    // S3 = s0 + dt * (k3 + f)
    this->launchRK4_K_Kernel( 0.5f*delta_t, delta_t, this->gpu_solution[INDX_INTERMEDIATE_1], this->gpu_feval[2], this->gpu_solution[INDX_IN], this->gpu_solution[INDX_INTERMEDIATE_1] ); 
   
    // Enforce boundary using GPU, but specify we want to use the intermediate buffer
    this->enforceBoundaryConditions(U_G, this->gpu_solution[INDX_INTERMEDIATE_1], cur_time+delta_t); 
    // Since our syncSet****(..) routines ONLY sync the sets at the tail end of
    // the solution (i.e., sets O and R), 
    // we'll just re-use U_G as scratch space. So long as we dont copy U_G to
    // the GPU calling syncSet*** on our INDX_OUT will overwrite any
    // intermediate values stored there temporarily
    // If we want to match the GPU we
    // should do: syncCPUtoGPU()
    if (useDouble) {
        this->syncSetODouble(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]);
    } else {
        this->syncSetOSingle(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]); 
    }

    // Should send intermediate steps by copying down from GPU, sending, then
    // copying back up to GPU
    this->sendrecvUpdates(this->U_G, "intermediate_U_G");
   
    if (useDouble) {
        this->syncSetRDouble(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]);
    } else {
        this->syncSetRSingle(this->U_G, gpu_solution[INDX_INTERMEDIATE_1]); 
    }

 // K3 t_n = cur_time + 0.5*dt
    // S3 = s0 + dt * (k3 + f)
    this->launchRK4_Final_Kernel( 0.5f*delta_t, delta_t, this->gpu_solution[INDX_IN], this->gpu_feval[0], this->gpu_feval[1], this->gpu_feval[2], this->gpu_solution[INDX_OUT] ); 
   
    // reset boundary solution on INDX_OUT
    this->enforceBoundaryConditions(U_G, this->gpu_solution[INDX_OUT], cur_time); 

    if (useDouble) {
        this->syncSetODouble(this->U_G, gpu_solution[INDX_OUT]);
    } else {
        this->syncSetOSingle(this->U_G, gpu_solution[INDX_OUT]); 
    }

    queue.finish();

//    this->syncCPUtoGPU(); 
         
#if 0
    for (int i = 0; i < nb_nodes; i++) {
        std::cout << "u[" << i << "] = " << U_G[i] << std::endl;
    }
#endif 

    // synchronize();
    this->sendrecvUpdates(U_G, "U_G");

    //exit(EXIT_FAILURE);

    swap(INDX_IN, INDX_OUT);
#endif 
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::loadKernels(std::string& local_sources) {
    tm["loadAttach"]->start(); 

#if 0
    this->loadBCKernel(local_sources);
    this->loadStepKernel(local_sources); 
#endif 
    this->loadRK4Kernels(local_sources); 
    tm["loadAttach"]->stop(); 
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::loadRK4Kernels(std::string& local_sources) {

    std::string rk4_k_kernel_name = "evaluate_K_RK4"; 
    std::string rk4_final_kernel_name = "final_RK4"; 

    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false;
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        useDouble = false;
    } 

    std::string my_source = local_sources; 
    if(useDouble) {
#define FLOAT double 
#include "cl_kernels/rk4_heat.cl"
        my_source.append(kernel_source);
#undef FLOAT
    }else {
#define FLOAT float
#include "cl_kernels/rk4_heat.cl"
        my_source.append(kernel_source);
#undef FLOAT
    }

    //std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n"; 
    this->loadProgram(my_source, useDouble); 

    try{
        std::cout << "Loading kernel \""<< rk4_k_kernel_name << "\" with double precision = " << useDouble << "\n"; 
        rk4_k_kernel = cl::Kernel(program, rk4_k_kernel_name.c_str(), &err);
        rk4_final_kernel = cl::Kernel(program, rk4_final_kernel_name.c_str(), &err);
        std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
    }
}

//----------------------------------------------------------------------
//
void TimeDependentPDE_CL::allocateGPUMem() {
#if 1
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int nb_stencils = grid_ref.getStencilsSize();
    unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();

    cout << "Allocating GPU memory for TimeDependentPDE\n";

    unsigned int solution_mem_bytes = nb_nodes * this->getFloatSize();

    unsigned int bytesAllocated = 0;

    gpu_solution[INDX_IN] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 
    gpu_solution[INDX_OUT] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 
    gpu_solution[INDX_INTERMEDIATE_1] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 
    
    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
#endif 
}

//----------------------------------------------------------------------
//

