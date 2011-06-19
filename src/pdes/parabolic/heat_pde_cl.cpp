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

    unsigned int nb_nodes = grid_ref.G.size();
    unsigned int solution_mem_bytes = nb_nodes*this->getFloatSize(); 
   
    std::vector<double> diffusivity(nb_nodes, 0.);

    //FIXME: we're assuming float type on diffusivity. IF we need double, we'll
    //have to move this down.
    this->fillDiffusion(diffusivity, U_G, 0., nb_nodes);

    std::cout << "[HeatPDE_CL] Writing initial conditions to GPU\n"; 
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

    std::cout << "[HeatPDE_CL] Done\n"; 
}

// Handle the boundary conditions however we want to. 
// NOTE: we must update the solution on the GPU too. 
void HeatPDE_CL::enforceBoundaryConditions(std::vector<SolutionType>& u_t, cl::Buffer& sol, double t)
{
    // FIXME: should we mirror the CPU first?
//    this->HeatPDE::enforceBoundaryConditions(u_t, t);
    
    unsigned int nb_stencils = grid_ref.getStencilsSize(); 
    unsigned int stencil_size = grid_ref.getMaxStencilSize(); 
    unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();
    unsigned int nb_nodes = grid_ref.getNodeListSize(); 
    float cur_time_f = (float) cur_time;
    
    try {
        bc_kernel.setArg(0, sol);                 // COPY_IN  / COPY OUT
        bc_kernel.setArg(1, this->gpu_boundary_indices);                 // COPY_IN 
        bc_kernel.setArg(2, sizeof(unsigned int), &nb_bnd);               // const 
        bc_kernel.setArg(3, sizeof(float), &cur_time_f);
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
    unsigned int safe_launch_size = (nb_bnd > 32) ? nb_bnd : 32;

    err = queue.enqueueNDRangeKernel(bc_kernel, /* offset */ cl::NullRange, 
            /* GLOBAL (work-groups in the grid)  */   cl::NDRange(safe_launch_size), 
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
    // This will avoid multiple writes to GPU if the latest version is already in place
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

        case MIDPOINT: 
            advanceSecondOrderMidpoint(delta_t);
            break;  
#if 0
        case RK4: 
            advanceRungeKutta4(delta_t); 
            break;
#endif 
        default: 
            std::cout << "[HeatPDE_CL] Invalid TimeScheme specified. Bailing...\n";
            exit(EXIT_FAILURE); 
            break; 
    };
    cur_time += delta_t; 
    tm["advance_gpu"]->stop(); 
}

void HeatPDE_CL::syncSetRSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
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


// General routine to copy the set R indices vec up to gpu_vec
void HeatPDE_CL::syncSetRDouble(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
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

// General routine to copy the set O indices from gpu_vec down to vec
void HeatPDE_CL::syncSetOSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
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


void HeatPDE_CL::syncSetODouble(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
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
// FIXME: this is a single precision version
void HeatPDE_CL::advanceFirstOrderEuler(double delta_t) {

    // Target (st5): 0.3991 ms
    //        (st33): 1.2 ms
    // GPU: 
    // Without diffusion, boundary or f(u) eval (st33): 0.3599
    // Without boundary or f(u) eval (st33): 0.3562
    // Without boundary (st33): 4.8389
    // no boundary, K*Laplacian only (no gradK . gradU) (st33): 1.1898

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

    // 1) Launch kernel for set QmD (will take a while, so in the meantime...)
    this->launchEulerSetQmDKernel(delta_t, this->gpu_solution[INDX_IN], this->gpu_solution[INDX_OUT]); 
    
    // NOTE: when run in serial only one kernel launch is required. 
    if (comm_ref.getSize() > 1) {
        std::cout << "INSIDE EULER set D STUFF\n";
        // 2) OVERLAP: Transfer set O from the input to the CPU for synchronization acros CPUs
        if (useDouble) {
            this->syncSetODouble(this->U_G, gpu_solution[INDX_IN]);
        } else {
            this->syncSetOSingle(this->U_G, gpu_solution[INDX_IN]); 
        }

        // 3) OVERLAP: Transmit between CPUs
        // NOTE: Require an MPI barrier here
        this->sendrecvUpdates(U_G, "U_G");


        // 4) OVERLAP: Update the input with set R
        if (useDouble) {
            this->syncSetRDouble(this->U_G, gpu_solution[INDX_IN]);
        } else {
            this->syncSetRSingle(this->U_G, gpu_solution[INDX_IN]); 
        }

        // 6) Launch a SECOND kernel to complete set D for this step (NOTE: in
        // higher order timeschemes we need to perform ADDITIONAL communication
        // here. Also, this MIGHT modify the boundary value so we should enforce
        // conditions AFTER this kernel)
        this->launchEulerSetDKernel(delta_t, this->gpu_solution[INDX_IN], this->gpu_solution[INDX_OUT]);
        queue.finish();
    }
    queue.finish();

    // 5) FINAL: reset boundary solution on INDX_OUT
    // COST: 0.3 ms
    this->enforceBoundaryConditions(U_G, this->gpu_solution[INDX_OUT], cur_time); 

    // Fire events to force the queue to execute.
    //queue.finish();

    // Flip our ping pong buffers. 
    swap(INDX_IN, INDX_OUT);
}


// Launch a kernel to perform the step: 
//      u(n+1) = u(n) + dt * f( u(n) )
// Params: 
//  dt => Timestep
//  sol_in => parameter u(n) above
//  sol_out => parameter u(n+1) above
void HeatPDE_CL::launchEulerSetQmDKernel( double dt, cl::Buffer& sol_in, cl::Buffer& sol_out) {
    this->launchStepKernel(dt, sol_in, sol_in, sol_out, grid_ref.QmD.size(), 0);
}

// Same as EulerSetQmDKernel, but it requires transfer of O and R to be complete. 
void HeatPDE_CL::launchEulerSetDKernel( double dt, cl::Buffer& sol_in, cl::Buffer& sol_out) {
    this->launchStepKernel(dt, sol_in, sol_in, sol_out, grid_ref.D.size(), grid_ref.QmD.size());
}


// Launch a kernel to perform the step: 
//      u(n+1) = u(n) + dt * f( U(n) )
// Params: 
//  dt => Timestep
//  sol_in => parameter u(n) above
//  deriv_sol_in => parameter U(n) above
//  sol_out => parameter u(n+1) above
//
void HeatPDE_CL::launchStepKernel( double dt, cl::Buffer& sol_in, cl::Buffer& deriv_sol_in, cl::Buffer& sol_out, unsigned int n_stencils, unsigned int offset_to_set) {

//    unsigned int nb_stencils = grid_ref.getStencilsSize(); 
    unsigned int stencil_size = grid_ref.getMaxStencilSize(); 
    unsigned int nb_nodes = grid_ref.getNodeListSize(); 
    float dt_f = (float) dt;
    float cur_time_f = (float) cur_time;

    try {
        step_kernel.setArg(0, der_ref_gpu.getGPUStencils()); 
        step_kernel.setArg(1, der_ref_gpu.getGPUWeights(RBFFD::LAPL)); 
        step_kernel.setArg(2, der_ref_gpu.getGPUWeights(RBFFD::X)); 
        step_kernel.setArg(3, der_ref_gpu.getGPUWeights(RBFFD::Y)); 
        step_kernel.setArg(4, der_ref_gpu.getGPUWeights(RBFFD::Z)); 
        step_kernel.setArg(5, sol_in);                 // COPY_IN 
        step_kernel.setArg(6, deriv_sol_in);                 // COPY_IN 
        step_kernel.setArg(7, this->gpu_diffusivity);                 // COPY_IN 
        step_kernel.setArg(8, sizeof(unsigned int), &offset_to_set);               // const 
        step_kernel.setArg(9, sizeof(unsigned int), &n_stencils);               // const 
        step_kernel.setArg(10, sizeof(unsigned int), &nb_nodes);                  // const 
        step_kernel.setArg(11, sizeof(unsigned int), &stencil_size);            // const
        step_kernel.setArg(12, sizeof(float), &dt_f);            // const
        step_kernel.setArg(13, sizeof(float), &cur_time_f);            // const
        step_kernel.setArg(14, sol_out);                 // COPY_IN / COPY_OUT
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    err = queue.enqueueNDRangeKernel(step_kernel, /* offset */ cl::NullRange, 
            /* GLOBAL (work-groups in the grid)  */   cl::NDRange(n_stencils), 
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
// FIXME: this is a single precision version
void HeatPDE_CL::advanceSecondOrderMidpoint(double delta_t) {
#if EVAN_UPDATE_THIS
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
    //      s1 = s0 + dt
    this->launchStepKernel( 0.5*delta_t, this->gpu_solution[INDX_IN], this->gpu_solution[INDX_IN], this->gpu_solution[INDX_INTERMEDIATE_1] ); 
    
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

    this->launchStepKernel( delta_t, this->gpu_solution[INDX_IN], this->gpu_solution[INDX_INTERMEDIATE_1], this->gpu_solution[INDX_OUT] ); 
    //-------- END OVERLAP -----------------------

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


#if 1
//----------------------------------------------------------------------
// FIXME: this is a single precision version
void HeatPDE_CL::advanceRungeKutta4(double delta_t) {

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

#endif 
//----------------------------------------------------------------------
//
void HeatPDE_CL::loadKernels(std::string& local_sources) {
    tm["loadAttach"]->start(); 

    this->loadBCKernel(local_sources);
    this->loadStepKernel(local_sources); 
#if EVAN_UPDATE_THESE
    this->loadRK4Kernels(local_sources); 
#endif 
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

void HeatPDE_CL::loadStepKernel(std::string& local_sources) {


    std::string kernel_name = "evaluateStep"; 

    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false;
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        useDouble = false;
    } 

    std::string my_source = local_sources; 
    if(useDouble) {
#define FLOAT double 
#include "cl_kernels/step_heat.cl"
        my_source.append(kernel_source);
#undef FLOAT
    }else {
#define FLOAT float
#include "cl_kernels/step_heat.cl"
        my_source.append(kernel_source);
#undef FLOAT
    }

    //std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n"; 
    this->loadProgram(my_source, useDouble); 

    try{
        std::cout << "Loading kernel \""<< kernel_name << "\" with double precision = " << useDouble << "\n"; 
        step_kernel = cl::Kernel(program, kernel_name.c_str(), &err);
        std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
    }
}

#if EVAN_UPDATE_THESE
void HeatPDE_CL::loadRK4Kernels(std::string& local_sources) {

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
#endif 


//----------------------------------------------------------------------
//
void HeatPDE_CL::allocateGPUMem() {
#if 1
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    unsigned int nb_stencils = grid_ref.getStencilsSize();
    unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();

    cout << "Allocating GPU memory for HeatPDE\n";

    unsigned int solution_mem_bytes = nb_nodes * this->getFloatSize();

    unsigned int bytesAllocated = 0;

    gpu_solution[INDX_IN] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 
    gpu_solution[INDX_OUT] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 
    gpu_solution[INDX_INTERMEDIATE_1] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
    bytesAllocated += solution_mem_bytes; 
    
    gpu_diffusivity = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);

    gpu_boundary_indices = cl::Buffer(context, CL_MEM_READ_ONLY, nb_bnd * sizeof(unsigned int), NULL, &err);

    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
#endif 
}

//----------------------------------------------------------------------
//

