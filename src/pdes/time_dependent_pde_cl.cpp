
#include "time_dependent_pde_cl.h"

//----------------------------------------------------------------------

void TimeDependentPDE_CL::setupTimers()
{
        tm["advance_gpu"] = new EB::Timer("Advance the PDE one step on the GPU") ;
        tm["loadAttach"] = new EB::Timer("Load the GPU Kernels for TimeDependentPDE_CL");
}

//----------------------------------------------------------------------

#if 1
void TimeDependentPDE_CL::fillInitialConditions(ExactSolution* exact) {
        // Fill U_G with initial conditions
        this->TimeDependentPDE::fillInitialConditions(exact);

        this->sendrecvUpdates(U_G, "U_G");

        unsigned int nb_nodes = grid_ref.G.size();
        unsigned int solution_mem_bytes = nb_nodes*this->getFloatSize();

#if 0
        std::vector<double> diffusivity(nb_nodes, 0.);

        //FIXME: we're assuming float type on diffusivity. IF we need double, we'll
        //have to move this down.
        this->fillDiffusion(diffusivity, U_G, 0., nb_nodes);

        err = queue.enqueueWriteBuffer(gpu_diffusivity, CL_TRUE, 0, solution_mem_bytes, &diffusivity[0], NULL, &event);
//        queue.flush();
        float* diffusivity_f = new float[nb_nodes];
        diffusivity_f[i] = (float)diffusivity[i];
        err = queue.enqueueWriteBuffer(gpu_diffusivity, CL_TRUE, 0, solution_mem_bytes, &diffusivity_f[0], NULL, &event);
//        queue.flush();
        delete [] diffusivity_f;
#endif

        try{
                std::cout << "[TimeDependentPDE_CL] Writing initial conditions to GPU\n";
                if (useDouble) {

                        // Fill GPU mem with initial solution
                        err = queue.enqueueWriteBuffer(gpu_solution[INDX_OUT], CL_TRUE, 0, solution_mem_bytes, &U_G[0], NULL, &event);
//                        queue.flush();

                        queue.finish();
                } else {

                        float* U_G_f = new float[nb_nodes];
                        for (unsigned int i = 0; i < nb_nodes; i++) {
                                U_G_f[i] = (float)U_G[i];
                        }
                        err = queue.enqueueWriteBuffer(gpu_solution[INDX_OUT], CL_TRUE, 0, solution_mem_bytes, &U_G_f[0], NULL, &event);
//                        queue.flush();

                        queue.finish();

                        delete [] U_G_f;
                }
        }
        catch (cl::Error er) {
                printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
        }
#if 0
        // FIXME: change all unsigned int to int. Or unsigned int. Size_t is not supported by GPU.
        std::vector<unsigned int>& bindices = grid_ref.getBoundaryIndices();
        unsigned int nb_bnd = bindices.size();
        err = queue.enqueueWriteBuffer(gpu_boundary_indices, CL_TRUE, 0, nb_bnd*sizeof(unsigned int), &bindices[0], NULL, &event);
//        queue.flush();
#endif
        std::cout << "[TimeDependentPDE_CL] Done\n";
}
#endif 


//----------------------------------------------------------------------

void TimeDependentPDE_CL::assemble() 
{
    std::cout << "Assemble\n";
        if (!weightsPrecomputed) {
                der_ref_gpu.computeAllWeightsForAllStencils();
                weightsPrecomputed = true;
        }
        // This will avoid multiple writes to GPU if the latest version is already in place
        // FIXME: allow this to finish later
        der_ref_gpu.updateWeightsOnGPU(false);

        // Flip our ping pong buffers.
        // NOTE: we need to initialize into INDX_OUT
        swap(INDX_IN, INDX_OUT);
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::syncSetRSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
    exit(EXIT_FAILURE);
        unsigned int nb_nodes = grid_ref.getNodeListSize();
        unsigned int set_G_size = grid_ref.G.size();
        unsigned int set_Q_size = grid_ref.Q.size();
        unsigned int set_R_size = grid_ref.R.size();

        unsigned int float_size = this->getFloatSize();

        // OUR SOLUTION IS ARRANGED IN THIS FASHION:
        //  { Q\B D O R } where B = union(D, O) and Q = union(Q\B D O)
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

                // Synchronize just the R part on GPU (CL_TRUE here indicates we dont block on write
                // NOTE: offset parameter to enqueueWriteBuffer is ONLY for the GPU side offset. The CPU offset needs to be managed directly on the CPU pointer: &U_G[offset_cpu]
                err = queue.enqueueWriteBuffer(gpu_vec, CL_TRUE, offset_to_set_R * float_size, set_R_bytes, &r_update_f[0], NULL, &event);
//                queue.flush();
                queue.finish();

        }
}

//----------------------------------------------------------------------

// General routine to copy the set R indices vec up to gpu_vec
void TimeDependentPDE_CL::syncSetRDouble(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {
    std::cout << "Upload Received\n"; 
        unsigned int nb_nodes = grid_ref.getNodeListSize();
        unsigned int set_G_size = grid_ref.G.size();
        unsigned int set_Q_size = grid_ref.Q.size();
        unsigned int set_R_size = grid_ref.R.size();

        unsigned int float_size = sizeof(double);

        // OUR SOLUTION IS ARRANGED IN THIS FASHION:
        //  { Q\B D O R } where B = union(D, O) and Q = union(Q\B D O)
        unsigned int offset_to_set_R = set_Q_size;

        unsigned int solution_mem_bytes = set_G_size*float_size;
        unsigned int set_R_bytes = set_R_size * float_size;

        if (set_R_size > 0) {

                // Synchronize just the R part on GPU (CL_TRUE here indicates we dont
                // block on write NOTE: offset parameter to enqueueWriteBuffer is ONLY
                // for the GPU side offset. The CPU offset needs to be managed directly
                // on the CPU pointer: &U_G[offset_cpu]
                err = queue.enqueueWriteBuffer(gpu_vec, CL_TRUE, offset_to_set_R * float_size, set_R_bytes, &vec[offset_to_set_R], NULL, &event);
//                queue.flush();
                queue.finish();

        }
}

//----------------------------------------------------------------------

// General routine to copy the set O indices from gpu_vec down to vec
void TimeDependentPDE_CL::syncSetOSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec) {

    exit(EXIT_FAILURE);
        unsigned int nb_nodes = grid_ref.getNodeListSize();
        unsigned int set_G_size = grid_ref.G.size();
        unsigned int set_Q_size = grid_ref.Q.size();
        unsigned int set_O_size = grid_ref.O.size();

        unsigned int float_size = this->getFloatSize();

        // OUR SOLUTION IS ARRANGED IN THIS FASHION:
        //  { Q\B D O R } where B = union(D, O) and Q = union(Q\B D O)
        unsigned int offset_to_set_O = (set_Q_size - set_O_size);

        unsigned int solution_mem_bytes = set_G_size*float_size;
        unsigned int set_O_bytes = set_O_size * float_size;

        // backup the current solution so we can perform intermediate steps
        std::vector<float> o_update_f(set_O_size,1.);


        if (set_O_size > 0) {
                // Pull only information required for neighboring domains back to the CPU
                err = queue.enqueueReadBuffer(gpu_vec, CL_TRUE, offset_to_set_O * float_size, set_O_bytes, &o_update_f[0], NULL, &event);
//                queue.flush();

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
    std::cout << "Download from GPU\n"; 
        unsigned int nb_nodes = grid_ref.getNodeListSize();
        unsigned int set_G_size = grid_ref.G.size();
        unsigned int set_Q_size = grid_ref.Q.size();
        unsigned int set_O_size = grid_ref.O.size();

        unsigned int float_size = sizeof(double);

        // OUR SOLUTION IS ARRANGED IN THIS FASHION:
        //  { Q\B D O R } where B = union(D, O) and Q = union(Q\B D O)
        unsigned int offset_to_set_O = (set_Q_size - set_O_size);

        unsigned int solution_mem_bytes = set_G_size*float_size;
        unsigned int set_O_bytes = set_O_size * float_size;

#if 0
        for (unsigned int i = offset_to_set_O - 5; i < set_G_size; i++) {
            std::cout << "vec[" << set_O_size << "," << i << "] = " << vec[i] << std::endl;
        }
#endif 

        if (set_O_size > 0) {
                // Pull only information required for neighboring domains back to the CPU
                err = queue.enqueueReadBuffer(gpu_vec, CL_TRUE, offset_to_set_O * float_size, set_O_bytes, &vec[offset_to_set_O], NULL, &event);
//                queue.flush();
//
                queue.finish();
        }


}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::syncCPUtoGPU() {

    if (cpu_dirty) {
        std::cout << "*************** FULL MEMCOPY DEVICE -> HOST ***************\n"; 
        unsigned int nb_nodes = grid_ref.getNodeListSize();
        unsigned int solution_mem_bytes = nb_nodes * this->getFloatSize();

        //std::cout << "SYNC CPU to GPU: " << solution_mem_bytes << " bytes, from index: " << INDX_IN << std::endl;

        if (useDouble) {
            err = queue.enqueueReadBuffer(gpu_solution[INDX_OUT], CL_TRUE, 0, solution_mem_bytes, &U_G[0], NULL, &event);
            //                queue.flush();
            queue.finish();
        } else {
            float* U_G_f = new float[nb_nodes];
            err = queue.enqueueReadBuffer(gpu_solution[INDX_OUT], CL_TRUE, 0, solution_mem_bytes, &U_G_f[0], NULL, &event);
            //                queue.flush();
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
        cpu_dirty = 0;
    }
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::advance(TimeScheme which, double delta_t) {
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
#endif 
        case RK4: 
                advanceRK4(delta_t);
                break;

        default: 
                std::cout << "[TimeDependentPDE_CL] Invalid TimeScheme specified. Bailing...\n";
         
                exit(EXIT_FAILURE);
                break;
        };
        cur_time += delta_t;
        cpu_dirty = 1;
        tm["advance_gpu"]->stop();
}

//----------------------------------------------------------------------

// FIXME: this is a single precision version
void TimeDependentPDE_CL::advanceFirstOrderEuler(double delta_t) {
    std::cout << "Advance Euler\n";
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

        queue.finish();

        // 1) Launch kernel for set QmD (will take a while, so in the meantime...)
        this->launchEulerKernel(0, grid_ref.Q.size(), delta_t, this->gpu_solution[INDX_IN], this->gpu_solution[INDX_OUT]);
        queue.finish();

        // Synchronize the solution output
        if (comm_ref.getSize() > 1) {
        //    std::cout << "INSIDE EULER set D STUFF (useDouble = " << useDouble << ")\n";
#if 1
            // 2) OVERLAP: Transfer set O from the input to the CPU for synchronization acros CPUs
            // (these are input values required by other procs)
            if (useDouble) {
                this->syncSetODouble(this->U_G, gpu_solution[INDX_OUT]);
            } else {
                this->syncSetOSingle(this->U_G, gpu_solution[INDX_OUT]);
            }
#endif 

            // 3) OVERLAP: Transmit between CPUs
            // NOTE: Require an MPI barrier here
            this->sendrecvUpdates(this->U_G, "Euler Output");
#if 1
            // 4) OVERLAP: Update the input with set R
            // (these are input values received from other procs)
            if (useDouble) {
                this->syncSetRDouble(this->U_G, gpu_solution[INDX_OUT]);
            } else {
                this->syncSetRSingle(this->U_G, gpu_solution[INDX_OUT]);
            }
#endif 
        }
        
        // Copies from INDX_IN to U_G
//        this->syncCPUtoGPU();
}

//----------------------------------------------------------------------
//
// FIXME: this is a single precision version
void TimeDependentPDE_CL::advanceSecondOrderMidpoint(double delta_t) {
#if 0
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
        this->launchStepKernel( 0.5*delta_t, this->gpu_solution[INDX_IN], this->gpu_solution[INDX_IN], this->gpu_solution[INDX_K1] );

        // Enforce boundary using GPU, but specify we want to use the intermediate buffer
        this->enforceBoundaryConditions(U_G, this->gpu_solution[INDX_K1], cur_time+0.5*delta_t);

        // Since our syncSet****(..) routines ONLY sync the sets at the tail end of
        // the solution (i.e., sets O and R),
        // we'll just re-use U_G as scratch space. So long as we dont copy U_G to
        // the GPU calling syncSet*** on our INDX_OUT will overwrite any
        // intermediate values stored there temporarily
        // If we want to match the GPU we
        // should do: syncCPUtoGPU()
        if (useDouble) {
                this->syncSetODouble(this->U_G, gpu_solution[INDX_K1]);
        } else {
                this->syncSetOSingle(this->U_G, gpu_solution[INDX_K1]);
        }

        // Should send intermediate steps by copying down from GPU, sending, then
        // copying back up to GPU
        this->sendrecvUpdates(this->U_G, "intermediate_U_G");

        if (useDouble) {
                this->syncSetRDouble(this->U_G, gpu_solution[INDX_K1]);
        } else {
                this->syncSetRSingle(this->U_G, gpu_solution[INDX_K1]);
        }

        this->launchStepKernel( delta_t, this->gpu_solution[INDX_IN], this->gpu_solution[INDX_K1], this->gpu_solution[INDX_OUT] );
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


//----------------------------------------------------------------------


void TimeDependentPDE_CL::advanceRK4(double delta_t) {

        // If we need to assemble a matrix L for solving implicitly, this is the routine to do that.
        // For explicit schemes we can just solve for our weights and have them stored in memory.
        this->assemble();

#if 1
    // NOTE: when run in serial only one kernel launch is required.
    if (comm_ref.getSize() > 1) {
        if (useDouble) {
            this->syncSetODouble(this->U_G, gpu_solution[INDX_IN]);
        } else {
            this->syncSetOSingle(this->U_G, gpu_solution[INDX_IN]);
        }
        this->sendrecvUpdates(this->U_G, "u_out");
        if (useDouble) {
            this->syncSetRDouble(this->U_G, gpu_solution[INDX_IN]);
        } else {
            this->syncSetRSingle(this->U_G, gpu_solution[INDX_IN]);
        }

        // At this point gpu_solution[indx_u_out] contains the complete
        // solution u_out[{Q R}] which would allow us to compute the next
        // iteration
    }
#endif 
        // ----------------------------------
        //
        //    k1 = dt*func(DM_Lambda, DM_Theta, H, u, t, nodes, useHV);
        //    k2 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F1, t+0.5*dt, nodes, useHV);
        //    k3 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F2, t+0.5*dt, nodes, useHV);
        //    k4 = dt*func(DM_Lambda, DM_Theta, H, u+F3, t+dt, nodes, useHV);
        //
        // ----------------------------------
        // NOTE: INDX_IN maps to "u", INDX_K1 to "F1" and INDX_TEMP to "u+0.5*F1"

        // Compute K1, K2, K3 and K4 in separate kernel launches (required to ensure global barrier between evaluations)
        evaluateRK4_WithComm(INDX_IN, INDX_IN, INDX_K1, INDX_TEMP1, delta_t, cur_time, 0.5);  
        // We use INDX_TEMP1 (== u+0.5*K1) to compute K2 and write INDX_TEMP2 with "u+0.5*K2"
        evaluateRK4_WithComm(INDX_IN, INDX_TEMP1, INDX_K2, INDX_TEMP2, delta_t, cur_time+0.5*delta_t, 0.5);  
        // We use INDX_TEMP2 (== u+0.5*K2) to compute K3 and write INDX_TEMP1 with "u+K3"
        evaluateRK4_WithComm(INDX_IN, INDX_TEMP2, INDX_K3, INDX_TEMP1, delta_t, cur_time+0.5*delta_t, 1.0);  
        // We use INDX_TEMP1 (== u+K3) to compute K4 and write INDX_TEMP1 with "u" (we can test our logic 
        evaluateRK4_WithComm(INDX_IN, INDX_TEMP1, INDX_K4, INDX_TEMP2, delta_t, cur_time+delta_t, 0.0);  

        // Finally, we combine all terms to get the update to u
        advanceRK4_WithComm(INDX_IN, INDX_K1, INDX_K2, INDX_K3, INDX_K4, INDX_OUT);
      
        queue.finish();

        //this->syncCPUtoGPU();
        //this->sendrecvUpdates(U_G, "U_G");

        //exit(EXIT_FAILURE);

        swap(INDX_IN, INDX_OUT);
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::loadKernels(std::string& local_sources) {
        tm["loadAttach"]->start();

#if 0
        this->loadBCKernel(local_sources);
#endif 

        this->loadEulerKernel(local_sources);
        this->loadRK4Kernels(local_sources);
#if 0
        this->loadMidpointKernel(local_sources);
#endif
        tm["loadAttach"]->stop();
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::loadEulerKernel(std::string& local_sources) {
        std::string kernel_name = "advanceEuler";

        if (!this->getDeviceFP64Extension().compare("")){
                useDouble = false;
        }
        if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
                useDouble = false;
        }

        // The true here specifies we search throught the dir specified by environment variable CL_KERNELS
        std::string my_source = this->loadFileContents("euler_general.cl", true);

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

void TimeDependentPDE_CL::loadMidpointKernel(std::string& local_sources) {
        std::string kernel_name = "advanceMidpoint";

        if (!this->getDeviceFP64Extension().compare("")){
                useDouble = false;
        }
        if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
                useDouble = false;
        }

#if 0
        std::string my_source = local_sources;
        if(useDouble) {
                // This keeps FLOAT scoped
#define FLOAT double
#include "cl_kernels/midpoint_general.cl"
                my_source.append(kernel_source);
#undef FLOAT
        }else {
#define FLOAT float
#include "cl_kernels/midpoint_general.cl"
                my_source.append(kernel_source);
#undef FLOAT
        }
#endif

        // The true here specifies we search throught the dir specified by environment variable CL_KERNELS
        std::string my_source = this->loadFileContents("midpoint_general.cl", true);
        ;
        //std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n";
        this->loadProgram(my_source, useDouble);

        try{
                std::cout << "Loading kernel \""<< kernel_name << "\" with double precision = " << useDouble << "\n";
                midpoint_kernel = cl::Kernel(program, kernel_name.c_str(), &err);
                std::cout << "Done attaching kernels!" << std::endl;
        }
        catch (cl::Error er) {
                printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
        }
}


//----------------------------------------------------------------------

void TimeDependentPDE_CL::loadRK4Kernels(std::string& local_sources) {

        std::string rk4_substep_kernel_name  = "evaluateRK4_classic";
        std::string rk4_advance_substep_kernel_name = "advanceRK4_classic";

        if (!this->getDeviceFP64Extension().compare("")){
                useDouble = false;
        }
        if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
                useDouble = false;
        }

        // The true here specifies we search throught the dir specified by environment variable CL_KERNELS
        std::string my_source = this->loadFileContents("rk4_classic.cl", true);

        //std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n";
        this->loadProgram(my_source, useDouble);

        try{
                std::cout << "Loading kernel \""<< rk4_substep_kernel_name << "\" with double precision = " << useDouble << "\n";
                rk4_substep_kernel = cl::Kernel(program, rk4_substep_kernel_name.c_str(), &err);

                std::cout << "Loading kernel \""<< rk4_advance_substep_kernel_name << "\" with double precision = " << useDouble << "\n";
                rk4_advance_substep_kernel = cl::Kernel(program, rk4_advance_substep_kernel_name.c_str(), &err);
                std::cout << "Done attaching kernels!" << std::endl;
        }
        catch (cl::Error er) {
                printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
        }
}

//----------------------------------------------------------------------
//
void TimeDependentPDE_CL::allocateGPUMem() {
        unsigned int nb_nodes = grid_ref.getNodeListSize();
        unsigned int nb_stencils = grid_ref.getStencilsSize();
        unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();

        cout << "Allocating GPU memory for TimeDependentPDE\n";

        unsigned int solution_mem_bytes = nb_nodes * this->getFloatSize();

        unsigned int bytesAllocated = 0;

        gpu_solution[INDX_IN] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
        bytesAllocated += solution_mem_bytes;
        std::cout << "Done with first buffer: " << nb_nodes << "*" << this->getFloatSize() << " bytes\n";
        gpu_solution[INDX_OUT] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_K1] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_K2] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_K3] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_K4] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_TEMP1] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_TEMP2] = cl::Buffer(context, CL_MEM_READ_WRITE, solution_mem_bytes, NULL, &err);
        bytesAllocated += solution_mem_bytes;

        std::cout << "[TimeDependentPDE_CL] Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::setAdvanceArgs(cl::Kernel kern, int argc_start) {

        unsigned int stencil_size = grid_ref.getMaxStencilSize();
        unsigned int nb_nodes = grid_ref.getNodeListSize();

        // Subtract 1 to make sure our ++ below works out correctly;
        int i = argc_start;
        std::cout << "SETTING THE ARGUMENTS FOR GPU TIME SCHEMES\n";

        try {
                kern.setArg(i++, der_ref_gpu.getGPUNodes());
                kern.setArg(i++, der_ref_gpu.getGPUStencils());

                kern.setArg(i++, der_ref_gpu.getGPUWeights(RBFFD::X));
                kern.setArg(i++, der_ref_gpu.getGPUWeights(RBFFD::Y));
                kern.setArg(i++, der_ref_gpu.getGPUWeights(RBFFD::Z));
                kern.setArg(i++, der_ref_gpu.getGPUWeights(RBFFD::LAPL));
                kern.setArg(i++, der_ref_gpu.getGPUWeights(RBFFD::R));
                kern.setArg(i++, der_ref_gpu.getGPUWeights(RBFFD::LAMBDA));
                kern.setArg(i++, der_ref_gpu.getGPUWeights(RBFFD::THETA));
                kern.setArg(i++, der_ref_gpu.getGPUWeights(RBFFD::HV));

                kern.setArg(i++, nb_nodes);
                kern.setArg(i++, stencil_size);
                kern.setArg(i++, der_ref.getUseHyperviscosity());

        } catch (cl::Error er) {
                printf("[TimeDependentPDE_CL::setAdvanceArgs] ERROR: %s(%s) (arg index: %d)\n", er.what(), oclErrorString(er.err()), i);
        }
}



//----------------------------------------------------------------------

// Launch a kernel to perform the step: 
//      u(n+1) = u(n) + dt * f( u(n) )
// Params: 
//  dt => Timestep
//  sol_in => parameter u(n) above
//  sol_out => parameter u(n+1) above
void TimeDependentPDE_CL::launchEulerKernel( unsigned int offset_to_set, unsigned int nb_stencils_to_compute, double dt, cl::Buffer& sol_in, cl::Buffer& sol_out) {

    std::cout << "Launch Euler Kernel\n";
        unsigned int n_stencils = grid_ref.getStencilsSize();
        float dt_f = (float) dt;
        float cur_time_f = (float) cur_time;

        int i = 0;

        try {
                // These will change each iteration
                euler_kernel.setArg(i++, sol_in);
                euler_kernel.setArg(i++, sol_out);
                euler_kernel.setArg(i++, dt);
                euler_kernel.setArg(i++, cur_time);

                // We should only do this on the first iter
                if (!euler_args_set) {
                        euler_kernel.setArg(i++, offset_to_set);
                        euler_kernel.setArg(i++, nb_stencils_to_compute);

                        this->setAdvanceArgs(euler_kernel, i);
                        euler_args_set++;
                }
                err = queue.enqueueNDRangeKernel(euler_kernel, /* offset */ cl::NullRange,
                                                 /* GLOBAL (work-groups in the grid)  */   cl::NDRange(nb_stencils_to_compute),
                                                 /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);

                queue.finish();
//                err = queue.flush();
                if (err != CL_SUCCESS) {
                        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
                                     " failed (" << err << ")\n";
                        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
                        exit(EXIT_FAILURE);
                }
        } catch (cl::Error er) {
                printf("[launchEulerKernel] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

}

//----------------------------------------------------------------------


// Launch a kernel to perform the step: 
//      u(n+1) = u(n) + dt * f( U(n) )
// Params: 
//  dt => Timestep
//  sol_in => parameter u(n) above
//  deriv_sol_in => parameter U(n) above
//  sol_out => parameter u(n+1) above
//
void TimeDependentPDE_CL::launchStepKernel( double dt, cl::Buffer& sol_in, cl::Buffer& deriv_sol_in, cl::Buffer& sol_out, unsigned int n_stencils, unsigned int offset_to_set) {
    exit(EXIT_FAILURE);
#if 0
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
//        queue.flush();

        //    err = queue.finish();
        if (err != CL_SUCCESS) {
                std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
                             " failed (" << err << ")\n";
                std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
                exit(EXIT_FAILURE);
        }
#endif
}

//----------------------------------------------------------------------
// NOTE: the communciation in this routine is to synchronize the input to the INTERMEDIATE STEPS, not the solution
void TimeDependentPDE_CL::evaluateRK4_WithComm(int indx_u_in, int indx_u_plus_scaled_k_in, int indx_k_out, int indx_u_plus_scaled_k_out, double del_t, double adjusted_time, double substep_scale)
{
    // NOTE: when run in serial only one kernel launch is required.
    if (comm_ref.getSize() > 1) {

        // Since our syncSet****(..) routines ONLY sync the sets at the tail end of
        // the solution (i.e., sets O and R),
        // we'll just re-use U_G as scratch space. So long as we dont copy U_G to
        // the GPU calling syncSet*** on our INDX_OUT will overwrite any
        // intermediate values stored there temporarily
        // If we want to match the GPU we
        // should do: syncCPUtoGPU()
        if (useDouble) {
            this->syncSetODouble(this->U_G, gpu_solution[indx_u_plus_scaled_k_in]);
        } else {
            this->syncSetOSingle(this->U_G, gpu_solution[indx_u_plus_scaled_k_in]);
        }
        queue.finish(); 

        // Should send intermediate steps by copying down from GPU, sending, then
        // copying back up to GPU. 
        // NOTE: we use U_G here because its scratch space
        this->sendrecvUpdates(this->U_G, "u_plus_scaled_k_in");

        if (useDouble) {
            this->syncSetRDouble(this->U_G, gpu_solution[indx_u_plus_scaled_k_in]);
        } else {
            this->syncSetRSingle(this->U_G, gpu_solution[indx_u_plus_scaled_k_in]);
        }
        queue.finish(); 

#if 0
        this->launchRK4_eval(grid_ref.QmD.size(), grid_ref.D.size(), adjusted_time, del_t, this->gpu_solution[indx_u_in], this->gpu_solution[indx_u_plus_scaled_k_in], this->gpu_solution[indx_k_out], this->gpu_solution[indx_u_plus_scaled_k_out], substep_scale);
        
        queue.finish();
#endif 
    }

#if 0
    this->launchRK4_eval(0, grid_ref.QmD.size(), adjusted_time, del_t, this->gpu_solution[indx_u_in], this->gpu_solution[indx_u_plus_scaled_k_in], this->gpu_solution[indx_k_out], this->gpu_solution[indx_u_plus_scaled_k_out], substep_scale);
#else 
    this->launchRK4_eval(0, grid_ref.Q.size(), adjusted_time, del_t, this->gpu_solution[indx_u_in], this->gpu_solution[indx_u_plus_scaled_k_in], this->gpu_solution[indx_k_out], this->gpu_solution[indx_u_plus_scaled_k_out], substep_scale);
#endif 

    queue.finish();


}

//----------------------------------------------------------------------
// NOTE: the communciation in this routine is to synchronize the solution OUTPUT (not intermediate steps)
void TimeDependentPDE_CL::advanceRK4_WithComm( int indx_u_in, int indx_k1, int indx_k2, int indx_k3, int indx_k4, int indx_u_out ) {

    // Advane only needs K1[{Q}], K2[{Q}], K3[{Q}] and K4[{Q}]
    // RK4 requires U[{Q, R}] going into the method. This implies we need to transfer U[{R}] at the end of the iteration.
    this->launchRK4_adv(0, grid_ref.Q.size(), this->gpu_solution[indx_u_in], this->gpu_solution[indx_k1],this->gpu_solution[indx_k2],this->gpu_solution[indx_k3],this->gpu_solution[indx_k4],this->gpu_solution[indx_u_out]);
        
    queue.finish();

#if 0
    // Enforce boundary using GPU, but specify we want to use the intermediate buffer
    this->enforceBoundaryConditions(U_G, this->gpu_solution[INDX_K1], cur_time+0.5*delta_t);
#endif 

#if 0
    // NOTE: when run in serial only one kernel launch is required.
    if (comm_ref.getSize() > 1) {
        if (useDouble) {
            this->syncSetODouble(this->U_G, gpu_solution[indx_u_out]);
        } else {
            this->syncSetOSingle(this->U_G, gpu_solution[indx_u_out]);
        }
        this->sendrecvUpdates(this->U_G, "u_out");
        if (useDouble) {
            this->syncSetRDouble(this->U_G, gpu_solution[indx_u_out]);
        } else {
            this->syncSetRSingle(this->U_G, gpu_solution[indx_u_out]);
        }

        // At this point gpu_solution[indx_u_out] contains the complete
        // solution u_out[{Q R}] which would allow us to compute the next
        // iteration
    }
#endif 
}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::launchRK4_eval( unsigned int offset_to_set, unsigned int nb_stencils_to_compute, double adjusted_t, double dt, cl::Buffer& u_in, cl::Buffer& u_plus_scaled_k_in,  cl::Buffer& k_out,  cl::Buffer& u_plus_scaled_k_out, double substep_scale) {

        unsigned int n_stencils = grid_ref.getStencilsSize();
        float dt_f = (float) dt;
        float cur_time_f = (float) adjusted_t;
        float substep_scale_f = (float) substep_scale;

        int i = 0;

        try {
                // These will change each iteration
                rk4_substep_kernel.setArg(i++, u_in);
                rk4_substep_kernel.setArg(i++, u_plus_scaled_k_in);
                rk4_substep_kernel.setArg(i++, k_out);
                rk4_substep_kernel.setArg(i++, u_plus_scaled_k_out);
                rk4_substep_kernel.setArg(i++, dt);
                rk4_substep_kernel.setArg(i++, adjusted_t);
                rk4_substep_kernel.setArg(i++, substep_scale);

                // We should only do this on the first iter
                if (!rk4_sub_args_set) {
                        rk4_substep_kernel.setArg(i++, offset_to_set); 
                        rk4_substep_kernel.setArg(i++, nb_stencils_to_compute);

                        this->setAdvanceArgs( rk4_substep_kernel, i);
                        rk4_sub_args_set++;
                }
                err = queue.enqueueNDRangeKernel( rk4_substep_kernel, /* offset */ cl::NullRange,
                                                 /* GLOBAL (work-groups in the grid)  */   cl::NDRange(nb_stencils_to_compute),
                                                 /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);

//                err = queue.flush();
                if (err != CL_SUCCESS) {
                        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
                                     " failed (" << err << ")\n";
                        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
                        exit(EXIT_FAILURE);
                }
        } catch (cl::Error er) {
                printf("[launchRK4_eval] ERROR: %s(%s) (arg index: %d)\n", er.what(), oclErrorString(er.err()), i);
        }

}

//----------------------------------------------------------------------

void TimeDependentPDE_CL::launchRK4_adv( unsigned int offset_to_set, unsigned int nb_stencils_to_compute, cl::Buffer& u_in, cl::Buffer& k1, cl::Buffer k2, cl::Buffer& k3, cl::Buffer& k4, cl::Buffer& u_out)
{
        int i = 0;

        try {
                // These will change each iteration
                rk4_advance_substep_kernel.setArg(i++, u_in);
                rk4_advance_substep_kernel.setArg(i++, k1);
                rk4_advance_substep_kernel.setArg(i++, k2);
                rk4_advance_substep_kernel.setArg(i++, k3);
                rk4_advance_substep_kernel.setArg(i++, k4);
                rk4_advance_substep_kernel.setArg(i++, u_out);

                // We should only do this on the first iter
                if (!rk4_adv_args_set) {
                        rk4_advance_substep_kernel.setArg(i++, offset_to_set);
                        rk4_advance_substep_kernel.setArg(i++, nb_stencils_to_compute);

                        this->setAdvanceArgs( rk4_advance_substep_kernel, i);
                        rk4_adv_args_set++;
                }
                err = queue.enqueueNDRangeKernel( rk4_advance_substep_kernel, /* offset */ cl::NullRange,
                                                 /* GLOBAL (work-groups in the grid)  */   cl::NDRange(nb_stencils_to_compute),
                                                 /* LOCAL (work-items per work-group) */    cl::NullRange, NULL, &event);

//                err = queue.flush();
                if (err != CL_SUCCESS) {
                        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
                                     " failed (" << err << ")\n";
                        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
                        exit(EXIT_FAILURE);
                }
        } catch (cl::Error er) {
                printf("[launchRK4_adv] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

}

//----------------------------------------------------------------------

