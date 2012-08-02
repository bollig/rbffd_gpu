
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/prod.hpp> 
#include <viennacl/io/matrix_market.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp> 
#include <viennacl/vector_proxy.hpp> 
#include <viennacl/linalg/vector_operations.hpp> 


#include "time_dependent_pde_vcl.h"

#include <iomanip>
#include <iostream>
#include <sstream> 
#include <map>
#include <fstream> 
#include <typeinfo> 
using namespace std;

//----------------------------------------------------------------------

void TimeDependentPDE_VCL::setupTimers()
{
        tm["initialize"] = new EB::Timer("[T_PDE_VCL] Fill initial conditions, weights on GPU");
        tm["gpu_dump"] = new EB::Timer("[T_PDE_VCL] Full copy solution GPU to CPU");
        tm["advance_gpu"] = new EB::Timer("[T_PDE_VCL] Advance the PDE one step on the GPU") ;
        tm["assemble_gpu"] = new EB::Timer("[T_PDE_VCL] Assemble (flip ping-pong buffers)");
        tm["rk4_adv_gpu"] = new EB::Timer("[T_PDE_VCL] RK4 Advance on GPU") ;
        tm["rk4_eval_gpu"] = new EB::Timer("[T_PDE_VCL] RK4 Evaluate Substep on GPU"); 
        tm["rk4_adv_setargs"] = new EB::Timer("[T_PDE_VCL] RK4 Adv Setargs") ;
        tm["rk4_eval_setargs"] = new EB::Timer("[T_PDE_VCL] RK4 Eval Setargs"); 
        tm["rk4_adv_kern"] = new EB::Timer("[T_PDE_VCL] RK4 Adv Kernel Launch (w/ q.finish)") ;
        tm["rk4_eval_kern"] = new EB::Timer("[T_PDE_VCL] RK4 Eval Kernel Launch (w/ q.finish)"); 
        tm["rk4_full_comm"] = new EB::Timer("[T_PDE_VCL] RK4 Communicate GPU>CPU>CPU>GPU"); 
        tm["rk4_O"] = new EB::Timer("[T_PDE_VCL] RK4 Transfer Set O (GPU to CPU)"); 
        tm["rk4_R"] = new EB::Timer("[T_PDE_VCL] RK4 Transfer Set R (CPU to GPU)"); 
        tm["rk4_mpi_comm_cl"] = new EB::Timer("[T_PDE_VCL] RK4 MPI Comm (Including Wait)");
        tm["loadAttach"] = new EB::Timer("[T_PDE_VCL] Load the GPU Kernels for TimeDependentPDE_VCL");
}

//----------------------------------------------------------------------

#if 1
void TimeDependentPDE_VCL::fillInitialConditions(ExactSolution* exact) {
    tm["initialize"]->start();
    // Fill U_G with initial conditions
    this->TimeDependentPDE::fillInitialConditions(exact);

// EB: this is unnecessary
// this->sendrecvUpdates(U_G, "U_G");

    unsigned int nb_nodes = grid_ref.G.size();
    unsigned int solution_mem_bytes = nb_nodes*this->getFloatSize();

    std::cout << "[TimeDependentPDE_VCL] Writing initial conditions to GPU\n";
    // Fill GPU mem with initial solution
    //err = queue.enqueueWriteBuffer(gpu_solution[INDX_OUT], CL_TRUE, 0, solution_mem_bytes, &U_G[0], NULL, &event);
    viennacl::copy(U_G, *(gpu_solution[INDX_OUT]));


    if (!this->assembled) {
        if (!weightsPrecomputed) {
            der_ref_gpu.computeAllWeightsForAllStencils();
            weightsPrecomputed = true;
        }
        // This will avoid multiple writes to GPU if the latest version is already in place
        // FIXME: allow this to finish later
        der_ref_gpu.updateWeightsOnGPU(false);
        assembled = true;
    }


    tm["initialize"]->stop();
    std::cout << "[TimeDependentPDE_VCL] Done\n";
}
#endif 


//----------------------------------------------------------------------

void TimeDependentPDE_VCL::assemble() 
{

   tm["assemble_gpu"]->start(); 
    // Flip our ping pong buffers.
    // NOTE: we need to initialize into INDX_OUT
    swap(INDX_IN, INDX_OUT);
    tm["assemble_gpu"]->stop(); 
}

//----------------------------------------------------------------------

int TimeDependentPDE_VCL::sendrecvBuf(VCL_VEC_t* buf, std::string label) {

    tm["rk4_full_comm"]->start();
    // Synchronize the solution output
    if (comm_ref.getSize() > 1) {
        // 2) OVERLAP: Transfer set O from the input to the CPU for synchronization acros CPUs
        // (these are input values required by other procs)
        tm["rk4_O"]->start();
        this->syncSetODouble(this->cpu_buf, buf);
        tm["rk4_O"]->stop();

        // 3) OVERLAP: Transmit between CPUs
        // NOTE: Require an MPI barrier here
        tm["rk4_mpi_comm_cl"]->start(); 
        this->sendrecvUpdates(this->cpu_buf, label);
        tm["rk4_mpi_comm_cl"]->stop(); 

        tm["rk4_R"]->start();
        // 4) OVERLAP: Update the input with set R
        // (these are input values received from other procs)
        this->syncSetRDouble(this->cpu_buf, buf);
        tm["rk4_R"]->stop();
    }
    tm["rk4_full_comm"]->stop();

    return 0; 
}

//----------------------------------------------------------------------

// General routine to copy the set R indices vec up to gpu_vec
void TimeDependentPDE_VCL::syncSetRDouble(std::vector<SolutionType>& vec, VCL_VEC_t* gpu_vec) {
        //unsigned int nb_nodes = grid_ref.getNodeListSize();
        //unsigned int set_G_size = grid_ref.G.size();
        unsigned int set_Q_size = grid_ref.Q.size();
        //unsigned int set_O_size = grid_ref.O.size();
        unsigned int set_R_size = grid_ref.R.size();

        unsigned int float_size = sizeof(double);

        // OUR SOLUTION IS ARRANGED IN THIS FASHION:
        //  { Q\B D O R } where B = union(D, O) and Q = union(Q\B D O)
        unsigned int offset_to_set_R = set_Q_size;

        unsigned int set_R_bytes = set_R_size * float_size; //set_R_size * float_size;

        if (set_R_size > 0) {

                // Synchronize just the R part on GPU (CL_TRUE here indicates we dont
                // block on write NOTE: offset parameter to enqueueWriteBuffer is ONLY
                // for the GPU side offset. The CPU offset needs to be managed directly
                // on the CPU pointer: &U_G[offset_cpu]
//                err = queue.enqueueWriteBuffer(gpu_vec, CL_TRUE, offset_to_set_R * float_size, set_R_bytes, &vec[offset_to_set_R], NULL, &event);
//                queue.flush();
//                                *gpu_vec_view_R = *cpu_vec_view_R; 
//                queue.finish();


        }
#if 0

        for (unsigned int i = offset_to_set_O - 5; i < set_G_size; i++) {
            std::cout << "vec[" << set_O_size << "," << i << "] = " << vec[i] << std::endl;
        }
#endif 
}

//----------------------------------------------------------------------

void TimeDependentPDE_VCL::syncSetODouble(std::vector<SolutionType>& vec, VCL_VEC_t* gpu_vec) {
//    std::cout << "Download from GPU\n"; 
        //unsigned int nb_nodes = grid_ref.getNodeListSize();
        //unsigned int set_G_size = grid_ref.G.size();
        unsigned int set_Q_size = grid_ref.Q.size();
        //unsigned int set_QmB_size = grid_ref.QmB.size();
        //unsigned int set_BmO_size = grid_ref.BmO.size();
        //unsigned int set_B_size = grid_ref.B.size();
        //unsigned int set_D_size = grid_ref.D.size();
        unsigned int set_O_size = grid_ref.O.size();

        unsigned int float_size = sizeof(double);

        // OUR SOLUTION IS ARRANGED IN THIS FASHION:
        //  { Q\B D O R } where B = union(D, O) and Q = union(Q\B D O)
        //  Minus 1 because we start indexing at 0 

        unsigned int offset_to_set_O = set_Q_size - set_O_size;
        unsigned int set_O_bytes = set_O_size * float_size; 

        if (set_O_size > 0) {
                // Pull only information required for neighboring domains back to the CPU
                //               cpu_view_O = gpu_view_O
                //queue.finish();
        }
}

//----------------------------------------------------------------------

void TimeDependentPDE_VCL::advance(TimeScheme which, double delta_t) {
        tm["advance_gpu"]->start();
        switch (which)
        {
#if 0
            // Only supports one thread per stencil: 
        case EULER: 
                advanceFirstOrderEuler(delta_t);
                break;
#endif 
#if 0
                // INCOMPLETE: 
        case MIDPOINT: 
                advanceSecondOrderMidpoint(delta_t);
                break;
#endif 
        case RK4: 
                advanceRK4(delta_t);
                break;

        default: 
                std::cout << "[TimeDependentPDE_VCL] Invalid TimeScheme specified. Bailing...\n";
         
                exit(EXIT_FAILURE);
                break;
        };
        cur_time += delta_t;
        cpu_dirty = 1;
        tm["advance_gpu"]->stop();
}

//----------------------------------------------------------------------


void TimeDependentPDE_VCL::advanceRK4(double delta_t) {
#if 0

        // If we need to assemble a matrix L for solving implicitly, this is the routine to do that.
        // For explicit schemes we can just solve for our weights and have them stored in memory.
        this->assemble();

        // ----------------------------------
        //
        //    k1 = dt*func(DM_Lambda, DM_Theta, H, u, t, nodes, useHV);
        //    k2 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F1, t+0.5*dt, nodes, useHV);
        //    k3 = dt*func(DM_Lambda, DM_Theta, H, u+0.5*F2, t+0.5*dt, nodes, useHV);
        //    k4 = dt*func(DM_Lambda, DM_Theta, H, u+F3, t+dt, nodes, useHV);
        //
        // ----------------------------------
        // NOTE: INDX_IN maps to "u", INDX_K1 to "F1" and INDX_TEMP to "u+0.5*F1"

//        std::cout << "Eval k1\n";
        // Compute K1, K2, K3 and K4 in separate kernel launches (required to ensure global barrier between evaluations)
        evaluateRK4_NoComm(INDX_IN, INDX_IN, INDX_K1, INDX_TEMP1, delta_t, cur_time, 0.5);  
        this->sendrecvBuf(gpu_solution[INDX_TEMP1]);

 //       std::cout << "Eval k2\n";
        // We use INDX_TEMP1 (== u+0.5*K1) to compute K2 and write INDX_TEMP2 with "u+0.5*K2"
        evaluateRK4_NoComm(INDX_IN, INDX_TEMP1, INDX_K2, INDX_TEMP2, delta_t, cur_time+0.5*delta_t, 0.5);  
        this->sendrecvBuf(gpu_solution[INDX_TEMP2]);

  //      std::cout << "Eval k3\n";
        // We use INDX_TEMP2 (== u+0.5*K2) to compute K3 and write INDX_TEMP1 with "u+K3"
        evaluateRK4_NoComm(INDX_IN, INDX_TEMP2, INDX_K3, INDX_TEMP1, delta_t, cur_time+0.5*delta_t, 1.0);  
        this->sendrecvBuf(gpu_solution[INDX_TEMP1]);

   //     std::cout << "Eval k4\n";
        // We use INDX_TEMP1 (== u+K3) to compute K4 (NOTE: no communication is required at this step since we
        // wont be evaluating anymore)
        evaluateRK4_NoComm(INDX_IN, INDX_TEMP1, INDX_K4, INDX_TEMP2, delta_t, cur_time+delta_t, 0.0);  

    //    std::cout << "Advance u_n\n";
        // Finally, we combine all terms to get the update to u
        advanceRK4_NoComm(INDX_IN, INDX_K1, INDX_K2, INDX_K3, INDX_K4, INDX_OUT);
        this->sendrecvBuf(gpu_solution[INDX_OUT]);
#endif 
}

//----------------------------------------------------------------------
//
void TimeDependentPDE_VCL::allocateGPUMem() {
        unsigned int nb_nodes = grid_ref.getNodeListSize();
        //unsigned int nb_stencils = grid_ref.getStencilsSize();
        //unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();

        cout << "Allocating GPU memory for TimeDependentPDE\n";

        unsigned int solution_mem_bytes = nb_nodes * this->getFloatSize();

        unsigned int bytesAllocated = 0;

        gpu_solution[INDX_IN] = new VCL_VEC_t(nb_nodes); 
        bytesAllocated += solution_mem_bytes;
        std::cout << "Done with first buffer: " << nb_nodes << "*" << this->getFloatSize() << " bytes\n";
        gpu_solution[INDX_OUT] = new VCL_VEC_t(nb_nodes);
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_K1] = new VCL_VEC_t(nb_nodes); 
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_K2] = new VCL_VEC_t(nb_nodes); 
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_K3] =  new VCL_VEC_t(nb_nodes); 
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_K4] =  new VCL_VEC_t(nb_nodes); 
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_TEMP1] =  new VCL_VEC_t(nb_nodes); 
        bytesAllocated += solution_mem_bytes;
        gpu_solution[INDX_TEMP2] =  new VCL_VEC_t(nb_nodes); 
        bytesAllocated += solution_mem_bytes;

        std::cout << "[TimeDependentPDE_VCL] Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
}



//----------------------------------------------------------------------
