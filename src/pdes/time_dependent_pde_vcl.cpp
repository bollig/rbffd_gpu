
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
    viennacl::copy(U_G.begin(), U_G.end(), gpu_solution[INDX_OUT]->begin());
    viennacl::copy(U_G.begin(), U_G.end(), gpu_solution[INDX_IN]->begin());


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

#if 0
    std::vector<double> u_t(nb_nodes, 1.);
    std::vector<double> dh_dlambda(nb_nodes, 1.);

    der_ref_gpu.applyWeightsForDeriv(RBFFD::LAMBDA, u_t, dh_dlambda, true);

    double tot = 0.0;
    for (int i = 0; i < nb_nodes; i++) {
        std::cout << "dh_dlambda[" << i << "] = " << dh_dlambda[i] << std::endl;
        tot += dh_dlambda[i];
    }
    std::cout << "l1 norm: " << tot / nb_nodes << std::endl;
#endif

    tm["initialize"]->stop();
    std::cout << "[TimeDependentPDE_VCL] Initial Conditions Done\n";
}
#endif


//----------------------------------------------------------------------

void TimeDependentPDE_VCL::assemble()
{

    std::cout << "*********** ASSEMBLE (Swap PingPong Solution Buffer)\n";
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

// This is the NON-Overlapped version
void TimeDependentPDE_VCL::advanceRK4(double delta_t) {

    unsigned int nb_stencils = grid_ref.getStencilsSize();
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    //std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    //std::vector<double>& input_sol = this->U_G;
    VCL_VEC_t& input_sol = *(gpu_solution[INDX_IN]);

    //std::vector<double> s(nb_nodes, 0.);

    //---------------------------------
    // f(t_n, y_n)
    // f(t_n + 0.5dt, y_n + 0.5dt*k1)
    // f(t_n + 0.5dt, y_n + 0.5dt*k2)
    // f(t_n + dt, y_n + dt*k3)
    //---------------------------------
    VCL_VEC_t& k1 = *(gpu_solution[INDX_K1]);
    VCL_VEC_t& k2 = *(gpu_solution[INDX_K2]);
    VCL_VEC_t& k3 = *(gpu_solution[INDX_K3]);
    VCL_VEC_t& k4 = *(gpu_solution[INDX_K4]);

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that.
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble();

    //NOTE: between *****'s are kernel and comm.
    //*********
    // ------------------- K1 ------------------------
    // This routine will apply our weights to "s" in however many intermediate steps are required
    // and store the results in k1
    tm["rk4_eval"]->start();
    this->solve(input_sol, k1, nb_stencils, nb_nodes, cur_time);

#if 0
    // ------------------- K2 ------------------------
    for (unsigned int i = 0; i < nb_stencils; i++) {
        //NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = input_sol[i] + 0.5*dt * ( k1[i] + f);
    }
    tm["rk4_eval"]->stop();

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt);

    // FIXME: might not be necessary unless we strictly believe that enforcing
    // BC can only be done by controlling CPU
    // NOTE: increases s from nb_stencils to nb_nodes
    tm["rk4_mpi_comm"]->start();
    this->sendrecvUpdates(s, "k1");
    tm["rk4_mpi_comm"]->stop();
    //*********
    // y*(t) = y(t) + h * k2
    // but k2 = lapl[ y(t) + h/2 * k1 ] (content between [..] was computed above)
    tm["rk4_eval"]->start();
    this->solve(s, &k2, nb_stencils, nb_nodes, cur_time+0.5*dt);

    // ------------------- K3 ------------------------
    for (unsigned int i = 0; i < nb_stencils; i++) {
        //NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = input_sol[i] + 0.5*dt * ( k2[i] + f);
    }
    tm["rk4_eval"]->stop();

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt);

    // FIXME: might not be necessary unless we strictly believe that enforcing
    // BC can only be done by controlling CPU
    tm["rk4_mpi_comm"]->start();
    this->sendrecvUpdates(s, "k2");
    tm["rk4_mpi_comm"]->stop();

    //*********
    tm["rk4_eval"]->start();
    this->solve(s, &k3, nb_stencils, nb_nodes, cur_time+0.5*dt);

    // ------------------- K4 ------------------------
    for (unsigned int i = 0; i < nb_stencils; i++) {
        //NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = input_sol[i] + dt * ( k3[i] + f);
    }
    tm["rk4_eval"]->stop();

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+dt);

    // FIXME: might not be necessary unless we strictly believe that enforcing
    // BC can only be done by controlling CPU
    tm["rk4_mpi_comm"]->start();
    this->sendrecvUpdates(s, "K3");
    tm["rk4_mpi_comm"]->stop();

    //*********
    tm["rk4_eval"]->start();
    this->solve(s, &k4, nb_stencils, nb_nodes, cur_time+dt);
    tm["rk4_eval"]->stop();
    // NOTE: No more communication is required for evaluations:

    // ------------------- FINAL ------------------------
    // FINAL STEP: y_n+1 = y_n + 1/6 * h * (k1 + 2*k2 + 2*k3 + k4)
    //
    tm["rk4_adv"]->start();
    for (unsigned int i = 0; i < nb_stencils; i++) {
        //NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term
        //double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * feval1
        input_sol[i] = input_sol[i] + (dt/6.) * ( k1[i] + 2.*k2[i] + 2.*k3[i] + k4[i]);
    }
    tm["rk4_adv"]->stop();

    // Make sure any boundary conditions are met.
    this->enforceBoundaryConditions(input_sol, cur_time+dt);

    // Ensure we have consistent values across the board
    tm["rk4_mpi_comm"]->start();
    this->sendrecvUpdates(input_sol, "U_G");
    tm["rk4_mpi_comm"]->stop();

#endif
    tm["rk4_eval_gpu"]->stop();
    exit(-1);
}


#if 0

// This is the Overlapped version. NEed to debug for VCL

void TimeDependentPDE_VCL::advanceRK4(double delta_t) {
#if 1

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
#endif


//----------------------------------------------------------------------
// NOTE: the communciation in this routine is to synchronize the input to the INTERMEDIATE STEPS, not the solution
void TimeDependentPDE_VCL::evaluateRK4_NoComm(int indx_u_in, int indx_u_plus_scaled_k_in, int indx_k_out, int indx_u_plus_scaled_k_out, double del_t, double adjusted_time, double substep_scale)
{
    tm["rk4_eval_gpu"]->start();

    std::cout << "Entered evaluateRK4_NoComm\n";
//    this->launchRK4_classic_eval(0, grid_ref.Q.size(), adjusted_time, del_t, this->gpu_solution[indx_u_in], this->gpu_solution[indx_u_plus_scaled_k_in], this->gpu_solution[indx_k_out], this->gpu_solution[indx_u_plus_scaled_k_out], substep_scale);

#if 0
    VCL_MAT_t& A = *(der_ref_gpu.getGPUWeights(RBFFD::LAMBDA));
    VCL_VEC_t& u_in = *(gpu_solution[indx_u_in]);
    VCL_VEC_t& u_plus_scaled_k_in = *(gpu_solution[indx_u_plus_scaled_k_in]);
    VCL_VEC_t& k_out = *(gpu_solution[indx_k_out]);
    VCL_VEC_t& u_plus_scaled_k_out = *(gpu_solution[indx_u_plus_scaled_k_out]);

    u_in + 0.5 * dt * u_plus_scaled_k_in
    A * u_plus_scaled_k_in

        unsigned int nb_stencils = grid_ref.getStencilsSize();
    unsigned int nb_nodes = grid_ref.getNodeListSize();
    //std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    std::vector<double>& input_sol = this->U_G;

    std::vector<double> s(nb_nodes, 0.);

    VCL_VEC_t k1(nb_stencils); // f(t_n, y_n)
    VCL_VEC_t k2(nb_stencils); // f(t_n + 0.5dt, y_n + 0.5dt*k1)
    VCL_VEC_t k3(nb_stencils,0.); // f(t_n + 0.5dt, y_n + 0.5dt*k2)
    VCL_VEC_t k4(nb_stencils,0.); // f(t_n + dt, y_n + dt*k3)

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that.
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble();

    //NOTE: between *****'s are kernel and comm.
    //*********
    // ------------------- K1 ------------------------
    // This routine will apply our weights to "s" in however many intermediate steps are required
    // and store the results in k1
    tm["rk4_eval"]->start();
    this->solve(input_sol, &k1, nb_stencils, nb_nodes, cur_time);

    // ------------------- K2 ------------------------
    for (unsigned int i = 0; i < nb_stencils; i++) {
        //NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = input_sol[i] + 0.5*dt * ( k1[i] + f);
    }
    tm["rk4_eval"]->stop();

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt);

    // FIXME: might not be necessary unless we strictly believe that enforcing
    // BC can only be done by controlling CPU
    // NOTE: increases s from nb_stencils to nb_nodes
    tm["rk4_mpi_comm"]->start();
    this->sendrecvUpdates(s, "k1");
    tm["rk4_mpi_comm"]->stop();
    //*********
    // y*(t) = y(t) + h * k2
    // but k2 = lapl[ y(t) + h/2 * k1 ] (content between [..] was computed above)
    tm["rk4_eval"]->start();
    this->solve(s, &k2, nb_stencils, nb_nodes, cur_time+0.5*dt);

    // ------------------- K3 ------------------------
    for (unsigned int i = 0; i < nb_stencils; i++) {
        //NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = input_sol[i] + 0.5*dt * ( k2[i] + f);
    }
    tm["rk4_eval"]->stop();

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+0.5*dt);

    // FIXME: might not be necessary unless we strictly believe that enforcing
    // BC can only be done by controlling CPU
    tm["rk4_mpi_comm"]->start();
    this->sendrecvUpdates(s, "k2");
    tm["rk4_mpi_comm"]->stop();

    //*********
    tm["rk4_eval"]->start();
    this->solve(s, &k3, nb_stencils, nb_nodes, cur_time+0.5*dt);

    // ------------------- K4 ------------------------
    for (unsigned int i = 0; i < nb_stencils; i++) {
        //NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term
        double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * k1
        s[i] = input_sol[i] + dt * ( k3[i] + f);
    }
    tm["rk4_eval"]->stop();

    // reset boundary solution
    this->enforceBoundaryConditions(s, cur_time+dt);

    // FIXME: might not be necessary unless we strictly believe that enforcing
    // BC can only be done by controlling CPU
    tm["rk4_mpi_comm"]->start();
    this->sendrecvUpdates(s, "K3");
    tm["rk4_mpi_comm"]->stop();

    //*********
    tm["rk4_eval"]->start();
    this->solve(s, &k4, nb_stencils, nb_nodes, cur_time+dt);
    tm["rk4_eval"]->stop();
    // NOTE: No more communication is required for evaluations:

    // ------------------- FINAL ------------------------
    // FINAL STEP: y_n+1 = y_n + 1/6 * h * (k1 + 2*k2 + 2*k3 + k4)
    //
    tm["rk4_adv"]->start();
    for (unsigned int i = 0; i < nb_stencils; i++) {
        //NodeType& v = nodes[i];
        // FIXME: allow the use of a forcing term
        //double f = 0.;//force(i, v, time*dt);
        // y(t) + h/2 * feval1
        input_sol[i] = input_sol[i] + (dt/6.) * ( k1[i] + 2.*k2[i] + 2.*k3[i] + k4[i]);
    }
    tm["rk4_adv"]->stop();

    // Make sure any boundary conditions are met.
    this->enforceBoundaryConditions(input_sol, cur_time+dt);

    // Ensure we have consistent values across the board
    tm["rk4_mpi_comm"]->start();
    this->sendrecvUpdates(input_sol, "U_G");
    tm["rk4_mpi_comm"]->stop();

#endif
    tm["rk4_eval_gpu"]->stop();
    exit(-1);
}

//----------------------------------------------------------------------
/// NOTE: the communciation in this routine is to synchronize the solution OUTPUT (not intermediate steps)
void TimeDependentPDE_VCL::advanceRK4_NoComm( int indx_u_in, int indx_k1, int indx_k2, int indx_k3, int indx_k4, int indx_u_out ) {

    // Advane only needs K1[{Q}], K2[{Q}], K3[{Q}] and K4[{Q}]
    // RK4 requires U[{Q, R}] going into the method. This implies we need to transfer U[{R}] at the end of the iteration.
    tm["rk4_adv_gpu"]->start();

#if 0
    this->launchRK4_adv(0, grid_ref.Q.size(), this->gpu_solution[indx_u_in], this->gpu_solution[indx_k1],this->gpu_solution[indx_k2],this->gpu_solution[indx_k3],this->gpu_solution[indx_k4],this->gpu_solution[indx_u_out]);
#endif
    tm["rk4_adv_gpu"]->stop();
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

//    UBLAS_VEC_t zero(nb_nodes,0.);
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

