#ifndef __TIME_DEPENDENT_PDE_CL_H_
#define __TIME_DEPENDENT_PDE_CL_H__

#include "time_dependent_pde.h"

#include "utils/opencl/cl_base_class.h"
#include "rbffd/rbffd_cl.h"

class TimeDependentPDE_CL : public TimeDependentPDE, public CLBaseClass 
{
        // --------------------------------------
        //  GPU related properties and routines
        // --------------------------------------
        
    protected:         
        int INDX_IN;
        int INDX_OUT;
        // Useful for RK4
        int INDX_K1;
        int INDX_K2;
        int INDX_K3;
        int INDX_K4;
        int INDX_TEMP1; 
        int INDX_TEMP2; 


        int euler_args_set; 
        int rk4_sub_args_set;
        int rk4_adv_args_set;

        RBFFD_CL& der_ref_gpu;

        bool weightsPrecomputed; 
        bool assembled; 
        bool useDouble;

        // These are buffers for RK4 evaluations. 
        // Each of these is NB_STENCILS long
        // K1, K2, K3 and K4
        //cl::Buffer gpu_feval[4];
        // IN and OUT
        cl::Buffer gpu_solution[8];


        // RK4 requires 4 substeps with a barrier between each. 
        // To minimize barriers, we use two kernels. First, a 
        // kernel which evaluates a substep (k1, k2, k3), scales it and adds
        // it to the solution--this is the input for the next evaluation 
        // using the same routine (except k3). 
        cl::Kernel rk4_substep_kernel;
        cl::Kernel rk4_substep_block_kernel;
        // The second kernel takes the 3 substep outputs from the above kernel
        // scales, and addes them to the solution to output the new solution
        // at t+dt.
        cl::Kernel rk4_advance_substep_kernel;


        // This is for Euler and Midpoint method substeps
        cl::Kernel euler_kernel;

        cl::Kernel midpoint_kernel;

        std::vector<SolutionType> cpu_buf; 

        // Flag to indicate if syncCPUtoGPU needs to copy down from GPU (should be set to 1 if advance is called)
        int cpu_dirty;

        std::string kernel_source_file;

        int gpuType; 

    public: 
        // USE_GPU=0 (pass over this constructor), =1 (use a block approach per stencil), =2 (use a thread approach per stencil)
        TimeDependentPDE_CL(Domain* grid, RBFFD_CL* der, Communicator* comm, int gpu_type, bool weightsComputed=false) 
            : TimeDependentPDE(grid, der, comm),
              INDX_IN(0), INDX_OUT(1),
              INDX_K1(2), INDX_K2(3), INDX_K3(4), INDX_K4(5),
              INDX_TEMP1(6), INDX_TEMP2(7),
              euler_args_set(0), rk4_sub_args_set(0), rk4_adv_args_set(0),
              useDouble(true),
            // We maintain a ref to der here so we can keep it cast as an OpenCL RBFFD class
            der_ref_gpu(*der), weightsPrecomputed(weightsComputed), 
            cpu_buf(grid->G.size(), -0.00000000001),
            cpu_dirty(0), assembled(false),
            gpuType(gpu_type)
        {;}

        void setGPUType(int type) {
            gpuType=type;
        }
        
        // Fill in the initial conditions of the PDE. (overwrite the solution)
        virtual void fillInitialConditions(ExactSolution* exact=NULL);

        void initialize(const char* solve_source_file)
        {
            this->kernel_source_file = solve_source_file;
            this->setupTimers();
            this->loadKernels(); 
            this->allocateGPUMem(); 
        }

        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t) {
            // We dont actually solve independent from the time stepper. The
            // stepper will internally call to a GPU device kernel to apply the
            // DM and "solve" 
            std::cout << "[TimeDependentPDE_CL] Error: solve should not be called. The time stepper should call a device kernel for solving\n";
            exit(EXIT_FAILURE);
        };

        // Interfaces: 
        virtual void loadKernels(std::string local_solve_source=""); 
        virtual void allocateGPUMem();
        // Set the default set of arguments for a kernel
        virtual int setAdvanceArgs(cl::Kernel kern, int start_indx);

        // Use RK4 scheme to advance
        virtual void advance(TimeScheme which, double delta_t); 
        // Build DM (essentially call RBFFD_CL to compute weights and update them on the GPU)  
        virtual void assemble();

        // --------------------------------------
        //  MPI+GPU related properties and routines
        // --------------------------------------
    protected:
        virtual int sendrecvBuf(cl::Buffer& buf, std::string label=""); 

        // Sync set R from the vec into the gpu_vec (host to device)
        void syncSetRSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec); 
        void syncSetRDouble(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec);

        // Sync set O from the gpu_vec to the vec (device to host) 
        void syncSetOSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec); 
        void syncSetODouble(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec); 

        // Sync the solution from GPU to the CPU (device to host)
        // NOTE: copies FULL solution to CPU.
        virtual void syncCPUtoGPU(); 


        // These should be generic for any PDE, so long as we have the solve()
        // routine custom to the PDE we can build the RK4 kernel generically. 
        virtual void advanceRK4(double dt);
        virtual void advanceFirstOrderEuler(double dt);
        virtual void advanceSecondOrderMidpoint(double dt);
#if 0
        virtual void advanceEuler(double dt);
        virtual void advanceMidpoint(double dt);
#endif 

        // --------------------------------------
        // Runge Kutta 4
        // --------------------------------------
        virtual void loadRK4Kernels(std::string& local_sources); 
        virtual void evaluateRK4_WithComm(int indx_u_in, int indx_u_plus_scaled_k_in, int indx_k_out, int indx_u_plus_scaled_k_out, double del_t, double current_time, double substep_scale); 
        virtual void advanceRK4_WithComm( int indx_u_in, int indx_k1, int indx_k2, int indx_k3, int indx_k4, int indx_u_out ); 

        virtual void loadEulerKernel(std::string& local_sources);
        virtual void loadMidpointKernel(std::string& local_sources);

        // Launch a kernel to do u(n+1) = u(n) + dt * f( U(n) ) over the
        // n_stencils_in_set starting at offset_to_set index in solution u.  
        // For example, if set D starts at index 15 and is 5 elements wide,
        // then call n_stencils_in_set=5, offset_to_set=15 (this routine
        // calculates the BYTE offset on its own. 
        void launchStepKernel( double dt, cl::Buffer& sol_in, cl::Buffer& deriv_sol, cl::Buffer& sol_out, unsigned int n_stencils_in_set, unsigned int offset_to_set);
        void launchEulerKernel( unsigned int offset_to_set, unsigned int nb_stencils_to_compute, double dt, cl::Buffer& sol_in, cl::Buffer& sol_out);

        void launchRK4_classic_eval( unsigned int offset_to_set, unsigned int nb_stencils_to_compute, double adjusted_t, double dt, cl::Buffer& u_in, cl::Buffer& u_plus_scaled_k_in,  cl::Buffer& k_out,  cl::Buffer& u_plus_scaled_k_out, double substep_scale);
        void launchRK4_block_eval( unsigned int offset_to_set, unsigned int nb_stencils_to_compute, double adjusted_t, double dt, cl::Buffer& u_in, cl::Buffer& u_plus_scaled_k_in,  cl::Buffer& k_out,  cl::Buffer& u_plus_scaled_k_out, double substep_scale);
        void launchRK4_adv( unsigned int offset_to_set, unsigned int nb_stencils_to_compute, cl::Buffer& u_in, cl::Buffer& k1, cl::Buffer k2, cl::Buffer& k3, cl::Buffer& k4, cl::Buffer& u_out);

        virtual std::string className() {return "heat_cl";}

    private: 
        virtual void setupTimers(); 
        void swap(int& a, int& b) { 
            int temp = a; a = b; b = temp; 
        }
        unsigned int getFloatSize() { if (useDouble) { return sizeof(double); } return sizeof(float); }

}; 
#endif 
