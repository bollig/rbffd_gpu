#ifndef __HEAT_PDE_CL_H__
#define __HEAT_PDE_CL_H__

#include "utils/opencl/cl_base_class.h"
#include "pdes/parabolic/heat_pde.h"
#include "rbffd/rbffd_cl.h"

class HeatPDE_CL : public HeatPDE, public CLBaseClass 
{
    protected: 

        cl::Kernel rk4_k_kernel;
        cl::Kernel rk4_final_kernel;
        cl::Kernel step_kernel;
        // Kernel for boundary conditions
        cl::Kernel bc_kernel;


        // Euler needs: 
        //      solution in
        //      diffusion
        //      solution out
        //      
        // We use a ping pong buffer scheme here for solution to avoid copying one to the other
        // Each of these is NB_NODES long
        cl::Buffer gpu_solution[3]; 

        // These are buffers for diff_op evaluations. For example: RK4 requires k1 = f(x,t)
        // k2 = f(x+0.5dt*k1, t+0.5*dt) etc.
        // Each of these is NB_STENCILS long
        cl::Buffer gpu_feval[4]; 

        cl::Buffer gpu_diffusivity;

        // Midpoint needs: 
        //      solution in
        //      diffusion
        //      intermediate solution out
        //      ---- COMM --- 
        //      inermediate solution in
        //      intermediate diffusion
        //      solution out
        //  

        // Boundary conditions need: 
        //  solution in
        //  boundary indices
        //  solution out
        cl::Buffer gpu_boundary_indices;

        RBFFD_CL& der_ref_gpu; 
        bool useDouble;

        int INDX_IN;
        int INDX_OUT;
        int INDX_INTERMEDIATE_1;
        int INDX_INTERMEDIATE_2;
        int INDX_INTERMEDIATE_3;

        EB::Timer* t_advance_gpu;
        EB::Timer* t_load_attach;
        
    public: 
        // Note: we specifically require the OpenCL version of RBFFD
        HeatPDE_CL(Domain* grid, RBFFD_CL* der, Communicator* comm, std::string& local_cl_sources, bool useUniformDiffusion, bool weightsComputed=false) 
            : HeatPDE(grid, der, comm, useUniformDiffusion, weightsComputed), 
            der_ref_gpu(*der), useDouble(true),
              INDX_IN(0), INDX_OUT(1),
              INDX_INTERMEDIATE_1(2),
              INDX_INTERMEDIATE_2(3),
              INDX_INTERMEDIATE_3(4)
        { 
            this->setupTimers(); 
            this->loadKernels(local_cl_sources); 
            this->allocateGPUMem();
        }

        ~HeatPDE_CL () {
            t_advance_gpu->print(); delete(t_advance_gpu);
            t_load_attach->print(); delete(t_load_attach);
        }

        // Build DM (essentially call RBFFD_CL to compute weights and update them on the GPU)  
        virtual void assemble();
        
        // This will apply the weights appropriately for an explicit (del_u =
        // L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t)
        {
            // We done actually solve independent from the time stepper. The
            // stepper will internally call to a GPU device kernel to apply the
            // DM and "solve" 
            std::cout << "[HeatPDE_CL] Error: solve should not be called. The time stepper should call a device kernel for solving\n";
            exit(EXIT_FAILURE); 
        }
        
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes)
        { 
            std::cout << "[HeatPDE_CL] Error: solve should not be called. The time stepper should call a device kernel for solving\n";
            exit(EXIT_FAILURE); 
        } 
    
        virtual void fillInitialConditions(ExactSolution* exact=NULL);
        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, cl::Buffer& sol, double t);
        
        virtual void loadKernels(std::string& local_sources); 

        virtual void allocateGPUMem(); 

        virtual void advance(TimeScheme which, double delta_t); 

    private: 
        virtual void setupTimers(); 

        void swap(int& a, int& b) { int temp = a; a = b; b = temp; }
        unsigned int getFloatSize() { if (useDouble) { return sizeof(double); } return sizeof(float); }

    protected: 
        virtual std::string className() {return "heat_cl";}
#if EVAN_UPDATE_THESE
        virtual void loadRK4Kernels(std::string& local_sources); 
        void launchRK4_K_Kernel( double solveDT, double advanceDT, cl::Buffer solve_in, cl::Buffer solve_out, cl::Buffer advance_in, cl::Buffer advance_out);
        void launchRK4_Final_Kernel( double solveDT, double advanceDT, cl::Buffer k1, cl::Buffer k2, cl::Buffer k3, cl::Buffer advance_in, cl::Buffer advance_out);
#endif 
        virtual void loadStepKernel(std::string& local_sources); 
        virtual void loadBCKernel(std::string& local_sources); 


        // Launch a kernel to do u(n+1) = u(n) + dt * f( U(n) ) over the
        // n_stencils_in_set starting at offset_to_set index in solution u.  
        // For example, if set D starts at index 15 and is 5 elements wide,
        // then call n_stencils_in_set=5, offset_to_set=15 (this routine
        // calculates the BYTE offset on its own. 
        void launchStepKernel( double dt, cl::Buffer& sol_in, cl::Buffer& deriv_sol, cl::Buffer& sol_out, unsigned int n_stencils_in_set, unsigned int offset_to_set);
        void launchEulerSetQmDKernel( double dt, cl::Buffer& sol_in, cl::Buffer& sol_out);
        void launchEulerSetDKernel( double dt, cl::Buffer& sol_in, cl::Buffer& sol_out);

        // Sync set R from the vec into the gpu_vec
        void syncSetRSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec); 
        void syncSetRDouble(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec);

        // Sync set O from the gpu_vec to the vec
        void syncSetOSingle(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec); 
        void syncSetODouble(std::vector<SolutionType>& vec, cl::Buffer& gpu_vec); 

        virtual void syncCPUtoGPU(); 

        // Call kernel to advance using first order euler
        virtual void advanceFirstOrderEuler(double dt);
        virtual void advanceSecondOrderMidpoint(double dt);
        virtual void advanceRungeKutta4(double dt);
        
        
    protected: 
        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t) {
            std::cout << "[HeatPDE_CL] Error wrong enforce boundary conditions called \n"; exit(EXIT_FAILURE); 
        }

}; 
#endif 

