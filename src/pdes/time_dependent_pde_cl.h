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
        int INDX_INTERMEDIATE_1;
        int INDX_INTERMEDIATE_2;
        int INDX_INTERMEDIATE_3;

        RBFFD_CL& der_ref_gpu;

        bool weightsPrecomputed; 
        bool useDouble;

        // These are buffers for RK4 evaluations. 
        // Each of these is NB_STENCILS long
        // K1, K2, K3 and K4
        cl::Buffer gpu_feval[4]; 
        // IN and OUT
        cl::Buffer gpu_solution[2]; 

        // We have two kernels: 
        // the first is a straight RK4 scheme for nodes without cross CPU deps
        cl::Kernel rk4_k_kernel;
        // The second pauses after each RK4 substep to perform MPI comm
        cl::Kernel rk4_final_kernel;


    public: 

        TimeDependentPDE_CL(Domain* grid, RBFFD_CL* der, Communicator* comm, bool weightsComputed=false) 
            : TimeDependentPDE(grid, der, comm),
            // We maintain a ref to der here so we can keep it cast as an OpenCL RBFFD class
            der_ref_gpu(*der), weightsPrecomputed(weightsComputed)
        {;}
        
        void initialize(std::string solve_source) 
        {
            this->setupTimers();
            this->loadKernels(solve_source); 
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
        virtual void loadKernels(std::string& local_solve_source); 
        virtual void allocateGPUMem(); 

        // Use RK4 scheme to advance
        virtual void advance(TimeScheme which, double delta_t); 
        // Build DM (essentially call RBFFD_CL to compute weights and update them on the GPU)  
        virtual void assemble();

        // --------------------------------------
        //  MPI+GPU related properties and routines
        // --------------------------------------
    protected:
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
#if 0
        virtual void advanceEuler(double dt);
        virtual void advanceMidpoint(double dt);
#endif 

        // --------------------------------------
        // Runge Kutta 4
        // --------------------------------------
        virtual void loadRK4Kernels(std::string& local_sources); 
        void launchRK4_K_Kernel( double solveDT, double advanceDT, cl::Buffer solve_in, cl::Buffer solve_out, cl::Buffer advance_in, cl::Buffer advance_out);
        void launchRK4_Final_Kernel( double solveDT, double advanceDT, cl::Buffer k1, cl::Buffer k2, cl::Buffer k3, cl::Buffer advance_in, cl::Buffer advance_out);




        virtual std::string className() {return "heat_cl";}

    private: 
        virtual void setupTimers(); 
        void swap(int& a, int& b) { int temp = a; a = b; b = temp; }
        unsigned int getFloatSize() { if (useDouble) { return sizeof(double); } return sizeof(float); }

}; 
#endif 
