#ifndef __TIME_DEPENDENT_PDE_VCL_H_
#define __TIME_DEPENDENT_PDE_VCL_H__

#include <mpi.h>
#include "rbffd/rbffd_vcl.h"
#include "time_dependent_pde.h"


#include "utils/opencl/viennacl_typedefs.h"

#include "viennacl/io/kernel_parameters.hpp"

class TimeDependentPDE_VCL : public TimeDependentPDE
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

        bool useDouble;

        RBFFD_VCL& der_ref_gpu;

        bool weightsPrecomputed; 

        std::vector<SolutionType> cpu_buf; 

        // Flag to indicate if syncCPUtoGPU needs to copy down from GPU (should be set to 1 if advance is called)
        int cpu_dirty;

        bool assembled; 

        // These are buffers for RK4 evaluations. 
        // Each of these is NB_STENCILS long
        // K1, K2, K3 and K4
        //VCL_VEC_t* gpu_feval[4];
        // IN and OUT
        VCL_VEC_t* gpu_solution[8];

        // The solution used within this class
        UBLAS_VEC_t* cpu_solution;

        std::string kernel_source_file;

        int gpuType; 

    public: 
        // USE_GPU=0 (pass over this constructor), =1 (use a block approach per stencil), =2 (use a thread approach per stencil)
        TimeDependentPDE_VCL(Domain* grid, RBFFD_VCL* der, Communicator* comm, int gpu_type, bool weightsComputed=false) 
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
    {
        // Borrowed from the VCL demo for reading tuned parameters (thanks Karl): 
        // -----------------------------------------
        std::cout << std::endl;
        std::cout << "----------------------------------------------" << std::endl;
        std::cout << "               Device Info" << std::endl;
        std::cout << "----------------------------------------------" << std::endl;

        std::cout << viennacl::ocl::current_device().info() << std::endl;

        try {
            viennacl::io::read_kernel_parameters< VCL_VEC_t >("vector_parameters.xml");
            viennacl::io::read_kernel_parameters< VCL_DENSE_MAT_t >("matrix_parameters.xml");
            viennacl::io::read_kernel_parameters< VCL_CSR_MAT_t >("sparse_parameters.xml");
            // -----------------------------------------

            //check:
            std::cout << "vector add:" << std::endl;
            std::cout << viennacl::ocl::get_kernel("d_vector_1", "add").local_work_size() << std::endl;
            std::cout << viennacl::ocl::get_kernel("d_vector_1", "add").global_work_size() << std::endl;

            std::cout << "matrix vec_mul:" << std::endl;
            std::cout << viennacl::ocl::get_kernel("d_matrix_row_1", "vec_mul").local_work_size() << std::endl;
            std::cout << viennacl::ocl::get_kernel("d_matrix_row_1", "vec_mul").global_work_size() << std::endl;

            std::cout << "compressed_matrix vec_mul:" << std::endl;
            std::cout << viennacl::ocl::get_kernel("d_compressed_matrix_1", "vec_mul").local_work_size() << std::endl;
            std::cout << viennacl::ocl::get_kernel("d_compressed_matrix_1", "vec_mul").global_work_size() << std::endl;
        } catch (...) {
            std::cout << "[TPDE_VCL] There was an exception thrown when reading optimal parameters. Please ensure that files vector_parameters.xml, matrix_parametes.xml and sparse_parameters.xml are in the proper directiry (PWD)\n";
        } 
    }
        

        virtual ~TimeDependentPDE_VCL() {
            this->tm.printAll(); 
            this->tm.clear(); 
            std::cout << "TPDE_VCL destroyed\n";
        }

        void setGPUType(int type) {
            gpuType=type;
        }

        // Fill in the initial conditions of the PDE. (overwrite the solution)
        virtual void fillInitialConditions(ExactSolution* exact=NULL);

        void initialize()
        {
            this->setupTimers();
            this->allocateGPUMem(); 
        }

        virtual void solve(VCL_VEC_t& y_t, VCL_VEC_t& f_out, unsigned int n_stencils, unsigned int n_nodes, double t)=0; 

        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t) {
            // We dont actually solve independent from the time stepper. The
            // stepper will internally call to a GPU device kernel to apply the
            // DM and "solve" 
            std::cout << "[T_PDE_VCL] Error: solve should not be called. The time stepper should call a device kernel for solving\n";
            exit(EXIT_FAILURE);
        };

        // Interfaces: 
        virtual void allocateGPUMem();

        // Use RK4 scheme to advance
        virtual void advance(TimeScheme which, double delta_t); 
        // Build DM (essentially call RBFFD_CL to compute weights and update them on the GPU)  
        virtual void assemble();

        // --------------------------------------
        //  MPI+GPU related properties and routines
        // --------------------------------------
    protected:
        virtual int sendrecvBuf(VCL_VEC_t* buf, std::string label=""); 

        // Sync set R from the vec into the gpu_vec (host to device)
        void syncSetRDouble(std::vector<SolutionType>& vec, VCL_VEC_t* gpu_vec);

        // Sync set O from the gpu_vec to the vec (device to host) 
        void syncSetODouble(std::vector<SolutionType>& vec, VCL_VEC_t* gpu_vec); 

        // Sync the solution from GPU to the CPU (device to host)
        // NOTE: copies FULL solution to CPU.
        virtual void syncCPUtoGPU() {

            std::cout << "*************** FULL MEMCOPY DEVICE -> HOST ***************\n"; 
            viennacl::copy(gpu_solution[INDX_IN]->begin(), gpu_solution[INDX_IN]->end(), &U_G[0]);            
        }

        void advanceRK4(double delta_t); 
        void advanceRK4_OverlapComm(double delta_t); 
        virtual void evaluateRK4_NoComm(int indx_u_in, int indx_u_plus_scaled_k_in, int indx_k_out, int indx_u_plus_scaled_k_out, double del_t, double current_time, double substep_scale); 
        virtual void advanceRK4_NoComm( int indx_u_in, int indx_k1, int indx_k2, int indx_k3, int indx_k4, int indx_u_out ); 

        virtual std::string className() {return "T_PDE_VCL";}

    private: 
        virtual void setupTimers(); 
        void swap(int& a, int& b) { 
            int temp = a; a = b; b = temp; 
        }
        unsigned int getFloatSize() { if (useDouble) { return sizeof(double); } return sizeof(float); }


    protected: 
        // We'll hide this routine because we want one based on time (see above)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes) 
        { std::cout << "[T_PDE_VCL] ERROR! SHOULD CALL THE TIME BASE SOLVE\n"; exit(EXIT_FAILURE); } 

        // Do nothing for the boundary by default. Can override this in subclasses
        virtual void enforceBoundaryConditions(VCL_VEC_t& sol, double cur_time) {
            // NOTE: be sure to modify the sol buffer directly. 
        }
}; 
#endif 
