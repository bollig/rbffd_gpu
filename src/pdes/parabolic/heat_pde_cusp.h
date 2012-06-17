#ifndef __HEAT_PDE_CUSP_H__
#define __HEAT_PDE_CUSP_H__

#include <cusp/multiply.h>
#include <cusp/blas.h>
#include <cusp/hyb_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/array1d.h>
#include <cusp/print.h>


//#include "rbffd/rbffd.h"


#include "pdes/parabolic/heat_pde.h"

class RBFFD;  

typedef cusp::csr_matrix<unsigned int, double, cusp::device_memory> GPU_MAT_t; 
typedef cusp::csr_matrix<unsigned int, double, cusp::host_memory> CPU_CSR_MAT_t; 
typedef cusp::coo_matrix<unsigned int, double, cusp::host_memory> CPU_COO_MAT_t; 
typedef cusp::array1d<double, cusp::device_memory> GPU_VEC_t; 
typedef cusp::array1d<double, cusp::host_memory> CPU_VEC_t; 


class HeatPDE_CUSP : public HeatPDE
{
    protected: 
        GPU_MAT_t* x_weights_gpu; 
        GPU_MAT_t* y_weights_gpu;
        GPU_MAT_t* z_weights_gpu;
        GPU_MAT_t* l_weights_gpu;

        // Euler needs: 
        //      solution in
        //      diffusion
        //      solution out
        //      
        // We use a ping pong buffer scheme here for solution to avoid copying one to the other
        // Each of these is NB_NODES long
        GPU_VEC_t gpu_solution[3]; 
        GPU_VEC_t gpu_diffusivity;

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
        cusp::array1d<unsigned int, cusp::device_memory> gpu_boundary_indices;

        int INDX_IN;
        int INDX_OUT;
        int INDX_INTERMEDIATE_1;
        int INDX_INTERMEDIATE_2;
        int INDX_INTERMEDIATE_3;

        bool useDouble;
        bool assembled;

    public: 
        // Note: we specifically require the OpenCL version of RBFFD
        HeatPDE_CUSP(Domain* grid, RBFFD* der, Communicator* comm, std::string& local_cl_sources, bool useUniformDiffusion, bool weightsComputed=false) 
            : HeatPDE(grid, der, comm, useUniformDiffusion, weightsComputed), 
                useDouble(true),
                assembled(false),
              INDX_IN(0), INDX_OUT(1),
              INDX_INTERMEDIATE_1(2),
              INDX_INTERMEDIATE_2(3),
              INDX_INTERMEDIATE_3(4)
        { 
            this->setupTimers(); 
            this->allocateGPUMem();
        }

        // Build DM (essentially call RBFFD_CUSP to compute weights and update them on the GPU)  
        virtual void assemble();
        
        // This will apply the weights appropriately for an explicit (del_u =
        // L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t)        
        {
            // We done actually solve independent from the time stepper. The
            // stepper will internally call to a GPU device kernel to apply the
            // DM and "solve" 
            std::cout << "[HeatPDE_CUSP] Error: solve should not be called. The time stepper should call a device kernel for solving\n";
        };
    
        virtual void fillInitialConditions(ExactSolution* exact=NULL);
        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, GPU_VEC_t& sol, double t);
        
        virtual void allocateGPUMem(); 

        virtual void advance(TimeScheme which, double delta_t); 

    private: 
        virtual void setupTimers(); 

        void swap(int& a, int& b) { int temp = a; a = b; b = temp; }
        unsigned int getFloatSize() { if (useDouble) { return sizeof(double); } return sizeof(float); }

    protected: 
        virtual std::string className() {return "heat_cl";}

        // Sync set R from the vec into the gpu_vec
        void syncSetRSingle(std::vector<SolutionType>& vec, GPU_VEC_t& gpu_vec); 
        void syncSetRDouble(std::vector<SolutionType>& vec, GPU_VEC_t& gpu_vec);

        // Sync set O from the gpu_vec to the vec
        void syncSetOSingle(std::vector<SolutionType>& vec, GPU_VEC_t& gpu_vec); 
        void syncSetODouble(std::vector<SolutionType>& vec, GPU_VEC_t& gpu_vec); 

        virtual void syncCPUtoGPU(); 

        // Call kernel to advance using first order euler
        virtual void advanceFirstOrderEuler(double dt);
        virtual void advanceSecondOrderMidpoint(double dt);
        virtual void advanceRungeKutta4(double dt);


        void launchEulerSetQmDKernel( double dt, GPU_VEC_t& sol_in, GPU_VEC_t& sol_out);
        void launchEulerSetDKernel( double dt, GPU_VEC_t& sol_in, GPU_VEC_t& sol_out);



        void fillGPUMat(RBFFD::DerType which, GPU_MAT_t*& gpu_buffer);

         
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes) 
        { std::cout << "[HeatPDECusp] ERROR! SHOULD CALL THE TIME BASED SOLVE ROUTINE\n"; exit(EXIT_FAILURE); } 


        virtual void solve() 
        { std::cout << "[HeatPDECusp] ERROR! SHOULD CALL THE TIME BASED SOLVE ROUTINE\n"; exit(EXIT_FAILURE); } 

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t) 
        {
            std::cout << "[HeatPDECusp] ERROR! SHOULD CALL THE OTHER enforceBoundaryConditions!\n";
        } 
        
}; 
#endif 

