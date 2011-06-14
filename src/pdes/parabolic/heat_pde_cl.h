#ifndef __HEAT_PDE_CL_H__
#define __HEAT_PDE_CL_H__

#include "utils/opencl/cl_base_class.h"
#include "pdes/parabolic/heat_pde.h"
#include "rbffd/rbffd_cl.h"

class HeatPDE_CL : public HeatPDE, public CLBaseClass 
{
    protected: 
        cl::Buffer gpu_weights[NUM_DERIV_TYPES]; 
        RBFFD_CL& der_ref_gpu; 
        bool useDouble;

    public: 
        // Note: we specifically require the OpenCL version of RBFFD
        HeatPDE_CL(Domain* grid, RBFFD_CL* der, Communicator* comm, bool useUniformDiffusion, bool weightsComputed=false) 
            : HeatPDE(grid, der, comm, useUniformDiffusion, weightsComputed), 
            der_ref_gpu(*der), useDouble(true)
        { 
            this->setupTimers(); 
            this->loadKernels(); 
            this->allocateGPUMem();
        }

        // Build DM (essentially call RBFFD_CL to compute weights and update them on the GPU)  
        virtual void assemble();
        
        // This will apply the weights appropriately for an explicit (del_u =
        // L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, size_t n, double t) {
            // We done actually solve independent from the time stepper. The
            // stepper will internally call to a GPU device kernel to apply the
            // DM and "solve" 
            std::cout << "[HeatPDE_CL] Error: solve should not be called. The time stepper should call a device kernel for solving\n";
        };
    
        
        virtual void loadKernels(); 

        virtual void allocateGPUMem(); 

        virtual void advance(TimeScheme which, double delta_t); 

    private: 
        virtual void setupTimers(); 

    protected: 
        virtual std::string className() {return "heat_cl";}

        virtual void loadEulerKernel(); 

        // Call kernel to advance using first order euler
        virtual void advanceFirstOrderEuler(double dt);
}; 
#endif 

