#ifndef __COSINE_BELL_CL_H__
#define __COSINE_BELL_CL_H__

#include "pdes/time_dependent_pde_cl.h"

class CosineBell_CL : public TimeDependentPDE_CL
{
    public:
        CosineBell_CL(Domain* grid, RBFFD_CL* der, Communicator* comm, double earth_radius, double velocity_angle, double one_revolution_in_seconds, int useOneThreadPerStencil, int useHyperviscosity, bool weightsComputed=false) 
            :
            TimeDependentPDE_CL(grid, der, comm, useOneThreadPerStencil, weightsComputed)
    {
        // Fill in constants
        // Allocate GPU buffers for velocity
        // load solve kernel
        //    std::string solve_str = #include "cosine_bell_solve.cl"
        // initialize the TimeDependentPDE_CL superclass
        this->initialize("cosine_bell_solve.cl");
    }

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t) { 
            //DO NOTHING;
        }


    protected:
        virtual std::string className() {return "cosine_bell_cl";}
}; 
#endif 

//,  public CosineBell 
//CosineBell(grid, der, comm, earth_radius, velocity_angle, one_revolution_in_seconds, useHyperviscosity, weightsComputed), 
