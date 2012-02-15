#ifndef __COSINE_BELL_CL_H__
#define __COSINE_BELL_CL_H__

#include "pdes/time_dependent_pde_cl.h"

class CosineBell_CL : public TimeDependentPDE_CL
{
    protected: 
        // RADIUS OF THE SPHERE: 
        double a;
        // RADIUS OF THE BELL
        double R;
        // ANGLE OF ADVECTION AROUND SPHERE (from Equator and Greenwhich Mean Time) 
        double alpha;
        // The initial velocity (NOTE: scalar in denom is 12 days in seconds)
        double u0;
        double time_for_revolution; 



    public:
        CosineBell_CL(Domain* grid, RBFFD_CL* der, Communicator* comm, double earth_radius, double velocity_angle, double one_revolution_in_seconds, int gpuType, int useHyperviscosity, bool weightsComputed=false) 
            :
            TimeDependentPDE_CL(grid, der, comm, gpuType, weightsComputed), 

            a(1.), //6.37122*10^6; % radius of earth in meters
            // ANGLE OF ADVECTION AROUND SPHERE (from Equator and Greenwhich Mean Time) 
            alpha(M_PI/2.),
            // Time in seconds to complete one revolution
            time_for_revolution(1036800.)

    {

        // RADIUS OF THE BELL
        R = a/3.;
        // The initial velocity (NOTE: scalar in denom is 12 days in seconds)
        u0 = (2.*M_PI*a)/time_for_revolution; 

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

        virtual double getMaxVelocity(double at_time) { 
            // The angular velocity is constant and will never be more than this. 
            // (At theta = pi/2, lambda = 0 and alpha = pi/2, the velocity is u0
            return u0; 
        }


    protected:
        virtual std::string className() {return "cosine_bell_cl";}
        virtual void solve() {;}
}; 
#endif 

//,  public CosineBell 
//CosineBell(grid, der, comm, earth_radius, velocity_angle, one_revolution_in_seconds, useHyperviscosity, weightsComputed), 
