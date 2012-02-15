#ifndef __COSINE_ROLLUP_H__
#define __COSINE_ROLLUP_H__

#include "pdes/time_dependent_pde.h"

#if 0
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp> 
#endif 

// TODO: extend this class and compute diffusion in two terms: lapl(y(t)) = div(y(t)) .dot. grad(y(t))
class CosineBell : public TimeDependentPDE
{
    protected: 
        // T/F : are the weights already computed so we can avoid that cost?
        bool weightsPrecomputed;

#if 0
        boost::numeric::ublas::compressed_matrix<FLOAT> D_N;
#endif 
        int useHyperviscosity; 

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
        CosineBell(Domain* grid, RBFFD* der, Communicator* comm, double earth_radius, double velocity_angle, double one_revolution_in_seconds, int useHyperviscosity, bool weightsComputed=false) 
            : TimeDependentPDE(grid, der, comm), weightsPrecomputed(weightsComputed), 
               useHyperviscosity(useHyperviscosity), 
                   // RADIUS OF THE SPHERE: 
                   a(earth_radius), //6.37122*10^6; % radius of earth in meters
                   // ANGLE OF ADVECTION AROUND SPHERE (from Equator and Greenwhich Mean Time) 
                   alpha(velocity_angle),
                   // Time in seconds to complete one revolution
                   time_for_revolution(one_revolution_in_seconds)
        { 
            // RADIUS OF THE BELL
            R = a/3.;
            // The initial velocity (NOTE: scalar in denom is 12 days in seconds)
            u0 = (2.*M_PI*a)/time_for_revolution; 
        }

        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble(); 
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t);

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t) { 
            //DO NOTHING;
        }

        virtual double getMaxVelocity(double at_time) { 
            // The angular velocity is constant and will never be more than this. 
            return u0; 
        }

#if 0
        virtual void advance(TimeScheme which, double delta_t);
#endif 
    private: 
        void setupTimers(); 

    protected: 
        virtual std::string className() {return "cosine_bell";}
        
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes)
        { std::cout << "[CosineBell] ERROR! SHOULD CALL THE TIME BASED SOLVE ROUTINE\n"; exit(EXIT_FAILURE); } 

        virtual void solve() {;}
}; 
#endif 

