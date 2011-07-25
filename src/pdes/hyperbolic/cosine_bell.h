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

    public: 
        CosineBell(Domain* grid, RBFFD* der, Communicator* comm, int useHyperviscosity, bool weightsComputed=false) 
            : TimeDependentPDE(grid, der, comm), weightsPrecomputed(weightsComputed), 
               useHyperviscosity(useHyperviscosity)
        { ; }

        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble(); 
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t);

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t) { 
            //DO NOTHING;
        }

#if 0
        virtual void advance(TimeScheme which, double delta_t);
#endif 
    private: 
        void setupTimers(); 

    protected: 
        virtual std::string className() {return "vortex_rollup";}
}; 
#endif 

