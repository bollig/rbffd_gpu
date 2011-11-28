#ifndef __STOKES_STEADY_PDE_H__
#define __STOKES_STEADY_PDE_H__

#include "pdes/pde.h"

#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

typedef boost::numeric::ublas::compressed_matrix<FLOAT> MatType;
typedef boost::numeric::ublas::vector<FLOAT>            VecType;

class StokesSteadyPDE : public PDE
{

    MatType *L_host; 
    VecType *F_host; 

    public: 
        StokesSteadyPDE(Domain* grid, RBFFD* der, Communicator* comm) 
            : PDE(grid, der, comm) 
        {
            std::cout << "INSIDE STOKES CONSTRUCTOR" << std::endl;
        }

        virtual void assemble();  
        
        virtual void solve(std::vector<SolutionType>& y, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes) 
        {
            std::cout << "Solving...." << std::endl;
            // Solve L u = F
            // Write solution to disk
            
            // Update U_G with the content from U
            // (reshape to fit into U_G. For stokes we assume we're in a vector type <u,v,w,p>)
            // SCALAR  std::copy(u_vec.begin(), u_vec.end(), U_G.begin()); 





        }


        virtual std::string className() { return "stokes_steady"; }

};

#endif  // __STOKES_STEADY_PDE_H__
