#ifndef __STOKES_STEADY_PDE_H__
#define __STOKES_STEADY_PDE_H__

#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "pdes/pde.h"

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

        virtual void assemble() {
        
            std::cout << "Assembling...." << std::endl;

            unsigned int nb_stencils = grid_ref.getStencilsSize();
            unsigned int nb_nodes = grid_ref.getNodeListSize(); 

            unsigned int nrows = 4 * nb_stencils; 
            unsigned int ncols = 4 * nb_nodes; 

            L_host = new MatType(nrows, ncols); 
            F_host = new VecType(ncols);
#if 0
            // Fill L
            for (unsigned int i = 0; i < nb_stencils; i++) {
                StencilType& st = grid_ref.getStencil(i);

                // TODO: change these to *SFC weights (when computed)
                std::vector<double*>& ddx = getStencilWeights(RBFFD::X, i);
                std::vector<double*>& ddy = getStencilWeights(RBFFD::Y, i); 
                std::vector<double*>& ddz = getStencilWeights(RBFFD::Z, i); 
                for (unsigned int j = 0; j < st.size(); j++) {
                    L_host(i,st[j]) = ddx[j];  
                }
            }


            // Fill F
            // Write both to disk (spy them in Matlab)
#endif
        }

        
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
