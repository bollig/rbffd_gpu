#ifndef __POISSON1D_VCL_H__
#define __POISSON1D_VCL_H__

#include "pdes/pde.h"
#include "grids/domain.h"
#include "utils/comm/communicator.h"


class ImplicitPDE : public PDE
{

    public: 
        ImplicitPDE(Domain* grid, RBFFD* der, Communicator* comm, unsigned int solution_dim) 
            // The 1 here indicates the solution will have one component
            : PDE(grid, der, comm, solution_dim) 
        {   
        }

        virtual void assemble() =0; 
        virtual void solve() =0; 

        template <typename VecT>
            void write_to_file(VecT vec, std::string filename)
            {
                std::ofstream fout;
                fout.open(filename.c_str());
                for (size_t i = 0; i < vec.size(); i++) {
                    fout << std::setprecision(10) << vec[i] << std::endl;
                }
                fout.close();
                std::cout << "Wrote " << filename << std::endl;
            }
};


#include "utils/mathematica_base.h"

class ManufacturedSolution : public MathematicaBase
{
    public: 
            virtual double operator()(double Xx, double Yy, double Zz) { return this->eval(Xx, Yy, Zz); }
            virtual double eval(double Xx, double Yy, double Zz) {
                return Sin(Pi*Xx) ;
            }
            virtual double lapl(double Xx, double Yy, double Zz) {
                return -(Power(Pi,2)*Sin(Pi*Xx)); 
            }
};



#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/prod.hpp> 
#include <viennacl/io/matrix_market.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp> 
#include <viennacl/vector_proxy.hpp> 
#include <viennacl/linalg/vector_operations.hpp> 

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/lu.hpp>


class Poisson1D_VCL : public ImplicitPDE
{

    typedef boost::numeric::ublas::compressed_matrix<double> UBLAS_MAT_t; 
    typedef boost::numeric::ublas::vector<double> UBLAS_VEC_t; 

    protected:
        UBLAS_MAT_t* LHS_host; 
        UBLAS_VEC_t* RHS_host; 
        UBLAS_VEC_t* U_host; 

        // Problem size
        unsigned int N;
        // Stencil size
        unsigned int n;
        // Number of boundary nodes
        unsigned int nb_bnd;
    
    public: 
        Poisson1D_VCL(Domain* grid, RBFFD* der, Communicator* comm) 
            // The 1 here indicates the solution will have one component
            : ImplicitPDE(grid, der, comm, 1) 
        {   
            N = grid_ref.getNodeListSize(); 

            unsigned int N = grid_ref.getNodeListSize(); 
            unsigned int n = grid_ref.getMaxStencilSize(); 
            unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();
            LHS_host = new UBLAS_MAT_t(N-nb_bnd,N-nb_bnd,(N-nb_bnd)*n); 
            RHS_host = new UBLAS_VEC_t(N-nb_bnd); 
            U_host = new UBLAS_VEC_t(N-nb_bnd); 
        }

        ~Poisson1D_VCL() {
            delete(LHS_host); 
            delete(RHS_host); 
            delete(U_host);
        }

        virtual void assemble() 
        {

            UBLAS_MAT_t& A = *LHS_host; 
            UBLAS_VEC_t& F = *RHS_host;
            UBLAS_VEC_t& U_exact = *U_host;

            std::cout << "Boundary nodes: " << nb_bnd << std::endl;

            //------ RHS ----------

            ManufacturedSolution UU; 

            std::vector<NodeType>& nodes = grid_ref.getNodeList(); 

            // We want U_exact to have the FULL solution. 
            // F should only have the compressed problem. 
            for (unsigned int i = 0; i < nb_bnd; i++) {
                NodeType& node = nodes[i]; 
                double Xx = node.x(); 
                double Yy = node.y(); 
                double Zz = node.z(); 

                U_exact[i] = UU.eval(Xx, Yy, Zz);// + 2*M_PI; 
            }

            for (unsigned int i = nb_bnd; i < N; i++) {
                NodeType& node = nodes[i]; 
                double Xx = node.x(); 
                double Yy = node.y(); 
                double Zz = node.z(); 

                U_exact[i] = UU.eval(Xx, Yy, Zz); // + 2*M_PI; 
                // Solving -lapl(u + const) = f = -lapl(u) + 0
                // of course the lapl(const) is 0, so we will have a test to verify
                // that our null space is closed. 
                F[i-nb_bnd] = -UU.lapl(Xx, Yy, Zz); 
            }

            //------ LHS ----------

            // NOTE: assumes the boundary is sorted to the top of the node indices
            for (unsigned int i = nb_bnd; i < N; i++) {
                StencilType& sten = grid_ref.getStencil(i); 
                double* lapl = der_ref.getStencilWeights(RBFFD::LAPL, i); 

                for (unsigned int j = 0; j < n; j++) {
                    if (sten[j] < (int)nb_bnd) { 
                        // Subtract the solution*weight from the element of the RHS. 
                        F[i-nb_bnd] -= (U_exact[sten[j]] * ( -lapl[j] )); 
                        //                std::cout << "Node " << i << " depends on boundary\n"; 
                    } else {
                        // Offset by nb_bnd so we crop off anything related to the boundary
                        A(i-nb_bnd,sten[j]-nb_bnd) = -lapl[j]; 
                    }
                }
            }    
        }



        virtual void solve()
        {

        }

    protected:  
        void write_System ( UBLAS_MAT_t& A, UBLAS_VEC_t& F, UBLAS_VEC_t& U_exact )
        {
            write_to_file(F, "output/F.mtx"); 
            write_to_file(U_exact, "output/U_exact.mtx"); 
            viennacl::io::write_matrix_market_file(A,"output/LHS.mtx"); 
        }
#if 0
        void write_Solution( UBLAS_VEC_t& U_exact, VCL_VEC_t& U_approx_gpu ) 
        {
            unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();

            // IF we want to write details we need to copy back to host. 
            UBLAS_VEC_t U_approx(U_exact.size());
            copy(U_exact.begin(), U_exact.begin()+nb_bnd, U_approx.begin());
            copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin()+nb_bnd);

            write_to_file(U_approx, "output/U_gpu.mtx"); 
        }
#endif 

};

#endif 
