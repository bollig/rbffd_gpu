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

        virtual void solve(std::vector<SolutionType>& y, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes) {}; 
        virtual std::string className() {
            return "ImplicitPDE"; 
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



// This is needed to make UBLAS variants of norm_* and GMRES
#define VIENNACL_HAVE_UBLAS 1

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
#include <boost/filesystem.hpp>

#include "precond/ilu0.hpp"

class Poisson1D_PDE_VCL : public ImplicitPDE
{

    typedef boost::numeric::ublas::compressed_matrix<double> UBLAS_MAT_t; 
    typedef boost::numeric::ublas::vector<double> UBLAS_VEC_t; 

    protected:
    UBLAS_MAT_t* LHS_host; 
    UBLAS_VEC_t* RHS_host; 
    UBLAS_VEC_t* U_exact_host; 

    // Problem size (Num Rows)
    unsigned int N;
    // Problem size (Num Cols)
    unsigned int M;
    // Stencil size
    unsigned int n;
    // Number of boundary nodes
    unsigned int nb_bnd;
    // N - nb_bnd
    unsigned int NN;
    // M - nb_bnd
    unsigned int MM;

    public: 
    Poisson1D_PDE_VCL(Domain* grid, RBFFD* der, Communicator* comm) 
        // The 1 here indicates the solution will have one component
        : ImplicitPDE(grid, der, comm, 1) 
    {   
        N = grid_ref.getStencilsSize(); 
        M = grid_ref.getNodeListSize(); 
        n = grid_ref.getMaxStencilSize(); 
        nb_bnd = grid_ref.getBoundaryIndicesSize();
        NN = N - nb_bnd; 
        MM = M - nb_bnd; 

        LHS_host = new UBLAS_MAT_t(NN,MM,(NN)*n); 
        RHS_host = new UBLAS_VEC_t(NN); 
        U_exact_host = new UBLAS_VEC_t(M); 
    }

    ~Poisson1D_PDE_VCL() {
        delete(LHS_host); 
        delete(RHS_host); 
        delete(U_exact_host);
    }

    virtual void assemble() 
    {

        UBLAS_MAT_t& A = *LHS_host; 
        UBLAS_VEC_t& F = *RHS_host;
        UBLAS_VEC_t& U_exact = *U_exact_host;

        std::cout << "Boundary nodes: " << nb_bnd << std::endl;

        //------ RHS ----------

        ManufacturedSolution UU; 

        std::vector<NodeType>& nodes = grid_ref.getNodeList(); 

        // We want U_exact to have the FULL solution. 
        // F should only have the compressed problem. 
        // NOTE: I am assuming the boundary nodes are first in the list. This
        // works well for Dirichlet conditions. Perhaps we should elegantly
        // handle them? 
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

        for (unsigned int i = N; i < M; i++) {
            U_exact[i] = -1.; // Indicate that we need to receive data from other procs
        }

        // This should get values from neighboring processors into U_exact. 
        this->sendrecvUpdates(U_exact,"U_exact"); 

        //------ LHS ----------

        // NOTE: assumes the boundary is sorted to the top of the node indices
        for (unsigned int i = nb_bnd; i < N; i++) {
            StencilType& sten = grid_ref.getStencil(i); 
            double* lapl = der_ref.getStencilWeights(RBFFD::LAPL, i); 

            for (unsigned int j = 0; j < n; j++) {
                if (sten[j] < (int)nb_bnd) { 
                    // Subtract the solution*weight from the element of the RHS. 
                    F[i-nb_bnd] -= (U_exact[sten[j]] * ( -lapl[j] )); 
                } else {
                    // Offset by nb_bnd so we crop off anything related to the boundary
                    A(i-nb_bnd,sten[j]-nb_bnd) = -lapl[j]; 
                }
            }
        }    
    }

    // Use GMRES to solve the linear system. 
    virtual void solve()
    {
        viennacl::linalg::gmres_tag tag(1e-8, 10, 2); 

#if 0
        // vandermond matrix test. PASS
        UBLAS_MAT_t AA(5,5,25); 
        AA(0,0) = 1;   AA(0,1) = 1;   AA(0,2) = 1;   AA(0,3) = 1;   AA(0,4) = 1; 
        AA(1,0) = 16;   AA(1,1) = 8;   AA(1,2) = 4;   AA(1,3) = 2;   AA(1,4) = 1; 
        AA(2,0) = 81;   AA(2,1) = 27;   AA(2,2) = 9;   AA(2,3) = 3;   AA(2,4) = 1; 
        AA(3,0) = 256;   AA(3,1) = 64;   AA(3,2) = 16;   AA(3,3) = 4;   AA(3,4) = 1; 
        AA(4,0) = 625;   AA(4,1) = 125;   AA(4,2) = 25;   AA(4,3) = 5;   AA(4,4) = 1; 
        VCL_MAT_t AA_gpu; 
        copy(AA,AA_gpu);
        viennacl::linalg::ilu0_precond< VCL_MAT_t > vcl_ilu( AA_gpu, viennacl::linalg::ilu0_tag() ); 
        viennacl::io::write_matrix_market_file(vcl_ilu.LU,"output/ILU.mtx"); 
        exit(-1);
#endif 
        viennacl::linalg::ilu0_precond< UBLAS_MAT_t > vcl_ilu0( *LHS_host, viennacl::linalg::ilu0_tag() ); 
#if 1
        viennacl::io::write_matrix_market_file(vcl_ilu0.LU,"output/ILU.mtx"); 
        std::cout << "Wrote preconditioner to output/ILU.mtx\n";
        UBLAS_VEC_t U_approx_gpu = viennacl::linalg::solve(*LHS_host, *RHS_host, tag, vcl_ilu0); 
#endif 
    }

    void write_System ( )
    {
        char dir[10]; 
        sprintf(dir, "output/%d_of_%d/", comm_ref.getRank()+1, comm_ref.getSize()); 
        std::string dir_str(dir); 
        std::cout << dir_str << std::endl;
        boost::filesystem::create_directories(dir_str);
        write_to_file(*RHS_host, dir_str + "F.mtx"); 
        write_to_file(*U_exact_host, dir_str + "U_exact.mtx"); 
        viennacl::io::write_matrix_market_file(*LHS_host,dir_str + "LHS.mtx"); 
    }

    void write_Solution()
    {
        unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();

        // IF we want to write details we need to copy back to host. 
        UBLAS_VEC_t U_approx(M, 0); 
//        copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin()+nb_bnd);

        write_to_file(U_approx, "output/U_gpu.mtx"); 
    }

    virtual std::string className() {
        return "Poisson1D_PDE_VCL"; 
    }
};

#endif 
