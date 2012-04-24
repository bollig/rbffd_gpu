#ifndef __POISSON1D_VCL_H__
#define __POISSON1D_VCL_H__

#include "pdes/pde.h"
#include "grids/domain.h"
#include "utils/comm/communicator.h"

#include <iostream> 
#include <fstream> 
#include <iomanip> 

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
#include "linalg/parallel_gmres.hpp"

class Poisson1D_PDE_VCL : public ImplicitPDE
{

    typedef boost::numeric::ublas::compressed_matrix<double> UBLAS_MAT_t; 
    typedef boost::numeric::ublas::vector<double> UBLAS_VEC_t; 
    typedef viennacl::compressed_matrix<double> VCL_MAT_t; 
    typedef viennacl::vector<double> VCL_VEC_t; 


    protected:
    UBLAS_MAT_t* LHS_host; 
    UBLAS_VEC_t* RHS_host; 
    UBLAS_VEC_t* U_exact_host; 
    UBLAS_VEC_t* U_approx_host; 

    VCL_MAT_t* LHS_dev; 
    VCL_VEC_t* RHS_dev; 
    VCL_VEC_t* U_exact_dev; 
    VCL_VEC_t* U_approx_dev; 


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

    std::string dir_str; 

    int solve_on_gpu; 

    public: 
    Poisson1D_PDE_VCL(Domain* grid, RBFFD* der, Communicator* comm, int use_gpu=0) 
        // The 1 here indicates the solution will have one component
        : ImplicitPDE(grid, der, comm, 1) , solve_on_gpu(use_gpu)
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
        U_approx_host = new UBLAS_VEC_t(N, 0); 

        if (solve_on_gpu) {
            LHS_dev = new VCL_MAT_t(NN,MM,(NN)*n); 
            RHS_dev = new VCL_VEC_t(NN); 
            U_exact_dev = new VCL_VEC_t(M); 
            U_approx_dev = new VCL_VEC_t(N); 
        }

        char dir[FILENAME_MAX]; 
        sprintf(dir, "output/%d_of_%d/", comm_ref.getRank()+1, comm_ref.getSize()); 

        dir_str = string(dir); 
        std::cout << "Making output dir: " << dir_str << std::endl;
        boost::filesystem::create_directories(dir_str);
    }

    ~Poisson1D_PDE_VCL() {
        delete(LHS_host); 
        delete(RHS_host); 
        delete(U_exact_host);
        delete(U_approx_host);

        if (solve_on_gpu) { 
            delete(LHS_dev); 
            delete(RHS_dev); 
            delete(U_exact_dev);
            delete(U_approx_dev);
        }
    }

    // Catch the call to "solve()" and allow users to specify "solve_on_gpu=1"
    // which requires memory allocation, copy to, solve, copy back operators
    virtual void solve() {
        if (solve_on_gpu) {
            this->solve(*LHS_dev, *RHS_dev, *U_exact_dev, *U_approx_dev); 
        } else {
            this->solve(*LHS_host, *RHS_host, *U_exact_host, *U_approx_host); 
        }
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

        std::cout << "N = " << N << ", M = " << M << std::endl;
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
//            double lapl_fd[3] = {2, -1, -1}; 


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
        
        if (solve_on_gpu) 
        { 
            // Put all of the known system on the GPU. 
            copy(A, *LHS_dev);
            viennacl::copy(RHS_host->begin(),RHS_host->end(), RHS_dev->begin());
            viennacl::copy(U_exact_host->begin(),U_exact_host->end(), U_exact_dev->begin());
            viennacl::copy(U_approx_host->begin(),U_approx_host->end(), U_approx_dev->begin());
        }
    }

    // Use GMRES to solve the linear system. 
    template <class MAT_t, class VEC_t>
        void solve(MAT_t& LHS, VEC_t& RHS, VEC_t& U_exact, VEC_t& U_approx_out)
        {
            // Solve on the CPU
            int restart = 5; 
            int krylov = 10;
            double tol = 1e-8; 

            // Tag has (tolerance, total iterations, number iterations between restarts)
            viennacl::linalg::parallel_gmres_tag tag(comm_ref, grid_ref, tol, restart*krylov, krylov); 

//            viennacl::linalg::ilu0_precond< MAT_t > vcl_ilu0( LHS, viennacl::linalg::ilu0_tag() ); 
#if 0
            viennacl::io::write_matrix_market_file(vcl_ilu0.LU, dir_str + "ILU.mtx"); 
            std::cout << "Wrote preconditioner to ILU.mtx\n";
#endif        
            std::cout << "GMRES Max Number of Iterations: " << tag.max_iterations() << std::endl;
            std::cout << "GMRES Krylov Dim: " << tag.krylov_dim() << std::endl;
            std::cout << "GMRES Max Number of Restarts (max_iter/krylov_dim - 1): " << tag.max_restarts() << std::endl;
            std::cout << "GMRES Tolerance: " << tag.tolerance() << std::endl;

            U_approx_out = viennacl::linalg::solve(LHS, RHS, tag);//, vcl_ilu0); 
            
            if (tag.iters() < tag.max_iterations()) { 
                std::cout << "\n[+++] Solver converged "; //<< tag.error() << " relative tolerance";       
                std::cout << " after " << tag.iters() << " iterations" << std::endl << std::endl;
            }
            else
            {
                std::cout << "\n[XXX] Solver reached iteration limit " << tag.iters() << " before converging\n\n";
            //    std::cout << " to " << tag.tolerance() << " relative tolerance " << std::endl << std::endl;
            }


            std::cout << "GMRES Iterations: " << tag.iters() << std::endl;
            std::cout << "GMRES Iteration Limit: " << tag.max_iterations() << std::endl;
            std::cout << "GMRES Residual Norm: " << tag.error() << std::endl;

#if 0
            std::cout << "GMRES Relative Tol: " << monitor.relative_tolerance() << std::endl;
            std::cout << "GMRES Absolute Tol: " << monitor.absolute_tolerance() << std::endl;
            std::cout << "GMRES Target Residual (Abs + Rel*norm(F)): " << monitor.tolerance() << std::endl;
#endif 
            checkNorms(U_approx_out, U_exact);
        }

    void checkNorms(VCL_VEC_t& sol, VCL_VEC_t& exact) {
        VCL_VEC_t diff(sol.size()); 

        viennacl::vector_range<VCL_VEC_t> exact_view( exact, viennacl::range(exact.size() - sol.size(), exact.size())); 

        viennacl::linalg::sub(sol, exact_view, diff); 

        std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff) / viennacl::linalg::norm_1(exact) << std::endl;  
        std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff) / viennacl::linalg::norm_2(exact) << std::endl;  
        std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff) / viennacl::linalg::norm_inf(exact) << std::endl;  
    }


    void checkNorms(UBLAS_VEC_t& sol, UBLAS_VEC_t& exact) {
        UBLAS_VEC_t diff = sol; 

        boost::numeric::ublas::vector_range<UBLAS_VEC_t> exact_view( exact, boost::numeric::ublas::range(exact.size() - sol.size(), exact.size())); 

        diff -= exact_view; 

        std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff) / viennacl::linalg::norm_1(exact) << std::endl;  
        std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff) / viennacl::linalg::norm_2(exact) << std::endl;  
        std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff) / viennacl::linalg::norm_inf(exact) << std::endl;  
    }

    void write_System ( )
    {
        write_to_file(*RHS_host, dir_str + "F.mtx"); 
        write_to_file(*U_exact_host, dir_str + "U_exact.mtx"); 
        viennacl::io::write_matrix_market_file(*LHS_host,dir_str + "LHS.mtx"); 
    }

    void write_Solution()
    {
        unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();

        // IF we want to write details we need to copy back to host. 
        UBLAS_VEC_t U_approx(M, 0); 
        if (solve_on_gpu) { 
            copy(U_approx_dev->begin(), U_approx_dev->end(), U_approx.begin()+nb_bnd);
            write_to_file(U_approx, dir_str + "U_gpu.mtx"); 
        } else { 
            write_to_file(*U_approx_host, dir_str + "U_gpu.mtx"); 
        }
    }

    virtual std::string className() {
        return "Poisson1D_PDE_VCL"; 
    }
};

#endif 
