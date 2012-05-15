#ifndef __STOKES_STEADY_H__
#define __STOKES_STEADY_H__

// This is needed to make UBLAS variants of norm_* and GMRES
#define VIENNACL_HAVE_UBLAS 1

#define STOKES_CONSTRAINTS 1

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
#if 1
#include "linalg/parallel_gmres.hpp"
#else
#include "linalg/parallel_gmres_vcl.hpp"
#endif 

#include "linalg/parallel_norm_1.hpp"
#include "linalg/parallel_norm_2.hpp"
#include "linalg/parallel_norm_inf.hpp"

#include "manufactured_solution.h"
#include "utils/spherical_harmonics.h"
#include "pdes/implicit_pde.h"
#include "timer_eb.h"

class StokesSteady_PDE_VCL : public ImplicitPDE
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


    EB::TimerList tlist; 

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
    StokesSteady_PDE_VCL(Domain* grid, RBFFD* der, Communicator* comm, int use_gpu=0) 
        // The 1 here indicates the solution will have one component
        : ImplicitPDE(grid, der, comm, 1) , solve_on_gpu(use_gpu)
    {   
        setupTimers(); 

        N = grid_ref.getStencilsSize(); 
        M = grid_ref.getNodeListSize(); 
        n = grid_ref.getMaxStencilSize(); 
        nb_bnd = grid_ref.getBoundaryIndicesSize();
        NN = N - nb_bnd; 
        MM = M - nb_bnd; 

        tlist["allocate"]->start();
#ifdef STOKES_CONSTRAINTS
        unsigned int NNZ = 9*n*NN+2*(4*NN)+2*(3*NN);  
        LHS_host = new UBLAS_MAT_t(4*NN+4,4*MM+4,NNZ); 
        RHS_host = new UBLAS_VEC_t(4*NN+4); 
        U_exact_host = new UBLAS_VEC_t(4*M+4); 
        U_approx_host = new UBLAS_VEC_t(4*NN+4, 0); 

        if (solve_on_gpu) {
            LHS_dev = new VCL_MAT_t(4*NN+4,4*MM+4,NNZ);
            RHS_dev = new VCL_VEC_t(4*NN+4); 
            U_exact_dev = new VCL_VEC_t(4*M+4); 
            U_approx_dev = new VCL_VEC_t(4*NN+4); 
        }
#else
        unsigned int NNZ = 9*n*NN;  
        LHS_host = new UBLAS_MAT_t(4*NN,4*MM,NNZ); 
        RHS_host = new UBLAS_VEC_t(4*NN); 
        U_exact_host = new UBLAS_VEC_t(4*M); 
        U_approx_host = new UBLAS_VEC_t(4*NN, 0); 

        if (solve_on_gpu) {
            LHS_dev = new VCL_MAT_t(4*NN,4*MM,NNZ);
            RHS_dev = new VCL_VEC_t(4*NN); 
            U_exact_dev = new VCL_VEC_t(4*M); 
            U_approx_dev = new VCL_VEC_t(4*NN); 
        }
#endif 
        tlist["allocate"]->stop(); 

        char dir[FILENAME_MAX]; 
        sprintf(dir, "output/%d_of_%d/", comm_ref.getRank()+1, comm_ref.getSize()); 

        dir_str = string(dir); 
        std::cout << "Making output dir: " << dir_str << std::endl;
        boost::filesystem::create_directories(dir_str);
    }

    ~StokesSteady_PDE_VCL() {
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

        tlist.printAllNonStatic();
        tlist.clear();
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
        tlist["assemble"]->start(); 
        std::cout << "Assembling... \n";

        // Choose either grouped by component or interleaved components
        // Number of components to interleave
        unsigned int NC = 4; 
        UBLAS_MAT_t& A = *LHS_host; 
        UBLAS_VEC_t& F = *RHS_host;
        UBLAS_VEC_t& U_exact = *U_exact_host;

        std::cout << "Boundary nodes: " << nb_bnd << std::endl;

        //------ RHS ----------

        tlist["assemble_rhs"]->start(); 
        ManufacturedSolution MS; 

        double eta = 1.;

        std::vector<NodeType>& nodes = grid_ref.getNodeList(); 

        //------------- Fill F -------------
        double sumU = 0.;
        double sumV = 0.;
        double sumW = 0.;
        double sumP = 0.;
        for (unsigned int j = 0; j < MM; j++) {
            unsigned int row_ind = j*NC;
            NodeType& node = nodes[j]; 
            double Xx = node.x(); 
            double Yy = node.y(); 
            double Zz = node.z(); 

            U_exact(row_ind+0) = MS.U(Xx,Yy,Zz); 
            U_exact(row_ind+1) = MS.V(Xx,Yy,Zz); 
            U_exact(row_ind+2) = MS.W(Xx,Yy,Zz); 
            U_exact(row_ind+3) = MS.P(Xx,Yy,Zz); 
            F(row_ind+0) = MS.RHS_U(Xx,Yy,Zz); 
            F(row_ind+1) = MS.RHS_V(Xx,Yy,Zz); 
            F(row_ind+2) = MS.RHS_W(Xx,Yy,Zz); 
            F(row_ind+3) = MS.RHS_P(Xx,Yy,Zz); 
            sumU += U_exact(row_ind+0); 
            sumV += U_exact(row_ind+1); 
            sumW += U_exact(row_ind+2); 
            sumP += U_exact(row_ind+3); 
        }

#ifdef STOKES_CONSTRAINTS
        // Sum of U
        F(4*N+0) = sumU; 

        // Sum of V
        F(4*N+1) = sumV; 

        // Sum of W
        F(4*N+2) = sumW; 

        // Sum of P
        F(4*N+3) = sumP;
#endif 

        // This should get values from neighboring processors into U_exact. 
        this->sendrecvUpdates(U_exact,"U_exact"); 
        tlist["assemble_rhs"]->stop();

        // -----------------  Fill LHS --------------------
        //
        tlist["assemble_lhs"]->start();

        // U (block)  row
        for (unsigned int i = 0; i < NN; i++) {
            StencilType& st = grid_ref.getStencil(i);

            double* ddx = der_ref.getStencilWeights(RBFFD::XSFC, i);
            double* ddy = der_ref.getStencilWeights(RBFFD::YSFC, i);
            double* ddz = der_ref.getStencilWeights(RBFFD::ZSFC, i);
            double* lapl = der_ref.getStencilWeights(RBFFD::LSFC, i); 

            unsigned int diag_row_ind = i * NC;

            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j]*NC;
                A(diag_row_ind+0, diag_col_ind+0) = -eta * lapl[j];  
                A(diag_row_ind+0, diag_col_ind+3) = ddx[j];  
#ifdef STOKES_CONSTRAINTS
                A(diag_row_ind+0, 4*NN+0) = 1.;
#endif
            }

            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j]*NC;
                A(diag_row_ind+1, diag_col_ind+1) = -eta * lapl[j];  
                A(diag_row_ind+1, diag_col_ind+3) = ddy[j];  
#ifdef STOKES_CONSTRAINTS
                A(diag_row_ind+1, 4*NN+1) = 1.;
#endif 
            }

            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j]*NC;
                A(diag_row_ind+2, diag_col_ind+2) = -eta * lapl[j];  
                A(diag_row_ind+2, diag_col_ind+3) = ddz[j];  
#ifdef STOKES_CONSTRAINTS
                A(diag_row_ind+2, 4*NN+2) = 1.;
#endif 
            }

            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j]*NC;
                A(diag_row_ind+3, diag_col_ind+0) = ddx[j];  
                A(diag_row_ind+3, diag_col_ind+1) = ddy[j];  
                A(diag_row_ind+3, diag_col_ind+2) = ddz[j];  
#ifdef STOKES_CONSTRAINTS
                A(diag_row_ind+3, 4*NN+3) = 1.;
#endif 
            }
        }

#ifdef STOKES_CONSTRAINTS
        for (unsigned int j = 0; j < NN; j++) {
            A(4*NN+0, (j*NC)+0) = 1.;  
        }
        for (unsigned int j = 0; j < NN; j++) {
            A(4*NN+1, (j*NC)+1) = 1.;  
        }
        for (unsigned int j = 0; j < NN; j++) {
            A(4*NN+2, (j*NC)+2) = 1.;  
        }
        for (unsigned int j = 0; j < NN; j++) {
            A(4*NN+3, (j*NC)+3) = 1.;  
        }
#endif 

        tlist["assemble_lhs"]->stop();

        if (solve_on_gpu) 
        { 
            std::cout << "Sending system to GPU...\n";
            tlist["assemble_gpu"]->start();
            // Put all of the known system on the GPU. 
            copy(A, *LHS_dev);
            viennacl::copy(RHS_host->begin(),RHS_host->end(), RHS_dev->begin());
            viennacl::copy(U_exact_host->begin(),U_exact_host->end(), U_exact_dev->begin());
            viennacl::copy(U_approx_host->begin(),U_approx_host->end(), U_approx_dev->begin());
            tlist["assemble_gpu"]->stop();
        }
        std::cout << "done.\n";

        tlist["assemble"]->stop();
    }


    // Use GMRES to solve the linear system. 
    template <class MAT_t, class VEC_t>
        void solve(MAT_t& LHS, VEC_t& RHS, VEC_t& U_exact, VEC_t& U_approx_out)
        {
            // Solve on the CPU
            int restart = 20; 
            int krylov = 120;
            double tol = 1e-8; 

            // Tag has (tolerance, total iterations, number iterations between restarts)
#if 0
            viennacl::linalg::parallel_gmres_tag tag(comm_ref, grid_ref, tol, restart*krylov, krylov); 
#else
            viennacl::linalg::gmres_tag tag(tol, restart*krylov, krylov); 
#endif 
#if 1
            tlist["solve"]->start();
            U_approx_out = viennacl::linalg::solve(LHS, RHS, tag); 
            tlist["solve"]->stop();
#else 
            tlist["precond"]->start();
            std::cout << "Generating preconditioner...\n";
            viennacl::linalg::ilu0_precond< MAT_t > vcl_ilu0( LHS, viennacl::linalg::ilu0_tag(0,3*NN) ); 
            tlist["precond"]->stop();
#if 0
            viennacl::io::write_matrix_market_file(vcl_ilu0.LU, dir_str + "ILU.mtx"); 
            std::cout << "Wrote preconditioner to ILU.mtx\n";
#endif        
            std::cout << "GMRES Max Number of Iterations: " << tag.max_iterations() << std::endl;
            std::cout << "GMRES Krylov Dim: " << tag.krylov_dim() << std::endl;
            std::cout << "GMRES Max Number of Restarts (max_iter/krylov_dim - 1): " << tag.max_restarts() << std::endl;
            std::cout << "GMRES Tolerance: " << tag.tolerance() << std::endl;

            tlist["solve"]->start();
            U_approx_out = viennacl::linalg::solve(LHS, RHS, tag, vcl_ilu0); 
            tlist["solve"]->stop();
#endif 

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
        tlist["checkNorms"]->start();
        unsigned int start_indx = 0; 
        unsigned int end_indx = NN; 

        for (unsigned int i =0; i < 4; i++) {
            VCL_VEC_t diff = viennacl::vector_range<VCL_VEC_t>(sol, viennacl::range(start_indx, end_indx)); 

            viennacl::vector_range<VCL_VEC_t> exact_view( exact, viennacl::range(start_indx + nb_bnd, end_indx+nb_bnd)); 

            diff -= exact_view; 

            std::cout << "COMPONENT [" << i << " (" << start_indx << "->" << end_indx-1 << " of " << sol.size() << ")]\n";
            std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff, comm_ref) / viennacl::linalg::norm_1(exact_view, comm_ref) << std::endl;  
            std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff, comm_ref) / viennacl::linalg::norm_2(exact_view, comm_ref) << std::endl;  
            std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff, comm_ref) / viennacl::linalg::norm_inf(exact_view, comm_ref) << std::endl;  

            start_indx = end_indx; 
            end_indx = start_indx + NN; 
        }

        // Global difference
        VCL_VEC_t g_diff = viennacl::vector_range<VCL_VEC_t>(sol, viennacl::range(0, sol.size()-4)); 

        viennacl::vector_range<VCL_VEC_t> g_exact_view( exact, viennacl::range(0 + nb_bnd, g_diff.size()+nb_bnd)); 

        g_diff -= g_exact_view; 

        std::cout << "GLOBAL ERROR " << NN << "\n";
        std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(g_diff, comm_ref) / viennacl::linalg::norm_1(g_exact_view, comm_ref) << std::endl;  
        std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(g_diff, comm_ref) / viennacl::linalg::norm_2(g_exact_view, comm_ref) << std::endl;  
        std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(g_diff, comm_ref) / viennacl::linalg::norm_inf(g_exact_view, comm_ref) << std::endl;  

        tlist["checkNorms"]->stop();
    }


    void checkNorms(UBLAS_VEC_t& sol, UBLAS_VEC_t& exact) {
        tlist["checkNorms"]->start();
        unsigned int start_indx = 0; 
        unsigned int end_indx = NN; 

        for (unsigned int i =0; i < 4; i++) {
            UBLAS_VEC_t diff = boost::numeric::ublas::vector_range<UBLAS_VEC_t>(sol, boost::numeric::ublas::range(start_indx, end_indx)); 

            boost::numeric::ublas::vector_range<UBLAS_VEC_t> exact_view( exact, boost::numeric::ublas::range(start_indx + nb_bnd, end_indx+nb_bnd)); 

            diff -= exact_view; 

            std::cout << "COMPONENT [" << i << " (" << start_indx << "->" << end_indx-1 << " of " << sol.size() << ")]\n";
            std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff, comm_ref) / viennacl::linalg::norm_1(exact_view, comm_ref) << std::endl;  
            std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff, comm_ref) / viennacl::linalg::norm_2(exact_view, comm_ref) << std::endl;  
            std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff, comm_ref) / viennacl::linalg::norm_inf(exact_view, comm_ref) << std::endl;  

            start_indx = end_indx; 
            end_indx = start_indx + NN; 
        }

        // Global difference
        UBLAS_VEC_t g_diff = boost::numeric::ublas::vector_range<UBLAS_VEC_t>(sol, boost::numeric::ublas::range(0, sol.size()-4)); 

        boost::numeric::ublas::vector_range<UBLAS_VEC_t> g_exact_view( exact, boost::numeric::ublas::range(0 + nb_bnd, g_diff.size()+nb_bnd)); 

        g_diff -= g_exact_view; 

        std::cout << "GLOBAL ERROR " << NN << "\n";
        std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(g_diff, comm_ref) / viennacl::linalg::norm_1(g_exact_view, comm_ref) << std::endl;  
        std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(g_diff, comm_ref) / viennacl::linalg::norm_2(g_exact_view, comm_ref) << std::endl;  
        std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(g_diff, comm_ref) / viennacl::linalg::norm_inf(g_exact_view, comm_ref) << std::endl;  

        tlist["checkNorms"]->stop();
    }




    void write_System ( )
    {
        tlist["writeSystem"]->start();
        write_to_file(*RHS_host, dir_str + "F.mtx"); 
        write_to_file(*U_exact_host, dir_str + "U_exact.mtx");
        if (NN < 4000) {
            viennacl::io::write_matrix_market_file(*LHS_host,dir_str + "LHS.mtx"); 
        }

        UBLAS_VEC_t temp = boost::numeric::ublas::prod(*LHS_host, *U_exact_host); 
        write_to_file(temp, dir_str + "RHS_discrete.mtx");
        write_to_file(temp-*RHS_host, dir_str + "RHS_err.mtx");
        tlist["writeSystem"]->stop(); 
    }

    void write_Solution()
    {
        tlist["writeSolution"]->start(); 
        // IF we want to write details we need to copy back to host. 
        UBLAS_VEC_t U_approx(U_exact_host->size(), 0); 
        if (solve_on_gpu) { 
            copy(U_exact_host->begin(), U_exact_host->end(), U_approx.begin());
            copy(U_approx_dev->begin(), U_approx_dev->end(), U_approx.begin()+nb_bnd);
            write_to_file(U_approx, dir_str + "U_gpu.mtx"); 
        } else { 
            write_to_file(*U_approx_host, dir_str + "U_gpu.mtx"); 
        }
        tlist["writeSolution"]->stop(); 
    }

    virtual std::string className() {
        return "StokesSteady_PDE_VCL"; 
    }


    protected: 

    void setupTimers() {
        tlist["allocate"] = new EB::Timer("Allocate GPU vectors and matrices"); 
        tlist["assemble"] = new EB::Timer("Assemble both RHS and LHS");
        tlist["assemble_lhs"] = new EB::Timer("Assemble LHS");
        tlist["assemble_rhs"] = new EB::Timer("Assemble RHS");
        tlist["assemble_gpu"] = new EB::Timer("Send assembled system to GPU");
        tlist["solve"] = new EB::Timer("Solve system with GMRES");
        tlist["precond"] = new EB::Timer("Construct preconditioner");
        tlist["writeSystem"] = new EB::Timer("Write system to disk"); 
        tlist["writeSolution"] = new EB::Timer("Write solution to disk"); 
        tlist["checkNorms"] = new EB::Timer("Check solution norms");
    }
};

#endif 
