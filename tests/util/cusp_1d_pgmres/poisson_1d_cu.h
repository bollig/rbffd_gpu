#ifndef __POISSON1D_CU_H__
#define __POISSON1D_CU_H__

#include "manufactured_solution.h"
#include "pdes/implicit_pde.h"
#include "grids/domain.h"

#include <cusp/hyb_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/gmres.h>
#include <cusp/gallery/poisson.h>
#include <cusp/print.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>
#include <cusp/io/matrix_market.h>
#include <cusp/precond/diagonal.h>
#include <cusp/precond/ainv.h>
#include <cusp/precond/smoothed_aggregation.h>
#include <cusp/precond/aggregate.h>
#include <cusp/precond/smooth.h>
#include <cusp/precond/strength.h>

#include <thrust/host_vector.h> 
#include <thrust/device_vector.h>
#include <thrust/generate.h>

#include <boost/filesystem.hpp>

class Poisson1D_PDE_CU : public ImplicitPDE
{

    typedef cusp::array1d<double, cusp::host_memory> HOST_VEC_t; 
    typedef cusp::array1d<double, cusp::device_memory> DEV_VEC_t; 
    typedef cusp::csr_matrix<unsigned int, double, cusp::host_memory> HOST_MAT_t; 
    typedef cusp::csr_matrix<unsigned int, double, cusp::device_memory> DEV_MAT_t; 



    protected:
    HOST_MAT_t* LHS_host; 
    HOST_VEC_t* RHS_host; 
    HOST_VEC_t* U_exact_host; 
    HOST_VEC_t* U_approx_host; 

    DEV_MAT_t* LHS_dev; 
    DEV_VEC_t* RHS_dev; 
    DEV_VEC_t* U_exact_dev; 
    DEV_VEC_t* U_approx_dev; 


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
    Poisson1D_PDE_CU(Domain* grid, RBFFD* der, Communicator* comm, int use_gpu=0) 
        // The 1 here indicates the solution will have one component
        : ImplicitPDE(grid, der, comm, 1) , solve_on_gpu(use_gpu)
    {   
        N = grid_ref.getStencilsSize(); 
        M = grid_ref.getNodeListSize(); 
        n = grid_ref.getMaxStencilSize(); 
        nb_bnd = grid_ref.getBoundaryIndicesSize();
        NN = N - nb_bnd; 
        MM = M - nb_bnd; 

        LHS_host = new HOST_MAT_t(NN,MM,(NN)*n); 
        RHS_host = new HOST_VEC_t(NN); 
        U_exact_host = new HOST_VEC_t(M, 0.); 
        U_approx_host = new HOST_VEC_t(NN, 0.); 

        if (solve_on_gpu) {
            LHS_dev = new DEV_MAT_t(NN,MM,(NN)*n); 
            RHS_dev = new DEV_VEC_t(NN); 
            U_exact_dev = new DEV_VEC_t(M); 
            U_approx_dev = new DEV_VEC_t(N); 
        }

        char dir[FILENAME_MAX]; 
        sprintf(dir, "output/%d_of_%d/", comm_ref.getRank()+1, comm_ref.getSize()); 

        dir_str = string(dir); 
        std::cout << "Making output dir: " << dir_str << std::endl;
        boost::filesystem::create_directories(dir_str);
    }

    ~Poisson1D_PDE_CU() {
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

        HOST_MAT_t& A = *LHS_host; 
        HOST_VEC_t& F = *RHS_host;
        HOST_VEC_t& U_exact = *U_exact_host;

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

        unsigned int ind = 0; 
        // NOTE: assumes the boundary is sorted to the top of the node indices
        for (unsigned int i = nb_bnd; i < N; i++) {
            StencilType& sten = grid_ref.getStencil(i); 
            double* lapl = der_ref.getStencilWeights(RBFFD::LAPL, i); 

            A.row_offsets[i-nb_bnd] = ind;

            for (unsigned int j = 0; j < n; j++) {
                if (sten[j] < (int)nb_bnd) { 
                    // Subtract the solution*weight from the element of the RHS. 
                    F[i-nb_bnd] -= (U_exact[sten[j]] * ( -lapl[j] )); 
                    // std::cout << "Node " << i << " depends on boundary\n"; 
                } else {
                    // Offset by nb_bnd so we crop off anything related to the boundary
                    A.column_indices[ind] = sten[j]-nb_bnd; 
                    A.values[ind] = -lapl[j]; 
                    ind++; 
                }
            }
        }    

        // VERY IMPORTANT. UNSPECIFIED LAUNCH FAILURES ARE CAUSED BY FORGETTING THIS!
        A.row_offsets[N-nb_bnd] = ind; 

        if (solve_on_gpu) 
        { 
            // Put all of the known system on the GPU. 
            *LHS_dev = *LHS_host; 
            *RHS_dev = *RHS_host; 
            *U_exact_dev = *U_exact_host; 
            *U_approx_dev = *U_approx_host; 
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

            try {
#if 0
                cusp::convergence_monitor<double> monitor( RHS, -1, tol); 
#else 
#if 0
                cusp::default_monitor<double> monitor( RHS, -1, tol );
#else 
                cusp::verbose_monitor<double> monitor( RHS, restart*krylov, tol );
#endif
#endif 


                std::cout << "GMRES Starting Residual Norm: " << monitor.residual_norm() << std::endl;

#if 0
                std::cout << "RHS = "; 
                for (int i = 0; i < RHS.size(); i++) {
                    std::cout << RHS[i] <<",";
                }
                std::cout << "\n";
#endif 

                cusp::krylov::gmres(LHS, U_approx_out, RHS, krylov, monitor); 
                cudaThreadSynchronize(); 

                if (monitor.converged())
                {
                    std::cout << "\n[+++] Solver converged to " << monitor.relative_tolerance() << " relative tolerance";       
                    std::cout << " after " << monitor.iteration_count() << " iterations" << std::endl << std::endl;
                }
                else
                {
                    std::cout << "\n[XXX] Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
                    std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << std::endl << std::endl;
                }

                std::cout << "GMRES Iterations: " << monitor.iteration_count() << std::endl;
                std::cout << "GMRES Iteration Limit: " << monitor.iteration_limit() << std::endl;
                std::cout << "GMRES Residual Norm: " << monitor.residual_norm() << std::endl;
                std::cout << "GMRES Relative Tol: " << monitor.relative_tolerance() << std::endl;
                std::cout << "GMRES Absolute Tol: " << monitor.absolute_tolerance() << std::endl;
                std::cout << "GMRES Target Residual (Abs + Rel*norm(F)): " << monitor.tolerance() << std::endl;
            }
            catch(std::bad_alloc &e)
            {
                std::cerr << "Ran out of memory trying to compute GMRES: " << e.what() << std::endl;
                exit(-1);
            }
            catch(thrust::system_error &e)
            {
                std::cerr << "Some other error happened during GMRES: " << e.what() << std::endl;
                exit(-1);
            }

            checkNorms(U_approx_out, U_exact);
        }

        void checkNorms(DEV_VEC_t& sol, DEV_VEC_t& exact) {

            try {
                typedef cusp::array1d<double, cusp::device_memory>::view VEC_VIEW_t; 
#if 1
                VEC_VIEW_t U_approx_view(exact.begin()+(exact.size() - sol.size()), exact.end()); 
                DEV_VEC_t diff(sol); 

                cusp::blas::axpy(U_approx_view, diff, -1); 
                std::cout << "Rel l1   Norm: " << cusp::blas::nrm1(diff) / cusp::blas::nrm1(exact) << std::endl;  
                std::cout << "Rel l2   Norm: " << cusp::blas::nrm2(diff) / cusp::blas::nrm2(exact) << std::endl;  
                std::cout << "Rel linf Norm: " << cusp::blas::nrmmax(diff) / cusp::blas::nrmmax(exact) << std::endl;  
#endif 
            }
            catch(std::bad_alloc &e)
            {
                std::cerr << "Ran out of memory trying to compute Error Norms: " << e.what() << std::endl;
                exit(-1);
            }
            catch(thrust::system_error &e)
            {
                std::cerr << "Some other error happened during Error Norms: " << e.what() << std::endl;
                exit(-1);
            }
        }


        void checkNorms(HOST_VEC_t& sol, HOST_VEC_t& exact) {

            try {
                typedef cusp::array1d<double, cusp::host_memory>::view VEC_VIEW_t; 
#if 1
                VEC_VIEW_t U_approx_view(exact.begin()+(exact.size() - sol.size()), exact.end()); 
                HOST_VEC_t diff(sol); 

                cusp::blas::axpy(U_approx_view, diff, -1); 
                std::cout << "Rel l1   Norm: " << cusp::blas::nrm1(diff) / cusp::blas::nrm1(exact) << std::endl;  
                std::cout << "Rel l2   Norm: " << cusp::blas::nrm2(diff) / cusp::blas::nrm2(exact) << std::endl;  
                std::cout << "Rel linf Norm: " << cusp::blas::nrmmax(diff) / cusp::blas::nrmmax(exact) << std::endl;  
#endif 
            }
            catch(std::bad_alloc &e)
            {
                std::cerr << "Ran out of memory trying to compute Error Norms: " << e.what() << std::endl;
                exit(-1);
            }
            catch(thrust::system_error &e)
            {
                std::cerr << "Some other error happened during Error Norms: " << e.what() << std::endl;
                exit(-1);
            }
        }

    void write_System ( )
    {
        write_to_file(*RHS_host, dir_str + "F.mtx"); 
        write_to_file(*U_exact_host, dir_str + "U_exact.mtx"); 
        cusp::io::write_matrix_market_file(*LHS_host,dir_str + "LHS.mtx"); 
    }

    void write_Solution()
    {
        unsigned int nb_bnd = grid_ref.getBoundaryIndicesSize();

        // IF we want to write details we need to copy back to host. 
        HOST_VEC_t U_approx(M, 0); 
        if (solve_on_gpu) { 
            //copy(U_approx_dev->begin(), U_approx_dev->end(), U_approx.begin()+nb_bnd);
            *U_approx_host = *U_approx_dev;
            write_to_file(U_approx, dir_str + "U_gpu.mtx"); 
        } else { 
            write_to_file(*U_approx_host, dir_str + "U_gpu.mtx"); 
        }
    }

    virtual std::string className() {
        return "Poisson1D_PDE_CU"; 
    }
};

#endif 
