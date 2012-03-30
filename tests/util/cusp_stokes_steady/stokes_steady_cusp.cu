// TODO : test this: 
//#define CUSP_USE_TEXTURE_MEMORY

// THIS IS adapted from verbose_monitor.cu
// PROVIDED BY THE CUSP v0.1 EXAMPLES

#include "grids/grid_reader.h"
#include "rbffd/rbffd.h"
#include "timer_eb.h" 

#include "stokes_steady_cusp.h"

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


#include "utils/spherical_harmonics.h"

#include <iomanip>
#include <iostream>
#include <sstream> 
#include <map>
#include <fstream> 
#include <typeinfo> 
using namespace std;

namespace cusp
{
    StokesSteady::StokesSteady(RBFFD& der_ref, Grid& grid_ref, int PrimeGPU) 
        : der(der_ref), 
        grid(grid_ref),
        primeGPU(PrimeGPU) 
    {
    // N should be number of stencils in domain
    // n should be number of nodes per stencil
    // nb_bnd should be number of boundary nodes in domain
        N = grid.getNodeListSize(); 
        n = grid.getMaxStencilSize(); 
        nb_bnd = grid.getBoundaryIndicesSize();
        nrows = 4 * N + 4; 
        ncols = 4 * N + 4; 
        NNZ = 9*n*N+2*(4*N)+2*(3*N);  

        setupTimers();
    }


    void StokesSteady::setupTimers() {

        if (primeGPU) {
            sprintf(test_name, "%u PRIMING THE GPU", N);  
            sprintf(assemble_timer_name, "%u Primer Assemble", N);
            sprintf(copy_timer_name,     "%u Primer Copy To CUSP_DEVICE_CSR", N); 
            sprintf(test_timer_name, "%u Primer GMRES test", N); 
        } else { 
            sprintf(test_name, "%u GMRES GPU (CUSP_DEVICE_CSR)", N);  
            sprintf(assemble_timer_name, "%u CUSP_HOST_CSR Assemble", N);
            sprintf(copy_timer_name,     "%u CUSP_HOST_CSR Copy To CUSP_DEVICE_CSR", N); 
            sprintf(test_timer_name, "%u GPU GMRES test", N); 
        }

        if (!tm.contains(assemble_timer_name)) { tm[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
        if (!tm.contains(copy_timer_name)) { tm[copy_timer_name] = new EB::Timer(copy_timer_name); } 
        if (!tm.contains(test_timer_name)) { tm[test_timer_name] = new EB::Timer(test_timer_name); } 

    }


    void StokesSteady::SpMV_Device(DEVICE_MAT_t& A, DEVICE_VEC_t& F, DEVICE_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu) {

    }


    // Perform GMRES on GPU
    void StokesSteady::GMRES_Device(DEVICE_MAT_t& A, DEVICE_VEC_t& F, DEVICE_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu) {
#if 1
        size_t restart = 300; 
        int max_iters = 1000; 
        double rel_tol = 1e-6; 
#else 
        // Maximum number of iterations (total) 
        size_t max_iters = 500; 
        // restart the process every "restart" iterations
        size_t restart = 200; 
        double rel_tol = 1e-8; 
#endif 

        try {

            //    cusp::convergence_monitor<double> monitor( F, max_iters, 0, 1e-3); 
            cusp::default_monitor<double> monitor( F, max_iters, rel_tol ); //, max_iters, rel_tol);// , 1e-3); 
            //cusp::default_monitor<double> monitor( F, -1, rel_tol ); //, max_iters, rel_tol);// , 1e-3); 

            std::cout << "GMRES Starting Residual Norm: " << monitor.residual_norm() << std::endl;

            // 1e-8, 10000, 300); 
            int precondType = -1; 
            switch (precondType) {
                case 0: 
                    {
                        // Jacobi Preconditioning (DIAGONAL)
                        // Probably wont work well for RBF-FD since we're not diagonally dominant
                        cusp::precond::diagonal<double, cusp::device_memory> M(A);
                        cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor, M); 
                    }
                    break;
                case 1:
                    {
                        // Smoothed Aggregation (Algebraic MultiGrid. Works for Nonsym?)
                        cusp::precond::smoothed_aggregation<int, double, cusp::device_memory> M(A);
                        cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor, M); 
                    }
                    break; 
#if 0 
                    // ONLY SPD MATRICES
                case 0: 
                    // AINV using static dropping
                    cusp::precond::scaled_bridson_ainv<double, cusp::device_memory> M(A, 0, 10);
                    cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor, M); 
                    break; 
#endif 
#if 0 
                    // ONLY SPD MATRICES
                case 1: 
                    // AINV using standard drop tolerance
                    cusp::precond::scaled_bridson_ainv<double, cusp::device_memory> M(A, .1);
                    cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor, M); 
                    break; 
#endif 
#if 0
                    // ONLY FOR SPD MATRICES
                case 2: 
                    // AINV using novel cusp dropping strategy (TODO: lookup) 
                    cusp::precond::bridson_ainv<double, cusp::device_memory> M(A, 0, -1, true, 2);
                    cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor, M); 
#endif 
                case 2: 
                    {
                        // AINV using novel cusp dropping strategy 
                        // assumes that sparsity pattern of precond is same as A, plus
                        // 2 extra nonzeros per row 
                        // VERY SLOW TO BUILD; DOES NOT CONVERGE
                        cusp::precond::nonsym_bridson_ainv<double, cusp::device_memory> M(A, 0, -1, true, 2);
                        cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor, M); 
                    }
                case 3: 
                    {
                        // AINV using novel cusp dropping strategy 
                        // Assume 40 nonzeros per row, drop everthing else. 
                        // VERY SLOW TO BUILD; DOES NOT CONVERGE
                        cusp::precond::nonsym_bridson_ainv<double, cusp::device_memory> M(A, 0.1, 10, false, 0);
                        cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor, M); 
                    }
                default: 
                    // Solve unpreconditioned Au = F
                    cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor); 
            }
            cudaThreadSynchronize(); 

            //    monitor.print();

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


        try {

            typedef cusp::array1d<double, DEVICE_VEC_t>::view DEVICE_VEC_VIEW_t; 

            DEVICE_VEC_VIEW_t U_approx_view(U_exact.begin()+(U_exact.size() - F.size()), U_exact.end()); 

            DEVICE_VEC_t diff(U_approx_gpu); 

            //cusp::blas::axpy(U_exact.begin()+(U_exact.size() - F.size()), U_exact.end(), diff.begin(),  -1); 
            cusp::blas::axpy(U_approx_view, diff, -1); 

            std::cout << "Rel l1   Norm: " << cusp::blas::nrm1(diff) / cusp::blas::nrm1(U_exact) << std::endl;  
            std::cout << "Rel l2   Norm: " << cusp::blas::nrm2(diff) / cusp::blas::nrm2(U_exact) << std::endl;  
            std::cout << "Rel linf Norm: " << cusp::blas::nrmmax(diff) / cusp::blas::nrmmax(U_exact) << std::endl;  
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

    //---------------------------------

    void StokesSteady::assemble_System_Stokes( RBFFD& der, Grid& grid, HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact){
        double eta = 1.;
        //double Ra = 1.e6;

        // We have different nb_stencils and nb_nodes when we parallelize. The subblocks might not be full
        unsigned int nb_stencils = grid.getStencilsSize();
        unsigned int nb_nodes = grid.getNodeListSize(); 
        unsigned int max_stencil_size = grid.getMaxStencilSize();
        unsigned int N = nb_nodes;
        // ---------------------------------------------------

        //------------- Fill the RHS of the System -------------
        // This is our manufactured solution:
        SphericalHarmonic::Sph32 UU; 
        SphericalHarmonic::Sph32105 VV; 
        SphericalHarmonic::Sph32 WW; 
        SphericalHarmonic::Sph32 PP; 

        std::vector<NodeType>& nodes = grid.getNodeList(); 

        //------------- Fill F -------------

        // U
        for (unsigned int j = 0; j < N; j++) {
            unsigned int row_ind = j + 0*N;
            NodeType& node = nodes[j]; 
            double Xx = node.x(); 
            double Yy = node.y(); 
            double Zz = node.z(); 

            U_exact[row_ind] = UU.eval(Xx,Yy,Zz); 
            F[row_ind] = -UU.lapl(Xx,Yy,Zz) + PP.d_dx(Xx,Yy,Zz);  
        }
#if 1

        // V
        for (unsigned int j = 0; j < N; j++) {
            unsigned int row_ind = j + 1*N;
            NodeType& node = nodes[j]; 
            double Xx = node.x(); 
            double Yy = node.y(); 
            double Zz = node.z(); 
            //double rr = sqrt(node.x()*node.x() + node.y()*node.y() + node.z()*node.z());
            //double dir = node.y();

            // F[row_ind] = (Ra * Temperature(j) * dir) / rr;  
            U_exact[row_ind] = VV.eval(Xx,Yy,Zz); 
            F[row_ind] = -VV.lapl(Xx,Yy,Zz) + PP.d_dy(Xx,Yy,Zz);  
        }

        // W
        for (unsigned int j = 0; j < N; j++) {
            unsigned int row_ind = j + 2*N;
            NodeType& node = nodes[j];
            double Xx = node.x(); 
            double Yy = node.y(); 
            double Zz = node.z(); 

            U_exact[row_ind] = WW.eval(Xx,Yy,Zz); 
            F[row_ind] = -WW.lapl(Xx,Yy,Zz) + PP.d_dz(Xx,Yy,Zz);  
        }

        // P
        for (unsigned int j = 0; j < N; j++) {
            unsigned int row_ind = j + 3*N;
            NodeType& node = nodes[j]; 
            double Xx = node.x(); 
            double Yy = node.y(); 
            double Zz = node.z(); 

            U_exact[row_ind] = PP.eval(Xx,Yy,Zz); 
            F[row_ind] = UU.d_dx(Xx,Yy,Zz) + VV.d_dy(Xx,Yy,Zz) + WW.d_dz(Xx,Yy,Zz);  
        }
#endif
        // Sum of U
        F[4*N+0] = 0.;

        // Sum of V
        F[4*N+1] = 0.;

        // Sum of W
        F[4*N+2] = 0.;

        // Sum of P
        F[4*N+3] = 0.;




        unsigned int ind = 0; 

        // -----------------  Fill LHS --------------------
        //
        // U (block)  row
        for (unsigned int i = 0; i < nb_stencils; i++) {
            StencilType& st = grid.getStencil(i);

            // TODO: change these to *SFC weights (when computed)
            double* ddx = der.getStencilWeights(RBFFD::XSFC, i);
            double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

            unsigned int diag_row_ind = i + 0*N;

            A.row_offsets[diag_row_ind] = ind; 


            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j] + 0*N;

                A.column_indices[ind] = diag_col_ind; 
                A.values[ind]  = -eta * lapl[j];  
                ind++; 
            }
            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j] + 3*N;

                A.column_indices[ind] = diag_col_ind; 
                A.values[ind]  = ddx[j];  
                ind++; 
            }

            // Added constraint to square mat and close nullspace
            A.column_indices[ind] = 4*N+0; 
            A.values[ind]  = 1;  
            ind++; 
        }

        // V (block)  row
        for (unsigned int i = 0; i < nb_stencils; i++) {
            StencilType& st = grid.getStencil(i);

            // TODO: change these to *SFC weights (when computed)
            double* ddy = der.getStencilWeights(RBFFD::YSFC, i);
            double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

            unsigned int diag_row_ind = i + 1*N;
            A.row_offsets[diag_row_ind] = ind; 

            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j] + 1*N;

                A.column_indices[ind] = diag_col_ind; 
                A.values[ind]  = -eta * lapl[j];  
                ind++; 
            }
            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j] + 3*N;

                A.column_indices[ind] = diag_col_ind; 
                A.values[ind]  = ddy[j];  
                ind++; 
            }

            // Added constraint to square mat and close nullspace
            A.column_indices[ind] = 4*N+1; 
            A.values[ind]  = 1;  
            ind++; 
        }

        // W (block)  row
        for (unsigned int i = 0; i < nb_stencils; i++) {
            StencilType& st = grid.getStencil(i);

            // TODO: change these to *SFC weights (when computed)
            double* ddz = der.getStencilWeights(RBFFD::ZSFC, i);
            double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

            unsigned int diag_row_ind = i + 2*N;
            A.row_offsets[diag_row_ind] = ind; 

            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j] + 2*N;

                A.column_indices[ind] = diag_col_ind; 
                A.values[ind]  = -eta * lapl[j];  
                ind++; 
            }
            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j] + 3*N;

                A.column_indices[ind] = diag_col_ind; 
                A.values[ind]  = ddz[j];
                ind++; 
            }

            // Added constraint to square mat and close nullspace
            A.column_indices[ind] = 4*N+2; 
            A.values[ind]  = 1;  
            ind++; 
        }


        // P (block)  row
        for (unsigned int i = 0; i < nb_stencils; i++) {
            StencilType& st = grid.getStencil(i);

            // TODO: change these to *SFC weights (when computed)
            double* ddx = der.getStencilWeights(RBFFD::XSFC, i);
            double* ddy = der.getStencilWeights(RBFFD::YSFC, i);
            double* ddz = der.getStencilWeights(RBFFD::ZSFC, i);

            unsigned int diag_row_ind = i + 3*N;
            A.row_offsets[diag_row_ind] = ind; 

            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j] + 0*N;

                A.column_indices[ind] = diag_col_ind; 
                A.values[ind]  = ddx[j]; 
                ind++; 
            }
            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j] + 1*N;

                A.column_indices[ind] = diag_col_ind; 
                A.values[ind]  = ddy[j]; 
                ind++; 
            }
            for (unsigned int j = 0; j < st.size(); j++) {
                unsigned int diag_col_ind = st[j] + 2*N;

                A.column_indices[ind] = diag_col_ind; 
                A.values[ind]  = ddz[j]; 
                ind++; 
            }

            // Added constraint to square mat and close nullspace
            A.column_indices[ind] = 4*N+3; 
            A.values[ind]  = 1;  
            ind++; 
        }

        // ------ EXTRA CONSTRAINT ROWS -----
        unsigned int diag_row_ind = 4*N;
        A.row_offsets[diag_row_ind] = ind;
        // U
        for (unsigned int j = 0; j < N; j++) {
            unsigned int diag_col_ind = j + 0*N;

            A.column_indices[ind] = diag_col_ind; 
            A.values[ind]  = 1;  
            ind++; 
        }

        diag_row_ind++; 
        A.row_offsets[diag_row_ind] = ind; 
        // V
        for (unsigned int j = 0; j < N; j++) {
            unsigned int diag_col_ind = j + 1*N;

            A.column_indices[ind] = diag_col_ind; 
            A.values[ind]  = 1;  
            ind++; 
        }

        diag_row_ind++; 
        A.row_offsets[diag_row_ind] = ind; 
        // W
        for (unsigned int j = 0; j < N; j++) {
            unsigned int diag_col_ind = j + 2*N;

            A.column_indices[ind] = diag_col_ind; 
            A.values[ind]  = 1;  
            ind++; 
        }

        diag_row_ind++; 
        A.row_offsets[diag_row_ind] = ind; 
        // P
        for (unsigned int j = 0; j < N; j++) {
            unsigned int diag_col_ind = j + 3*N;

            A.column_indices[ind] = diag_col_ind; 
            A.values[ind]  = 1;  
            ind++; 
        }

        // VERY IMPORTANT. UNSPECIFIED LAUNCH FAILURES ARE CAUSED BY FORGETTING THIS!
        A.row_offsets[4*N+4] = ind; 
    }


    void StokesSteady::write_System ( HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact )
    {
        write_to_file(F, "output/F.mtx"); 
        write_to_file(U_exact, "output/U_exact.mtx"); 
        cusp::io::write_matrix_market_file(A,"output/LHS.mtx"); 
    }

    void StokesSteady::write_Solution( Grid& grid, HOST_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu ) 
    {
        // IF we want to write details we need to copy back to host. 
        HOST_VEC_t U_approx(U_exact.size());

        if (U_approx_gpu.size() == U_exact.size()) {
            thrust::copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin());
        } else {
            thrust::copy(U_exact.begin(), U_exact.begin()+nb_bnd, U_approx.begin());
            thrust::copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin()+nb_bnd);
        }

        write_to_file(U_approx, "output/U_gpu.mtx"); 
    }

    void StokesSteady::assemble() {
        std::cout << "Assembling: " << test_name << std::endl;

        // ----- ASSEMBLE -----
        tm[assemble_timer_name]->start(); 
        A = new HOST_MAT_t(nrows, ncols, NNZ); 
        F = new HOST_VEC_t(nrows, 0);
        U_exact = new HOST_VEC_t(nrows, 0);
        assemble_System_Stokes(der, grid, *A, *F, *U_exact); 

        tm[assemble_timer_name]->stop(); 

        if (!primeGPU) {
            //write_System(*A, *F, *U_exact); 
        }
        // ----- SOLVE -----

        tm[copy_timer_name]->start();

        A_gpu = new DEVICE_MAT_t(*A); 
        F_gpu = new DEVICE_VEC_t(*F); 
        U_exact_gpu = new DEVICE_VEC_t(*U_exact); 
        U_approx_gpu = new DEVICE_VEC_t(F->size(), 0);

        tm[copy_timer_name]->stop();

    }

    void StokesSteady::solve() {
        std::cout << "Solving: " << test_name << std::endl;

        tm[test_timer_name]->start();
        // Use GMRES to solve A*u = F
        #if 1
        GMRES_Device(*A_gpu, *F_gpu, *U_exact_gpu, *U_approx_gpu);
        #else 
// Start by testing the parallel SpMV
        SpMV_Device(*A_gpu, *F_gpu, *U_exact_gpu, *U_approx_gpu);   
        #endif 
        tm[test_timer_name]->stop();

        if (!primeGPU) {
            write_Solution(grid, *U_exact, *U_approx_gpu); 
        }
    }

    StokesSteady::~StokesSteady() {
        std::cout << "Cleanup aisle three..."; 
        // Cleanup
        delete(A);
        delete(A_gpu);
        delete(F);
        delete(U_exact);
        delete(F_gpu);
        delete(U_exact_gpu);
        delete(U_approx_gpu);
        std::cout << "Done\n";
    }

};
