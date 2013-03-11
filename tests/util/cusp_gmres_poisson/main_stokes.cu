// TODO : test this: 
//#define CUSP_USE_TEXTURE_MEMORY

// THIS IS adapted from verbose_monitor.cu
// PROVIDED BY THE CUSP v0.1 EXAMPLES

#include "grids/grid_reader.h"
#include "rbffd/rbffd.h"
#include "timer_eb.h" 

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


typedef std::vector< std::map< unsigned int, double> > STL_MAT_t; 
typedef std::vector<double> STL_VEC_t; 


typedef cusp::array1d<double, cusp::host_memory> HOST_VEC_t; 
typedef cusp::array1d<double, cusp::device_memory> DEVICE_VEC_t; 
typedef cusp::csr_matrix<unsigned int, double, cusp::host_memory> HOST_MAT_t; 
typedef cusp::csr_matrix<unsigned int, double, cusp::device_memory> DEVICE_MAT_t; 

EB::TimerList timers;

//---------------------------------

// Perform GMRES on GPU
void GMRES_Device(DEVICE_MAT_t& A, DEVICE_VEC_t& F, DEVICE_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu) {
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

void assemble_System_Stokes( RBFFD& der, Grid& grid, HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact){
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


void write_System ( HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact )
{
    write_to_file(F, "output/F.mtx"); 
    write_to_file(U_exact, "output/U_exact.mtx"); 
    cusp::io::write_matrix_market_file(A,"output/LHS.mtx"); 
}

void write_Solution( Grid& grid, HOST_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu ) 
{
    unsigned int nb_bnd = grid.getBoundaryIndicesSize();

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


//---------------------------------

void gpuTest(RBFFD& der, Grid& grid, int primeGPU=0) {
    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 
    unsigned int nrows = 4 * N + 4; 
    unsigned int ncols = 4 * N + 4; 
    unsigned int NNZ = 9*n*N+2*(4*N)+2*(3*N);  
 
    char test_name[256]; 
    char assemble_timer_name[256]; 
    char copy_timer_name[512]; 
    char test_timer_name[256]; 

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

    if (!timers.contains(assemble_timer_name)) { timers[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
    if (!timers.contains(copy_timer_name)) { timers[copy_timer_name] = new EB::Timer(copy_timer_name); } 
    if (!timers.contains(test_timer_name)) { timers[test_timer_name] = new EB::Timer(test_timer_name); } 


    std::cout << test_name << std::endl;


    // ----- ASSEMBLE -----
    timers[assemble_timer_name]->start(); 
    HOST_MAT_t* A = new HOST_MAT_t(nrows, ncols, NNZ); 
    HOST_VEC_t* F = new HOST_VEC_t(nrows, 0);
    HOST_VEC_t* U_exact = new HOST_VEC_t(nrows, 0);
    assemble_System_Stokes(der, grid, *A, *F, *U_exact); 
    
    timers[assemble_timer_name]->stop(); 

    if (!primeGPU) {
        //write_System(*A, *F, *U_exact); 
    }
    // ----- SOLVE -----

    timers[copy_timer_name]->start();

    DEVICE_MAT_t* A_gpu = new DEVICE_MAT_t(*A); 
    DEVICE_VEC_t* F_gpu = new DEVICE_VEC_t(*F); 
    DEVICE_VEC_t* U_exact_gpu = new DEVICE_VEC_t(*U_exact); 
    DEVICE_VEC_t* U_approx_gpu = new DEVICE_VEC_t(F->size(), 0);

    timers[copy_timer_name]->stop();

    timers[test_timer_name]->start();
    // Use GMRES to solve A*u = F
    GMRES_Device(*A_gpu, *F_gpu, *U_exact_gpu, *U_approx_gpu);
    timers[test_timer_name]->stop();

    if (!primeGPU) {
        write_Solution(grid, *U_exact, *U_approx_gpu); 
    }
    // Cleanup
    delete(A);
    delete(A_gpu);
    delete(F);
    delete(U_exact);
    delete(F_gpu);
    delete(U_exact_gpu);
    delete(U_approx_gpu);
}


int main(void)
{
    bool writeIntermediate = true; 
    bool primed = false; 

    std::vector<std::string> grids; 

    //grids.push_back("~/GRIDS/md/md005.00036"); 

    //    grids.push_back("~/GRIDS/md/md165.27556"); 
    //grids.push_back("~/GRIDS/md/md063.04096"); 
#if 1 
    grids.push_back("~/GRIDS/md/md031.01024"); 
    grids.push_back("~/GRIDS/md/md050.02601"); 
    grids.push_back("~/GRIDS/md/md063.04096"); 
    grids.push_back("~/GRIDS/md/md089.08100"); 
    grids.push_back("~/GRIDS/md/md127.16384"); 
    grids.push_back("~/GRIDS/md/md165.27556"); 
#endif 
#if 0
    grids.push_back("~/GRIDS/geoff/scvtimersesh_100k_nodes.ascii"); 
    grids.push_back("~/GRIDS/geoff/scvtimersesh_500k_nodes.ascii"); 
    grids.push_back("~/GRIDS/geoff/scvtimersesh_100k_nodes.ascii"); 
    grids.push_back("~/GRIDS/geoff/scvtimersesh_500k_nodes.ascii"); 
    grids.push_back("~/GRIDS/geoff/scvtimersesh_1m_nodes.ascii"); 
#endif 
    //grids.push_back("~/GRIDS/geoff/scvtimersesh_1m_nodes.ascii"); 

    for (size_t i = 0; i < grids.size(); i++) {
        std::string& grid_name = grids[i]; 

        std::string weight_timer_name = grid_name + " Calc Weights";  

        timers[weight_timer_name] = new EB::Timer(weight_timer_name.c_str()); 

        // Get contours from rbfzone.blogspot.com to choose eps_c1 and eps_c2 based on stencil_size (n)
        unsigned int stencil_size = 40;
        double eps_c1 = 0.027;
        double eps_c2 = 0.274;


        GridReader* grid = new GridReader(grid_name, 4); 
        grid->setMaxStencilSize(stencil_size); 
        // We do not read until generate is called: 

        Grid::GridLoadErrType err = grid->loadFromFile(); 
        if (err == Grid::NO_GRID_FILES) 
        {
            grid->generate();
            // NOTE: We force at least one node in the domain to be a boundary. 
            //-----------------------------
            // We will set the first node as a boundary/ground point. We know
            // the normal because we're on teh sphere centered at (0,0,0)
            unsigned int nodeIndex = 0; 
            NodeType& node = grid->getNode(nodeIndex); 
            Vec3 nodeNormal = node - Vec3(0,0,0); 
            grid->appendBoundaryIndex(nodeIndex, nodeNormal); 
            //-----------------------------
            if (writeIntermediate) {
                grid->writeToFile(); 
            }
        } 
        std::cout << "Generate Stencils\n";
        Grid::GridLoadErrType st_err = grid->loadStencilsFromFile(); 
        if (st_err == Grid::NO_STENCIL_FILES) {
            //            grid->generateStencils(Grid::ST_BRUTE_FORCE);   
#if 1
            grid->generateStencils(Grid::ST_KDTREE);   
#else 
            grid->setNSHashDims(50, 50,50);  
            grid->generateStencils(Grid::ST_HASH);   
#endif 
            if (writeIntermediate) {
                grid->writeToFile(); 
            }
        }


        std::cout << "Generate RBFFD Weights\n"; 
        timers[weight_timer_name]->start(); 
        RBFFD der(RBFFD::LSFC | RBFFD::XSFC | RBFFD::YSFC | RBFFD::ZSFC, grid, 3, 0); 
        der.setEpsilonByParameters(eps_c1, eps_c2);
        int der_err = der.loadAllWeightsFromFile(); 
        if (der_err) {
            der.computeAllWeightsForAllStencils(); 
            timers[weight_timer_name]->stop(); 

#if 0
            // Im finding that its more efficient to compute the weights than write and load from disk. 
            if (writeIntermediate) {
                der.writeAllWeightsToFile(); 
            }
#endif 
        }

        if (!primed)  {
            std::cout << "\n\n"; 
            cout << "Priming GPU with dummy operations (removes compile from benchmarks)\n";
            gpuTest(der,*grid, 1);
            primed = true; 
            std::cout << "\n\n"; 
        } 

        // No support for GMRES on the CPU yet. 
        //cpuTest(der,*grid);  
        gpuTest(der,*grid);  

        delete(grid); 
    }

    timers.printAll();
    timers.writeToFile();
    return EXIT_SUCCESS;
}

