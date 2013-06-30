#include "rbffd/rbffd.h"
#include "grids/grid_reader.h"
#include "timer_eb.h"
#include "precond/ilu0.hpp"

#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>
#include <viennacl/linalg/row_scaling.hpp>
#if 0
// TODO: SPAI and AMG (experimental in VCL 1.2.0 and didnt work for us in CUSP
#endif
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

#include "utils/spherical_harmonics.h"

#include <CL/opencl.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>
#include <fstream>
#include <typeinfo>
using namespace std;


typedef std::vector< std::map< unsigned int, double> > STL_MAT_t;
typedef boost::numeric::ublas::compressed_matrix<double> UBLAS_MAT_t;
typedef boost::numeric::ublas::coordinate_matrix<double> UBLAS_ALT_MAT_t;
typedef viennacl::compressed_matrix<double> VCL_MAT_t;
typedef viennacl::coordinate_matrix<double> VCL_ALT_MAT_t;

typedef std::vector<double> STL_VEC_t;
typedef boost::numeric::ublas::vector<double> UBLAS_VEC_t;
typedef viennacl::vector<double> VCL_VEC_t;

EB::TimerList timers;


//---------------------------------


// Perform GMRES on GPU
void GMRES_Device(VCL_MAT_t& A, VCL_VEC_t& F, VCL_VEC_t& U_exact, VCL_VEC_t& U_approx_gpu, unsigned int N, unsigned int nb_bnd) {
    //viennacl::linalg::gmres_tag tag;
    //viennacl::linalg::gmres_tag tag(1e-8, 10000, 300);
    viennacl::linalg::gmres_tag tag(1e-8, 2000, 250);

    int precond = 1;
    switch(precond) {
        case 0:
            {
                // ILU with 0 fill-in
                viennacl::linalg::ilu0_precond< VCL_MAT_t > vcl_ilu0( A, viennacl::linalg::ilu0_tag());
//                viennacl::io::write_matrix_market_file(vcl_ilu0.LU,"output/ILU.mtx");
             //   std::cout << "Wrote preconditioner to output/ILU.mtx\n";
                U_approx_gpu = viennacl::linalg::solve(A, F, tag, vcl_ilu0);
            }
            break;
        case 1:
            {
                // ILU with 0 fill-in on a subset of the input matrix
                viennacl::linalg::ilu0_precond< VCL_MAT_t > vcl_ilu0( A, viennacl::linalg::ilu0_tag(0, 3*(N-nb_bnd)));
//                viennacl::io::write_matrix_market_file(vcl_ilu0.LU,"output/ILU.mtx");
           //     std::cout << "Wrote preconditioner to output/ILU.mtx\n";
                U_approx_gpu = viennacl::linalg::solve(A, F, tag, vcl_ilu0);
            }
            break;
        case 2:
            {
                //compute ILUT preconditioner (NOT zero fill. This does fill-in according to tag defaults.):
                viennacl::linalg::ilut_precond< VCL_MAT_t > vcl_ilut( A, viennacl::linalg::ilut_tag() );
                U_approx_gpu = viennacl::linalg::solve(A, F, tag, vcl_ilut);
            }
            break;
        case 3:
            {
                //compute ILUT preconditioner with 20 nonzeros per row
                viennacl::linalg::ilut_precond< VCL_MAT_t > vcl_ilut( A, viennacl::linalg::ilut_tag(10) );
                U_approx_gpu = viennacl::linalg::solve(A, F, tag, vcl_ilut);
            }
            break;
        default:
            // Solve Au = F
            U_approx_gpu = viennacl::linalg::solve(A, F, tag);
    };
    std::cout << "GMRES Iterations: " << tag.iters() << std::endl;
    std::cout << "GMRES Error Estimate: " << tag.error() << std::endl;
    std::cout << "GMRES Krylov Dim: " << tag.krylov_dim() << std::endl;
    std::cout << "GMRES Max Number of Restarts (max_iter/krylov_dim): " << tag.max_restarts() << std::endl;
    std::cout << "GMRES Max Number of Iterations: " << tag.max_iterations() << std::endl;
    std::cout << "GMRES Tolerance: " << tag.tolerance() << std::endl;

    VCL_VEC_t diff(F.size());
    viennacl::linalg::sub(U_approx_gpu, U_exact, diff);

    std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff) / viennacl::linalg::norm_1(U_exact) << std::endl;
    std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff) / viennacl::linalg::norm_2(U_exact) << std::endl;
    std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff) / viennacl::linalg::norm_inf(U_exact) << std::endl;
    std::cout << "----------------------------\n";

    for (int i = 0; i < 4; i++) {
        unsigned int NN = N - nb_bnd;
        VCL_VEC_t uu(NN);
        viennacl::copy(U_approx_gpu.begin()+i*NN, U_approx_gpu.begin()+((i*NN)+(NN)), uu.begin());

        VCL_VEC_t uu_exact(NN);
        viennacl::copy(U_exact.begin()+i*NN, U_exact.begin()+((i*NN)+(NN)), uu_exact.begin());
        VCL_VEC_t diff = uu - uu_exact;

        std::cout << "==> Component " << i << "\n";
        std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff) / viennacl::linalg::norm_1(uu_exact) << std::endl;
        std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff) / viennacl::linalg::norm_2(uu_exact) << std::endl;
        std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff) / viennacl::linalg::norm_inf(uu_exact) << std::endl;
        std::cout << "----------------------------\n";
    }

}

//---------------------------------

void assemble_System_Stokes( RBFFD& der, Grid& grid, UBLAS_MAT_t& A, UBLAS_VEC_t& F, UBLAS_VEC_t& U_exact, UBLAS_VEC_t& U_exact_compressed){
    double eta = 1.;
    //double Ra = 1.e6;

    // We have different nb_stencils and nb_nodes when we parallelize. The subblocks might not be full
    unsigned int nb_stencils = grid.getStencilsSize();
   // unsigned int nb_nodes = grid.getNodeListSize();
   // unsigned int max_stencil_size = grid.getMaxStencilSize();
    unsigned int N = nb_stencils;
    unsigned int nb_bnd = grid.getBoundaryIndicesSize();
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
    for (unsigned int j = 0; j < nb_bnd; j++) {
        unsigned int row_ind = j + 0*(N);
        NodeType& node = nodes[j];
        double Xx = node.x();
        double Yy = node.y();
        double Zz = node.z();

        U_exact(row_ind) = UU.eval(Xx, Yy, Zz);
    }
    for (unsigned int j = nb_bnd; j < N; j++) {
        unsigned int row_ind = j + 0*(N-nb_bnd);
        unsigned int uncompressed_row_ind = j + 0*(N);
        NodeType& node = nodes[j];
        double Xx = node.x();
        double Yy = node.y();
        double Zz = node.z();

        U_exact(uncompressed_row_ind) = UU.eval(Xx,Yy,Zz);
        U_exact_compressed(row_ind-nb_bnd) = UU.eval(Xx,Yy,Zz);
        F(row_ind-nb_bnd) = -eta * UU.lapl(Xx,Yy,Zz) + PP.d_dx(Xx,Yy,Zz);
    }

    // V
    for (unsigned int j = 0; j < nb_bnd; j++) {
        unsigned int row_ind = j + 1*(N);
        NodeType& node = nodes[j];
        double Xx = node.x();
        double Yy = node.y();
        double Zz = node.z();

        U_exact(row_ind) = VV.eval(Xx,Yy,Zz);
    }
    for (unsigned int j = nb_bnd; j < N; j++) {
        unsigned int row_ind = j + 1*(N-nb_bnd);
        unsigned int uncompressed_row_ind = j + 1*(N);
        NodeType& node = nodes[j];
        double Xx = node.x();
        double Yy = node.y();
        double Zz = node.z();

        U_exact(uncompressed_row_ind) = VV.eval(Xx,Yy,Zz);
        U_exact_compressed(row_ind-nb_bnd) = VV.eval(Xx,Yy,Zz);
        F(row_ind-nb_bnd) = -eta * VV.lapl(Xx,Yy,Zz) + PP.d_dy(Xx,Yy,Zz);
    }


    // W
    for (unsigned int j = 0; j < nb_bnd; j++) {
        unsigned int row_ind = j + 2*(N);
        NodeType& node = nodes[j];
        double Xx = node.x();
        double Yy = node.y();
        double Zz = node.z();

        U_exact(row_ind) = WW.eval(Xx,Yy,Zz);
    }

    for (unsigned int j = nb_bnd; j < N; j++) {
        unsigned int row_ind = j + 2*(N-nb_bnd);
        unsigned int uncompressed_row_ind = j + 2*(N);
        NodeType& node = nodes[j];
        double Xx = node.x();
        double Yy = node.y();
        double Zz = node.z();

        U_exact(uncompressed_row_ind) = WW.eval(Xx,Yy,Zz);
        U_exact_compressed(row_ind-nb_bnd) = WW.eval(Xx,Yy,Zz);
        F(row_ind-nb_bnd) = -eta * WW.lapl(Xx,Yy,Zz) + PP.d_dz(Xx,Yy,Zz);
    }

    // P
    for (unsigned int j = 0; j < nb_bnd; j++) {
        unsigned int row_ind = j + 3*(N);
        NodeType& node = nodes[j];
        double Xx = node.x();
        double Yy = node.y();
        double Zz = node.z();

        U_exact(row_ind) = PP.eval(Xx,Yy,Zz);
    }


    for (unsigned int j = nb_bnd; j < N; j++) {
        unsigned int row_ind = j + 3*(N-nb_bnd);
        unsigned int uncompressed_row_ind = j + 3*(N);
        NodeType& node = nodes[j];
        double Xx = node.x();
        double Yy = node.y();
        double Zz = node.z();

        U_exact(uncompressed_row_ind) = PP.eval(Xx,Yy,Zz);
        U_exact_compressed(row_ind-nb_bnd) = PP.eval(Xx,Yy,Zz);
        F(row_ind-nb_bnd) = UU.d_dx(Xx,Yy,Zz) + VV.d_dy(Xx,Yy,Zz) + WW.d_dz(Xx,Yy,Zz);
    }

    // -----------------  Fill LHS --------------------
    //
    // U (block)  row
    for (unsigned int i = nb_bnd; i < N; i++) {
        StencilType& st = grid.getStencil(i);

        // System has form:
        // -lapl(U) + grad(P) = f
        // div(U) = 0

        // TODO: change these to *SFC weights (when computed)
        double* ddx = der.getStencilWeights(RBFFD::XSFC, i);
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i);

        unsigned int diag_row_ind = i + 0*(N-nb_bnd);

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 0*(N-nb_bnd);

            if (st[j] < (int)nb_bnd) {
                // Need the exact solution at stencil node j
                NodeType& node = nodes[st[j]];
                double Xx = node.x();
                double Yy = node.y();
                double Zz = node.z();

                F[diag_row_ind-nb_bnd] -= -eta * UU.lapl(Xx, Yy, Zz);
            } else {
                A(diag_row_ind-nb_bnd, diag_col_ind-nb_bnd) = -eta * lapl[j];
            }
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*(N-nb_bnd);

            if (st[j] < (int)nb_bnd) {
                // Need the exact solution at stencil node j
                NodeType& node = nodes[st[j]];
                double Xx = node.x();
                double Yy = node.y();
                double Zz = node.z();

                F[diag_row_ind-nb_bnd] -= PP.d_dx(Xx, Yy, Zz);
            } else {
                A(diag_row_ind-nb_bnd, diag_col_ind-nb_bnd) = ddx[j];
            }
        }
    }


    // V (block)  row
    for (unsigned int i = nb_bnd; i < N; i++) {
        StencilType& st = grid.getStencil(i);
        double* ddy = der.getStencilWeights(RBFFD::YSFC, i);
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i);

        unsigned int diag_row_ind = i + 1*(N-nb_bnd);

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 1*(N-nb_bnd);

            if (st[j] < (int)nb_bnd) {
                // Need the exact solution at stencil node j
                NodeType& node = nodes[st[j]];
                double Xx = node.x();
                double Yy = node.y();
                double Zz = node.z();

                F[diag_row_ind-nb_bnd] -= -eta * VV.lapl(Xx, Yy, Zz);
            } else {
                A(diag_row_ind-nb_bnd, diag_col_ind-nb_bnd) = -eta * lapl[j];
            }
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*(N-nb_bnd);

            if (st[j] < (int)nb_bnd) {
                // Need the exact solution at stencil node j
                NodeType& node = nodes[st[j]];
                double Xx = node.x();
                double Yy = node.y();
                double Zz = node.z();

                F[diag_row_ind-nb_bnd] -= PP.d_dy(Xx, Yy, Zz);
            } else {
                A(diag_row_ind-nb_bnd, diag_col_ind-nb_bnd) = ddy[j];
            }
        }
    }

    // W (block)  row
    for (unsigned int i = nb_bnd; i < N; i++) {
        StencilType& st = grid.getStencil(i);
        double* ddz = der.getStencilWeights(RBFFD::ZSFC, i);
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i);

        unsigned int diag_row_ind = i + 2*(N-nb_bnd);

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 2*(N-nb_bnd);

            if (st[j] < (int)nb_bnd) {
                NodeType& node = nodes[st[j]];
                double Xx = node.x();
                double Yy = node.y();
                double Zz = node.z();

                F[diag_row_ind-nb_bnd] -= -eta * WW.lapl(Xx, Yy, Zz);
            } else {
                A(diag_row_ind-nb_bnd, diag_col_ind-nb_bnd) = -eta * lapl[j];
            }
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*(N-nb_bnd);

            if (st[j] < (int)nb_bnd) {
                NodeType& node = nodes[st[j]];
                double Xx = node.x();
                double Yy = node.y();
                double Zz = node.z();

                F[diag_row_ind-nb_bnd] -= PP.d_dz(Xx, Yy, Zz);
            } else {
                A(diag_row_ind-nb_bnd, diag_col_ind-nb_bnd) = ddz[j];
            }
        }
    }

    // P (block)  row
    for (unsigned int i = nb_bnd; i < N; i++) {
        StencilType& st = grid.getStencil(i);
        double* ddx = der.getStencilWeights(RBFFD::XSFC, i);
        double* ddy = der.getStencilWeights(RBFFD::YSFC, i);
        double* ddz = der.getStencilWeights(RBFFD::ZSFC, i);

        unsigned int diag_row_ind = i + 3*(N-nb_bnd);

        // ddx(U)-component
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 0*(N-nb_bnd);

            if (st[j] < (int)nb_bnd) {
                NodeType& node = nodes[st[j]];
                double Xx = node.x();
                double Yy = node.y();
                double Zz = node.z();

                F[diag_row_ind-nb_bnd] -= UU.d_dx(Xx, Yy, Zz);
            } else {
                A(diag_row_ind-nb_bnd, diag_col_ind-nb_bnd) = ddx[j];
            }
        }

        // ddy(V)-component
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 1*(N-nb_bnd);

            if (st[j] < (int)nb_bnd) {
                NodeType& node = nodes[st[j]];
                double Xx = node.x();
                double Yy = node.y();
                double Zz = node.z();

                F[diag_row_ind-nb_bnd] -= VV.d_dx(Xx, Yy, Zz);
            } else {
                A(diag_row_ind-nb_bnd, diag_col_ind-nb_bnd) = ddy[j];
            }
        }

        // ddz(W)-component
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 2*(N-nb_bnd);

            if (st[j] < (int)nb_bnd) {
                NodeType& node = nodes[st[j]];
                double Xx = node.x();
                double Yy = node.y();
                double Zz = node.z();

                F[diag_row_ind-nb_bnd] -= WW.d_dx(Xx, Yy, Zz);
            } else {
                A(diag_row_ind-nb_bnd, diag_col_ind-nb_bnd) = ddz[j];
            }
        }
    }
}





    template <typename VecT>
void write_to_file(VecT vec, std::string filename)
{
    std::ofstream fout;
    fout.open(filename.c_str());
    for (size_t i = 0; i < vec.size(); i++) {
        //   fout << i << "\t" << std::setprecision(10) << vec[i] << std::endl;
        fout << std::setprecision(10) << vec[i] << std::endl;
    }
    fout.close();
    std::cout << "Wrote " << filename << std::endl;
}


void write_System ( UBLAS_MAT_t& A, UBLAS_VEC_t& F, UBLAS_VEC_t& U_exact )
{
    write_to_file(F, "output/F.mtx");
    write_to_file(U_exact, "output/U_exact.mtx");
//    viennacl::io::write_matrix_market_file(A,"output/LHS.mtx");
}

void write_Solution( Grid& grid, UBLAS_VEC_t& U_exact_compressed, VCL_VEC_t& U_approx_gpu )
{
    //unsigned int nb_bnd = grid.getBoundaryIndicesSize();

    // IF we want to write details we need to copy back to host.
    UBLAS_VEC_t U_approx(U_exact_compressed.size());
    copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin());

    write_to_file(U_approx, "output/U_gpu.mtx");
}


//---------------------------------

void gpuTest(RBFFD& der, Grid& grid, int primeGPU=0) {
    unsigned int N = grid.getStencilsSize();
    unsigned int n = grid.getMaxStencilSize();
    unsigned int nb_bnd = grid.getBoundaryIndicesSize();
    unsigned int n_unknowns = 4 * N;
    // We subtract off the unknowns for the boundary
    unsigned int nrows = 4 * N - 4*nb_bnd;
    unsigned int ncols = 4 * N - 4*nb_bnd;
    unsigned int NNZ = 9*n*N+2*(4*N)+2*(3*N);

    char test_name[256];
    char assemble_timer_name[256];
    char copy_timer_name[512];
    char test_timer_name[256];

    if (primeGPU) {
        sprintf(test_name, "%u PRIMING THE GPU", N);
        sprintf(assemble_timer_name, "%u Primer Assemble", N);
        sprintf(copy_timer_name,     "%u Primer Copy To VCL_CSR", N);
        sprintf(test_timer_name, "%u Primer GMRES test", N);
    } else {
        sprintf(test_name, "%u GMRES GPU (VCL_CSR)", N);
        sprintf(assemble_timer_name, "%u UBLAS_CSR Assemble", N);
        sprintf(copy_timer_name,     "%u UBLAS_CSR Copy To VCL_CSR", N);
        sprintf(test_timer_name, "%u GPU GMRES test", N);
    }

    if (!timers.contains(assemble_timer_name)) { timers[assemble_timer_name] = new EB::Timer(assemble_timer_name); }
    if (!timers.contains(copy_timer_name)) { timers[copy_timer_name] = new EB::Timer(copy_timer_name); }
    if (!timers.contains(test_timer_name)) { timers[test_timer_name] = new EB::Timer(test_timer_name); }


    std::cout << test_name << std::endl;


    // ----- ASSEMBLE -----
    timers[assemble_timer_name]->start();
    // Compress system to remove boundary rows
    UBLAS_MAT_t* A = new UBLAS_MAT_t(nrows, ncols, NNZ);
    UBLAS_VEC_t* F = new UBLAS_VEC_t(nrows, 0);
    UBLAS_VEC_t* U_exact = new UBLAS_VEC_t(n_unknowns, 0);
    UBLAS_VEC_t* U_exact_compressed = new UBLAS_VEC_t(nrows, 0);
    assemble_System_Stokes(der, grid, *A, *F, *U_exact, *U_exact_compressed);
    timers[assemble_timer_name]->stop();

    write_System(*A, *F, *U_exact);
    write_to_file(*U_exact_compressed, "output/U_exact_compressed.mtx");

    UBLAS_VEC_t F_discrete = prod(*A, *U_exact_compressed);
    write_to_file(F_discrete, "output/F_discrete.mtx");

    // ----- SOLVE -----

    timers[copy_timer_name]->start();

    VCL_MAT_t* A_gpu = new VCL_MAT_t(A->size1(), A->size2());
    copy(*A, *A_gpu);

    VCL_VEC_t* F_gpu = new VCL_VEC_t(F->size());
    VCL_VEC_t* U_exact_gpu = new VCL_VEC_t(U_exact_compressed->size());
    VCL_VEC_t* U_approx_gpu = new VCL_VEC_t(F->size());

    viennacl::copy(F->begin(), F->end(), F_gpu->begin());
    viennacl::copy(U_exact_compressed->begin(), U_exact_compressed->end(), U_exact_gpu->begin());
    timers[copy_timer_name]->stop();

    timers[test_timer_name]->start();
    // Use GMRES to solve A*u = F
    GMRES_Device(*A_gpu, *F_gpu, *U_exact_gpu, *U_approx_gpu, N, nb_bnd);
    timers[test_timer_name]->stop();

    write_Solution(grid, *U_exact_compressed, *U_approx_gpu);

    // Cleanup
    delete(A);
    delete(A_gpu);
    delete(F);
    delete(U_exact);
    delete(U_exact_compressed);
    delete(F_gpu);
    delete(U_exact_gpu);
    delete(U_approx_gpu);
}



int main(void)
{
    bool writeIntermediate = true;
    bool primed = false;

    std::vector<std::string> grids;

    //    grids.push_back("~/GRIDS/md/md005.00036");

    //grids.push_back("~/GRIDS/md/md165.27556");
    //grids.push_back("~/GRIDS/md/md031.01024");
    grids.push_back("~/GRIDS/md/md089.08100");
#if 0
    grids.push_back("~/GRIDS/md/md031.01024");
    grids.push_back("~/GRIDS/md/md050.02601");
    grids.push_back("~/GRIDS/md/md063.04096");
    grids.push_back("~/GRIDS/md/md089.08100");
    grids.push_back("~/GRIDS/md/md127.16384");
    grids.push_back("~/GRIDS/md/md165.27556");
#endif
#if 0
    grids.push_back("~/GRIDS/geoff/scvtmesh_100k_nodes.ascii");
    grids.push_back("~/GRIDS/geoff/scvtmesh_500k_nodes.ascii");
    grids.push_back("~/GRIDS/geoff/scvtmesh_1m_nodes.ascii");
#endif
    //grids.push_back("~/GRIDS/geoff/scvtmesh_1m_nodes.ascii");

    for (size_t i = 0; i < grids.size(); i++) {
        std::string& grid_name = grids[i];

        std::string weight_timer_name = grid_name + " Calc Weights";

        timers[weight_timer_name] = new EB::Timer(weight_timer_name.c_str());

        // Get contours from rbfzone.blogspot.com to choose eps_c1 and eps_c2 based on stencil_size (n)
#if 0
        // Too ill-conditioned? Doesnt converge in GMRES + ILU0
        unsigned int stencil_size = 40;
        double eps_c1 = 0.027;
        double eps_c2 = 0.274;
#else
        unsigned int stencil_size = 31;
        double eps_c1 = 0.035;
        double eps_c2 = 0.1;
#endif


        GridReader* grid = new GridReader(grid_name, 4);
        grid->setMaxStencilSize(stencil_size);
        // We do not read until generate is called:

        Grid::GridLoadErrType err = grid->loadFromFile();
        if (err == Grid::NO_GRID_FILES)
        {
            grid->generate();
#if 1
            // NOTE: We force at least one node in the domain to be a boundary.
            //-----------------------------
            // We will set the first node as a boundary/ground point. We know
            // the normal because we're on teh sphere centered at (0,0,0)
            for (unsigned int nodeIndex = 0; nodeIndex < 1; nodeIndex++) {
                NodeType& node = grid->getNode(nodeIndex);
                Vec3 nodeNormal = node - Vec3(0,0,0);
                grid->appendBoundaryIndex(nodeIndex, nodeNormal);
            }
#endif
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
        //TODO:         der.setWeightType(RBFFD::ContourSVD);
        der.setEpsilonByParameters(eps_c1, eps_c2);
        int der_err = der.loadAllWeightsFromFile();
        if (der_err) {
            der.computeAllWeightsForAllStencils();

            timers[weight_timer_name]->stop();
#if 0
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

