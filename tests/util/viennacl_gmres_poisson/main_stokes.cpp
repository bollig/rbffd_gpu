#include "grids/grid_reader.h"
#include "rbffd/rbffd.h"
#include "timer_eb.h" 

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
#include "ilu0.hpp"

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

EB::TimerList tm;


//---------------------------------


// Perform GMRES on GPU
void GMRES_Device(VCL_MAT_t& A, VCL_VEC_t& F, VCL_VEC_t& U_exact, VCL_VEC_t& U_approx_gpu, unsigned int N, unsigned int n) {
    //viennacl::linalg::gmres_tag tag;
    //viennacl::linalg::gmres_tag tag(1e-8, 10000, 300); 
    //viennacl::linalg::gmres_tag tag(1e-8, 1000, 3*n); 
    viennacl::linalg::gmres_tag tag(1e-6, 1000, 80); 

    int precond = 0; 
    switch(precond) {
        case 0: 
            {
                //compute ILUT preconditioner (NOT zero fill. This does fill-in according to tag defaults.):
                viennacl::linalg::ilu0_precond< VCL_MAT_t > vcl_ilu0( A, viennacl::linalg::ilu0_tag(0, 3*N)); 
#if 0
                viennacl::io::write_matrix_market_file(vcl_ilu0.LU,"output/ILU.mtx"); 
                std::cout << "Wrote preconditioner to output/ILU.mtx\n";
#endif 
                //solve (e.g. using conjugate gradient solver)
                U_approx_gpu = viennacl::linalg::solve(A, F, tag, vcl_ilu0);
            }
            break; 
        case 1: 
            {
                //compute ILUT preconditioner (NOT zero fill. This does fill-in according to tag defaults.):
                viennacl::linalg::ilut_precond< VCL_MAT_t > vcl_ilut( A, viennacl::linalg::ilut_tag() );
                //solve (e.g. using conjugate gradient solver)
                U_approx_gpu = viennacl::linalg::solve(A, F, tag, vcl_ilut);
            }
            break; 
        case 2: 
            { 
                //compute ILUT preconditioner with 20 nonzeros per row 
                viennacl::linalg::ilut_precond< VCL_MAT_t > vcl_ilut( A, viennacl::linalg::ilut_tag(10) );
                //solve (e.g. using conjugate gradient solver)
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
#if 0
    viennacl::vector_range<VCL_VEC_t> U_exact_view(U_exact, viennacl::range(U_exact.size() - F.size(),U_exact.size()));
#endif 
    VCL_VEC_t diff(F.size()); 
    if (F.size() != U_exact.size()) { 
    exit(-1); 
    }
    viennacl::linalg::sub(U_approx_gpu, U_exact, diff); 

    std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff) / viennacl::linalg::norm_1(U_exact) << std::endl;  
    std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff) / viennacl::linalg::norm_2(U_exact) << std::endl;  
    std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff) / viennacl::linalg::norm_inf(U_exact) << std::endl;  
    std::cout << "----------------------------\n";

    for (int i = 0; i < 4; i++) { 
        VCL_VEC_t uu(N);
        viennacl::copy(U_approx_gpu.begin()+i*N, U_approx_gpu.begin()+((i*N)+N), uu.begin()); 

        VCL_VEC_t uu_exact(N);
        viennacl::copy(U_exact.begin()+i*N, U_exact.begin()+((i*N)+N), uu_exact.begin()); 

        std::cout << "==> Component " << i << "\n"; 
        std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(uu - uu_exact) / viennacl::linalg::norm_1(uu_exact) << std::endl;  
        std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(uu - uu_exact) / viennacl::linalg::norm_2(uu_exact) << std::endl;  
        std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(uu - uu_exact) / viennacl::linalg::norm_inf(uu_exact) << std::endl;  
        std::cout << "----------------------------\n";
    }
}

//---------------------------------

void assemble_System_Stokes( RBFFD& der, Grid& grid, UBLAS_MAT_t& A, UBLAS_VEC_t& F, UBLAS_VEC_t& U_exact){
    double eta = 1.;
    //double Ra = 1.e6;

    // We have different nb_stencils and nb_nodes when we parallelize. The subblocks might not be full
    unsigned int nb_stencils = grid.getStencilsSize();
    unsigned int nb_nodes = grid.getNodeListSize(); 
    //unsigned int max_stencil_size = grid.getMaxStencilSize();
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

        U_exact(row_ind) = UU.eval(Xx,Yy,Zz); 
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

        // F(row_ind) = (Ra * Temperature(j) * dir) / rr;  
        U_exact(row_ind) = VV.eval(Xx,Yy,Zz); 
        F(row_ind) = -VV.lapl(Xx,Yy,Zz) + PP.d_dy(Xx,Yy,Zz);  
    }

    // W
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 2*N;
        NodeType& node = nodes[j];
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        U_exact(row_ind) = WW.eval(Xx,Yy,Zz); 
        F(row_ind) = -WW.lapl(Xx,Yy,Zz) + PP.d_dz(Xx,Yy,Zz);  
    }

    // P
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 3*N;
        NodeType& node = nodes[j]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        U_exact(row_ind) = PP.eval(Xx,Yy,Zz); 
        F(row_ind) = UU.d_dx(Xx,Yy,Zz) + VV.d_dy(Xx,Yy,Zz) + WW.d_dz(Xx,Yy,Zz);  
    }
#endif
    // Sum of U
    F(4*N+0) = 0.;

    // Sum of V
    F(4*N+1) = 0.;

    // Sum of W
    F(4*N+2) = 0.;

    // Sum of P
    F(4*N+3) = 0.;
 




    // -----------------  Fill LHS --------------------
    //
    // U (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddx = der.getStencilWeights(RBFFD::XSFC, i);
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        unsigned int diag_row_ind = i + 0*N;

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 0*N;

            A(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*N;

            A(diag_row_ind, diag_col_ind) = ddx[j];  
        }

        // Added constraint to square mat and close nullspace
        A(diag_row_ind, 4*N+0) = 1.; 
    }

    // V (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddy = der.getStencilWeights(RBFFD::YSFC, i);
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        unsigned int diag_row_ind = i + 1*N;

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 1*N;

            A(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*N;

            A(diag_row_ind, diag_col_ind) = ddy[j];  
        }

        // Added constraint to square mat and close nullspace
        A(diag_row_ind, 4*N+1) = 1.; 
    }

    // W (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddz = der.getStencilWeights(RBFFD::ZSFC, i);
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        unsigned int diag_row_ind = i + 2*N;

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 2*N;

            A(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*N;

            A(diag_row_ind, diag_col_ind) = ddz[j];  
        }

        // Added constraint to square mat and close nullspace
        A(diag_row_ind, 4*N+2) = 1.; 
    }


    // P (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddx = der.getStencilWeights(RBFFD::XSFC, i);
        double* ddy = der.getStencilWeights(RBFFD::YSFC, i);
        double* ddz = der.getStencilWeights(RBFFD::ZSFC, i);

        unsigned int diag_row_ind = i + 3*N;

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 0*N;

            A(diag_row_ind, diag_col_ind) = ddx[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 1*N;

            A(diag_row_ind, diag_col_ind) = ddy[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 2*N;

            A(diag_row_ind, diag_col_ind) = ddz[j];  
        }

        // Added constraint to square mat and close nullspace
        A(diag_row_ind, 4*N+3) = 1.;  
    }

    // ------ EXTRA CONSTRAINT ROWS -----
    unsigned int diag_row_ind = 4*N;
    // U
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 0*N;

        A(diag_row_ind, diag_col_ind) = 1.;  
    }

    diag_row_ind++; 
    // V
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 1*N;

        A(diag_row_ind, diag_col_ind) = 1.;  
    }

    diag_row_ind++; 
    // W
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 2*N;

        A(diag_row_ind, diag_col_ind) = 1.;  
    }

    diag_row_ind++; 
    // P
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 3*N;

        A(diag_row_ind, diag_col_ind) = 1.;  
    }

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


void write_System ( UBLAS_MAT_t& A, UBLAS_VEC_t& F, UBLAS_VEC_t& U_exact )
{
    write_to_file(F, "output/F.mtx"); 
    write_to_file(U_exact, "output/U_exact.mtx"); 
    viennacl::io::write_matrix_market_file(A,"output/LHS.mtx"); 
}

void write_Solution( Grid& grid, UBLAS_VEC_t& U_exact, VCL_VEC_t& U_approx_gpu ) 
{
    //unsigned int nb_bnd = grid.getBoundaryIndicesSize();

    // IF we want to write details we need to copy back to host. 
    UBLAS_VEC_t U_approx(U_exact.size());
    copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin());

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
        sprintf(copy_timer_name,     "%u Primer Copy To VCL_CSR", N); 
        sprintf(test_timer_name, "%u Primer GMRES test", N); 
    } else { 
        sprintf(test_name, "%u GMRES GPU (VCL_CSR)", N);  
        sprintf(assemble_timer_name, "%u UBLAS_CSR Assemble", N);
        sprintf(copy_timer_name,     "%u UBLAS_CSR Copy To VCL_CSR", N); 
        sprintf(test_timer_name, "%u GPU GMRES test", N); 
    }

    if (!tm.contains(assemble_timer_name)) { tm[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
    if (!tm.contains(copy_timer_name)) { tm[copy_timer_name] = new EB::Timer(copy_timer_name); } 
    if (!tm.contains(test_timer_name)) { tm[test_timer_name] = new EB::Timer(test_timer_name); } 


    std::cout << test_name << std::endl;


    // ----- ASSEMBLE -----
    tm[assemble_timer_name]->start(); 
    // Compress system to remove boundary rows
    UBLAS_MAT_t* A = new UBLAS_MAT_t(nrows, ncols, NNZ); 
    UBLAS_VEC_t* F = new UBLAS_VEC_t(nrows, 0);
    UBLAS_VEC_t* U_exact = new UBLAS_VEC_t(nrows, 0);
    assemble_System_Stokes(der, grid, *A, *F, *U_exact); 
    tm[assemble_timer_name]->stop(); 

    write_System(*A, *F, *U_exact); 

    // ----- SOLVE -----

    tm[copy_timer_name]->start();

    VCL_MAT_t* A_gpu = new VCL_MAT_t(A->size1(), A->size2()); 
    copy(*A, *A_gpu);

    VCL_VEC_t* F_gpu = new VCL_VEC_t(F->size());
    VCL_VEC_t* U_exact_gpu = new VCL_VEC_t(U_exact->size());
    VCL_VEC_t* U_approx_gpu = new VCL_VEC_t(F->size());

    viennacl::copy(F->begin(), F->end(), F_gpu->begin());
    viennacl::copy(U_exact->begin(), U_exact->end(), U_exact_gpu->begin());
    tm[copy_timer_name]->stop();

    tm[test_timer_name]->start();
    // Use GMRES to solve A*u = F
    GMRES_Device(*A_gpu, *F_gpu, *U_exact_gpu, *U_approx_gpu, N, n);
    tm[test_timer_name]->stop();

    write_Solution(grid, *U_exact, *U_approx_gpu); 

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

    grids.push_back("~/GRIDS/md/md031.01024"); 
    //grids.push_back("~/GRIDS/md/md089.08100"); 
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

        tm[weight_timer_name] = new EB::Timer(weight_timer_name.c_str()); 

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
#if 0
            // NOTE: We force at least one node in the domain to be a boundary. 
            //-----------------------------
            // We will set the first node as a boundary/ground point. We know
            // the normal because we're on teh sphere centered at (0,0,0)
            unsigned int nodeIndex = 0; 
            NodeType& node = grid->getNode(nodeIndex); 
            Vec3 nodeNormal = node - Vec3(0,0,0); 
            grid->appendBoundaryIndex(nodeIndex, nodeNormal); 
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
        tm[weight_timer_name]->start(); 
        RBFFD der(RBFFD::LSFC | RBFFD::XSFC | RBFFD::YSFC | RBFFD::ZSFC, grid, 3, 0); 
//TODO:         der.setWeightType(RBFFD::ContourSVD);
        der.setEpsilonByParameters(eps_c1, eps_c2);
        int der_err = der.loadAllWeightsFromFile(); 
        if (der_err) {
            der.computeAllWeightsForAllStencils(); 

            tm[weight_timer_name]->stop(); 
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

    tm.printAll();
    tm.writeToFile();
    return EXIT_SUCCESS;
}

