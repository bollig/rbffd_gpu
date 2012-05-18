// THIS IS adapted from verbose_monitor.cu
// PROVIDED BY THE CUSP v0.1 EXAMPLES

#include "grids/grid_reader.h"
#include "rbffd/rbffd.h"
#include "timer_eb.h" 

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

// Perform GMRES on CPU
void GMRES_Host(UBLAS_MAT_t& A, UBLAS_VEC_t& F, UBLAS_VEC_t& U_exact) {
    std::cout << "ERROR: UBLAS is not supported by ViennaCL GMRES.\n"; 
    exit(-1); 
#if 0
    UBLAS_VEC_t U_approx(U_exact.size());
    viennacl::linalg::gmres_tag tag(1e-8, 100); 
    U_approx = viennacl::linalg::solve(A, U_exact, tag); 

    std::cout << "Rel l1   Norm: " << boost::numeric::ublas::norm_1(U_approx - U_exact) / boost::numeric::ublas::norm_1(U_exact) << std::endl;  
    std::cout << "Rel l2   Norm: " << boost::numeric::ublas::norm_2(U_approx - U_exact) / boost::numeric::ublas::norm_2(U_exact) << std::endl;  
    std::cout << "Rel linf Norm: " << boost::numeric::ublas::norm_inf(U_approx - U_exact) / boost::numeric::ublas::norm_inf(U_exact) << std::endl;  
#endif 
}

// Perform GMRES on GPU
void GMRES_Device(VCL_MAT_t& A, VCL_VEC_t& F, VCL_VEC_t& U_exact) {
#if 1
    VCL_VEC_t U_approx_gpu(U_exact.size());
    U_approx_gpu.clear(); 
    //viennacl::linalg::gmres_tag tag;
    viennacl::linalg::gmres_tag tag(1e-8, 1000, 20); 
    //viennacl::linalg::gmres_tag tag(1e-10, 1000, 20); 

    // Solve Au = F
    U_approx_gpu = viennacl::linalg::solve(A, F, tag); 

    std::cout << "GMRES Iterations: " << tag.iters() << std::endl;
    std::cout << "GMRES Error Estimate: " << tag.error() << std::endl;
    std::cout << "GMRES Krylov Dim: " << tag.krylov_dim() << std::endl;
    std::cout << "GMRES Max Number of Restarts (max_iter/krylov_dim): " << tag.max_restarts() << std::endl;
    std::cout << "GMRES Max Number of Iterations: " << tag.max_iterations() << std::endl;
    std::cout << "GMRES Tolerance: " << tag.tolerance() << std::endl;

    VCL_VEC_t diff = U_approx_gpu - U_exact; 

    std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff) / viennacl::linalg::norm_1(U_exact) << std::endl;  
    std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff) / viennacl::linalg::norm_2(U_exact) << std::endl;  
    std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff) / viennacl::linalg::norm_inf(U_exact) << std::endl;  
#endif 

    // IF we want to write details we need to copy back to host. 
#if 1
    UBLAS_VEC_t U_approx(U_exact.size());
    copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin());

    std::ofstream f_out("output/U_gpu.mtx"); 
    for (unsigned int i = 0; i < U_exact.size(); i++) {
        f_out << U_approx[i] << std::endl;
    }
    f_out.close();
#endif 

}

//---------------------------------

// Assembly depends on the input type. This one for UBLAS_CSR
void assemble_LHS(RBFFD& der, Grid& grid, STL_MAT_t& A){
    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    unsigned int n_bnd = grid.getBoundaryIndicesSize();
    std::cout << "Boundary nodes: " << n_bnd << std::endl;

    for (unsigned int i = n_bnd; i < N; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A[i][sten[j]] = -lapl[j]; 
        }
    }
}

// Assembly depends on the input type. This one for UBLAS_CSR
void assemble_LHS( RBFFD& der, Grid& grid, UBLAS_MAT_t& A){
    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    unsigned int n_bnd = grid.getBoundaryIndicesSize();
    std::cout << "Boundary nodes: " << n_bnd << std::endl;

    for (unsigned int i = n_bnd; i < N; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        for (unsigned int j = 0; j < n; j++) {
            A(i,sten[j]) = -lapl[j]; 
        }
    }
}

//---------------------------------

// This assembly is the same regardless of STL/UBlas vector
template <class MatType, class VecType>
void assemble_RHS ( RBFFD& der, Grid& grid, VecType& F, VecType& U_exact){
    SphericalHarmonic::Sph105 UU; 

    unsigned int N = grid.getNodeListSize(); 
    //unsigned int n = grid.getMaxStencilSize(); 
    std::vector<NodeType>& nodes = grid.getNodeList(); 

    for (unsigned int i = 0; i < N; i++) {
        NodeType& node = nodes[i]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        U_exact[i] = UU.eval(Xx, Yy, Zz) + 2*M_PI; 
        // Solving -lapl(u + const) = f = -lapl(u) + 0
        // of course the lapl(const) is 0, so we will have a test to verify
        // that our null space is closed. 
        F[i] = -UU.lapl(Xx, Yy, Zz); 
    }
}

//---------------------------------

void gpuTest(RBFFD& der, Grid& grid, int primeGPU=0) {
    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

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

    UBLAS_MAT_t* A = NULL; 
    VCL_MAT_t* A_op = NULL; 

    // Assemble the matrix
    // ----------------------
    tm[assemble_timer_name]->start(); 
    A = new UBLAS_MAT_t(N, N, n*N); 
    assemble_LHS(der, grid, *A);  

    UBLAS_VEC_t* F = new UBLAS_VEC_t(N, 1);
    UBLAS_VEC_t* U_exact = new UBLAS_VEC_t(N, 1);
    assemble_RHS<UBLAS_VEC_t>(der, grid, *F, *U_exact);  
    tm[assemble_timer_name]->stop(); 

    tm[copy_timer_name]->start();
    A_op = new VCL_MAT_t(N,N); 
    copy(*A, *A_op);

    VCL_VEC_t* F_op = new VCL_VEC_t(N);
    VCL_VEC_t* U_exact_op = new VCL_VEC_t(N);
    viennacl::copy(F->begin(), F->end(), F_op->begin());
    viennacl::copy(U_exact->begin(), U_exact->end(), U_exact_op->begin());
    tm[copy_timer_name]->stop();

#if 1
    std::ofstream f_out("output/U_exact.mtx"); 
    std::ofstream f_out2("output/F.mtx"); 
    for (unsigned int i = 0; i < N; i++) {
        f_out << (*U_exact)[i] << std::endl;
        f_out2 << (*F)[i] << std::endl;
    }
    f_out.close();
    f_out2.close();
#endif 


    tm[test_timer_name]->start();
    // Use GMRES to solve A*u = F
    GMRES_Device(*A_op, *F_op, *U_exact_op);
    tm[test_timer_name]->stop();

    // Cleanup
    delete(A);
    delete(A_op);
    delete(F);
    delete(U_exact);
    delete(F_op);
    delete(U_exact_op);
}

void cpuTest(RBFFD& der, Grid& grid) {

    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    char test_name[256]; 
    char assemble_timer_name[256]; 
    char test_timer_name[256]; 

    sprintf(test_name, "%u GMRES CPU (VCL_CSR)", N);  
    sprintf(assemble_timer_name, "%u UBLAS_CSR Assemble", N);
    sprintf(test_timer_name, "%u CPU GMRES test", N); 

    if (!tm.contains(assemble_timer_name)) { tm[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
    if (!tm.contains(test_timer_name)) { tm[test_timer_name] = new EB::Timer(test_timer_name); } 

    std::cout << test_name << std::endl;

    // Assemble the matrix
    // ----------------------
    tm[assemble_timer_name]->start(); 
    UBLAS_MAT_t* A = new UBLAS_MAT_t(N,N, n*N); 
    assemble_LHS(der, grid, *A);  

    UBLAS_VEC_t* F = new UBLAS_VEC_t(N, 1);
    UBLAS_VEC_t* U_exact = new UBLAS_VEC_t(N, 1);
    assemble_RHS<UBLAS_VEC_t>(der, grid, *F, *U_exact);  
    tm[assemble_timer_name]->stop(); 

#if 0
    std::ofstream f_out("F.mtx"); 
    for (unsigned int i = 0; i < N; i++) {
        f_out << (*F)[i] << std::endl;
    }
    f_out.close();
#endif 

    tm[test_timer_name]->start();
    GMRES_Host(*A, *F, *U_exact);
    tm[test_timer_name]->stop();

    // Cleanup
    delete(A);
}



int main(void)
{
    bool writeIntermediate = true; 
    bool primed = false; 

    std::vector<std::string> grids; 

    //grids.push_back("~/GRIDS/md/md005.00036"); 

    grids.push_back("~/GRIDS/md/md165.27556"); 
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
        der.setEpsilonByParameters(eps_c1, eps_c2);
        int der_err = der.loadAllWeightsFromFile(); 
        if (der_err) {
            der.computeAllWeightsForAllStencils(); 

            tm[weight_timer_name]->start(); 
            if (writeIntermediate) {
                der.writeAllWeightsToFile(); 
            }
        }

        if (!primed)  {
            cout << "Priming GPU with dummy operations (removes compile from benchmarks)\n";
            gpuTest(der,*grid, 1);
            primed = true; 
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

