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

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "utils/spherical_harmonics.h"

#include <CL/opencl.h>

#include <iostream>
#include <sstream> 
#include <map>
#include <fstream> 
#include <typeinfo> 
using namespace std;

#define stringify( name ) # name

// Define a couple types that we will work with. 
// Then we can templatize most routines and specialize as necessary 

typedef std::vector< std::map< unsigned int, double> > STL_Sparse_Mat; 
typedef viennacl::compressed_matrix<double> VCL_CSR_Mat; 
typedef viennacl::coordinate_matrix<double> VCL_COO_Mat; 

//typedef std::vector<double> STL_Vec; 
typedef boost::numeric::ublas::vector<double> STL_Vec; 
typedef viennacl::vector<double> VCL_Vec; 

enum MatrixType : int
{
    COO_CPU=0, COO_GPU, CSR_CPU, CSR_GPU, DUMMY
};

const char* assemble_t_eStrings[] = 
{
    //stringify( COO_CPU ), //STL_Sparse_Mat ), 
    stringify( STL_Sparse_Mat ), 
    stringify( VCL_COO_GPU ), 
    stringify( STL_Sparse_Mat ), 
    //stringify( CSR_CPU ), //STL_Sparse_Mat ), 
    stringify( VCL_CSR_GPU ), 
    stringify( DUMMY )
};


// TODO: 
// Sort CSR, ELL, HYB by column. (use std::pair<unsigned int, unsigned int>
// (sten[j], j) and sort on sten[j]. Then use the sorted j's to index sten[]
// and lapl[]
// NOTE: I did this sorting and benchmarked. There was no difference in timing.
// The STL maps auto sort and the assembly on the GPU sorts as well. 

EB::TimerList tm;


//---------------------------------

template <typename MatT, typename VecT>
void benchmark_GMRES_Host(MatT& A, VecT& F, VecT& U_exact) {
    VecT F_discrete(A.size(), 1);
    F_discrete = viennacl::linalg::prod(A, U_exact); 


    std::cout << "Rel l1   Norm: " << boost::numeric::ublas::norm_1(F_discrete - F) / boost::numeric::ublas::norm_1(F) << std::endl;  
    std::cout << "Rel l2   Norm: " << boost::numeric::ublas::norm_2(F_discrete - F) / boost::numeric::ublas::norm_2(F) << std::endl;  
    std::cout << "Rel linf Norm: " << boost::numeric::ublas::norm_inf(F_discrete - F) / boost::numeric::ublas::norm_inf(F) << std::endl;  
}

template <typename MatT, typename VecT>
void benchmark_GMRES_Device(MatT& A, VecT& F, VecT& U_exact) {
    VecT F_discrete(F.size());
    F_discrete = viennacl::linalg::prod(A, U_exact); 

#if 0
    std::vector<double> b_host(A.size1(), 1);
    viennacl::copy(b.begin(), b.end(), b_host.begin());
#endif 
    //viennacl::ocl::current_context().get_queue().finish();

    std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(F_discrete - F) / viennacl::linalg::norm_1(F) << std::endl;  
    std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(F_discrete - F) / viennacl::linalg::norm_2(F) << std::endl;  
    std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(F_discrete - F) / viennacl::linalg::norm_inf(F) << std::endl;  
}

template <class MatType=STL_Sparse_Mat>
void assemble_LHS ( RBFFD& der, Grid& grid, MatType& A){

    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    //A_ptr = new MatType( N ); 
    //    MatType& A     = *A_ptr; 

    for (unsigned int i = 0; i < N; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A[i][sten[j]] = -lapl[j]; 
        }
    }
}

template <class MatType, class VecType=STL_Vec>
void assemble_RHS ( RBFFD& der, Grid& grid, VecType& F, VecType& U_exact){
    SphericalHarmonic::Sph32 UU; 

    unsigned int N = grid.getNodeListSize(); 
    //unsigned int n = grid.getMaxStencilSize(); 
    std::vector<NodeType>& nodes = grid.getNodeList(); 

    for (unsigned int i = 0; i < N; i++) {
        NodeType& node = nodes[i]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        U_exact[i] = UU.eval(Xx, Yy, Zz); 
        // Solving -lapl(u) = f
        F[i] = -UU.lapl(Xx, Yy, Zz); 
    }
}

template <class MatType, class OpMatType, MatrixType assemble_t_e, MatrixType operate_t_e>
void run_SpMV(RBFFD& der, Grid& grid) {
    unsigned int N = grid.getNodeListSize(); 

    char test_name[256]; 
    char assemble_timer_name[256]; 
    char copy_timer_name[512]; 
    char test_timer_name[256]; 

    sprintf(test_name, "%u SpMV (%s -> %s)", N, assemble_t_eStrings[assemble_t_e], assemble_t_eStrings[operate_t_e]); 
    sprintf(assemble_timer_name, "%u %s Assemble", N, assemble_t_eStrings[assemble_t_e]); 
    sprintf(copy_timer_name,     "%u %s Copy To %s", N, assemble_t_eStrings[assemble_t_e], assemble_t_eStrings[operate_t_e]); 
    sprintf(test_timer_name, "%u %s test", N, assemble_t_eStrings[operate_t_e]);

    if (!tm.contains(assemble_timer_name)) { tm[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
    if (!tm.contains(copy_timer_name)) { tm[copy_timer_name] = new EB::Timer(copy_timer_name); } 
    if (!tm.contains(test_timer_name)) { tm[test_timer_name] = new EB::Timer(test_timer_name); } 


    std::cout << test_name << std::endl;

    MatType* A = NULL; 
    OpMatType* A_op = NULL; 

    // Assemble the matrix
    // ----------------------
    tm[assemble_timer_name]->start(); 
    A = new MatType(N); 
    assemble_LHS<MatType>(der, grid, *A);  
    
    STL_Vec* F = new STL_Vec(N, 1);
    STL_Vec* U_exact = new STL_Vec(N, 1);
    assemble_RHS<MatType>(der, grid, *F, *U_exact);  
    tm[assemble_timer_name]->stop(); 

    tm[copy_timer_name]->start();
    A_op = new OpMatType(N,N); 
    copy(*A, *A_op);

    VCL_Vec* F_op = new VCL_Vec(N);
    VCL_Vec* U_exact_op = new VCL_Vec(N);
    viennacl::copy(F->begin(), F->end(), F_op->begin());
    viennacl::copy(U_exact->begin(), U_exact->end(), U_exact_op->begin());
    tm[copy_timer_name]->stop();

    tm[test_timer_name]->start();
    // Use GMRES to solve A*u = F
    benchmark_GMRES_Device<OpMatType>(*A_op, *F_op, *U_exact_op);
    tm[test_timer_name]->stop();

    // Cleanup
    delete(A);
    delete(A_op);
    delete(F);
    delete(U_exact);
    delete(F_op);
    delete(U_exact_op);
}

template <class MatType, MatrixType assemble_t_e, MatrixType operate_t_e>
void run_SpMV(RBFFD& der, Grid& grid) {

    unsigned int N = grid.getNodeListSize(); 

    char test_name[256]; 
    char assemble_timer_name[256]; 
    char test_timer_name[256]; 

    sprintf(test_name, "%u SpMV (%s -> %s)", N, assemble_t_eStrings[assemble_t_e], assemble_t_eStrings[operate_t_e]); 
    sprintf(assemble_timer_name, "%u %s Assemble", N, assemble_t_eStrings[assemble_t_e]); 
    sprintf(test_timer_name, "%u %s test", N, assemble_t_eStrings[assemble_t_e]);

    if (!tm.contains(assemble_timer_name)) { tm[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
    if (!tm.contains(test_timer_name)) { tm[test_timer_name] = new EB::Timer(test_timer_name); } 

    std::cout << test_name << std::endl;

    // Assemble the matrix
    // ----------------------
    tm[assemble_timer_name]->start(); 
    MatType* A = new MatType(N); 
    assemble_LHS<MatType>(der, grid, *A);  

    STL_Vec* F = new STL_Vec(N, 1);
    STL_Vec* U_exact = new STL_Vec(N, 1);
    assemble_RHS<MatType>(der, grid, *F, *U_exact);  
    tm[assemble_timer_name]->stop(); 

#if 0
    std::ofstream f_out("F.mtx"); 
    for (unsigned int i = 0; i < N; i++) {
        f_out << (*F)[i] << std::endl;
    }
    f_out.close();
#endif 

    tm[test_timer_name]->start();
    benchmark_GMRES_Host<MatType>(*A, *F, *U_exact);
    tm[test_timer_name]->stop();

    // Cleanup
    delete(A);
    }

template <class MatType, MatrixType assemble_t_e, MatrixType operate_t_e>
void run_test(RBFFD& der, Grid& grid) {
    switch (operate_t_e) {
        case COO_GPU: 
            run_SpMV<MatType, VCL_COO_Mat, assemble_t_e, operate_t_e>(der, grid); 
            break;
        case CSR_GPU:
            run_SpMV<MatType, VCL_CSR_Mat, assemble_t_e, operate_t_e>(der, grid); 
            break; 
        case COO_CPU: 
        case CSR_CPU: 
            run_SpMV<MatType, assemble_t_e, operate_t_e>(der, grid); 
            break; 
        default: 
            std::cout << "ERROR! Unsupported GMRES type\n"; 
    }
}

template <MatrixType assemble_t_e, MatrixType operate_t_e>
void run_test(RBFFD& der, Grid& grid) {
    switch (assemble_t_e) {
        case COO_CPU:
        case CSR_CPU:
            run_test<STL_Sparse_Mat, assemble_t_e, operate_t_e>(der, grid); 
            break;
        case DUMMY: 
            run_SpMV<STL_Sparse_Mat, VCL_COO_Mat, assemble_t_e, assemble_t_e>(der, grid); 
            break; 
        default: 
            std::cout << "ERROR! Unsupported assembly type\n"; 
    }
}

int main(void)
{
    bool writeIntermediate = true; 
    bool primed = false; 

    std::vector<std::string> grids; 

    //grids.push_back("~/GRIDS/md/md005.00036"); 
    grids.push_back("~/GRIDS/md/md031.01024"); 
#if 1 
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
            run_test<DUMMY, DUMMY>(der, *grid); 
            primed = true; 
        } 

        cout << "Running Tests\n" << std::endl;
        {
            run_test<COO_CPU, COO_CPU>(der, *grid); 
            run_test<COO_CPU, COO_GPU>(der, *grid); 
#if 1
            run_test<CSR_CPU, CSR_CPU>(der, *grid); 
            run_test<CSR_CPU, CSR_GPU>(der, *grid); 
            run_test<COO_CPU, CSR_GPU>(der, *grid); 
            run_test<CSR_CPU, COO_GPU>(der, *grid); 
#endif 
        }

        delete(grid); 
    }

    tm.printAll();
    tm.writeToFile();
    return EXIT_SUCCESS;
}

