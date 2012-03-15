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
#include <boost/numeric/ublas/io.hpp>

#include <CL/opencl.h>

#include <iostream>
#include <sstream> 
#include <map>
#include <typeinfo> 
using namespace std;

#define stringify( name ) # name

// Define a couple types that we will work with. 
// Then we can templatize most routines and specialize as necessary 

typedef std::vector< std::map< unsigned int, double> > STL_Sparse_Mat; 
typedef viennacl::compressed_matrix<double> VCL_CSR_Mat; 
typedef viennacl::coordinate_matrix<double> VCL_COO_Mat; 

enum MatrixType : int
{
    COO_CPU=0, COO_GPU, CSR_CPU, CSR_GPU, DUMMY
};

const char* matTypeStrings[] = 
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

template <typename MatT>
void benchmarkMultiplyHost(MatT& A) {
    // If we multiply a vector of 1s we should see our result equal 0 (if our
    // RBF-FD weights are good)
    std::vector<double> x(A.size(), 1);
    std::vector<double> b(A.size(), 1);
    b = viennacl::linalg::prod(A, x); 

    std::cout << "l1   Norm: " << viennacl::linalg::norm_1(b) << std::endl;  
    std::cout << "l2   Norm: " << viennacl::linalg::norm_2(b) << std::endl;  
    std::cout << "linf Norm: " << viennacl::linalg::norm_inf(b) << std::endl;  
}

template <typename MatT>
void benchmarkMultiplyDevice(MatT& A) {
    // If we multiply a vector of 1s we should see our result equal 0 (if our
    // RBF-FD weights are good)
    std::vector<double> x_host(A.size1(), 1);
    viennacl::vector<double> x(A.size1());
    viennacl::copy(x_host.begin(), x_host.end(), x.begin());

    viennacl::vector<double> b(A.size1());
    b = viennacl::linalg::prod(A, x); 

#if 0
    std::vector<double> b_host(A.size1(), 1);
    viennacl::copy(b.begin(), b.end(), b_host.begin());
#endif 
    //viennacl::ocl::current_context().get_queue().finish();

    std::cout << "l1   Norm: " << viennacl::linalg::norm_1(b) << std::endl;  
    std::cout << "l2   Norm: " << viennacl::linalg::norm_2(b) << std::endl;  
    std::cout << "linf Norm: " << viennacl::linalg::norm_inf(b) << std::endl;  
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

template <class MatType, class MultMatType, MatrixType matType, MatrixType multType>
void run_SpMV(RBFFD& der, Grid& grid) {
    unsigned int N = grid.getNodeListSize(); 

    char test_name[256]; 
    char assemble_timer_name[256]; 
    char copy_timer_name[512]; 
    char multiply_timer_name[256]; 

    sprintf(test_name, "%u SpMV (%s -> %s)", N, matTypeStrings[matType], matTypeStrings[multType]); 
    sprintf(assemble_timer_name, "%u %s Assemble", N, matTypeStrings[matType]); 
    sprintf(copy_timer_name,     "%u %s Copy To %s", N, matTypeStrings[matType], matTypeStrings[multType]); 
    sprintf(multiply_timer_name, "%u %s Multiply", N, matTypeStrings[multType]);

    if (!tm.contains(assemble_timer_name)) { tm[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
    if (!tm.contains(copy_timer_name)) { tm[copy_timer_name] = new EB::Timer(copy_timer_name); } 
    if (!tm.contains(multiply_timer_name)) { tm[multiply_timer_name] = new EB::Timer(multiply_timer_name); } 


    std::cout << test_name << std::endl;

    MatType* A = NULL; 
    MultMatType* A_mult = NULL; 

    // Assemble the matrix
    // ----------------------
    tm[assemble_timer_name]->start(); 
    A = new MatType(N); 
    assemble_LHS<MatType>(der, grid, *A);  
    tm[assemble_timer_name]->stop(); 

    tm[copy_timer_name]->start();
    A_mult = new MultMatType(N,N); 
    copy(*A, *A_mult);
    tm[copy_timer_name]->stop();

    tm[multiply_timer_name]->start();
    benchmarkMultiplyDevice<MultMatType>(*A_mult);
    tm[multiply_timer_name]->stop();

    // Cleanup
    delete(A);
    delete(A_mult);
}

template <class MatType, MatrixType matType, MatrixType multType>
void run_SpMV(RBFFD& der, Grid& grid) {

    unsigned int N = grid.getNodeListSize(); 

    char test_name[256]; 
    char assemble_timer_name[256]; 
    char multiply_timer_name[256]; 

    sprintf(test_name, "%u SpMV (%s -> %s)", N, matTypeStrings[matType], matTypeStrings[multType]); 
    sprintf(assemble_timer_name, "%u %s Assemble", N, matTypeStrings[matType]); 
    sprintf(multiply_timer_name, "%u %s Multiply", N, matTypeStrings[matType]);

    if (!tm.contains(assemble_timer_name)) { tm[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
    if (!tm.contains(multiply_timer_name)) { tm[multiply_timer_name] = new EB::Timer(multiply_timer_name); } 

    std::cout << test_name << std::endl;

    // Assemble the matrix
    // ----------------------
    tm[assemble_timer_name]->start(); 
    MatType* A = new MatType(N); 
    assemble_LHS<MatType>(der, grid, *A);  
    tm[assemble_timer_name]->stop(); 

    
    tm[multiply_timer_name]->start();
    benchmarkMultiplyHost<MatType>(*A);
    tm[multiply_timer_name]->stop();

    // Cleanup
    delete(A);
    }

template <class MatType, MatrixType matType, MatrixType multType>
void run_test(RBFFD& der, Grid& grid) {
    switch (multType) {
        case COO_GPU: 
            run_SpMV<MatType, VCL_COO_Mat, matType, multType>(der, grid); 
            break;
        case CSR_GPU:
            run_SpMV<MatType, VCL_CSR_Mat, matType, multType>(der, grid); 
            break; 
        case COO_CPU: 
        case CSR_CPU: 
            run_SpMV<MatType, matType, multType>(der, grid); 
            break; 
        default: 
            std::cout << "ERROR! Unsupported multiply type\n"; 
    }
}

template <MatrixType matType, MatrixType multType>
void run_test(RBFFD& der, Grid& grid) {
    switch (matType) {
        case COO_CPU:
        case CSR_CPU:
            run_test<STL_Sparse_Mat, matType, multType>(der, grid); 
            break;
        case DUMMY: 
            run_SpMV<STL_Sparse_Mat, VCL_COO_Mat, matType, matType>(der, grid); 
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

#if 1 
    //grids.push_back("~/GRIDS/md/md005.00036"); 
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
            run_test<DUMMY, DUMMY>(der, *grid); 
            primed = true; 
        } 

        cout << "Running Tests\n" << std::endl;
        {
            run_test<COO_CPU, COO_CPU>(der, *grid); 
            run_test<COO_CPU, COO_GPU>(der, *grid); 
            run_test<CSR_CPU, CSR_CPU>(der, *grid); 
            run_test<CSR_CPU, CSR_GPU>(der, *grid); 
            run_test<COO_CPU, CSR_GPU>(der, *grid); 
            run_test<CSR_CPU, COO_GPU>(der, *grid); 
        }

        delete(grid); 
    }

    tm.printAll();
    tm.writeToFile();
    return EXIT_SUCCESS;
}

