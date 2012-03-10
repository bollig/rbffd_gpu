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
using namespace std;

// TODO: 
// Sort CSR, ELL, HYB by column. (use std::pair<unsigned int, unsigned int>
// (sten[j], j) and sort on sten[j]. Then use the sorted j's to index sten[]
// and lapl[]

EB::TimerList tm;

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
    b.clear();
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

void test_COO ( RBFFD& der, Grid& grid, int platform) {
    //typedef boost::numerica::ublas::coordinate_matrix<double> 
    typedef std::vector< std::map< unsigned int, double> > MatType; 
    typedef viennacl::coordinate_matrix<double> MatTypeGPU; 

    char *matString = "COO"; 
    char platformString[4]; 
    if (platform) {
        sprintf(platformString, "GPU"); 
    } else {
        sprintf(platformString, "CPU"); 
    }

    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    char assemble_timer_name[256]; 
    char copy_timer_name[256]; 
    char multiply_timer_name[256]; 

    sprintf(assemble_timer_name, "%u %s %s Assemble", N, matString, platformString); 
    sprintf(copy_timer_name,     "%u %s %s Send2Dev", N, matString, platformString); 
    sprintf(multiply_timer_name, "%u %s %s Multiply", N, matString, platformString); 

    if (!tm.contains(assemble_timer_name)) {
        tm[assemble_timer_name] = new EB::Timer(assemble_timer_name);  
        tm[copy_timer_name] = new EB::Timer(copy_timer_name);  
        tm[multiply_timer_name] = new EB::Timer(multiply_timer_name);
    }
    std::cout << "WORKING ON: " << assemble_timer_name << std::endl;
    tm[assemble_timer_name]->start();

    MatType A( N ); // , N , N*n ); 

    for (unsigned int i = 0; i < N; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A[i][sten[j]] = -lapl[j]; 
        }
    }
    tm[assemble_timer_name]->stop();
    std::cout << "\t\t\tMultiply\n";

    if (platform) {
        tm[copy_timer_name]->start();
        MatTypeGPU A_gpu(N,N); 
        copy(A, A_gpu); 
        tm[copy_timer_name]->stop();
        tm[multiply_timer_name]->start();
        benchmarkMultiplyDevice<MatTypeGPU>(A_gpu); 
    } else { 
        tm[multiply_timer_name]->start();
        benchmarkMultiplyHost<MatType>(A); 
    }
    tm[multiply_timer_name]->stop();
}

void test_CSR ( RBFFD& der, Grid& grid, int platform) {
    //typedef boost::numerica::ublas::coordinate_matrix<double> 
    typedef std::vector< std::map< unsigned int, double> > MatType; 
    typedef viennacl::compressed_matrix<double> MatTypeGPU; 

    char *matString = "CSR"; 
    char platformString[4]; 
    if (platform) {
        sprintf(platformString, "GPU"); 
    } else {
        sprintf(platformString, "CPU"); 
    }

    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    char assemble_timer_name[256]; 
    char copy_timer_name[256]; 
    char multiply_timer_name[256]; 

    sprintf(assemble_timer_name, "%u %s %s Assemble", N, matString, platformString); 
    sprintf(copy_timer_name,     "%u %s %s Send2Dev", N, matString, platformString); 
    sprintf(multiply_timer_name, "%u %s %s Multiply", N, matString, platformString); 

    if (!tm.contains(assemble_timer_name)) {
        tm[assemble_timer_name] = new EB::Timer(assemble_timer_name);  
        tm[copy_timer_name] = new EB::Timer(copy_timer_name);  
        tm[multiply_timer_name] = new EB::Timer(multiply_timer_name);
    }
    std::cout << "WORKING ON: " << assemble_timer_name << std::endl;
    tm[assemble_timer_name]->start();

    MatType A( N ); // , N , N*n ); 

    for (unsigned int i = 0; i < N; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A[i][sten[j]] = -lapl[j]; 
        }
    }
    tm[assemble_timer_name]->stop();
    std::cout << "\t\t\tMultiply\n";

    if (platform) {
        tm[copy_timer_name]->start();
        MatTypeGPU A_gpu(N,N); 
        copy(A, A_gpu); 
        tm[copy_timer_name]->stop();
        tm[multiply_timer_name]->start();
        benchmarkMultiplyDevice<MatTypeGPU>(A_gpu); 
    } else { 
        tm[multiply_timer_name]->start();
        benchmarkMultiplyHost<MatType>(A); 
    }
    tm[multiply_timer_name]->stop();
}

#if 0

void test_ELL ( RBFFD& der, Grid& grid, int platform) {
    typedef cusp::ell_matrix<int, double, cusp::host_memory> MatType; 
    typedef cusp::ell_matrix<int, double, cusp::device_memory> MatTypeGPU; 

    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    char *matString = "ELL"; 
    char platformString[4]; 
    if (platform) {
        sprintf(platformString, "GPU"); 
    } else {
        sprintf(platformString, "CPU"); 
    }

    char assemble_timer_name[256]; 
    char copy_timer_name[256]; 
    char multiply_timer_name[256]; 

    sprintf(assemble_timer_name, "%u %s %s Assemble", N, matString, platformString); 
    sprintf(copy_timer_name, "%u %s %s Send2Dev", N, matString, platformString); 
    sprintf(multiply_timer_name, "%u %s %s Multiply", N, matString, platformString); 

    if (!tm.contains(assemble_timer_name)) {
        tm[assemble_timer_name] = new EB::Timer(assemble_timer_name);  
        tm[copy_timer_name] = new EB::Timer(copy_timer_name);  
        tm[multiply_timer_name] = new EB::Timer(multiply_timer_name);
    }
    std::cout << "WORKING ON: " << assemble_timer_name << std::endl;

    tm[assemble_timer_name]->start();

    // Allocate a (N,N) matrix with (N*n) total nonzeros and at most (n) nonzero per row
    MatType A( N , N , N*n , n ); 

    for (int i = 0; i < A.num_rows; i++) {
        StencilType& sten = grid.getStencil(i); 
        // std::vector<unsigned int> sort_ind = grid.getSortedStencilInd(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A.column_indices(i, j) =  sten[j]; 
            A.values(i, j) = -lapl[j]; 
        }
    }
    tm[assemble_timer_name]->stop();
#if 0
    std::cout << "N = " << N << "\t n = " << n << std::endl;
    cusp::array2d<double, cusp::host_memory> A_full(A); 
    cusp::print(A_full); 
    cusp::print(A); 
#endif 
    std::cout << "\t\t\tMultiply\n";
    if (platform) {
        tm[copy_timer_name]->start();
        MatTypeGPU A_gpu(A); 
        tm[copy_timer_name]->stop();
        tm[multiply_timer_name]->start();
        benchmarkMultiplyDevice<MatTypeGPU>(A_gpu); 
    } else { 
        tm[multiply_timer_name]->start();
        benchmarkMultiplyHost<MatType>(A); 
    }
    tm[multiply_timer_name]->stop();
}

void test_HYB ( RBFFD& der, Grid& grid, int platform) {

    // The HYB format has both an ELL (where ALL rows have n nonzeros) and a
    // COO (surplus nonzeros per row. In our case we know we will ALWAYS have n
    // nonzeros for stencil weights per row, unless a weight computes to 0.
    // This means HYB is equivalent to ELL for us. If we convert from ELL to
    // HYB we *might* see a performance boost if their constructor is smart
    // enough to check for 0's, but I doubt it. We will still fill a HYB matrix
    // and test performance. perhaps there are other efficiency differences
    // between the two formats. 

    typedef cusp::hyb_matrix<int, double, cusp::host_memory> MatType; 
    typedef cusp::hyb_matrix<int, double, cusp::device_memory> MatTypeGPU; 

    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    char *matString = "HYB"; 
    char platformString[4]; 
    if (platform) {
        sprintf(platformString, "GPU"); 
    } else {
        sprintf(platformString, "CPU"); 
    }

    char assemble_timer_name[256]; 
    char copy_timer_name[256]; 
    char multiply_timer_name[256]; 

    sprintf(assemble_timer_name, "%u %s %s Assemble", N, matString, platformString); 
    sprintf(copy_timer_name, "%u %s %s Send2Dev", N, matString, platformString); 
    sprintf(multiply_timer_name, "%u %s %s Multiply", N, matString, platformString); 

    if (!tm.contains(assemble_timer_name)) {
        tm[assemble_timer_name] = new EB::Timer(assemble_timer_name);  
        tm[copy_timer_name] = new EB::Timer(copy_timer_name);  
        tm[multiply_timer_name] = new EB::Timer(multiply_timer_name);
    }
    std::cout << "WORKING ON: " << assemble_timer_name << std::endl;

    tm[assemble_timer_name]->start();
    // Allocate a (N,N) matrix with (N*n) total nonzeros and at most (n) nonzero per row
    // and 0 extra non-zeros per row
    MatType A( N , N , N*n , 0 , n ); 

    for (int i = 0; i < A.num_rows; i++) {
        StencilType& sten = grid.getStencil(i); 
        // std::vector<unsigned int> sort_ind = grid.getSortedStencilInd(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A.ell.column_indices(i, j) =  sten[j]; 
            A.ell.values(i, j) = -lapl[j]; 
            // A.coo.row_indices[ind] = 0; ...
        }
    }
    tm[assemble_timer_name]->stop();
#if 0
    std::cout << "N = " << N << "\t n = " << n << std::endl;
    cusp::array2d<double, cusp::host_memory> A_full(A); 
    cusp::print(A_full); 
    cusp::print(A); 
#endif 
    std::cout << "\t\t\tMultiply\n";

    if (platform) {
        tm[copy_timer_name]->start();
        MatTypeGPU A_gpu(A); 
        tm[copy_timer_name]->stop();
        tm[multiply_timer_name]->start();
        benchmarkMultiplyDevice<MatTypeGPU>(A_gpu); 
    } else { 
        tm[multiply_timer_name]->start();
        benchmarkMultiplyHost<MatType>(A); 
    } 
    tm[multiply_timer_name]->stop();
}


#endif 

void testSPMV(int MAT_TYPE, int PLATFORM, RBFFD& der, Grid& grid) { 
    switch (MAT_TYPE) {
        case 0:  
            test_COO(der, grid, PLATFORM); 
            break; 
        case 1: 
            test_CSR(der, grid, PLATFORM); 
            break; 
#if 0
        case 2: 
            test_ELL(der, grid, PLATFORM); 
            break; 
        case 3: 
            test_HYB(der, grid, PLATFORM); 
            break; 
#endif 
        default: 
            std::cout << "INVALID SPMV TYPE\n"; 
            break;  
    }
}


int main(void)
{
    bool writeIntermediate = true; 

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

        cout << "Allocating device memory\n" << std::endl;
#if 0
        for (int k = 0; k < 5; k++) 
#endif 
        {
            for (int j = 0; j < 4; j++) 
            {
                // CPU: 
                testSPMV(j, 0, der, *grid); 
                // GPU: 
                testSPMV(j, 1, der, *grid); 
            }
        }

        delete(grid); 
    }

    tm.printAll();
    tm.writeToFile();
    return EXIT_SUCCESS;
}

