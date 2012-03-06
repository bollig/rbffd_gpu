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


#include <thrust/host_vector.h> 
#include <thrust/device_vector.h>
#include <thrust/generate.h>

#include <iostream>
#include <sstream> 
using namespace std;

// TODO: 
//  benchmark assembly
//  benchmark spmv
//  benchmark CPU vs GPU spmv

EB::TimerList tm;

template <typename MatT>
void benchmarkMultiplyHost(MatT& A) {
    // If we multiply a vector of 1s we should see our result equal 0 (if our
    // RBF-FD weights are good)
    cusp::array1d<double, cusp::host_memory> x(A.num_rows, 1);
#if 0
    // generate random data on the host to make sure when we multiply we
    // actually replace b
    thrust::host_vector<double> h_vec(A.num_rows);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);

    // transfer to device and compute sum
    cusp::array1d<double, cusp::host_memory> b = h_vec;
    x = h_vec;
#else 
    cusp::array1d<double, cusp::host_memory> b(A.num_rows, 1);
#endif 
    cusp::multiply(A, x, b); 


    std::cout << "l1   Norm: " << cusp::blas::nrm1(b) << std::endl;  
    std::cout << "l2   Norm: " << cusp::blas::nrm2(b) << std::endl;  
    std::cout << "linf Norm: " << cusp::blas::nrmmax(b) << std::endl;  

#if 0
    cusp::array1d<double, cusp::host_memory> b_host = b;
    std::cout << "l1   Norm: " << cusp::blas::nrm1(b_host) << std::endl;  
    std::cout << "l2   Norm: " << cusp::blas::nrm2(b_host) << std::endl;  
    std::cout << "linf Norm: " << cusp::blas::nrmmax(b_host) << std::endl;  
#endif 
#if 0
    cusp::print(b); 
#endif 
}

template <typename MatT>
void benchmarkMultiplyDevice(MatT& A) {
    // If we multiply a vector of 1s we should see our result equal 0 (if our
    // RBF-FD weights are good)
    cusp::array1d<double, cusp::device_memory> x(A.num_rows, 1);
#if 0
    // generate random data on the host to make sure when we multiply we
    // actually replace b
    thrust::host_vector<double> h_vec(A.num_rows);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);

    // transfer to device and compute sum
    cusp::array1d<double, cusp::device_memory> b = h_vec;
    x = h_vec;
#else 
    cusp::array1d<double, cusp::device_memory> b(A.num_rows, 1);
#endif 
    cusp::multiply(A, x, b); 

#if 1
    std::cout << "l1   Norm: " << cusp::blas::nrm1(b) << std::endl;  
    std::cout << "l2   Norm: " << cusp::blas::nrm2(b) << std::endl;  
    std::cout << "linf Norm: " << cusp::blas::nrmmax(b) << std::endl;  
#endif 
    cudaThreadSynchronize();
#if 0
    cusp::print(b); 
#endif 
}

void test_COO ( RBFFD& der, Grid& grid, int platform) {
    typedef cusp::coo_matrix<int, double, cusp::host_memory> MatType; 
    typedef cusp::coo_matrix<int, double, cusp::device_memory> MatTypeGPU; 

    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    std::ostringstream timer_basic_name; 
    timer_basic_name << N; 
    timer_basic_name << "_COO"; 
    if (platform) {
        timer_basic_name << "_GPU"; 
    } else {
        timer_basic_name << "_CPU"; 
    }

    std::ostringstream assemble_timer_name(timer_basic_name.str()); 
    std::ostringstream multiply_timer_name(timer_basic_name.str()); 
    assemble_timer_name << "_assemble"; 
    std::string assemble_label = timer_basic_name.str() + " Assemble"; 
    multiply_timer_name << "_multiply"; 
    std::string multiply_label = timer_basic_name.str() + " Multiply"; 

    if (!tm.contains(assemble_timer_name.str())) {
        tm[assemble_timer_name.str()] = new EB::Timer(assemble_label.c_str());  
        tm[multiply_timer_name.str()] = new EB::Timer(multiply_label.c_str());
    }
    std::cout << "WORKING ON: " << assemble_label.c_str() << std::endl;
    tm[assemble_timer_name.str()]->start();

    MatType A( N , N , N*n ); 

    unsigned int ind = 0; 
    for (int i = 0; i < A.num_rows; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A.row_indices[ind] =  i; 
            A.column_indices[ind] =  sten[j]; 
            A.values[ind] =  -lapl[j]; 
            ind++; 
        }
    }
    tm[assemble_timer_name.str()]->stop();
#if 0
    std::cout << "N = " << N << "\t n = " << n << std::endl;
    cusp::array2d<double, cusp::host_memory> A_full(A); 
    cusp::print(A_full); 
    cusp::print(A); 
#endif
    std::cout << "\t\t\tMultiply\n";
    tm[multiply_timer_name.str()]->start();
    if (platform) {
        MatTypeGPU A_gpu(A); 
        benchmarkMultiplyDevice<MatTypeGPU>(A_gpu); 
    } else { 
        benchmarkMultiplyHost<MatType>(A); 
    }
    tm[multiply_timer_name.str()]->stop();
}

void test_CSR ( RBFFD& der, Grid& grid, int platform) {
    typedef cusp::csr_matrix<int, double, cusp::host_memory> MatType; 
    typedef cusp::csr_matrix<int, double, cusp::device_memory> MatTypeGPU; 

    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    std::ostringstream timer_basic_name; 
    timer_basic_name << N; 
    timer_basic_name << "_CSR"; 
    if (platform) {
        timer_basic_name << "_GPU"; 
    } else {
        timer_basic_name << "_CPU"; 
    }

    std::ostringstream assemble_timer_name(timer_basic_name.str()); 
    std::ostringstream multiply_timer_name(timer_basic_name.str()); 
    assemble_timer_name << "_assemble"; 
    std::string assemble_label = timer_basic_name.str() + " Assemble"; 
    multiply_timer_name << "_multiply"; 
    std::string multiply_label = timer_basic_name.str() + " Multiply"; 


    if (!tm.contains(assemble_timer_name.str())) {
        tm[assemble_timer_name.str()] = new EB::Timer(assemble_label.c_str());  
        tm[multiply_timer_name.str()] = new EB::Timer(multiply_label.c_str());
    }

    std::cout << "WORKING ON: " << assemble_label.c_str() << std::endl;
    tm[assemble_timer_name.str()]->start();

    MatType A( N , N , N*n ); 

    unsigned int ind = 0; 
    for (int i = 0; i < A.num_rows; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        A.row_offsets[i] = ind;

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A.column_indices[ind] =  sten[j]; 
            A.values[ind] =  -lapl[j]; 
            ind++; 
        }
    }
    tm[assemble_timer_name.str()]->stop();
    std::cout << "\t\t\tMultiply\n"; 
#if 0
    std::cout << "N = " << N << "\t n = " << n << std::endl;
    cusp::array2d<double, cusp::host_memory> A_full(A); 
    cusp::print(A_full); 
    cusp::print(A); 
#endif 
    tm[multiply_timer_name.str()]->start();
    if (platform) {
        MatTypeGPU A_gpu(A); 
        benchmarkMultiplyDevice<MatTypeGPU>(A_gpu); 
    } else { 
        benchmarkMultiplyHost<MatType>(A); 
    }
    tm[multiply_timer_name.str()]->stop();
}


void test_ELL ( RBFFD& der, Grid& grid, int platform) {
    typedef cusp::ell_matrix<int, double, cusp::host_memory> MatType; 
    typedef cusp::ell_matrix<int, double, cusp::device_memory> MatTypeGPU; 

    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    std::ostringstream timer_basic_name; 
    timer_basic_name << N; 
    timer_basic_name << "_ELL"; 
    if (platform) {
        timer_basic_name << "_GPU"; 
    } else {
        timer_basic_name << "_CPU"; 
    }

    std::ostringstream assemble_timer_name(timer_basic_name.str()); 
    std::ostringstream multiply_timer_name(timer_basic_name.str()); 
    assemble_timer_name << "_assemble"; 
    std::string assemble_label = timer_basic_name.str() + " Assemble"; 
    multiply_timer_name << "_multiply"; 
    std::string multiply_label = timer_basic_name.str() + " Multiply"; 

    if (!tm.contains(assemble_timer_name.str())) {
        tm[assemble_timer_name.str()] = new EB::Timer(assemble_label.c_str());  
        tm[multiply_timer_name.str()] = new EB::Timer(multiply_label.c_str());
    }

    std::cout << "WORKING ON: " << assemble_label.c_str() << std::endl;
    tm[assemble_timer_name.str()]->start();

    // Allocate a (N,N) matrix with (N*n) total nonzeros and at most (n) nonzero per row
    MatType A( N , N , N*n , n ); 

    for (int i = 0; i < A.num_rows; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A.column_indices(i, j) =  sten[j]; 
            A.values(i, j) =  -lapl[j]; 
        }
    }
    tm[assemble_timer_name.str()]->stop();
#if 0
    std::cout << "N = " << N << "\t n = " << n << std::endl;
    cusp::array2d<double, cusp::host_memory> A_full(A); 
    cusp::print(A_full); 
    cusp::print(A); 
#endif 
    std::cout << "\t\t\tMultiply\n";
    tm[multiply_timer_name.str()]->start();
    if (platform) {
        MatTypeGPU A_gpu(A); 
        benchmarkMultiplyDevice<MatTypeGPU>(A_gpu); 
    } else { 
        benchmarkMultiplyHost<MatType>(A); 
    }
    tm[multiply_timer_name.str()]->stop();
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

    std::ostringstream timer_basic_name; 
    timer_basic_name << N; 
    timer_basic_name << "_HYB"; 
    if (platform) {
        timer_basic_name << "_GPU"; 
    } else {
        timer_basic_name << "_CPU"; 
    }

    std::ostringstream assemble_timer_name(timer_basic_name.str()); 
    std::ostringstream multiply_timer_name(timer_basic_name.str()); 
    assemble_timer_name << "_assemble"; 
    std::string assemble_label = timer_basic_name.str() + " Assemble"; 
    multiply_timer_name << "_multiply"; 
    std::string multiply_label = timer_basic_name.str() + " Multiply"; 

    if (!tm.contains(assemble_timer_name.str())) {
        tm[assemble_timer_name.str()] = new EB::Timer(assemble_label.c_str());  
        tm[multiply_timer_name.str()] = new EB::Timer(multiply_label.c_str());
    }

    std::cout << "WORKING ON: " << assemble_label.c_str() << std::endl;
    tm[assemble_timer_name.str()]->start();
    // Allocate a (N,N) matrix with (N*n) total nonzeros and at most (n) nonzero per row
    // and 0 extra non-zeros per row
    MatType A( N , N , N*n , 0 , n ); 

    for (int i = 0; i < A.num_rows; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        // Off diagonals
        for (unsigned int j = 0; j < n; j++) {
            A.ell.column_indices(i, j) =  sten[j]; 
            A.ell.values(i, j) =  -lapl[j]; 
            // A.coo.row_indices[ind] = 0; ...
        }
    }
    tm[assemble_timer_name.str()]->stop();
#if 0
    std::cout << "N = " << N << "\t n = " << n << std::endl;
    cusp::array2d<double, cusp::host_memory> A_full(A); 
    cusp::print(A_full); 
    cusp::print(A); 
#endif 
    std::cout << "\t\t\tMultiply\n";
    tm[multiply_timer_name.str()]->start();
    if (platform) {
        MatTypeGPU A_gpu(A); 
        benchmarkMultiplyDevice<MatTypeGPU>(A_gpu); 
    } else { 
        benchmarkMultiplyHost<MatType>(A); 
    } 
    tm[multiply_timer_name.str()]->stop();
}



void testSPMV(int MAT_TYPE, int PLATFORM, RBFFD& der, Grid& grid) { 

    switch (MAT_TYPE) {
        case 0:  
            test_COO(der, grid, PLATFORM); 
            break; 
        case 1: 
            test_CSR(der, grid, PLATFORM); 
            break; 
        case 2: 
            test_ELL(der, grid, PLATFORM); 
            break; 
        case 3: 
            test_HYB(der, grid, PLATFORM); 
            break; 
        default: 
            std::cout << "INVALID SPMV TYPE\n"; 
            break;  
    }
}


int main(void)
{
    bool writeIntermediate = true; 

    std::vector<std::string> grids; 
#if 0
    grids.push_back("~/GRIDS/md/md003.00016"); 
    grids.push_back("~/GRIDS/md/md031.01024"); 
    grids.push_back("~/GRIDS/md/md050.02601"); 
    grids.push_back("~/GRIDS/md/md063.04096"); 
    grids.push_back("~/GRIDS/md/md089.08100"); 
    grids.push_back("~/GRIDS/md/md127.16384"); 
    grids.push_back("~/GRIDS/md/md165.27556"); 
#endif 

    grids.push_back("~/GRIDS/geoff/scvtmesh_100k_nodes.ascii"); 


    for (size_t i = 0; i < grids.size(); i++) {
        std::string& grid_name = grids[i]; 

        std::string weight_timer_name = grid_name + " Calc Weights";  

        tm[weight_timer_name] = new EB::Timer(weight_timer_name.c_str()); 


        // Get contours from rbfzone.blogspot.com to choose eps_c1 and eps_c2 based on stencil_size (n)
        unsigned int stencil_size = 40;
        double eps_c1 = 0.027;
        double eps_c2 = 0.274;


        GridReader grid(grid_name, 4); 
        grid.setMaxStencilSize(stencil_size); 
        // We do not read until generate is called: 

        Grid::GridLoadErrType err = grid.loadFromFile(); 
        if (err == Grid::NO_GRID_FILES) 
        {
            grid.generate();
            if (writeIntermediate) {
                grid.writeToFile(); 
            }
        } 
        std::cout << "Generate Stencils\n";
        Grid::GridLoadErrType st_err = grid.loadStencilsFromFile(); 
        if (st_err == Grid::NO_STENCIL_FILES) {
            //            grid.generateStencils(Grid::ST_BRUTE_FORCE);   
            //grid.generateStencils(Grid::ST_KDTREE);   
            grid.setNSHashDims(50, 50,50);  
            grid.generateStencils(Grid::ST_HASH);   
            if (writeIntermediate) {
                grid.writeToFile(); 
            }
        }


        std::cout << "Generate RBFFD Weights\n"; 
        tm[weight_timer_name]->start(); 
        RBFFD der(RBFFD::LSFC | RBFFD::XSFC | RBFFD::YSFC | RBFFD::ZSFC, &grid, 3, 0); 
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

        // enum MAT_TYPES {COO, CSR, ELL, HYB};
        // enum PLATFORMS {CPU, GPU}; 
        // j indexes MAT_TYPES. 
        for (int k = 0; k < 5; k++) {
#if 1
            for (int j = 0; j < 4; j++) 
#else 
                int j = 3;
#endif 
            {
                // CPU: 
                testSPMV(j, 1, der, grid); 
                // GPU: 
                testSPMV(j, 0, der, grid); 
            }
        }

#if 0
        cusp::array1d<double, cusp::host_memory> x(A.num_rows, 1); 
        cusp::array1d<double, cusp::host_memory> b = x; 

        std::cout << "Multiplying matrices\n";
        cusp::multiply(A,x,b); 
#endif 
    }
#if 0
    exit(-1);
    // Convert the 2D memory block to a sparse representation
    cusp::coo_matrix<int, double, cusp::device_memory> A_dev(A);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<double, cusp::device_memory> x_dev(A.num_rows, 0);
    cusp::array1d<double, cusp::device_memory> b_dev(A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-8
    cusp::verbose_monitor<double> monitor(b, 100, 1e-8);

    cout << "Starting GMRES\n" << std::endl;

    // solve the linear system A x = b
    cusp::krylov::gmres(A_dev, x_dev, b_dev, 30, monitor);

    cout << "GMRES complete\n" << std::endl;

    // monitor will report solver progress and results
#endif 
    tm.printAll();
    tm.writeToFile();
    return EXIT_SUCCESS;
}

