// THIS IS adapted from verbose_monitor.cu
// PROVIDED BY THE CUSP v0.1 EXAMPLES

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/gmres.h>
#include <cusp/gallery/poisson.h>
#include <cusp/print.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>

#include <iostream>
using namespace std;

int main(void)
{

    int nb_stencils = 10000;
    int stencil_size = 41;

    // Only in using the 2D memory block can we access elements directly for assignment
    // This is a limitation of CUSP at the moment.
    //    cusp::array2d<double, cusp::host_memory> A(nb_stencils, nb_stencils);
    cusp::coo_matrix<int, double, cusp::host_memory> A(nb_stencils,nb_stencils,nb_stencils*stencil_size);

    unsigned int ind = 0; 
    for (int i = 0; i < A.num_rows; i++) {
        // A(i,i) = stencil_size;
        A.row_indices[ind] =  i; 
        A.column_indices[ind] =  i; 
        A.values[ind] =  (double)stencil_size-1; 
        ind++; 

        // Off diagonals
        for (int j = 1; j < stencil_size; j++) {
            //            int k = rand() % A.num_cols;
            int k = (i + j) % A.num_cols;
            // A(i,k) = -1;
            A.row_indices[ind] =  i; 
            A.column_indices[ind] =  k; 
            A.values[ind] =  -1.f; 
            ind++; 
        }
    }

#if 0
    cusp::array2d<double, cusp::host_memory> A_full(A);
    cusp::print(A_full); 
    exit(-1);
#endif 
    cout << "Allocating device memory\n" << std::endl;

    cusp::array1d<double, cusp::host_memory> x(A.num_rows, 1); 
    cusp::array1d<double, cusp::host_memory> b = x; 

    std::cout << "Multiplying matrices\n";
    cusp::multiply(A,x,b); 

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

    return EXIT_SUCCESS;
}

