// THIS IS adapted from verbose_monitor.cu
// PROVIDED BY THE CUSP v0.1 EXAMPLES

#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/print.h>
#include <cusp/array2d.h>

#include <iostream>
using namespace std;

int main(void)
{

    int nb_stencils = 10;
    int stencil_size = 4;

    // Only in using the 2D memory block can we access elements directly for assignment
    // This is a limitation of CUSP at the moment.
    cusp::array2d<float, cusp::host_memory> A(nb_stencils, nb_stencils);

    for (int i = 0; i < A.num_rows; i++) {
        A(i,i) = stencil_size;  // Test values
    }

    for (int i = 0; i < A.num_cols; i++) {
	for (int j = 0; j < stencil_size; j++) {
	        int k = rand() % A.num_cols;
        	cout << "RAND: " << k  << endl;
        	A(i,k) = -1;
	}
    }

	cusp::print_matrix(A); 

    // Convert the 2D memory block to a sparse representation
    cusp::coo_matrix<int, float, cusp::device_memory> B(A);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-6
    cusp::verbose_monitor<float> monitor(b, 100, 1e-6);

    // solve the linear system A x = b
    cusp::krylov::cg(B, x, b, monitor);

    // monitor will report solver progress and results

    cusp::print_matrix(A);
    cusp::print_matrix(B);


    return EXIT_SUCCESS;
}

