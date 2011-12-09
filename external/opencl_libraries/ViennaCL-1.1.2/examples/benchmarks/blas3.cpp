/* =======================================================================
   Copyright (c) 2010, Institute for Microelectronics, TU Vienna.
   http://www.iue.tuwien.ac.at
                             -----------------
                     ViennaCL - The Vienna Computing Library
                             -----------------
                            
   authors:    Karl Rupp                          rupp@iue.tuwien.ac.at
               Florian Rudolf                     flo.rudy+viennacl@gmail.com
               Josef Weinbub                      weinbub@iue.tuwien.ac.at

   license:    MIT (X11), see file LICENSE in the ViennaCL base directory
======================================================================= */

//disable debug mechanisms to have a fair comparison with ublas:
#define NDEBUG
//#define VIENNACL_BUILD_INFO

//
// include necessary system headers
//
#include <iostream>

//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

// Some helper functions for this tutorial:
#include "../tutorial/Random.hpp"


#include "benchmark-utils.hpp"

/*
*   Tutorial: BLAS level 3 functionality
*   
*/

#define BLAS3_MATRIX_SIZE   1500

template<typename ScalarType>
int run_benchmark()
{
  Timer timer;
  double exec_time;

  //
  // One alternative: Put the matrices into a contiguous block of memory (allows to use viennacl::fast_copy(), avoiding temporary memory)
  //
  std::vector<ScalarType> stl_A(BLAS3_MATRIX_SIZE * BLAS3_MATRIX_SIZE);
  std::vector<ScalarType> stl_B(BLAS3_MATRIX_SIZE * BLAS3_MATRIX_SIZE);
  std::vector<ScalarType> stl_C(BLAS3_MATRIX_SIZE * BLAS3_MATRIX_SIZE);

  //
  // Fill the matrix
  //
  for (unsigned int i = 0; i < BLAS3_MATRIX_SIZE; ++i)
    for (unsigned int j = 0; j < BLAS3_MATRIX_SIZE; ++j)
      stl_A[i*BLAS3_MATRIX_SIZE + j] = random<ScalarType>();

  for (unsigned int i = 0; i < BLAS3_MATRIX_SIZE; ++i)
    for (unsigned int j = 0; j < BLAS3_MATRIX_SIZE; ++j)
      stl_B[i + j*BLAS3_MATRIX_SIZE] = random<ScalarType>();

  //
  // Set up some ViennaCL objects
  //
  viennacl::ocl::set_context_device_type(0, viennacl::ocl::gpu_tag());
  viennacl::matrix<ScalarType> vcl_A(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  viennacl::matrix<ScalarType, viennacl::column_major> vcl_B(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  viennacl::matrix<ScalarType> vcl_C(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  
  /////////////////////////////////////////////////
  //////////// Matrix-matrix products /////////////
  /////////////////////////////////////////////////
  
  //
  // Now iterate over all OpenCL devices in the context and compute the matrix-matrix product
  //
  std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();
  for (size_t i=0; i<devices.size(); ++i)
  {
    viennacl::ocl::current_context().switch_device(devices[i]);
    std::cout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;

    viennacl::fast_copy(&(stl_A[0]),
                        &(stl_A[0]) + stl_A.size(),
                        vcl_A);
    viennacl::fast_copy(&(stl_B[0]),
                        &(stl_B[0]) + stl_B.size(),
                        vcl_B);
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    viennacl::ocl::get_queue().finish();
    timer.start();
    vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
    viennacl::ocl::get_queue().finish();
    exec_time = timer.get();
    std::cout << " - Execution time on device (no setup time included): " << exec_time << std::endl;
    std::cout << " - GFLOPs: " << (vcl_A.size1() / 1000.0) * (vcl_A.size2() / 1000.0) * (vcl_B.size2() / 1000.0) / exec_time << std::endl;
    std::cout << std::endl;
  }
  
  return EXIT_SUCCESS;
}

int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  
  std::cout << viennacl::ocl::current_device().info() << std::endl;
  
  
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Benchmark :: Dense Matrix-Matrix product " << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  run_benchmark<float>();
  if( viennacl::ocl::current_device().double_support() )
  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    run_benchmark<double>();
  }
  return 0;
}
