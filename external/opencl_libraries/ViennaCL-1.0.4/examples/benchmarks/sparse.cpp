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

   file changelog: - May 28, 2010   New from scratch for first release
======================================================================= */

//#define VCL_BUILD_INFO
#define NDEBUG

#define VIENNACL_HAVE_UBLAS 1

// enable this to get double precision with AMD Stream SDK on CPUs (experimental!)
//#define VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_CPU
// enable this to get double precision with AMD Stream SDK on GPUs (experimental!)
//#define VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"


#include <iostream>
#include <vector>
#include "benchmark-utils.hpp"
#include "io.hpp"

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>

using std::cout;
using std::cin;
using std::endl;


/*
*   Benchmark 1:
*   Vector tests
*   
*/

#define BENCHMARK_VECTOR_SIZE   10000000
#define BENCHMARK_RUNS          10


template<typename ScalarType>
int run_benchmark()
{   
   Timer timer;
   double exec_time;
   
   ScalarType std_result = 0;
   
  ScalarType std_factor1 = static_cast<ScalarType>(3.1415);
  ScalarType std_factor2 = static_cast<ScalarType>(42.0);
  viennacl::scalar<ScalarType> vcl_factor1(std_factor1);
  viennacl::scalar<ScalarType> vcl_factor2(std_factor2);
  
  boost::numeric::ublas::vector<ScalarType> ublas_vec1;
  boost::numeric::ublas::vector<ScalarType> ublas_vec2;
  viennacl::vector<ScalarType> vcl_vec1(BENCHMARK_VECTOR_SIZE);
  viennacl::vector<ScalarType> vcl_vec2(BENCHMARK_VECTOR_SIZE); 
  viennacl::vector<ScalarType> vcl_vec3(BENCHMARK_VECTOR_SIZE); 

  #ifdef _MSC_VER
  if (!readVectorFromFile<ScalarType>("../../examples/testdata/rhs65025.txt", ublas_vec1))
  #else
  if (!readVectorFromFile<ScalarType>("../examples/testdata/rhs65025.txt", ublas_vec1))
  #endif
  {
    cout << "Error reading RHS file" << endl;
    return 0;
  }
  std::cout << "done reading rhs" << std::endl;
  ublas_vec2 = ublas_vec1;
  
  viennacl::compressed_matrix<ScalarType, 1> vcl_compressed_matrix_1;
  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  viennacl::compressed_matrix<ScalarType, 4> vcl_compressed_matrix_4;
  viennacl::compressed_matrix<ScalarType, 8> vcl_compressed_matrix_8;
  #endif
  viennacl::coordinate_matrix<ScalarType> vcl_coordinate_matrix_128;
  
  boost::numeric::ublas::compressed_matrix<ScalarType> ublas_matrix;
  #ifdef _MSC_VER
  if (!readMatrixFromFile("../../examples/testdata/matrix65025.txt", ublas_matrix))
  #else
  if (!readMatrixFromFile("../examples/testdata/matrix65025.txt", ublas_matrix))
  #endif
  {
    cout << "Error reading Matrix file" << endl;
    return 0;
  }
  //unsigned int cg_mat_size = cg_mat.size(); 
  std::cout << "done reading matrix" << std::endl;
  
  //cpu to gpu:
  copy(ublas_matrix, vcl_compressed_matrix_1);
  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  copy(ublas_matrix, vcl_compressed_matrix_4);
  copy(ublas_matrix, vcl_compressed_matrix_8);
  #endif
  copy(ublas_matrix, vcl_coordinate_matrix_128);
  copy(ublas_vec1, vcl_vec1);
  copy(ublas_vec2, vcl_vec2);

  
  ///////////// Matrix operations /////////////////
  
  std::cout << "------- Matrix-Vector product on CPU ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    ublas_vec1 = prod(ublas_matrix, ublas_vec2);
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << ublas_vec1[0] << std::endl;
  
  
  std::cout << "------- Matrix-Vector product with compressed_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::prod(vcl_compressed_matrix_1, vcl_vec2); //startup calculation
  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  vcl_vec1 = viennacl::linalg::prod(vcl_compressed_matrix_4, vcl_vec2); //startup calculation
  vcl_vec1 = viennacl::linalg::prod(vcl_compressed_matrix_8, vcl_vec2); //startup calculation
  #endif
  std_result = 0.0;
  
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::prod(vcl_compressed_matrix_1, vcl_vec2);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time align1: " << exec_time << std::endl;
  std::cout << "GPU align1 "; printOps(static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << vcl_vec1[0] << std::endl;

  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::prod(vcl_compressed_matrix_4, vcl_vec2);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time align4: " << exec_time << std::endl;
  std::cout << "GPU align4 "; printOps(static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << vcl_vec1[0] << std::endl;

  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::prod(vcl_compressed_matrix_8, vcl_vec2);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time align8: " << exec_time << std::endl;
  std::cout << "GPU align8 "; printOps(ublas_matrix.nnz(), exec_time / BENCHMARK_RUNS);
  std::cout << vcl_vec1[0] << std::endl;
  #endif
  
  // vector addition
  
/*  std::cout << "------- Matrix-Vector product with coordinate_matrix ----------" << std::endl;
  
  vcl_vec1 = viennacl::linalg::prod(vcl_coordinate_matrix_128, vcl_vec2); //startup calculation
  viennacl::ocl::finish();
  std_result = 0.0;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::prod(vcl_coordinate_matrix_128, vcl_vec2);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << vcl_vec1[0] << std::endl;
  std::cout << viennacl::linalg::norm_2(vcl_vec1) << std::endl;*/
  
   return 0;
}


int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  
  std::cout << viennacl::ocl::device().info() << std::endl;
  
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Benchmark :: Sparse" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  run_benchmark<float>();
  if( viennacl::ocl::device().double_support() )
  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    run_benchmark<double>();
  }
  return 0;
}

