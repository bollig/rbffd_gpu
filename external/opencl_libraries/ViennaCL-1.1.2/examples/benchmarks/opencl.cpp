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

//#define VIENNACL_BUILD_INFO
#define NDEBUG

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"

#include <iostream>
#include <vector>
#include "benchmark-utils.hpp"

using std::cout;
using std::cin;
using std::endl;


/*
*   Benchmark 4:
*   Profiling performance of current OpenCL implementation
*   
*/

#define BENCHMARK_VECTOR_SIZE   100000


template<typename ScalarType>
int run_benchmark()
{
   
   Timer timer;
   double exec_time;
   
  std::vector<ScalarType> std_vec1(BENCHMARK_VECTOR_SIZE);
  
  
  viennacl::ocl::get_queue().finish();
  
  timer.start();
  viennacl::scalar<ScalarType> vcl_s1;
  exec_time = timer.get();
  std::cout << "Time for building scalar kernels: " << exec_time << std::endl;
  
  timer.start();
  viennacl::vector<ScalarType> vcl_vec1(BENCHMARK_VECTOR_SIZE);
  exec_time = timer.get();
  viennacl::vector<ScalarType> vcl_vec2(BENCHMARK_VECTOR_SIZE);
  std::cout << "Time for building vector kernels: " << exec_time << std::endl;
  
  timer.start();
  viennacl::matrix<ScalarType> vcl_matrix(BENCHMARK_VECTOR_SIZE/100, BENCHMARK_VECTOR_SIZE/100);
  exec_time = timer.get();
  std::cout << "Time for building matrix kernels: " << exec_time << std::endl;
  
  timer.start();
  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(BENCHMARK_VECTOR_SIZE, BENCHMARK_VECTOR_SIZE);
  exec_time = timer.get();
  std::cout << "Time for building compressed_matrix kernels: " << exec_time << std::endl;
  

  
  ///////////// Vector operations /////////////////
  
  std_vec1[0] = 1.0;
  for (int i=1; i<BENCHMARK_VECTOR_SIZE; ++i)
    std_vec1[i] = std_vec1[i-1] * ScalarType(1.000001);

  viennacl::copy(std_vec1, vcl_vec1);
  
  double std_accumulate = 0;
  double vcl_accumulate = 0;

  timer.start();
  for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
    std_accumulate += std_vec1[i];
  exec_time = timer.get();
  std::cout << "Time for " << BENCHMARK_VECTOR_SIZE << " entry accesses on host: " << exec_time << std::endl;
  std::cout << "Time per entry: " << exec_time / BENCHMARK_VECTOR_SIZE << std::endl;
  std::cout << "Result of operation on host: " << std_accumulate << std::endl;

  vcl_accumulate = vcl_vec1[0];
  viennacl::ocl::get_queue().finish();
  vcl_accumulate = 0;
  timer.start();
  for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
    vcl_accumulate += vcl_vec1[i];
  exec_time = timer.get();
  std::cout << "Time for " << BENCHMARK_VECTOR_SIZE << " entry accesses via OpenCL: " << exec_time << std::endl;
  std::cout << "Time per entry: " << exec_time / BENCHMARK_VECTOR_SIZE << std::endl;
  std::cout << "Result of operation via OpenCL: " << vcl_accumulate << std::endl;
  
  return 0;
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
  std::cout << "## Benchmark :: OpenCL performance" << std::endl;
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

