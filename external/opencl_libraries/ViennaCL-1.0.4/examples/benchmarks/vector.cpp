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

// enable this to get double precision with AMD Stream SDK on CPUs (experimental!)
//#define VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_CPU
// enable this to get double precision with AMD Stream SDK on GPUs (experimental!)
//#define VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"

#include <iostream>
#include <vector>
#include "benchmark-utils.hpp"

using std::cout;
using std::cin;
using std::endl;


/*
*   Benchmark 1:
*   Vector tests
*   
*/

#define BENCHMARK_VECTOR_SIZE   3000000
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
  
  std::vector<ScalarType> std_vec1(BENCHMARK_VECTOR_SIZE);
  std::vector<ScalarType> std_vec2(BENCHMARK_VECTOR_SIZE);
  std::vector<ScalarType> std_vec3(BENCHMARK_VECTOR_SIZE);
  viennacl::vector<ScalarType> vcl_vec1(BENCHMARK_VECTOR_SIZE);
  viennacl::vector<ScalarType> vcl_vec2(BENCHMARK_VECTOR_SIZE); 
  viennacl::vector<ScalarType> vcl_vec3(BENCHMARK_VECTOR_SIZE); 

  
  ///////////// Vector operations /////////////////
  
  
  
  // inner product
  std::cout << "------- Vector inner products ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
      std_result += std_vec1[i] * std_vec2[i];
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << std_result << std::endl;
  
  
  std_result = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_factor2 = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
  // vector addition
  
  std::cout << "------- Vector addition ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
      std_vec3[i] = std_vec1[i] + std_vec2[i];
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
  
  vcl_vec3 = vcl_vec1 + vcl_vec2; //startup calculation
  viennacl::ocl::finish();
  std_result = 0.0;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec3 = vcl_vec1 + vcl_vec2;
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
  
  
  
  // multiply add:
  std::cout << "------- Vector multiply add ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
      std_vec1[i] += std_factor1 * std_vec2[i];
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
  
  vcl_vec1 += vcl_factor1 * vcl_vec2; //startup calculation
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 += vcl_factor1 * vcl_vec2;
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
 
  
  
  //complicated vector addition:
  std::cout << "------- Vector complicated expression ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    for (int i=0; i<BENCHMARK_VECTOR_SIZE; ++i)
      std_vec3[i] += std_vec2[i] / std_factor1 + std_factor2 * (std_vec1[i] - std_factor1 * std_vec2[i]);
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(3 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
  vcl_vec3 = vcl_vec2 / vcl_factor1 + vcl_factor2 * (vcl_vec1 - vcl_factor1*vcl_vec2); //startup calculation
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec3 = vcl_vec2 / vcl_factor1 + vcl_factor2 * (vcl_vec1 - vcl_factor1*vcl_vec2);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(3 * static_cast<double>(std_vec1.size()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  
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
  std::cout << "## Benchmark :: Vector" << std::endl;
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

