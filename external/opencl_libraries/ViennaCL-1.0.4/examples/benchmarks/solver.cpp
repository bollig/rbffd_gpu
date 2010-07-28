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
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"


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
*   Benchmark:
*   Iterative solver tests
*   
*/

#define BENCHMARK_RUNS          10


template <typename ScalarType>
ScalarType diff_inf(boost::numeric::ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   boost::numeric::ublas::vector<ScalarType> v2_cpu(v2.size());
   copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( fabs(v2_cpu[i]), fabs(v1[i]) ) > 0 )
         v2_cpu[i] = fabs(v2_cpu[i] - v1[i]) / std::max( fabs(v2_cpu[i]), fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   return norm_inf(v2_cpu);
}

template <typename ScalarType>
ScalarType diff_2(boost::numeric::ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   boost::numeric::ublas::vector<ScalarType> v2_cpu(v2.size());
   copy(v2.begin(), v2.end(), v2_cpu.begin());

   return norm_2(v1 - v2_cpu) / norm_2(v1);
}

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
  boost::numeric::ublas::vector<ScalarType> ublas_result;
  unsigned int solver_iters = 10;
  unsigned int solver_restarts = 3;
  double solver_tolerance = 1e-3;

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
  #ifdef _MSC_VER
  if (!readVectorFromFile<ScalarType>("../../examples/testdata/result65025.txt", ublas_result))
  #else
  if (!readVectorFromFile<ScalarType>("../examples/testdata/result65025.txt", ublas_result))
  #endif
  {
    cout << "Error reading result file" << endl;
    return 0;
  }
  std::cout << "done reading result" << std::endl;
  
  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(static_cast<unsigned int>(ublas_vec1.size()),
                                                                static_cast<unsigned int>(ublas_vec1.size()));
  viennacl::coordinate_matrix<ScalarType> vcl_coordinate_matrix(static_cast<unsigned int>(ublas_vec1.size()),
                                                                static_cast<unsigned int>(ublas_vec1.size()));

  viennacl::vector<ScalarType> vcl_vec1(static_cast<unsigned int>(ublas_vec1.size()));
  viennacl::vector<ScalarType> vcl_vec2(static_cast<unsigned int>(ublas_vec1.size())); 
  viennacl::vector<ScalarType> vcl_result(static_cast<unsigned int>(ublas_vec1.size())); 
  

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
  copy(ublas_matrix, vcl_compressed_matrix);
  copy(ublas_matrix, vcl_coordinate_matrix);
  copy(ublas_vec1, vcl_vec1);
  copy(ublas_vec2, vcl_vec2);
  copy(ublas_result, vcl_result);
  
  
  viennacl::linalg::ilut_precond< boost::numeric::ublas::compressed_matrix<ScalarType> >    ublas_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
  viennacl::linalg::ilut_precond< viennacl::compressed_matrix<ScalarType> > vcl_ilut(vcl_compressed_matrix, viennacl::linalg::ilut_tag());
  
  
  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////           ILUT preconditioner         //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  std::cout << "------- ILUT on CPU ----------" << std::endl;
  
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    ublas_ilut.apply(ublas_vec1);
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  
  std::cout << "------- ILUT on GPU ----------" << std::endl;
  
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_ilut.apply(vcl_vec1);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  

  

  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////              CG solver                //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  long cg_ops = static_cast<long>(static_cast<size_t>(solver_iters) * (static_cast<size_t>(ublas_matrix.nnz()) 
                                                                        + static_cast<size_t>(6) * static_cast<size_t>(ublas_vec2.size())));
  
  std::cout << "------- CG solver (no preconditioner) on CPU ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    ublas_vec1 = viennacl::linalg::solve(ublas_matrix, ublas_vec2, viennacl::linalg::cg_tag(1e-2, solver_iters));
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(cg_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << norm_2(prod(ublas_matrix, ublas_vec1) - ublas_vec2) / norm_2(ublas_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << norm_2(ublas_result - ublas_vec1) / norm_2(ublas_result) << std::endl;
  
  
  std::cout << "------- CG solver (no preconditioner) on GPU, compressed_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::cg_tag(solver_tolerance, solver_iters)); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::cg_tag(solver_tolerance, solver_iters));
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(cg_ops, exec_time / BENCHMARK_RUNS);
  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_compressed_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  #endif  
  std::cout << "Relative deviation from result: " << diff_2(ublas_result, vcl_vec1) << std::endl;
  
//   std::cout << "------- CG solver (no preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
//   
//   
//   vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::cg_tag(solver_tolerance, solver_iters)); //startup calculation
//   std_result = 0.0;
//   viennacl::ocl::finish();
//   timer.start();
//   for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
//   {
//     vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::cg_tag(solver_tolerance, solver_iters));
//   }
//   viennacl::ocl::finish();
//   exec_time = timer.get();
//   std::cout << "GPU time: " << exec_time << std::endl;
//   std::cout << "GPU "; printOps(cg_ops, exec_time / BENCHMARK_RUNS);
//   std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_coordinate_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
//   std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(vcl_result - vcl_vec1) / viennacl::linalg::norm_2(vcl_result) << std::endl;


  std::cout << "------- CG solver (ILUT preconditioner) on CPU ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    ublas_vec1 = viennacl::linalg::solve(ublas_matrix, ublas_vec2, viennacl::linalg::cg_tag(solver_tolerance, solver_iters), ublas_ilut);
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(cg_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << norm_2(prod(ublas_matrix, ublas_vec1) - ublas_vec2) / norm_2(ublas_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << norm_2(ublas_result - ublas_vec1) / norm_2(ublas_result) << std::endl;
  
  std::cout << "------- CG solver (ILUT preconditioner) on GPU, compressed_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::cg_tag(solver_tolerance, solver_iters), vcl_ilut); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::cg_tag(solver_tolerance, solver_iters), vcl_ilut);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(cg_ops, exec_time / BENCHMARK_RUNS);
  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_compressed_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  #endif  
  std::cout << "Relative deviation from result: " << diff_2(ublas_result, vcl_vec1) << std::endl;
  
/*  std::cout << "------- CG solver (ILUT preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::cg_tag(solver_tolerance, solver_iters), vcl_ilut); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::cg_tag(solver_tolerance, solver_iters), vcl_ilut);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(cg_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_coordinate_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(vcl_result - vcl_vec1) / viennacl::linalg::norm_2(vcl_result) << std::endl;*/
  
  
  
  
  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////           BiCGStab solver             //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  
  long bicgstab_ops = static_cast<long>(static_cast<size_t>(solver_iters) * (static_cast<size_t>(2) * static_cast<size_t>(ublas_matrix.nnz()) 
                                                                             + static_cast<size_t>(13) * static_cast<size_t>(ublas_vec2.size())));
  
  std::cout << "------- BiCGStab solver (no preconditioner) on CPU ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    ublas_vec1 = viennacl::linalg::solve(ublas_matrix, ublas_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters));
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(bicgstab_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << norm_2(prod(ublas_matrix, ublas_vec1) - ublas_vec2) / norm_2(ublas_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << norm_2(ublas_result - ublas_vec1) / norm_2(ublas_result) << std::endl;
  
  
  std::cout << "------- BiCGStab solver (no preconditioner) on GPU, compressed_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters)); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters));
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(bicgstab_ops, exec_time / BENCHMARK_RUNS);
  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_compressed_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  #endif  
  std::cout << "Relative deviation from result: " << diff_2(ublas_result, vcl_vec1) << std::endl;

  
/*  std::cout << "------- BiCGStab solver (no preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters)); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters));
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(bicgstab_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_coordinate_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(vcl_result - vcl_vec1) / viennacl::linalg::norm_2(vcl_result) << std::endl;*/
    

  std::cout << "------- BiCGStab solver (ILUT preconditioner) on CPU ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    ublas_vec1 = viennacl::linalg::solve(ublas_matrix, ublas_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters), ublas_ilut);
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(bicgstab_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << norm_2(prod(ublas_matrix, ublas_vec1) - ublas_vec2) / norm_2(ublas_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << norm_2(ublas_result - ublas_vec1) / norm_2(ublas_result) << std::endl;
  
  
  std::cout << "------- BiCGStab solver (ILUT preconditioner) on GPU, compressed_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters), vcl_ilut); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters), vcl_ilut);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(bicgstab_ops, exec_time / BENCHMARK_RUNS);
  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_compressed_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  #endif  
  std::cout << "Relative deviation from result: " << diff_2(ublas_result, vcl_vec1) << std::endl;

  
/*  std::cout << "------- BiCGStab solver (ILUT preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters), vcl_ilut); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::bicgstab_tag(solver_tolerance, solver_iters), vcl_ilut);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(bicgstab_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_coordinate_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(vcl_result - vcl_vec1) / viennacl::linalg::norm_2(vcl_result) << std::endl;*/
  
  ///////////////////////////////////////////////////////////////////////////////
  ///////////////////////            GMRES solver             ///////////////////
  ///////////////////////////////////////////////////////////////////////////////
  
  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  long gmres_ops = static_cast<long>(solver_restarts) * 
                   static_cast<long>(solver_iters) * 
                   static_cast<long>((ublas_matrix.nnz() + (static_cast<size_t>(solver_iters) + static_cast<size_t>(7)) * ublas_vec2.size()));
  
  std::cout << "------- GMRES solver (no preconditioner) on CPU ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    ublas_vec1 = viennacl::linalg::solve(ublas_matrix, ublas_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts));
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(gmres_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << norm_2(prod(ublas_matrix, ublas_vec1) - ublas_vec2) / norm_2(ublas_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << norm_2(ublas_result - ublas_vec1) / norm_2(ublas_result) << std::endl;
  
  
  std::cout << "------- GMRES solver (no preconditioner) on GPU, compressed_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts)); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts));
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(gmres_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_compressed_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(vcl_result - vcl_vec1) / viennacl::linalg::norm_2(vcl_result) << std::endl;

  
/*  std::cout << "------- GMRES solver (no preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts)); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts));
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(gmres_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_coordinate_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(vcl_result - vcl_vec1) / viennacl::linalg::norm_2(vcl_result) << std::endl;*/
  
  
  std::cout << "------- GMRES solver (ILUT preconditioner) on CPU ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    ublas_vec1 = viennacl::linalg::solve(ublas_matrix, ublas_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts), ublas_ilut);
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(gmres_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << norm_2(prod(ublas_matrix, ublas_vec1) - ublas_vec2) / norm_2(ublas_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << norm_2(ublas_result - ublas_vec1) / norm_2(ublas_result) << std::endl;
  
  
  std::cout << "------- GMRES solver (ILUT preconditioner) on GPU, compressed_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts), vcl_ilut); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts), vcl_ilut);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(gmres_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_compressed_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(vcl_result - vcl_vec1) / viennacl::linalg::norm_2(vcl_result) << std::endl;
  
  
/*  std::cout << "------- GMRES solver (ILUT preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
  
  
  vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts), vcl_ilut); //startup calculation
  std_result = 0.0;
  viennacl::ocl::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    vcl_vec1 = viennacl::linalg::solve(vcl_coordinate_matrix, vcl_vec2, viennacl::linalg::gmres_tag(solver_tolerance, solver_iters, solver_restarts), vcl_ilut);
  }
  viennacl::ocl::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(gmres_ops, exec_time / BENCHMARK_RUNS);
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(viennacl::linalg::prod(vcl_coordinate_matrix, vcl_vec1) - vcl_vec2) / viennacl::linalg::norm_2(vcl_vec2) << std::endl;
  std::cout << "Relative deviation from result: " << viennacl::linalg::norm_2(vcl_result - vcl_vec1) / viennacl::linalg::norm_2(vcl_result) << std::endl;*/
  #else
  std::cout << "GMRES leads to wrong results on ATI GPUs, thus skipping benchmark. Consider running GMRES on your CPU using ATI Stream SDK." << std::endl;
  #endif
  
  
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
  std::cout << "## Benchmark :: Solver" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << " ATTENTION: Please be aware that GMRES may not work on ATI GPUs." << std::endl;
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

