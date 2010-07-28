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

//
// include necessary system headers
//
#include <iostream>

// enable this to get double precision with AMD Stream SDK on CPUs (experimental!)
//#define VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_CPU
// enable this to get double precision with AMD Stream SDK on GPUs (experimental!)
//#define VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU

//
// Necessary to obtain a suitable performance in ublas
#define NDEBUG

//
// ublas includes
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_HAVE_UBLAS 1

//
// ViennaCL includes
//
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"


// Some helper functions for this tutorial:
#include "Random.hpp"
#include "vector-io.hpp"


/*
*
*   Tutorial no. 4:  Iterative solvers without OpenCL
*   
*/
using namespace boost::numeric;


int main()
{
  typedef float       ScalarType;
  
  //
  // Set up some ublas objects
  //
  ublas::vector<ScalarType> rhs;
  ublas::vector<ScalarType> rhs2;
  ublas::vector<ScalarType> ref_result;
  ublas::vector<ScalarType> result;
  ublas::compressed_matrix<ScalarType> ublas_matrix;
  
  //
  // Read system from file
  //
  #ifdef _MSC_VER
  if (!readMatrixFromFile("../../examples/testdata/matrix65025.txt", ublas_matrix))
  #else
  if (!readMatrixFromFile("../examples/testdata/matrix65025.txt", ublas_matrix))
  #endif
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return 0;
  }
  //unsigned int cg_mat_size = cg_mat.size(); 
  std::cout << "done reading matrix" << std::endl;

  #ifdef _MSC_VER
  if (!readVectorFromFile<ScalarType>("../../examples/testdata/rhs65025.txt", rhs))
  #else
  if (!readVectorFromFile<ScalarType>("../examples/testdata/rhs65025.txt", rhs))
  #endif
  {
    std::cout << "Error reading RHS file" << std::endl;
    return 0;
  }
  std::cout << "done reading rhs" << std::endl;

  #ifdef _MSC_VER
  if (!readVectorFromFile<ScalarType>("../../examples/testdata/result65025.txt", ref_result))
  #else
  if (!readVectorFromFile<ScalarType>("../examples/testdata/result65025.txt", ref_result))
  #endif
  {
    std::cout << "Error reading Result file" << std::endl;
    return 0;
  }
  std::cout << "done reading result" << std::endl;

  
  //
  // set up ILUT preconditioners for ViennaCL and ublas objects:
  // 
  viennacl::linalg::ilut_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
  

  //
  // Conjugate gradient solver:
  //
  std::cout << "----- CG Test -----" << std::endl;
  
  result = solve(ublas_matrix, rhs, viennacl::linalg::cg_tag());
  result = solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(20), ublas_ilut);

  
  //
  // Stabilized BiConjugate gradient solver:
  //
  std::cout << "----- BiCGStab Test -----" << std::endl;

  result = solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag());          //without preconditioner
  result = solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag(20), ublas_ilut); //with preconditioner
  
  //
  // GMRES solver:
  //
  std::cout << "----- GMRES Test -----" << std::endl;

  //
  // for ublas objects:
  //
  result = solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag());   //without preconditioner
  result = solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag(), ublas_ilut);//with preconditioner

  //
  //  That's it. See Tutorial No. 5 for writing custom compute kernels.
  //
  std::cout << "!!!! TUTORIAL 4 COMPLETED SUCCESSFULLY !!!!" << std::endl;
  
  return 0;
}

