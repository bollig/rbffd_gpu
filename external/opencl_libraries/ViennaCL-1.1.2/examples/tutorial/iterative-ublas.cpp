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

//
// include necessary system headers
//
#include <iostream>

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
#include "viennacl/io/matrix_market.hpp"

// Some helper functions for this tutorial:
#include "Random.hpp"
#include "vector-io.hpp"

/*
*
*   Tutorial:  Iterative solvers without OpenCL
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
  if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../../examples/testdata/mat65k.mtx"))
  #else
  if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../examples/testdata/mat65k.mtx"))
  #endif
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return 0;
  }
  //std::cout << "done reading matrix" << std::endl;

  #ifdef _MSC_VER
  if (!readVectorFromFile("../../examples/testdata/rhs65025.txt", rhs))
  #else
  if (!readVectorFromFile("../examples/testdata/rhs65025.txt", rhs))
  #endif
  {
    std::cout << "Error reading RHS file" << std::endl;
    return 0;
  }
  //std::cout << "done reading rhs" << std::endl;

  #ifdef _MSC_VER
  if (!readVectorFromFile("../../examples/testdata/result65025.txt", ref_result))
  #else
  if (!readVectorFromFile("../examples/testdata/result65025.txt", ref_result))
  #endif
  {
    std::cout << "Error reading Result file" << std::endl;
    return 0;
  }
  //std::cout << "done reading result" << std::endl;

  
  //
  // set up ILUT preconditioners for ViennaCL and ublas objects. Other preconditioners can also be used (see manual)
  // 
  viennacl::linalg::ilut_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
  
  //
  // Conjugate gradient solver:
  //
  std::cout << "----- CG Test -----" << std::endl;

  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag());
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(1e-6, 20), ublas_ilut);

  
  //
  // Stabilized BiConjugate gradient solver:
  //
  std::cout << "----- BiCGStab Test -----" << std::endl;

  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag());          //without preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), ublas_ilut); //with preconditioner
  
  //
  // GMRES solver:
  //
  std::cout << "----- GMRES Test -----" << std::endl;

  //
  // for ublas objects:
  //
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag());   //without preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag(1e-6, 20), ublas_ilut);//with preconditioner

  //
  //  That's it. 
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  
  return 0;
}

