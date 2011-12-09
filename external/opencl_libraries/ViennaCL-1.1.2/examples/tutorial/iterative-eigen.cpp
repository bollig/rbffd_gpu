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

#define NDEBUG

//
// Include Eigen headers
//
#include <Eigen/Core>
#include <Eigen/Sparse>

// Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Eigen objects
#define VIENNACL_HAVE_EIGEN 1

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
#include "../benchmarks/benchmark-utils.hpp"


int main(int, char *[])
{
  typedef float ScalarType;
  
  Eigen::SparseMatrix<ScalarType, Eigen::RowMajor> eigen_matrix(65025, 65025);
  Eigen::VectorXf eigen_rhs;
  Eigen::VectorXf eigen_result;
  Eigen::VectorXf ref_result;
  Eigen::VectorXf residual;
  
  //
  // Read system from file
  //
  eigen_matrix.startFill(65025 * 7);
  #ifdef _MSC_VER
  if (!viennacl::io::read_matrix_market_file(eigen_matrix, "../../examples/testdata/mat65k.mtx"))
  #else
  if (!viennacl::io::read_matrix_market_file(eigen_matrix, "../examples/testdata/mat65k.mtx"))
  #endif
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return 0;
  }
  eigen_matrix.endFill();
  //std::cout << "done reading matrix" << std::endl;

  #ifdef _MSC_VER
  if (!readVectorFromFile("../../examples/testdata/rhs65025.txt", eigen_rhs))
  #else
  if (!readVectorFromFile("../examples/testdata/rhs65025.txt", eigen_rhs))
  #endif
  {
    std::cout << "Error reading RHS file" << std::endl;
    return 0;
  }
  
  #ifdef _MSC_VER
  if (!readVectorFromFile("../../examples/testdata/result65025.txt", ref_result))
  #else
  if (!readVectorFromFile("../examples/testdata/result65025.txt", ref_result))
  #endif
  {
    std::cout << "Error reading Result file" << std::endl;
    return 0;
  }
  
  //CG solver:
  std::cout << "----- Running CG -----" << std::endl;
  eigen_result = viennacl::linalg::solve(eigen_matrix, eigen_rhs, viennacl::linalg::cg_tag());
  
  residual = eigen_matrix * eigen_result - eigen_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(eigen_rhs) << std::endl;

  //BiCGStab solver:
  std::cout << "----- Running BiCGStab -----" << std::endl;
  eigen_result = viennacl::linalg::solve(eigen_matrix, eigen_rhs, viennacl::linalg::bicgstab_tag());
  
  residual = eigen_matrix * eigen_result - eigen_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(eigen_rhs) << std::endl;

  //GMRES solver:
  std::cout << "----- Running GMRES -----" << std::endl;
  eigen_result = viennacl::linalg::solve(eigen_matrix, eigen_rhs, viennacl::linalg::gmres_tag());
  
  residual = eigen_matrix * eigen_result - eigen_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(eigen_rhs) << std::endl;
  
}

