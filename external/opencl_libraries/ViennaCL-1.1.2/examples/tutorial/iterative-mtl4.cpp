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

//#define NDEBUG

//
// Include MTL4 headers
//
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

// Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Eigen objects
#define VIENNACL_HAVE_MTL4 1

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

int main(int, char *[])
{
  
  mtl::compressed2D<double> mtl4_matrix;
  mtl4_matrix.change_dim(65025, 65025);
  set_to_zero(mtl4_matrix);  
  
  mtl::dense_vector<double> mtl4_rhs(65025, 0.0);
  mtl::dense_vector<double> mtl4_result(65025, 0.0);
  mtl::dense_vector<double> mtl4_ref_result(65025, 0.0);
  mtl::dense_vector<double> mtl4_residual(65025, 0.0);
  
  //
  // Read system from file
  //
  #ifdef _MSC_VER
  if (!viennacl::io::read_matrix_market_file(mtl4_matrix, "../../examples/testdata/mat65k.mtx"))
  #else
  if (!viennacl::io::read_matrix_market_file(mtl4_matrix, "../examples/testdata/mat65k.mtx"))
  #endif
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return 0;
  }
  std::cout << "done reading matrix" << std::endl;

  #ifdef _MSC_VER
  if (!readVectorFromFile("../../examples/testdata/rhs65025.txt", mtl4_rhs))
  #else
  if (!readVectorFromFile("../examples/testdata/rhs65025.txt", mtl4_rhs))
  #endif
  {
    std::cout << "Error reading RHS file" << std::endl;
    return 0;
  }
  
  #ifdef _MSC_VER
  if (!readVectorFromFile("../../examples/testdata/result65025.txt", mtl4_ref_result))
  #else
  if (!readVectorFromFile("../examples/testdata/result65025.txt", mtl4_ref_result))
  #endif
  {
    std::cout << "Error reading Result file" << std::endl;
    return 0;
  }
  
  //
  //CG solver:
  //
  std::cout << "----- Running CG -----" << std::endl;
  mtl4_result = viennacl::linalg::solve(mtl4_matrix, mtl4_rhs, viennacl::linalg::cg_tag());

  mtl4_residual = mtl4_matrix * mtl4_result - mtl4_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(mtl4_residual) / viennacl::linalg::norm_2(mtl4_rhs) << std::endl;
  
  //
  //BiCGStab solver:
  //
  std::cout << "----- Running BiCGStab -----" << std::endl;
  mtl4_result = viennacl::linalg::solve(mtl4_matrix, mtl4_rhs, viennacl::linalg::bicgstab_tag());
  
  mtl4_residual = mtl4_matrix * mtl4_result - mtl4_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(mtl4_residual) / viennacl::linalg::norm_2(mtl4_rhs) << std::endl;

  //GMRES solver:
  std::cout << "----- Running GMRES -----" << std::endl;
  mtl4_result = viennacl::linalg::solve(mtl4_matrix, mtl4_rhs, viennacl::linalg::gmres_tag());
  
  mtl4_residual = mtl4_matrix * mtl4_result - mtl4_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(mtl4_residual) / viennacl::linalg::norm_2(mtl4_rhs) << std::endl;

}

