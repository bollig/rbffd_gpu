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
// ublas includes
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>


// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_HAVE_UBLAS 1


//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"       //generic matrix-vector product
#include "viennacl/linalg/norm_2.hpp"     //generic l2-norm for vectors

// Some helper functions for this tutorial:
#include "Random.hpp"
#include "vector-io.hpp"

/*
*   Tutorial no. 2: BLAS level 2 functionality
*   
*/

using namespace boost::numeric;

int main()
{
  typedef float       ScalarType;
  
  //
  // Set up some ublas objects
  //
  ublas::vector<ScalarType> rhs(12);
  for (unsigned int i = 0; i < rhs.size(); ++i)
    rhs(i) = random<ScalarType>();
  ublas::vector<ScalarType> rhs2 = rhs;
  ublas::vector<ScalarType> result = ublas::zero_vector<ScalarType>(10);
  ublas::vector<ScalarType> result2 = result;
  ublas::vector<ScalarType> rhs_trans = rhs;
  rhs_trans.resize(result.size(), true);
  ublas::vector<ScalarType> result_trans = ublas::zero_vector<ScalarType>(rhs.size());

  
  ublas::matrix<ScalarType> matrix(result.size(),rhs.size());

  //
  // Fill the matrix
  //
  for (unsigned int i = 0; i < matrix.size1(); ++i)
    for (unsigned int j = 0; j < matrix.size2(); ++j)
      matrix(i,j) = random<ScalarType>();

  //
  // Set up some ViennaCL objects
  //
  viennacl::vector<ScalarType> vcl_rhs(static_cast<unsigned int>(rhs.size()));
  viennacl::vector<ScalarType> vcl_result(static_cast<unsigned int>(result.size())); 
  viennacl::matrix<ScalarType> vcl_matrix(static_cast<unsigned int>(rhs.size()), static_cast<unsigned int>(rhs.size()));

  copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  copy(matrix, vcl_matrix);

  
  /////////////////////////////////////////////////
  //////////// Matrix vector products /////////////
  /////////////////////////////////////////////////
  
  
  //
  // Compute matrix-vector products
  //
  std::cout << "----- Matrix-Vector product -----" << std::endl;
  result = prod(matrix, rhs);                                   //the ublas way
  vcl_result = viennacl::linalg::prod(vcl_matrix, vcl_rhs);     //the ViennaCL way
  

  
  //
  // Compute transposed matrix-vector products
  //
  std::cout << "----- Transposed Matrix-Vector product -----" << std::endl;
  result_trans = prod(trans(matrix), rhs_trans);
  
  viennacl::vector<ScalarType> vcl_rhs_trans(static_cast<unsigned int>(rhs_trans.size()));
  viennacl::vector<ScalarType> vcl_result_trans(static_cast<unsigned int>(result_trans.size())); 
  copy(rhs_trans.begin(), rhs_trans.end(), vcl_rhs_trans.begin());
  vcl_result_trans = viennacl::linalg::prod(trans(vcl_matrix), vcl_rhs_trans);
  
  
  
  /////////////////////////////////////////////////
  //////////////// Direct solver  /////////////////
  /////////////////////////////////////////////////
  
  
  //
  // Setup suitable matrices
  //
  ublas::matrix<ScalarType> tri_matrix(10,10);
  for (size_t i=0; i<tri_matrix.size1(); ++i)
  {
    for (size_t j=0; j<i; ++j)
      tri_matrix(i,j) = 0.0;

    for (size_t j=i; j<tri_matrix.size2(); ++j)
      tri_matrix(i,j) = matrix(i,j);
  }
  
  viennacl::matrix<ScalarType> vcl_tri_matrix(static_cast<unsigned int>(tri_matrix.size1()), static_cast<unsigned int>(tri_matrix.size2()));
  copy(tri_matrix, vcl_tri_matrix);
  
  rhs.resize(tri_matrix.size1(), true);
  rhs2.resize(tri_matrix.size1(), true);
  vcl_rhs.resize(static_cast<unsigned int>(tri_matrix.size1()), true);
  
  copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  vcl_result.resize(10);

  
  //
  // Triangular solver
  //
  std::cout << "----- Upper Triangular solve -----" << std::endl;
  result = solve(tri_matrix, rhs, ublas::upper_tag());                                           //ublas
  vcl_result = viennacl::linalg::solve(vcl_tri_matrix, vcl_rhs, viennacl::linalg::upper_tag());  //ViennaCL
  
  //
  // Inplace variants of the above
  //
  inplace_solve(tri_matrix, rhs, ublas::upper_tag());                                       //ublas
  viennacl::linalg::inplace_solve(vcl_tri_matrix, vcl_rhs, viennacl::linalg::upper_tag());  //ViennaCL
  

  //
  // Set up a full system for LU solver:
  // 
  std::cout << "----- LU factorization -----" << std::endl;
  unsigned int lu_dim = 300;
  ublas::matrix<ScalarType> square_matrix(lu_dim, lu_dim);
  ublas::vector<ScalarType> lu_rhs(lu_dim);
  viennacl::matrix<ScalarType> vcl_square_matrix(lu_dim, lu_dim);
  viennacl::vector<ScalarType> vcl_lu_rhs(lu_dim);

  for (size_t i=0; i<lu_dim; ++i)
    for (size_t j=0; j<lu_dim; ++j)
      square_matrix(i,j) = random<ScalarType>();

  //put some more weight on diagonal elements:
  for (size_t j=0; j<lu_dim; ++j)
  {
    square_matrix(j,j) += 10.0;
    lu_rhs(j) = random<ScalarType>();
  }
    
  copy(square_matrix, vcl_square_matrix);
  copy(lu_rhs, vcl_lu_rhs);
  viennacl::linalg::lu_factorize(vcl_square_matrix);
  viennacl::linalg::lu_substitute(vcl_square_matrix, vcl_lu_rhs);
  copy(square_matrix, vcl_square_matrix);
  copy(lu_rhs, vcl_lu_rhs);

  
  //
  // ublas:
  //
  lu_factorize(square_matrix);
  inplace_solve (square_matrix, lu_rhs, ublas::unit_lower_tag ());
  inplace_solve (square_matrix, lu_rhs, ublas::upper_tag ());


  //
  // ViennaCL:
  //
  viennacl::linalg::lu_factorize(vcl_square_matrix);
  viennacl::linalg::lu_substitute(vcl_square_matrix, vcl_lu_rhs);

  //
  //  That's it. Move on to the second tutorial, where sparse matrices and iterative solvers are explained.
  //
  std::cout << "!!!! TUTORIAL 2 COMPLETED SUCCESSFULLY !!!!" << std::endl;
  
  return 0;
}

