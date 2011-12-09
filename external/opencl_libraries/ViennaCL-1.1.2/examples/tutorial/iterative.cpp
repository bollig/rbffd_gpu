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
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"


// Some helper functions for this tutorial:
#include "Random.hpp"
#include "vector-io.hpp"


/*
*
*   Tutorial:  Iterative solvers
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
  // Set up some ViennaCL objects
  //
  size_t vcl_size = rhs.size();
  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix;
  viennacl::coordinate_matrix<ScalarType> vcl_coordinate_matrix;
  viennacl::vector<ScalarType> vcl_rhs(vcl_size); 
  viennacl::vector<ScalarType> vcl_result(vcl_size);
  viennacl::vector<ScalarType> vcl_ref_result(vcl_size);
  
  viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  viennacl::copy(ref_result.begin(), ref_result.end(), vcl_ref_result.begin());
  
  
  //
  // Transfer ublas-matrix to GPU:
  //
  viennacl::copy(ublas_matrix, vcl_compressed_matrix);
  
  //
  // alternative way: via STL. Sparse matrix as std::vector< std::map< unsigned int, ScalarType> >
  //
  std::vector< std::map< unsigned int, ScalarType> > stl_matrix(rhs.size());
  for (ublas::compressed_matrix<ScalarType>::iterator1 iter1 = ublas_matrix.begin1();
       iter1 != ublas_matrix.end1();
       ++iter1)
  {
    for (ublas::compressed_matrix<ScalarType>::iterator2 iter2 = iter1.begin();
         iter2 != iter1.end();
         ++iter2)
         stl_matrix[iter2.index1()][static_cast<unsigned int>(iter2.index2())] = *iter2;
  }
  viennacl::copy(stl_matrix, vcl_coordinate_matrix);
  viennacl::copy(vcl_coordinate_matrix, stl_matrix);
  
  //
  // set up ILUT preconditioners for ViennaCL and ublas objects:
  // 
  viennacl::linalg::ilut_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
  viennacl::linalg::ilut_precond< viennacl::compressed_matrix<ScalarType> > vcl_ilut(vcl_compressed_matrix, viennacl::linalg::ilut_tag());

  //
  // set up Jacobi preconditioners for ViennaCL and ublas objects:
  // 
  viennacl::linalg::jacobi_precond< ublas::compressed_matrix<ScalarType> >    ublas_jacobi(ublas_matrix, viennacl::linalg::jacobi_tag());
  viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<ScalarType> > vcl_jacobi(vcl_compressed_matrix, viennacl::linalg::jacobi_tag());
  
  //
  // Conjugate gradient solver:
  //
  std::cout << "----- CG Test -----" << std::endl;
  
  //
  // for ublas objects:
  //
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag());
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(1e-6, 20), ublas_ilut);
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(1e-6, 20), ublas_jacobi);

  
  //
  // for ViennaCL objects:
  //
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::cg_tag());
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::cg_tag(1e-6, 20), vcl_ilut);
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::cg_tag(1e-6, 20), vcl_jacobi);
  
  //
  // Stabilized BiConjugate gradient solver:
  //
  std::cout << "----- BiCGStab Test -----" << std::endl;

  //
  // for ublas objects:
  //
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag());          //without preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), ublas_ilut); //with preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), ublas_jacobi); //with preconditioner

  
  //
  // for ViennaCL objects:
  //
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::bicgstab_tag());   //without preconditioner
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), vcl_ilut); //with preconditioner
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), vcl_jacobi); //with preconditioner
  
  //
  // GMRES solver:
  //
  std::cout << "----- GMRES Test -----" << std::endl;
  std::cout << " ATTENTION: Please be aware that GMRES may not work on ATI GPUs when using Stream SDK v2.1." << std::endl;

  //
  // for ublas objects:
  //
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag());   //without preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag(1e-6, 20), ublas_ilut);//with preconditioner
  result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag(1e-6, 20), ublas_jacobi);//with preconditioner

  //
  // for ViennaCL objects:
  //
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::gmres_tag());   //without preconditioner
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::gmres_tag(1e-6, 20), vcl_ilut);//with preconditioner
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_rhs, viennacl::linalg::gmres_tag(1e-6, 20), vcl_jacobi);//with preconditioner

  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  
  return 0;
}

