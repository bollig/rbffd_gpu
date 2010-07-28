/* =======================================================================
   Copyright (c) 2010, Institute for Microelectronics, TU Vienna.
   http://www.iue.tuwien.ac.at
                             -----------------
                     ViennaCL - The Vienna Computing Library
                             -----------------
                            
   authors:    Karl Rupp                          rupp@iue.tuwien.ac.at
               Florian Rudolf
               Josef Weinbub                      weinbub@iue.tuwien.ac.at

   license:    MIT (X11), see file LICENSE in the ViennaCL base directory

   file changelog: - May 28, 2010   New from scratch for first release
======================================================================= */

#define NDEBUG

//
// *** System
//
#include <iostream>
//
// *** ViennaCL
//
//#define VCL_BUILD_INFO
#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "examples/tutorial/Random.hpp"
#include "examples/tutorial/vector-io.hpp"

//
// *** Boost
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
//
// -------------------------------------------------------------
//
using namespace boost::numeric;
//
// -------------------------------------------------------------
//
template <typename ScalarType>
ScalarType diff(ScalarType & s1, viennacl::scalar<ScalarType> & s2) 
{
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), fabs(s2));
   return 0;
}

template <typename ScalarType>
ScalarType diff(ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
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

//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;
   // --------------------------------------------------------------------------            
   ublas::vector<NumericT> rhs;
   ublas::vector<NumericT> result;
   ublas::compressed_matrix<NumericT> ublas_matrix;

    if (!readMatrixFromFile("../../examples/testdata/matrix65025.txt", ublas_matrix))
    {
      std::cout << "Error reading Matrix file" << std::endl;
      return EXIT_FAILURE;
    }
    //unsigned int cg_mat_size = cg_mat.size(); 
    std::cout << "done reading matrix" << std::endl;

    if (!readVectorFromFile<NumericT>("../../examples/testdata/rhs65025.txt", rhs))
    {
      std::cout << "Error reading RHS file" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "done reading rhs" << std::endl;

    if (!readVectorFromFile<NumericT>("../../examples/testdata/result65025.txt", result))
    {
      std::cout << "Error reading Result file" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "done reading result" << std::endl;
   

   viennacl::vector<NumericT> vcl_rhs(rhs.size());
   viennacl::vector<NumericT> vcl_result(result.size()); 
   viennacl::vector<NumericT> vcl_result2(result.size()); 
   viennacl::compressed_matrix<NumericT> vcl_compressed_matrix(rhs.size(), rhs.size());
   viennacl::coordinate_matrix<NumericT> vcl_coordinate_matrix(rhs.size(), rhs.size());

   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(ublas_matrix, vcl_compressed_matrix);
   copy(ublas_matrix, vcl_coordinate_matrix);

   // --------------------------------------------------------------------------          
   std::cout << "Benching products: ublas" << std::endl;
   result     = viennacl::linalg::prod(ublas_matrix, rhs);
   std::cout << "Benching products: compressed_matrix" << std::endl;
   vcl_result = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);
   
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with compressed_matrix" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   
/*   std::cout << "Benching products: coordinate_matrix" << std::endl;
   vcl_result = viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs);

   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with coordinate_matrix" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }*/
   
   // --------------------------------------------------------------------------            
   // --------------------------------------------------------------------------            
   NumericT alpha(2.786);
   NumericT beta(1.432);
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(result.begin(), result.end(), vcl_result.begin());
   copy(result.begin(), result.end(), vcl_result2.begin());

   std::cout << "Benching scaled additions of products and vectors" << std::endl;
   result     = alpha * viennacl::linalg::prod(ublas_matrix, rhs) + beta * result;
   vcl_result2 = alpha * viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs) + beta * vcl_result;

   if( fabs(diff(result, vcl_result2)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product (compressed_matrix) with scaled additions" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   
/*   vcl_result2 = alpha * viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs) + beta * vcl_result;

   if( fabs(diff(result, vcl_result2)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product (coordinate_matrix) with scaled additions" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result2)) << std::endl;
      retval = EXIT_FAILURE;
   }*/
   
   // --------------------------------------------------------------------------            
   return retval;
}
//
// -------------------------------------------------------------
//
int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Sparse Matrices" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = 1.0E-5;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   
/*   {
      typedef float NumericT;
      NumericT epsilon = 1.0E-6;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;*/
   
   if( viennacl::ocl::device().double_support() )
   {
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-10;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
            if( retval == EXIT_SUCCESS )
               std::cout << "# Test passed" << std::endl;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      
/*      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-15;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
      if( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;*/
   }
   return retval;
}
