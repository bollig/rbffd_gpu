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
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "examples/tutorial/Random.hpp"
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
   ublas::vector<NumericT> rhs(12);
   for (unsigned int i = 0; i < rhs.size(); ++i)
     rhs(i) = random<NumericT>();
   ublas::vector<NumericT> rhs2 = rhs;
   ublas::vector<NumericT> result = ublas::zero_vector<NumericT>(10);
   ublas::vector<NumericT> result2 = result;
   ublas::vector<NumericT> rhs_trans = rhs;
   rhs_trans.resize(result.size(), true);
   ublas::vector<NumericT> result_trans = ublas::zero_vector<NumericT>(rhs.size());

  
   ublas::matrix<NumericT> matrix(result.size(),rhs.size());
  
   for (unsigned int i = 0; i < matrix.size1(); ++i)
      for (unsigned int j = 0; j < matrix.size2(); ++j)
         matrix(i,j) = random<NumericT>();

   viennacl::vector<NumericT> vcl_rhs(rhs.size());
   viennacl::vector<NumericT> vcl_rhs_trans(rhs_trans.size());
   viennacl::vector<NumericT> vcl_result_trans(result_trans.size());
   viennacl::vector<NumericT> vcl_result(result.size()); 
   viennacl::matrix<NumericT> vcl_matrix(rhs.size(), rhs.size());

   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(matrix, vcl_matrix);

   // --------------------------------------------------------------------------            
   result     = viennacl::linalg::prod(matrix, rhs);
   vcl_result = viennacl::linalg::prod(vcl_matrix, vcl_rhs);
   
   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------            
   NumericT alpha(2.786);
   NumericT beta(1.432);
   copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
   copy(result.begin(), result.end(), vcl_result.begin());

   result     = alpha * viennacl::linalg::prod(matrix, rhs) + beta * result;
   vcl_result = alpha * viennacl::linalg::prod(vcl_matrix, vcl_rhs) + beta * vcl_result;

   if( fabs(diff(result, vcl_result)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << fabs(diff(result, vcl_result)) << std::endl;
      retval = EXIT_FAILURE;
   }
   // --------------------------------------------------------------------------            

   copy(rhs_trans.begin(), rhs_trans.end(), vcl_rhs_trans.begin());
   copy(result_trans.begin(), result_trans.end(), vcl_result_trans.begin());
   
   result_trans     = alpha * viennacl::linalg::prod(trans(matrix), rhs_trans) + beta * result_trans;  
   vcl_result_trans = alpha * viennacl::linalg::prod(trans(vcl_matrix), vcl_rhs_trans) + beta * vcl_result_trans;

   if( fabs(diff(result_trans, vcl_result_trans)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << fabs(diff(result_trans, vcl_result_trans)) << std::endl;
      retval = EXIT_FAILURE;
   }
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
   std::cout << "## Test :: Matrix" << std::endl;
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
   {
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
   std::cout << std::endl;
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
      {
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
      std::cout << std::endl;
   }
   return retval;
}
