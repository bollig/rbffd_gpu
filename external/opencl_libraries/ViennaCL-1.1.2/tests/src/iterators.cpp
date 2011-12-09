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
======================================================================= */

//
// *** System
//
#include <iostream>
#include <stdlib.h>

//
// *** ViennaCL
//
//#define VCL_BUILD_INFO
//#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"

//
// -------------------------------------------------------------
//
template< typename NumericT >
int test()
{
   int retval = EXIT_SUCCESS;
   // --------------------------------------------------------------------------
   typedef viennacl::vector<NumericT>  VclVector;

   VclVector vcl_cont(3);
   vcl_cont[0] = 1;
   vcl_cont[1] = 2;
   vcl_cont[2] = 3;

   typename VclVector::const_iterator const_iter_def_const;
   typename VclVector::iterator       iter_def_const;

   for(typename VclVector::const_iterator iter = vcl_cont.begin();
       iter != vcl_cont.end(); iter++)
   {
      std::cout << *iter << std::endl;
   }

   for(typename VclVector::iterator iter = vcl_cont.begin();
       iter != vcl_cont.end(); iter++)
   {
      std::cout << *iter << std::endl;
   }

   // --------------------------------------------------------------------------                        
   return retval;
}

int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Iterators" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>();
      if( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   if( viennacl::ocl::current_device().double_support() )
   {
      {
         typedef double NumericT;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>();
            if( retval == EXIT_SUCCESS )
              std::cout << "# Test passed" << std::endl;
            else
              return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
   }
   return retval;
}
