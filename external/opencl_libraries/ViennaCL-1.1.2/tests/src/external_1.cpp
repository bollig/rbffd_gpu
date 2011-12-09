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
// A check for the absence of external linkage (otherwise, library is not truly 'header-only')
//


//
// *** System
//
#include <iostream>


//
// *** ViennaCL
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"

#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/row_scaling.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/linalg/direct_solve.hpp"

#include "viennacl/io/matrix_market.hpp"


//defined in external_2.cpp
void other_func();

//
// -------------------------------------------------------------
//
int main()
{
  typedef float   NumericType;
  
  //doing nothing but instantiating a few types
  viennacl::scalar<NumericType>  s;
  viennacl::vector<NumericType>  v(10);
  viennacl::matrix<NumericType>  m(10, 10);
  viennacl::compressed_matrix<NumericType>  compr(10, 10);
  viennacl::coordinate_matrix<NumericType>  coord(10, 10);
  
  //this is the external linkage check:
  other_func();
  
  return EXIT_SUCCESS;
}
