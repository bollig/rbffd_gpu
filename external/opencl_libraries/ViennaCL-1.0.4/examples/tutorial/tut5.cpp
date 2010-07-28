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
#include <string>

//
// ViennaCL includes
//
#include "viennacl/ocl/program.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/vector.hpp"


// Some helper functions for this tutorial:
#include "Random.hpp"


/*
*
*   Tutorial no. 5:  Custom compute kernels
*   
*/



//
// A custom compute kernel which computes an elementwise product of two vectors
// Input: v1 ... vector
//        v2 ... vector
// Output: result ... vector
//
// Algorithm: set result[i] <- v1[i] * v2[i]
//            (in MATLAB notation this is something like 'result = v1 .* v2');
//

const char * my_compute_kernel = 
"__kernel void elementwise_prod(\n"
"          __global const float * vec1,\n"
"          __global const float * vec2, \n"
"          __global float * result,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec1[i] * vec2[i];\n"
"};\n";


int main()
{
  typedef float       ScalarType;

  unsigned int vector_size = 10;
  viennacl::vector<ScalarType>  vec1(vector_size);
  viennacl::vector<ScalarType>  vec2(vector_size);
  viennacl::vector<ScalarType>  result(vector_size);
  
  //
  // fill the operands vec1 and vec2:
  //
  for (unsigned int i=0; i<vector_size; ++i)
  {
    vec1[i] = static_cast<ScalarType>(i);
    vec2[i] = static_cast<ScalarType>(vector_size-i);
  }


  //
  // Set up the OpenCL program given in my_compute_kernel:
  // A program is one compilation unit an can hold many different compute kernels.
  //
  viennacl::ocl::program my_prog;
  my_prog.create(my_compute_kernel);
  
  //
  // Get the kernel 'elementwise_prod' from the program 'my_program'
  //
  viennacl::ocl::kernel my_kernel;
  my_kernel.prepareInit("elementwise_prod", my_prog);
  
  //
  // Prepare kernel launch: Set the kernel arguments first
  //
  unsigned int pos = 0;
  my_kernel.setArgument(pos++, vec1.handle());
  my_kernel.setArgument(pos++, vec2.handle());
  my_kernel.setArgument(pos++, result.handle());
  my_kernel.setArgument(pos++, vector_size);
  
  //
  // Launch the kernel with 'vector_size' threads in one work group
  //
  my_kernel.start1D(vector_size,    //number of global threads
                    vector_size);   //number of threads per work group
  
  //
  // Print the result:
  //
  std::cout << "        vec1: " << vec1 << std::endl;
  std::cout << "        vec2: " << vec2 << std::endl;
  std::cout << "vec1 .* vec2: " << result << std::endl;
  
  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL 5 COMPLETED SUCCESSFULLY !!!!" << std::endl;
  
  return 0;
}

