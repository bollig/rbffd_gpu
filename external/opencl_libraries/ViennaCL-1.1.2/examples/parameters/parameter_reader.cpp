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

//#define VIENNACL_DEBUG_ALL
//#define NDEBUG

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/io/kernel_parameters.hpp"


#include <iostream>
#include <vector>





int main(int argc, char *argv[])
{
  // -----------------------------------------
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  
  std::cout << viennacl::ocl::current_device().info() << std::endl;

  viennacl::io::read_kernel_parameters< viennacl::vector<float> >("vector_parameters.xml");
  viennacl::io::read_kernel_parameters< viennacl::matrix<float> >("matrix_parameters.xml");
  viennacl::io::read_kernel_parameters< viennacl::compressed_matrix<float> >("sparse_parameters.xml");
  // -----------------------------------------  

  //check:
  std::cout << "vector add:" << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_vector_1", "add").local_work_size() << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_vector_1", "add").global_work_size() << std::endl;

  std::cout << "matrix vec_mul:" << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_matrix_row_1", "vec_mul").local_work_size() << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_matrix_row_1", "vec_mul").global_work_size() << std::endl;
  
  std::cout << "compressed_matrix vec_mul:" << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_compressed_matrix_1", "vec_mul").local_work_size() << std::endl;
  std::cout << viennacl::ocl::get_kernel("f_compressed_matrix_1", "vec_mul").global_work_size() << std::endl;

 
  return 0;
}

