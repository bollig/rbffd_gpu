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

#ifndef _VIENNACL_COORDINATE_MATRIX_OPERATIONS_HPP_
#define _VIENNACL_COORDINATE_MATRIX_OPERATIONS_HPP_

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/kernels/coordinate_matrix_kernels.h"

namespace viennacl
{
  namespace linalg
  {
    // A * x
    /** @brief Returns a proxy class that represents matrix-vector multiplication with a coordinate_matrix
    *
    * This is used for the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param NUM_THREADS Number of threads per work group. Can be used for fine-tuning.
    */
    template<class SCALARTYPE, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
    viennacl::vector_expression<const viennacl::coordinate_matrix<SCALARTYPE, ALIGNMENT>,
                                const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT>, 
                                viennacl::op_prod > prod_impl(const viennacl::coordinate_matrix<SCALARTYPE, ALIGNMENT> & mat, 
                                                              const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & vec, 
                                                              unsigned int NUM_THREADS)
    {
      return viennacl::vector_expression<const viennacl::coordinate_matrix<SCALARTYPE, ALIGNMENT>,
                               const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT>, 
                               viennacl::op_prod >(mat, vec);
    }
    
    //namespace {
    /** @brief Carries out matrix-vector multiplication with a coordinate_matrix
    *
    * Implementation of the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    * @param NUM_THREADS Number of threads per work group. Can be used for fine-tuning.
    */
      template<class TYPE, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
      void prod_impl(const viennacl::coordinate_matrix<TYPE, ALIGNMENT> & mat, 
                     const viennacl::vector<TYPE, VECTOR_ALIGNMENT> & vec,
                           viennacl::vector<TYPE, VECTOR_ALIGNMENT> & result,
                      unsigned int NUM_THREADS = 0)
      {
        assert(mat.size1() == result.size());
        assert(mat.size2() == vec.size());
        result.clear();
        
        //std::cout << "prod(coordinate_matrix" << ALIGNMENT << ", vector) called with internal_nnz=" << mat.internal_nnz() << std::endl;
        
        //unsigned int thread_num = 128;
        unsigned int thread_num = 1;
        if (viennacl::ocl::device().type() == CL_DEVICE_TYPE_CPU)
        {
          thread_num = 1;
        }
        unsigned int pos = 0;
        viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.setArgument(pos++, mat.handle12());
        viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.setArgument(pos++, mat.handle());
        viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.setArgument(pos++, vec.handle());
        viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.setArgument(pos++, result.handle());
        //viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.setArgument(pos++, result2.handle());
        viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.setArgument(pos++, mat.nnz());
        //viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.setArgument(pos++, result.size());
        viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.setLocalBuffer(pos++, sizeof(unsigned int)*thread_num);
        viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.setLocalBuffer(pos++, sizeof(TYPE)*thread_num);
        
        viennacl::linalg::kernels::coordinate_matrix<TYPE, ALIGNMENT>::vec_mul.start1D(thread_num, thread_num);
        //clFinish(device().queue());
        
        //result += result2;

      }
    //};

  } //namespace linalg



    //v = A * x, TODO: Check for self-assignment
    /** @brief Implementation of the operation v1 = A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator=(const viennacl::vector_expression< const coordinate_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                          const viennacl::vector<SCALARTYPE, ALIGNMENT>,
                                                                                          viennacl::op_prod> & proxy) 
    {
      // check for the special case x = A * x
      if (proxy.get_lhs().handle() == this->handle())
      {
        viennacl::vector<SCALARTYPE, ALIGNMENT> result(proxy.get_rhs().size());
        viennacl::linalg::prod_impl(proxy.get_lhs(), proxy.get_rhs(), result);
        *this = result;
        return *this;
      }
      else
      {
        viennacl::linalg::prod_impl(proxy.get_lhs(), proxy.get_rhs(), *this);
        return *this;
      }
      return *this;
    }

    //v += A * x
    /** @brief Implementation of the operation v1 += A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator+=(const vector_expression< const coordinate_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                 const vector<SCALARTYPE, ALIGNMENT>,
                                                                                 op_prod> & proxy) 
    {
      vector<SCALARTYPE, ALIGNMENT> result(proxy.get_lhs().size1());
      viennacl::linalg::prod_impl(proxy.get_lhs(), proxy.get_rhs(), result);
      *this += result;
      return *this;
    }

    /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator-=(const vector_expression< const coordinate_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                 const vector<SCALARTYPE, ALIGNMENT>,
                                                                                 op_prod> & proxy) 
    {
      vector<SCALARTYPE, ALIGNMENT> result(proxy.get_lhs().size1());
      viennacl::linalg::prod_impl(proxy.get_lhs(), proxy.get_rhs(), result);
      *this -= result;
      return *this;
    }
    
    
    //free functions:
    /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator+(const vector_expression< const coordinate_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                const vector<SCALARTYPE, ALIGNMENT>,
                                                                                op_prod> & proxy) 
    {
      assert(proxy.get_lhs().size1() == size());
      vector<SCALARTYPE, ALIGNMENT> result(size());
      viennacl::linalg::prod_impl(proxy.get_lhs(), proxy.get_rhs(), result);
      result += *this;
      return result;
    }

    /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator-(const vector_expression< const coordinate_matrix<SCALARTYPE, MAT_ALIGNMENT>,
                                                                                const vector<SCALARTYPE, ALIGNMENT>,
                                                                                op_prod> & proxy) 
    {
      assert(proxy.get_lhs().size1() == size());
      vector<SCALARTYPE, ALIGNMENT> result(size());
      viennacl::linalg::prod_impl(proxy.get_lhs(), proxy.get_rhs(), result);
      result = *this - result;
      return result;
    }

} //namespace viennacl


#endif
