#ifndef VIENNACL_MATRIX_OPERATIONS_HPP_
#define VIENNACL_MATRIX_OPERATIONS_HPP_

/* =========================================================================
   Copyright (c) 2010-2011, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file matrix_operations.hpp
    @brief Implementations of dense matrix related operations. also matrix-vector products.
*/

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/tools/matrix_kernel_class_deducer.hpp"
#include "viennacl/tools/matrix_prod_kernel_class_deducer.hpp"
#include "viennacl/linalg/kernels/vector_kernels.h"
#include "viennacl/linalg/kernels/matrix_row_kernels.h"
#include "viennacl/linalg/kernels/matrix_col_kernels.h"

#include "viennacl/linalg/kernels/matrix_prod_col_col_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_col_col_row_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_col_row_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_col_row_row_kernels.h"

#include "viennacl/linalg/kernels/matrix_prod_row_col_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_row_col_row_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_row_row_col_kernels.h"
#include "viennacl/linalg/kernels/matrix_prod_row_row_row_kernels.h"

namespace viennacl
{
  namespace linalg
  {
    
    /** @brief Adds two dense matrices and writes the result to a third matrix
    *
    * This is the implementation of the convenience expression result = mat1 + mat2;
    *
    * @param mat1   The left hand side operand
    * @param mat2   The right hand side operand
    * @param result The resulting matrix
    */
    template<class TYPE, typename F, unsigned int ALIGNMENT>
    void add(const viennacl::matrix<TYPE, F, ALIGNMENT> & mat1, 
             const viennacl::matrix<TYPE, F, ALIGNMENT> & mat2,
             viennacl::matrix<TYPE, F, ALIGNMENT> & result)
    {
      assert(result.size1() == mat1.size1());
      assert(result.size2() == mat1.size2());
      assert(result.size1() == mat2.size1());
      assert(result.size2() == mat2.size2());

      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<TYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "add");
      assert( (mat1.internal_size() == mat2.internal_size())
             && "Operands must have same dimension and memory layout in this version of ViennaCL!");
      cl_uint size = std::min(mat1.internal_size(), mat2.internal_size());

      viennacl::ocl::enqueue(k(mat1, mat2, result, size));        
    }

    /** @brief Adds a dense matrix to another
    *
    * This is the implementation of the convenience expression result += mat1;
    *
    * @param mat2   The addend (either a matrix or a matrix_range)
    * @param result The resulting matrix  (either a matrix or a matrix_range)
    */
    template <typename M1, typename M2>
    typename viennacl::enable_if< viennacl::is_matrix<M1>::value
                                  && viennacl::is_matrix<M2>::value
                                >::type
    inplace_add(M1 & result, M2 const & mat2)
    {
      assert(viennacl::traits::size1(result) == viennacl::traits::size1(mat2));
      assert(viennacl::traits::size2(result) == viennacl::traits::size2(mat2));

      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< M1 >::ResultType    KernelClass;
      
      size_t block_size = 15;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "inplace_add");
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(viennacl::traits::size1(result), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(viennacl::traits::size2(result), block_size));
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);
      
      viennacl::ocl::enqueue(k(viennacl::traits::handle(result),
                                       cl_uint(viennacl::traits::start1(result)), cl_uint(viennacl::traits::start2(result)), 
                                       cl_uint(viennacl::traits::size1(result)), cl_uint(viennacl::traits::size2(result)),
                                       cl_uint(viennacl::traits::internal_size1(result)), cl_uint(viennacl::traits::internal_size2(result)),
                                viennacl::traits::handle(mat2), 
                                      cl_uint(viennacl::traits::start1(mat2)), cl_uint(viennacl::traits::start2(mat2)), 
                                      cl_uint(viennacl::traits::size1(mat2)), cl_uint(viennacl::traits::size2(mat2)),
                                      cl_uint(viennacl::traits::internal_size1(mat2)), cl_uint(viennacl::traits::internal_size2(mat2))
                              )
                            );
    }

    /** @brief Adds a dense matrix to another
    *
    * This is the implementation of the convenience expression result += mat1;
    *
    * @param mat1   The left hand side operand
    * @param mat2   The right hand side operand
    * @param result The resulting matrix
    */
    /*
    template <typename MatrixType>
    void inplace_add(viennacl::matrix_range<MatrixType> & result, 
                     const viennacl::matrix_range<MatrixType> & mat2)
    {
      assert(result.size1() == mat2.size1());
      assert(result.size2() == mat2.size2());

      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< MatrixType >::ResultType    KernelClass;
      
      size_t block_size = 15;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "inplace_add");
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(result.size1(), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(result.size2(), block_size));
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);

      viennacl::ocl::enqueue(k(result.get(), cl_uint(result.start1()), cl_uint(result.start2()), 
                                             cl_uint(result.size1()), cl_uint(result.size2()),
                                             cl_uint(result.get().internal_size1()), cl_uint(result.get().internal_size2()),
                                mat2.get(), cl_uint(mat2.start1()), cl_uint(mat2.start2()),
                                            cl_uint(mat2.size1()), cl_uint(mat2.size2()),
                                            cl_uint(mat2.get().internal_size1()), cl_uint(mat2.get().internal_size2())
                              )
                            );
    } */

    /** @brief Adds a dense matrix to another
    *
    * This is the implementation of the convenience expression result += mat1;
    *
    * @param mat1   The left hand side operand
    * @param mat2   The right hand side operand
    * @param result The resulting matrix
    */
    /*
    template<class TYPE, typename F, unsigned int ALIGNMENT>
    void inplace_add(viennacl::matrix<TYPE, F, ALIGNMENT> & result, 
                     const viennacl::matrix_range<viennacl::matrix<TYPE, F, ALIGNMENT> > & mat2)
    {
      viennacl::range r1(0, result.size1());
      viennacl::range r2(0, result.size2());
      viennacl::matrix_range<viennacl::matrix<TYPE, F, ALIGNMENT> > result_wrap(result, r1, r2);
      inplace_add(result_wrap, mat2);
    } */




    /** @brief Subtracts two dense matrices and writes the result to a third matrix
    *
    * This is the implementation of the convenience expression result = mat1 - mat2;
    *
    * @param mat1   The left hand side operand
    * @param mat2   The right hand side operand
    * @param result The resulting matrix
    */
    template<class TYPE, typename F, unsigned int ALIGNMENT>
    void sub(const viennacl::matrix<TYPE, F, ALIGNMENT> & mat1, 
             const viennacl::matrix<TYPE, F, ALIGNMENT> & mat2,
             viennacl::matrix<TYPE, F, ALIGNMENT> & result)
    {
      assert(result.size1() == mat1.size1());
      assert(result.size2() == mat1.size2());
      assert(result.size1() == mat2.size1());
      assert(result.size2() == mat2.size2());

      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<TYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "sub");
      assert( (mat1.internal_size() == mat2.internal_size())
             && "Operands must have same dimension and memory layout in this version of ViennaCL!");
      cl_uint size = std::min(mat1.internal_size(), mat2.internal_size());

      viennacl::ocl::enqueue(k(mat1, mat2, result, size));        
    }

    /** @brief Subtracts a dense matrix from another
    *
    * This is the implementation of the convenience expression mat1 -= mat2;
    *
    * @param mat2   The matrix to be subtracted
    * @param result The resulting matrix
    */
    template<class TYPE, typename F, unsigned int ALIGNMENT>
    void inplace_sub(viennacl::matrix<TYPE, F, ALIGNMENT> & result, 
                     const viennacl::matrix<TYPE, F, ALIGNMENT> & mat2)
    {
      assert(result.size1() == mat2.size1());
      assert(result.size2() == mat2.size2());

      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<TYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "inplace_sub");
      assert( (result.internal_size() == mat2.internal_size())
             && "Operands must have same dimension and memory layout in this version of ViennaCL!");
      cl_uint size = std::min(result.internal_size(), mat2.internal_size());

      viennacl::ocl::enqueue(k(result, mat2, size));        
    }

    /** @brief Multiplies a dense matrix by a scalar
    *
    * This is the implementation of the convenience expression matrix *= val;
    *
    * @param result The matrix to be manipulated
    * @param val    The CPU scalar by which all entries of the matrix are multiplied
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT>
    void inplace_mult(viennacl::matrix<SCALARTYPE, F, ALIGNMENT> & result, 
                      SCALARTYPE val)
    {
      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "cpu_inplace_mult");
      viennacl::ocl::enqueue(k(result, val, cl_uint(result.internal_size())));
    }


    /** @brief Multiplies a dense matrix by a scalar
    *
    * This is the implementation of the convenience expression matrix *= val;
    *
    * @param result The matrix to be manipulated
    * @param val    The scalar by which all entries of the matrix are multiplied
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT>
    void inplace_mult(viennacl::matrix<SCALARTYPE, F, ALIGNMENT> & result, 
                      viennacl::scalar<SCALARTYPE> const & val)
    {
      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "inplace_mult");
      viennacl::ocl::enqueue(k(result, val, cl_uint(result.internal_size())));
    }



    /** @brief Multiplies a dense matrix by a scalar
    *
    * This is the implementation of the convenience expression matrix /= val;
    *
    * @param result The matrix to be manipulated
    * @param val    The scalar by which all entries of the matrix are divided
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT>
    void inplace_divide(viennacl::matrix<SCALARTYPE, F, ALIGNMENT> & result, 
                        viennacl::scalar<SCALARTYPE> const & val)
    {
      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "inplace_divide");
      unsigned int size = result.internal_size();

      viennacl::ocl::enqueue(k(result, val, size));
    }

    // A * x
    /** @brief Returns a proxy class that represents matrix-vector multiplication
    *
    * This is used for the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
    viennacl::vector_expression<const viennacl::matrix<SCALARTYPE, F, ALIGNMENT>,
                                const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT>, 
                                op_prod > prod_impl(const viennacl::matrix<SCALARTYPE, F, ALIGNMENT> & mat, 
                                                    const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & vec)
    {
      return viennacl::vector_expression<const viennacl::matrix<SCALARTYPE, F, ALIGNMENT>,
                                         const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT>, 
                                         op_prod >(mat, vec);
    }
    
    /** @brief Carries out matrix-vector multiplication
    *
    * Implementation of the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    */
    template<class TYPE, typename F, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
    void prod_impl(const viennacl::matrix<TYPE, F, ALIGNMENT> & mat, 
                    const viennacl::vector<TYPE, VECTOR_ALIGNMENT> & vec, 
                          viennacl::vector<TYPE, VECTOR_ALIGNMENT> & result)
    {
      assert(mat.size2() == vec.size());
      // Inplace matrix-vector products like x = prod(A, x) are currently illegal: Introduce a temporary like y = prod(A, x); x = y; instead
      assert(vec.handle() != result.handle() && "No direct inplace matrix-vector product possible. Introduce a temporary!");
      result.resize(mat.size1());

      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<TYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "vec_mul");
      viennacl::ocl::enqueue(
                             k(mat, cl_uint(mat.size1()), cl_uint(mat.size2()),
                                    cl_uint(mat.internal_size1()), cl_uint(mat.internal_size2()), vec, result));    
    }



    // trans(A) * x
    /** @brief Returns a proxy class that represents matrix-vector multiplication with a transposed matrix
    *
    * This is used for the convenience expression result = trans(mat) * vec;
    *
    * @param proxy  The transposed matrix proxy
    * @param vec    The vector
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
    viennacl::vector_expression<const viennacl::matrix_expression< const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                   const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                   op_trans>,
                                const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT>, 
                                op_prod > prod_impl(const viennacl::matrix_expression< const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                                       const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                                       op_trans> & proxy, 
                                                    const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & vec)
    {
      return viennacl::vector_expression<const viennacl::matrix_expression< const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                            const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                                            op_trans>,
                                         const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT>, 
                                         op_prod >(proxy, vec);
    }

    /** @brief Unwraps the transposed matrix proxy and forwards to trans_prod_impl()
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
    void prod_impl(const viennacl::matrix_expression< const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                      const matrix<SCALARTYPE, F, ALIGNMENT>,
                                                      op_trans> & mat,
                    const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & vec, 
                          viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & result)
    {
      trans_prod_impl(mat.lhs(), vec, result);
    }
    
    /** @brief Carries out matrix-vector multiplication with a transposed matrix
    *
    * Implementation of the convenience expression result = trans(mat) * vec;
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT, unsigned int VECTOR_ALIGNMENT>
    void trans_prod_impl(const matrix<SCALARTYPE, F, ALIGNMENT> & mat,
                          const viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & vec, 
                                viennacl::vector<SCALARTYPE, VECTOR_ALIGNMENT> & result)
    {
      assert(mat.size1() == vec.size());  //remember: mat is transposed!
      // Inplace matrix-vector products like x = prod(A, x) are currently illegal: Introduce a temporary like y = prod(A, x); x = y; instead
      assert(vec.handle() != result.handle() && "No direct inplace matrix-vector product possible. Introduce a temporary!");
      result.resize(mat.size2());

      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "trans_vec_mul");
      
      viennacl::ocl::enqueue(k(mat, cl_uint(mat.size1()), cl_uint(mat.size2()),
                                    cl_uint(mat.internal_size1()), cl_uint(mat.internal_size2()), vec, result));        
    }



    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(A, B);
    *
    */
    template<class TYPE, typename F1, typename F2, typename F3, unsigned int ALIGNMENT>
    void prod_impl(const viennacl::matrix<TYPE, F1, ALIGNMENT> & A, 
                    const viennacl::matrix<TYPE, F2, ALIGNMENT> & B, 
                          viennacl::matrix<TYPE, F3, ALIGNMENT> & C, 
                          int block_size = 15) // [JW] added ability to set block size from outside ..
    {
      assert(A.size1() == C.size1());
      assert(A.size2() == B.size1());
      assert(B.size2() == C.size2());
      // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
      assert(C.handle() != A.handle() 
             && C.handle() != B.handle()
             && "No direct inplace matrix-matrix product possible. Introduce a temporary!");
      
      typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< viennacl::matrix<TYPE, F1, ALIGNMENT>,
                                                                          viennacl::matrix<TYPE, F2, ALIGNMENT>,
                                                                          viennacl::matrix<TYPE, F3, ALIGNMENT> >::ResultType    KernelClass;
      KernelClass::init();
      
      //std::cout << "KernelClass::program_name() : " << KernelClass::program_name() << std::endl;
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "prod_AA");
      
      /*k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1() / 2, block_size / 2));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2() / 2, block_size / 2));
      k.local_work_size(0, block_size / 2);
      k.local_work_size(1, block_size / 2);*/
      
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1(), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2(), block_size));
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);
      
      viennacl::ocl::enqueue(
                             k(A, cl_uint(0), cl_uint(0), 
                                  cl_uint(A.size1()), cl_uint(A.size2()),
                                  cl_uint(A.internal_size1()), cl_uint(A.internal_size2()),
                               B, cl_uint(0), cl_uint(0),
                                  cl_uint(B.size1()), cl_uint(B.size2()),
                                  cl_uint(B.internal_size1()), cl_uint(B.internal_size2()),
                               C, cl_uint(0), cl_uint(0), 
                                  cl_uint(C.size1()), cl_uint(C.size2()),
                                  cl_uint(C.internal_size1()), cl_uint(C.internal_size2()),
                               viennacl::ocl::local_mem(sizeof(TYPE) * block_size * block_size),
                               viennacl::ocl::local_mem(sizeof(TYPE) * block_size * block_size) ));        
    }


    /** @brief Carries out matrix-matrix multiplication for submatrices
    *
    * Implementation of C = prod(A, B); for submatrices
    *
    */
    template<typename T1, typename T2, typename T3>
    void prod_impl(const viennacl::matrix_range<T1> & A, 
                    const viennacl::matrix_range<T2> & B, 
                          viennacl::matrix_range<T3> & C, 
                          int block_size = 15) // [JW] added ability to set block size from outside ..
    {
      typedef typename T1::value_type::value_type   value_type;
      
      assert(A.size1() == C.size1());
      assert(A.size2() == B.size1());
      assert(B.size2() == C.size2());
      // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
      assert(C.get().handle() != A.get().handle() 
             && C.get().handle() != B.get().handle()
             && "No direct inplace matrix-matrix product possible. Introduce a temporary!");
      
      typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< T1, T2, T3 >::ResultType    KernelClass;
      KernelClass::init();
      
      //std::cout << "KernelClass::program_name() : " << KernelClass::program_name() << std::endl;
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "prod_AA");
      
      /*k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1() / 2, block_size / 2));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2() / 2, block_size / 2));
      k.local_work_size(0, block_size / 2);
      k.local_work_size(1, block_size / 2);*/
      
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1(), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2(), block_size));
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);
      
      viennacl::ocl::enqueue(
          k(A.get(), cl_uint(A.start1()), cl_uint(A.start2()),
                     cl_uint(A.size1()), cl_uint(A.size2()),
                     cl_uint(A.get().internal_size1()), cl_uint(A.get().internal_size2()),
            B.get(), cl_uint(B.start1()), cl_uint(B.start2()),
                     cl_uint(B.size1()), cl_uint(B.size2()),
                     cl_uint(B.get().internal_size1()), cl_uint(B.get().internal_size2()),
            C.get(), cl_uint(C.start1()), cl_uint(C.start2()),
                     cl_uint(C.size1()), cl_uint(C.size2()),
                     cl_uint(C.get().internal_size1()), cl_uint(C.get().internal_size2()),
            viennacl::ocl::local_mem(sizeof(value_type) * block_size * block_size),
            viennacl::ocl::local_mem(sizeof(value_type) * block_size * block_size) ));        
    }



    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(trans(A), B);
    *
    */
    template<class TYPE, typename F1, typename F2, typename F3, unsigned int ALIGNMENT>
    void prod_impl(const viennacl::matrix_expression< const matrix<TYPE, F1, ALIGNMENT>,
                                                      const matrix<TYPE, F1, ALIGNMENT>,
                                                      op_trans> & A, 
                    const viennacl::matrix<TYPE, F2, ALIGNMENT> & B, 
                          viennacl::matrix<TYPE, F3, ALIGNMENT> & C)
    {
      assert(A.size2() == C.size1());
      assert(A.size1() == B.size1());
      assert(B.size2() == C.size2());
      // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
      assert(C.handle() != A.lhs().handle() 
             && C.handle() != B.handle()
             && "No direct inplace matrix-matrix product possible. Introduce a temporary!");
      
      int block_size = 15;

      typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< viennacl::matrix<TYPE, F1, ALIGNMENT>,
                                                                          viennacl::matrix<TYPE, F2, ALIGNMENT>,
                                                                          viennacl::matrix<TYPE, F3, ALIGNMENT> >::ResultType    KernelClass;
      KernelClass::init();
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "prod_TA");
      
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1(), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2(), block_size));
      
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);
      viennacl::ocl::enqueue(
              k(A.lhs(), cl_uint(0), cl_uint(0), 
                         cl_uint(A.lhs().size1()), cl_uint(A.lhs().size2()),
                         cl_uint(A.lhs().internal_size1()), cl_uint(A.lhs().internal_size2()),
                B, cl_uint(0), cl_uint(0),
                   cl_uint(B.size1()), cl_uint(B.size2()),
                   cl_uint(B.internal_size1()), cl_uint(B.internal_size2()),
                C, cl_uint(0), cl_uint(0),
                   cl_uint(C.size1()), cl_uint(C.size2()),
                   cl_uint(C.internal_size1()), cl_uint(C.internal_size2()),
                viennacl::ocl::local_mem(sizeof(TYPE) * block_size * block_size),
                viennacl::ocl::local_mem(sizeof(TYPE) * block_size * block_size) )
                            );        
    }


    /** @brief Carries out matrix-matrix multiplication for submatrices
    *
    * Implementation of C = prod(trans(A), B); for submatrices
    *
    */
    template <typename M1, typename M2, typename M3>
    void prod_impl(const viennacl::matrix_expression< const matrix_range<M1>,
                                                      const matrix_range<M1>,
                                                      op_trans> & A_trans, 
                    const viennacl::matrix_range<M2> & B, 
                          viennacl::matrix_range<M3> & C)
    {
      typedef typename M1::value_type::value_type    value_type;
      assert(A_trans.size2() == C.size1());
      assert(A_trans.size1() == B.size1());
      assert(B.size2() == C.size2());
      // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
      assert(C.get().handle() != A_trans.lhs().get().handle() 
             && C.get().handle() != B.get().handle()
             && "No direct inplace matrix-matrix product possible. Introduce a temporary!");
      
      int block_size = 15;

      typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< M1, M2, M3 >::ResultType    KernelClass;
      KernelClass::init();
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "prod_TA");
      
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1(), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2(), block_size));
      
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);
      
      const matrix_range<M1> & A = A_trans.lhs();
      viennacl::ocl::enqueue(
              k(A.get(), cl_uint(A.start1()), cl_uint(A.start2()),
                         cl_uint(A.size1()), cl_uint(A.size2()),
                         cl_uint(A.get().internal_size1()), cl_uint(A.get().internal_size2()),
                B.get(), cl_uint(B.start1()), cl_uint(B.start2()), 
                         cl_uint(B.size1()), cl_uint(B.size2()),
                         cl_uint(B.get().internal_size1()), cl_uint(B.get().internal_size2()),
                C.get(), cl_uint(C.start1()), cl_uint(C.start2()),
                         cl_uint(C.size1()), cl_uint(C.size2()),
                         cl_uint(C.get().internal_size1()), cl_uint(C.get().internal_size2()),
                viennacl::ocl::local_mem(sizeof(value_type) * block_size * block_size),
                viennacl::ocl::local_mem(sizeof(value_type) * block_size * block_size) )
                            );        
    }




    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(A, trans(B));
    *
    */
    template<class TYPE, typename F1, typename F2, typename F3, unsigned int ALIGNMENT>
    void prod_impl(const viennacl::matrix<TYPE, F1, ALIGNMENT> & A, 
                   const viennacl::matrix_expression< const matrix<TYPE, F2, ALIGNMENT>,
                                                      const matrix<TYPE, F2, ALIGNMENT>,
                                                      op_trans> & B,
                   viennacl::matrix<TYPE, F3, ALIGNMENT> & C)
    {
      assert(A.size1() == C.size1());
      assert(A.size2() == B.size2());
      assert(B.size1() == C.size2());
      // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
      assert(C.handle() != A.handle() 
             && C.handle() != B.lhs().handle()
             && "No direct inplace matrix-matrix product possible. Introduce a temporary!");
      
      int block_size = 15;

      typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< viennacl::matrix<TYPE, F1, ALIGNMENT>,
                                                                          viennacl::matrix<TYPE, F2, ALIGNMENT>,
                                                                          viennacl::matrix<TYPE, F3, ALIGNMENT> >::ResultType    KernelClass;
      KernelClass::init();
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "prod_AT");
      
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1(), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2(), block_size));
      
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);
      viennacl::ocl::enqueue(
              k(A, cl_uint(0), cl_uint(0),
                   cl_uint(A.size1()), cl_uint(A.size2()),
                   cl_uint(A.internal_size1()), cl_uint(A.internal_size2()),
                B.lhs(), cl_uint(0), cl_uint(0),
                         cl_uint(B.lhs().size1()), cl_uint(B.lhs().size2()),
                         cl_uint(B.lhs().internal_size1()), cl_uint(B.lhs().internal_size2()),
                C, cl_uint(0), cl_uint(0),
                   cl_uint(C.size1()), cl_uint(C.size2()),
                   cl_uint(C.internal_size1()), cl_uint(C.internal_size2()),
                viennacl::ocl::local_mem(sizeof(TYPE) * block_size * block_size),
                viennacl::ocl::local_mem(sizeof(TYPE) * block_size * block_size) )
                            );        
    }


    /** @brief Carries out matrix-matrix multiplication for submatrices
    *
    * Implementation of C = prod(A, trans(B)); for submatrices
    *
    */
    template <typename M1, typename M2, typename M3>
    void prod_impl(const viennacl::matrix_range<M1> & A, 
                   const viennacl::matrix_expression< const matrix_range<M2>,
                                                      const matrix_range<M2>,
                                                      op_trans> & B_trans,
                   viennacl::matrix_range<M3> & C)
    {
      typedef typename M1::value_type::value_type    value_type;
      assert(A.size1() == C.size1());
      assert(A.size2() == B_trans.size2());
      assert(B_trans.size1() == C.size2());
      // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
      assert(C.get().handle() != A.get().handle() 
             && C.get().handle() != B_trans.lhs().get().handle()
             && "No direct inplace matrix-matrix product possible. Introduce a temporary!");
      
      int block_size = 15;

      typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< M1, M2, M3 >::ResultType    KernelClass;
      KernelClass::init();
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "prod_AT");
      
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1(), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2(), block_size));
      
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);
      const matrix_range<M2> & B = B_trans.lhs();
      viennacl::ocl::enqueue(
              k(A.get(), cl_uint(A.start1()), cl_uint(A.start2()),
                         cl_uint(A.size1()), cl_uint(A.size2()),
                         cl_uint(A.get().internal_size1()), cl_uint(A.get().internal_size2()),
                B.get(), cl_uint(B.start1()), cl_uint(B.start2()),
                         cl_uint(B.size1()), cl_uint(B.size2()),
                         cl_uint(B.get().internal_size1()), cl_uint(B.get().internal_size2()),
                C.get(), cl_uint(C.start1()), cl_uint(C.start2()),
                         cl_uint(C.size1()), cl_uint(C.size2()),
                         cl_uint(C.get().internal_size1()), cl_uint(C.get().internal_size2()),
                viennacl::ocl::local_mem(sizeof(value_type) * block_size * block_size),
                viennacl::ocl::local_mem(sizeof(value_type) * block_size * block_size) )
                            );        
    }









    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(trans(A), trans(B));
    *
    */
    template<class TYPE, typename F1, typename F2, typename F3, unsigned int ALIGNMENT>
    void prod_impl(const viennacl::matrix_expression< const matrix<TYPE, F1, ALIGNMENT>,
                                                      const matrix<TYPE, F1, ALIGNMENT>,
                                                      op_trans> & A,
                   const viennacl::matrix_expression< const matrix<TYPE, F2, ALIGNMENT>,
                                                      const matrix<TYPE, F2, ALIGNMENT>,
                                                      op_trans> & B,
                   viennacl::matrix<TYPE, F3, ALIGNMENT> & C)
    {
      assert(A.size2() == C.size1());
      assert(A.size1() == B.size2());
      assert(B.size1() == C.size2());
      // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
      assert(C.handle() != A.lhs().handle() 
             && C.handle() != B.lhs().handle()
             && "No direct inplace matrix-matrix product possible. Introduce a temporary!");
      
      int block_size = 15;

      typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< viennacl::matrix<TYPE, F1, ALIGNMENT>,
                                                                          viennacl::matrix<TYPE, F2, ALIGNMENT>,
                                                                          viennacl::matrix<TYPE, F3, ALIGNMENT> >::ResultType    KernelClass;
      KernelClass::init();
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "prod_TT");
      
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1(), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2(), block_size));
      
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);
      viennacl::ocl::enqueue(
            k(A.lhs(), cl_uint(0), cl_uint(0), 
                       cl_uint(A.lhs().size1()), cl_uint(A.lhs().size2()),
                       cl_uint(A.lhs().internal_size1()), cl_uint(A.lhs().internal_size2()),
              B.lhs(), cl_uint(0), cl_uint(0), 
                       cl_uint(B.lhs().size1()), cl_uint(B.lhs().size2()),
                       cl_uint(B.lhs().internal_size1()), cl_uint(B.lhs().internal_size2()),
              C, cl_uint(0), cl_uint(0), 
                 cl_uint(C.size1()), cl_uint(C.size2()),
                 cl_uint(C.internal_size1()), cl_uint(C.internal_size2()),
              viennacl::ocl::local_mem(sizeof(TYPE) * block_size * block_size),
              viennacl::ocl::local_mem(sizeof(TYPE) * block_size * block_size) )
                            );        
    }


    /** @brief Carries out matrix-matrix multiplication for submatrices
    *
    * Implementation of C = prod(trans(A), trans(B)); for submatrices
    *
    */
    template <typename M1, typename M2, typename M3>
    void prod_impl(const viennacl::matrix_expression< const matrix_range<M1>,
                                                      const matrix_range<M1>,
                                                      op_trans> & A_trans,
                   const viennacl::matrix_expression< const matrix_range<M2>,
                                                      const matrix_range<M2>,
                                                      op_trans> & B_trans,
                   viennacl::matrix_range<M3> & C)
    {
      typedef typename M1::value_type::value_type    value_type;
      assert(A_trans.size2() == C.size1());
      assert(A_trans.size1() == B_trans.size2());
      assert(B_trans.size1() == C.size2());
      // Inplace matrix-vector products like B = prod(A, B) are currently illegal: Introduce a temporary like C = prod(A, B); B = C; instead
      assert(C.get().handle() != A_trans.lhs().get().handle() 
             && C.get().handle() != B_trans.lhs().get().handle()
             && "No direct inplace matrix-matrix product possible. Introduce a temporary!");
      
      int block_size = 15;

      typedef typename viennacl::tools::MATRIX_PROD_KERNEL_CLASS_DEDUCER< M1, M2, M3 >::ResultType    KernelClass;
      KernelClass::init();
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "prod_TT");
      
      k.global_work_size(0, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size1(), block_size));
      k.global_work_size(1, viennacl::tools::roundUpToNextMultiple<unsigned int>(C.size2(), block_size));
      
      k.local_work_size(0, block_size);
      k.local_work_size(1, block_size);
      const matrix_range<M1> & A = A_trans.lhs();
      const matrix_range<M2> & B = B_trans.lhs();
      viennacl::ocl::enqueue(
            k(A.get(), cl_uint(A.start1()), cl_uint(A.start2()),
                       cl_uint(A.size1()), cl_uint(A.size2()),
                       cl_uint(A.get().internal_size1()), cl_uint(A.get().internal_size2()),
              B.get(), cl_uint(B.start1()), cl_uint(B.start2()),
                       cl_uint(B.size1()), cl_uint(B.size2()),
                       cl_uint(B.get().internal_size1()), cl_uint(B.get().internal_size2()),
              C.get(), cl_uint(C.start1()), cl_uint(C.start2()),
                       cl_uint(C.size1()), cl_uint(C.size2()),
                       cl_uint(C.get().internal_size1()), cl_uint(C.get().internal_size2()),
              viennacl::ocl::local_mem(sizeof(value_type) * block_size * block_size),
              viennacl::ocl::local_mem(sizeof(value_type) * block_size * block_size) )
                            );        
    }









    /** @brief Returns a proxy class for the operation mat += vec1 * vec2^T, i.e. a rank 1 update
    *
    * @param vec1    The first vector
    * @param vec2    The second vector
    */
    template<class SCALARTYPE, unsigned int VA1, unsigned int VA2>
    viennacl::matrix_expression< const viennacl::vector<SCALARTYPE, VA1>,
                                 const viennacl::vector<SCALARTYPE, VA2>,
                                 op_prod> outer_prod(const viennacl::vector<SCALARTYPE, VA1> & vec1, 
                                                     const viennacl::vector<SCALARTYPE, VA2> & vec2)
    {
      return viennacl::matrix_expression< const viennacl::vector<SCALARTYPE, VA1>,
                                          const viennacl::vector<SCALARTYPE, VA2>,
                                          op_prod>(vec1, vec2);
    }
    
    

    /** @brief The implementation of the operation mat += vec1 * vec2^T, i.e. a rank 1 update
    *
    * Implementation of the convenience expression result += outer_prod(vec1, vec2);
    *
    * @param mat1    The matrix to be updated
    * @param vec1    The first vector
    * @param vec2    The second vector
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT>
    void rank_1_update(viennacl::matrix<SCALARTYPE, F, ALIGNMENT> & mat1, 
                       const viennacl::vector<SCALARTYPE, ALIGNMENT> & vec1, 
                       const viennacl::vector<SCALARTYPE, ALIGNMENT> & vec2)
    {
      assert(mat1.size1() == vec1.size());
      assert(mat1.size2() == vec2.size());

      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "rank1_update");

      viennacl::ocl::enqueue(k(mat1, cl_uint(mat1.size1()), cl_uint(mat1.size2()),
                                     cl_uint(mat1.internal_size1()), cl_uint(mat1.internal_size2()), vec1, vec2));        
    }
    
    
    /** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
    *
    * Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
    *
    * @param mat1    The matrix to be updated
    * @param val     The scaling factor
    * @param vec1    The first vector
    * @param vec2    The second vector
    */
    template<class SCALARTYPE, typename F, unsigned int ALIGNMENT>
    void scaled_rank_1_update(viennacl::matrix<SCALARTYPE, F, ALIGNMENT> & mat1,
                              SCALARTYPE val,
                              const viennacl::vector<SCALARTYPE, ALIGNMENT> & vec1, 
                              const viennacl::vector<SCALARTYPE, ALIGNMENT> & vec2)
    {
      assert(mat1.size1() == vec1.size());
      assert(mat1.size2() == vec2.size());

      typedef typename viennacl::tools::MATRIX_KERNEL_CLASS_DEDUCER< matrix<SCALARTYPE, F, ALIGNMENT> >::ResultType    KernelClass;
      
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(KernelClass::program_name(), "scaled_rank1_update");

      viennacl::ocl::enqueue(k(mat1, cl_uint(mat1.size1()), cl_uint(mat1.size2()),
                                     cl_uint(mat1.internal_size1()), cl_uint(mat1.internal_size2()), 
                                                           val, vec1, vec2));        
    }
    
  } //namespace linalg


    //v = A * x
    /** @brief Implementation of the operation v1 = A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator=(const viennacl::vector_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                          const viennacl::vector<SCALARTYPE, ALIGNMENT>,
                                                                                          viennacl::op_prod> & proxy) 
    {
      // check for the special case x = A * x
      if (proxy.rhs().handle() == this->handle())
      {
        viennacl::vector<SCALARTYPE, ALIGNMENT> result(proxy.rhs().size());
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
        *this = result;
        return *this;
      }
      else
      {
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
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
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator+=(const vector_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                 const vector<SCALARTYPE, ALIGNMENT>,
                                                                                 op_prod> & proxy) 
    {
      vector<SCALARTYPE, ALIGNMENT> result(proxy.lhs().size1());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      *this += result;
      return *this;
    }

    /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator-=(const vector_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                 const vector<SCALARTYPE, ALIGNMENT>,
                                                                                 op_prod> & proxy) 
    {
      vector<SCALARTYPE, ALIGNMENT> result(proxy.lhs().size1());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      *this -= result;
      return *this;
    }
    
    
    //free functions:
    /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator+(const vector_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                const vector<SCALARTYPE, ALIGNMENT>,
                                                                                op_prod> & proxy) 
    {
      assert(proxy.lhs().size1() == size());
      vector<SCALARTYPE, ALIGNMENT> result(size());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      result += *this;
      return result;
    }

    /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator-(const vector_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                const vector<SCALARTYPE, ALIGNMENT>,
                                                                                op_prod> & proxy) 
    {
      assert(proxy.lhs().size1() == size());
      vector<SCALARTYPE, ALIGNMENT> result(size());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      result = *this - result;
      return result;
    }


    ////////// transposed_matrix_proxy


    //v = trans(A) * x
    /** @brief Implementation of the operation v1 = A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator=(const viennacl::vector_expression< const matrix_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                                   const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                                   op_trans>,
                                                                                          const viennacl::vector<SCALARTYPE, ALIGNMENT>,
                                                                                          viennacl::op_prod> & proxy) 
    {
      // check for the special case x = trans(A) * x
      if (proxy.rhs().handle() == this->handle())
      {
        viennacl::vector<SCALARTYPE, ALIGNMENT> result(proxy.rhs().size());
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
        *this = result;
        return *this;
      }
      else
      {
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
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
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator+=(const vector_expression< const matrix_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                          const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                          op_trans>,
                                                                                 const vector<SCALARTYPE, ALIGNMENT>,
                                                                                 op_prod> & proxy) 
    {
      vector<SCALARTYPE, ALIGNMENT> result(proxy.lhs().size1());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      *this += result;
      return *this;
    }

    /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> & 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator-=(const vector_expression< const matrix_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                          const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                          op_trans>,
                                                                                 const vector<SCALARTYPE, ALIGNMENT>,
                                                                                 op_prod> & proxy) 
    {
      vector<SCALARTYPE, ALIGNMENT> result(proxy.lhs().size1());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      *this -= result;
      return *this;
    }
    
    
    //free functions:
    /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator+(const vector_expression< const matrix_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                         const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                         op_trans>,
                                                                                const vector<SCALARTYPE, ALIGNMENT>,
                                                                                op_prod> & proxy) 
    {
      assert(proxy.lhs().size1() == size());
      vector<SCALARTYPE, ALIGNMENT> result(size());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      result += *this;
      return result;
    }

    /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
    *
    * @param proxy  An expression template proxy class.
    */
    template <typename SCALARTYPE, unsigned int ALIGNMENT>
    template <typename F, unsigned int MAT_ALIGNMENT>
    viennacl::vector<SCALARTYPE, ALIGNMENT> 
    viennacl::vector<SCALARTYPE, ALIGNMENT>::operator-(const vector_expression< const matrix_expression< const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                         const matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                                                                                         op_trans>,
                                                                                const vector<SCALARTYPE, ALIGNMENT>,
                                                                                op_prod> & proxy) 
    {
      assert(proxy.lhs().size1() == size());
      vector<SCALARTYPE, ALIGNMENT> result(size());
      viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
      result = *this - result;
      return result;
    }


} //namespace viennacl


#endif
