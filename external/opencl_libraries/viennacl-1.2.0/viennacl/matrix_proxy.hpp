#ifndef VIENNACL_MATRIX_PROXY_HPP_
#define VIENNACL_MATRIX_PROXY_HPP_

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

/** @file matrix_proxy.hpp
    @brief Proxy classes for matrices.
*/

#include "viennacl/forwards.h"
#include "viennacl/range.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/matrix_operations.hpp"

namespace viennacl
{

  template <typename MatrixType>
  class matrix_range
  {
    public:
      typedef typename MatrixType::value_type     value_type;
      typedef range::size_type                    size_type;
      typedef range::difference_type              difference_type;
      typedef value_type                          reference;
      typedef const value_type &                  const_reference;
      
      matrix_range(MatrixType & A, 
                   range const & row_range,
                   range const & col_range) : A_(A), row_range_(row_range), col_range_(col_range) {}
                   
      size_type start1() const { return row_range_.start(); }
      size_type size1() const { return row_range_.size(); }

      size_type start2() const { return col_range_.start(); }
      size_type size2() const { return col_range_.size(); }
      
      template <typename MatrixType1, typename MatrixType2>
      matrix_range<MatrixType> & operator = (const matrix_expression< MatrixType1,
                                                                      MatrixType2,
                                                                      op_prod > & proxy) 
      {
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
        return *this;
      }
      
      
      matrix_range<MatrixType> & operator += (matrix_range<MatrixType> const & other)
      {
        viennacl::linalg::inplace_add(*this, other);
        return *this;
      }
      
      template <typename MatrixType1, typename MatrixType2>
      matrix_range<MatrixType> & operator += (const matrix_expression< MatrixType1,
                                                                       MatrixType2,
                                                                       op_prod > & proxy)
      {
        MatrixType1 temp = proxy;
        viennacl::range r1(0, temp.size1());
        viennacl::range r2(0, temp.size2());
        viennacl::matrix_range<MatrixType> temp2(temp, r1, r2);
        viennacl::linalg::inplace_add(*this, temp2);
        return *this;
      }
      
      template <typename MatrixType1, typename MatrixType2>
      matrix_range<MatrixType> & operator += (const matrix_expression< const matrix_range<MatrixType1>,
                                                                       const matrix_range<MatrixType2>,
                                                                       op_prod > & proxy)
      {
        MatrixType1 temp(proxy.size1(), proxy.size2());
        viennacl::range r1(0, temp.size1());
        viennacl::range r2(0, temp.size2());
        viennacl::matrix_range<MatrixType> temp2(temp, r1, r2);
        temp2 = proxy;
        viennacl::linalg::inplace_add(*this, temp2);
        return *this;
      }

      //const_reference operator()(size_type i, size_type j) const { return A_(start1() + i, start2() + i); }
      //reference operator()(size_type i, size_type j) { return A_(start1() + i, start2() + i); }

      MatrixType & get() { return A_; }
      const MatrixType & get() const { return A_; }

    private:
      MatrixType & A_;
      range row_range_;
      range col_range_;
  };

  
  /** @brief Returns an expression template class representing a transposed matrix */
  template <typename MatrixType>
  matrix_expression< const matrix_range<MatrixType>,
                     const matrix_range<MatrixType>,
                     op_trans> trans(const matrix_range<MatrixType> & mat)
  {
    return matrix_expression< const matrix_range<MatrixType>,
                              const matrix_range<MatrixType>,
                              op_trans>(mat, mat);
  }
  
  
  
  
  /////////////////////////////////////////////////////////////
  ///////////////////////// CPU to GPU ////////////////////////
  /////////////////////////////////////////////////////////////
  
  //row_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(const CPU_MATRIX & cpu_matrix,
            matrix_range<matrix<SCALARTYPE, row_major, 1> > & gpu_matrix_range )
  {
    assert( (cpu_matrix.size1() == gpu_matrix_range.size1())
           && (cpu_matrix.size2() == gpu_matrix_range.size2()) );
    
     if ( gpu_matrix_range.start2() != 0 ||  gpu_matrix_range.size2() !=  gpu_matrix_range.get().size2())
     {
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size2());
       
       //copy each stride separately:
       for (size_t i=0; i < gpu_matrix_range.size1(); ++i)
       {
         for (size_t j=0; j < gpu_matrix_range.size2(); ++j)
           entries[j] = cpu_matrix(i,j);
         
         size_t start_offset = (gpu_matrix_range.start1() + i) * gpu_matrix_range.get().internal_size2() + gpu_matrix_range.start2();
         size_t num_entries = gpu_matrix_range.size2();
         cl_int err = clEnqueueWriteBuffer(viennacl::ocl::get_queue().handle(),
                                          gpu_matrix_range.get().handle(), CL_TRUE, 
                                          sizeof(SCALARTYPE)*start_offset,
                                          sizeof(SCALARTYPE)*num_entries,
                                          &(entries[0]), 0, NULL, NULL);
        VIENNACL_ERR_CHECK(err);
        //std::cout << "Strided copy worked!" << std::endl;
       }
     }
     else
     {
       //full block can be copied: 
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1()*gpu_matrix_range.size2());
       
       //copy each stride separately:
       for (size_t i=0; i < gpu_matrix_range.size1(); ++i)
         for (size_t j=0; j < gpu_matrix_range.size2(); ++j)
           entries[i*gpu_matrix_range.get().internal_size2() + j] = cpu_matrix(i,j);
       
       size_t start_offset = gpu_matrix_range.start1() * gpu_matrix_range.get().internal_size2();
       size_t num_entries = gpu_matrix_range.size1() * gpu_matrix_range.size2();
       //std::cout << "start_offset: " << start_offset << std::endl;
       cl_int err = clEnqueueWriteBuffer(viennacl::ocl::get_queue().handle(),
                                         gpu_matrix_range.get().handle(), CL_TRUE, 
                                         sizeof(SCALARTYPE)*start_offset,
                                         sizeof(SCALARTYPE)*num_entries,
                                         &(entries[0]), 0, NULL, NULL);
       VIENNACL_ERR_CHECK(err);
       //std::cout << "Block copy worked!" << std::endl;
     }
  }
  
  //column_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(const CPU_MATRIX & cpu_matrix,
            matrix_range<matrix<SCALARTYPE, column_major, 1> > & gpu_matrix_range )
  {
    assert( (cpu_matrix.size1() == gpu_matrix_range.size1())
           && (cpu_matrix.size2() == gpu_matrix_range.size2()) );
    
     if ( gpu_matrix_range.start1() != 0 ||  gpu_matrix_range.size1() != gpu_matrix_range.get().size1())
     {
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1());
       
       //copy each stride separately:
       for (size_t j=0; j < gpu_matrix_range.size2(); ++j)
       {
         for (size_t i=0; i < gpu_matrix_range.size1(); ++i)
           entries[i] = cpu_matrix(i,j);
         
         size_t start_offset = (gpu_matrix_range.start2() + j) * gpu_matrix_range.get().internal_size1() + gpu_matrix_range.start1();
         size_t num_entries = gpu_matrix_range.size1();
         cl_int err = clEnqueueWriteBuffer(viennacl::ocl::get_queue().handle(),
                                          gpu_matrix_range.get().handle(), CL_TRUE, 
                                          sizeof(SCALARTYPE)*start_offset,
                                          sizeof(SCALARTYPE)*num_entries,
                                          &(entries[0]), 0, NULL, NULL);
        VIENNACL_ERR_CHECK(err);
        //std::cout << "Strided copy worked!" << std::endl;
       }
     }
     else
     {
       //full block can be copied: 
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1()*gpu_matrix_range.size2());
       
       //copy each stride separately:
       for (size_t i=0; i < gpu_matrix_range.size1(); ++i)
         for (size_t j=0; j < gpu_matrix_range.size2(); ++j)
           entries[i + j*gpu_matrix_range.get().internal_size1()] = cpu_matrix(i,j);
       
       size_t start_offset = gpu_matrix_range.start2() * gpu_matrix_range.get().internal_size1();
       size_t num_entries = gpu_matrix_range.size1() * gpu_matrix_range.size2();
       //std::cout << "start_offset: " << start_offset << std::endl;
       cl_int err = clEnqueueWriteBuffer(viennacl::ocl::get_queue().handle(),
                                         gpu_matrix_range.get().handle(), CL_TRUE, 
                                         sizeof(SCALARTYPE)*start_offset,
                                         sizeof(SCALARTYPE)*num_entries,
                                         &(entries[0]), 0, NULL, NULL);
       VIENNACL_ERR_CHECK(err);
       //std::cout << "Block copy worked!" << std::endl;
     }
    
  }


  /////////////////////////////////////////////////////////////
  ///////////////////////// GPU to CPU ////////////////////////
  /////////////////////////////////////////////////////////////
  
  
  //row_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(matrix_range<matrix<SCALARTYPE, row_major, 1> > const & gpu_matrix_range,
            CPU_MATRIX & cpu_matrix)
  {
    assert( (cpu_matrix.size1() == gpu_matrix_range.size1())
           && (cpu_matrix.size2() == gpu_matrix_range.size2()) );
    
     if ( gpu_matrix_range.start2() != 0 ||  gpu_matrix_range.size2() !=  gpu_matrix_range.get().size2())
     {
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size2());
       
       //copy each stride separately:
       for (size_t i=0; i < gpu_matrix_range.size1(); ++i)
       {
         size_t start_offset = (gpu_matrix_range.start1() + i) * gpu_matrix_range.get().internal_size2() + gpu_matrix_range.start2();
         size_t num_entries = gpu_matrix_range.size2();
         cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle(),
                                          gpu_matrix_range.get().handle(), CL_TRUE, 
                                          sizeof(SCALARTYPE)*start_offset,
                                          sizeof(SCALARTYPE)*num_entries,
                                          &(entries[0]), 0, NULL, NULL);
        VIENNACL_ERR_CHECK(err);
        //std::cout << "Strided copy worked!" << std::endl;
        
        for (size_t j=0; j < gpu_matrix_range.size2(); ++j)
          cpu_matrix(i,j) = entries[j];
         
       }
     }
     else
     {
       //full block can be copied: 
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1()*gpu_matrix_range.size2());
       
       size_t start_offset = gpu_matrix_range.start1() * gpu_matrix_range.get().internal_size2();
       size_t num_entries = gpu_matrix_range.size1() * gpu_matrix_range.size2();
       //std::cout << "start_offset: " << start_offset << std::endl;
       cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle(),
                                         gpu_matrix_range.get().handle(), CL_TRUE, 
                                         sizeof(SCALARTYPE)*start_offset,
                                         sizeof(SCALARTYPE)*num_entries,
                                         &(entries[0]), 0, NULL, NULL);
       VIENNACL_ERR_CHECK(err);
       //std::cout << "Block copy worked!" << std::endl;

       for (size_t i=0; i < gpu_matrix_range.size1(); ++i)
         for (size_t j=0; j < gpu_matrix_range.size2(); ++j)
           cpu_matrix(i,j) = entries[i*gpu_matrix_range.get().internal_size2() + j];
    }
    
  }
  
  
  //column_major:
  template <typename CPU_MATRIX, typename SCALARTYPE>
  void copy(matrix_range<matrix<SCALARTYPE, column_major, 1> > const & gpu_matrix_range,
            CPU_MATRIX & cpu_matrix)
  {
    assert( (cpu_matrix.size1() == gpu_matrix_range.size1())
           && (cpu_matrix.size2() == gpu_matrix_range.size2()) );
    
     if ( gpu_matrix_range.start1() != 0 ||  gpu_matrix_range.size1() !=  gpu_matrix_range.get().size1())
     {
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1());
       
       //copy each stride separately:
       for (size_t j=0; j < gpu_matrix_range.size2(); ++j)
       {
         size_t start_offset = (gpu_matrix_range.start2() + j) * gpu_matrix_range.get().internal_size1() + gpu_matrix_range.start1();
         size_t num_entries = gpu_matrix_range.size1();
         cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle(),
                                          gpu_matrix_range.get().handle(), CL_TRUE, 
                                          sizeof(SCALARTYPE)*start_offset,
                                          sizeof(SCALARTYPE)*num_entries,
                                          &(entries[0]), 0, NULL, NULL);
        VIENNACL_ERR_CHECK(err);
        //std::cout << "Strided copy worked!" << std::endl;
        
        for (size_t i=0; i < gpu_matrix_range.size1(); ++i)
          cpu_matrix(i,j) = entries[i];
       }
     }
     else
     {
       //full block can be copied: 
       std::vector<SCALARTYPE> entries(gpu_matrix_range.size1()*gpu_matrix_range.size2());
       
       //copy each stride separately:
       size_t start_offset = gpu_matrix_range.start2() * gpu_matrix_range.get().internal_size1();
       size_t num_entries = gpu_matrix_range.size1() * gpu_matrix_range.size2();
       //std::cout << "start_offset: " << start_offset << std::endl;
       cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle(),
                                         gpu_matrix_range.get().handle(), CL_TRUE, 
                                         sizeof(SCALARTYPE)*start_offset,
                                         sizeof(SCALARTYPE)*num_entries,
                                         &(entries[0]), 0, NULL, NULL);
       VIENNACL_ERR_CHECK(err);
       //std::cout << "Block copy worked!" << std::endl;
       
       for (size_t i=0; i < gpu_matrix_range.size1(); ++i)
         for (size_t j=0; j < gpu_matrix_range.size2(); ++j)
           cpu_matrix(i,j) = entries[i + j*gpu_matrix_range.get().internal_size1()];
     }
    
  }


/*
  template<typename MatrixType>
  std::ostream & operator<<(std::ostream & s, matrix_range<MatrixType> const & proxy)
  {
    MatrixType temp(proxy.size1(), proxy.size2());
    viennacl::range r1(0, proxy.size1());
    viennacl::range r2(0, proxy.size2());
    matrix_range<MatrixType> temp2(temp, r1, r2);
    viennacl::copy(proxy, temp2);
    s << temp;
    return s;
  }*/


}

#endif