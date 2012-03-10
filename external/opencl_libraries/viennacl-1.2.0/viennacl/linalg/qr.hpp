#ifndef VIENNACL_LINALG_QR_HPP
#define VIENNACL_LINALG_QR_HPP

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

/** @file viennacl/linalg/qr.hpp
    @brief Proivdes a QR factorization using a block-based approach.  Experimental in 1.2.x.
*/

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <cmath>
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/matrix_expression.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

namespace viennacl
{
    namespace linalg
    {
      
        // orthogonalises j-th column of A
        template <typename MatrixType, typename VectorType>
        typename MatrixType::value_type setup_householder_vector(MatrixType const & A, VectorType & v, size_t j)
        {
          typedef typename MatrixType::value_type   ScalarType;
          
          //compute norm of column below diagonal:
          ScalarType sigma = 0;
          ScalarType beta = 0;
          for (size_t k = j+1; k<A.size1(); ++k)
            sigma += A(k, j) * A(k, j);

          //get v from A:
          for (size_t k = j+1; k<A.size1(); ++k)
            v[k] = A(k, j);
          
          if (sigma == 0)
            return 0;
          else
          {
            ScalarType mu = sqrt(sigma + A(j,j)*A(j,j));
            //std::cout << "mu: " << mu << std::endl;
            //std::cout << "sigma: " << sigma << std::endl;
            
            ScalarType v1;
            if (A(j,j) <= 0)
              v1 = A(j,j) - mu;
            else
              v1 = -sigma / (A(j,j) + mu);
            
            beta = 2.0 * v1 * v1 / (sigma + v1 * v1);
            
            //divide v by its diagonal element v[j]
            v[j] = 1;
            for (size_t k = j+1; k<A.size1(); ++k)
              v[k] /= v1;
          }
            
          return beta;
        }

        // Apply (I - beta v v^T) to the k-th column of A, where v is the reflector starting at j-th row/column
        template <typename MatrixType, typename VectorType, typename ScalarType>
        void householder_reflect(MatrixType & A, VectorType & v, ScalarType beta, size_t j, size_t k)
        {
          ScalarType v_in_col = A(j,k);
          for (size_t i=j+1; i<A.size1(); ++i)
            v_in_col += v[i] * A(i,k);

          for (size_t i=j; i<A.size1(); ++i)
            A(i,k) -= beta * v_in_col * v[i];
        }

        // Apply (I - beta v v^T) to A, where v is the reflector starting at j-th row/column
        template <typename MatrixType, typename VectorType, typename ScalarType>
        void householder_reflect(MatrixType & A, VectorType & v, ScalarType beta, size_t j)
        {
          size_t column_end = A.size2();
          
          for (size_t k=j; k<column_end; ++k) //over columns
            householder_reflect(A, v, beta, j, k);
        }
        
        
        template <typename MatrixType, typename VectorType>
        void write_householder_to_A(MatrixType & A, VectorType const & v, size_t j)
        {
          for (size_t i=j+1; i<A.size1(); ++i)
            A(i,j) = v[i];
        }
        
        
        //takes an inplace QR matrix A and generates Q and R explicitly
        template <typename MatrixType, typename VectorType>
        void recoverQ(MatrixType const & A, VectorType const & betas, MatrixType & Q, MatrixType & R)
        {
          typedef typename MatrixType::value_type   ScalarType;
          
          std::vector<ScalarType> v(A.size1());

          Q.clear();
          R.clear();

          //
          // Recover R from upper-triangular part of A:
          //
          size_t i_max = std::min(R.size1(), R.size2());
          for (size_t i=0; i<i_max; ++i)
            for (size_t j=i; j<R.size2(); ++j)
              R(i,j) = A(i,j);
         
          //
          // Recover Q by applying all the Householder reflectors to the identity matrix:
          //
          for (size_t i=0; i<Q.size1(); ++i)
            Q(i,i) = 1.0;

          size_t j_max = std::min(A.size1(), A.size2());
          for (size_t j=0; j<j_max; ++j)
          {
            size_t col_index = j_max - j - 1;
            v[col_index] = 1.0;
            for (size_t i=col_index+1; i<A.size1(); ++i)
              v[i] = A(i, col_index);
            
            /*std::cout << "Recovery with beta = " << betas[col_index] << ", j=" << col_index << std::endl;
            std::cout << "v: ";
            for (size_t i=0; i<v.size(); ++i)
              std::cout << v[i] << ", ";
            std::cout << std::endl;*/

            if (betas[col_index] != 0)
              householder_reflect(Q, v, betas[col_index], col_index);
          }
        }
       
        /*template<typename MatrixType>
        std::vector<typename MatrixType::value_type> qr(MatrixType & A)
        {
          typedef typename MatrixType::value_type   ScalarType;
          
          std::vector<ScalarType> betas(A.size2());
          std::vector<ScalarType> v(A.size1());

          //copy A to Q:
          for (size_t j=0; j<A.size2(); ++j)
          {
             betas[j] = setup_householder_vector(A, v, j);
             householder_reflect(A, v, betas[j], j);
             write_householder_to_A(A, v, j);
          }
          
          return betas;
        }*/
        
        
        
        class range
        {
          public:
            range(size_t start, size_t end) : start_(start), end_(end) {}
            
            size_t lower() const { return start_; }
            size_t upper() const { return end_; }
            
          private:
            size_t start_;
            size_t end_;
        };

        template <typename MatrixType>
        class sub_matrix
        {
          public:
            typedef typename MatrixType::value_type value_type;
            
            sub_matrix(MatrixType & mat,
                       range row_range,
                       range col_range) : mat_(mat), row_range_(row_range), col_range_(col_range) {}
                       
            value_type operator()(size_t row, size_t col) const
            {
              assert(row < size1());
              assert(col < size2());
              return mat_(row + row_range_.lower(), col + col_range_.lower()); 
            }
                       
            size_t size1() const { return row_range_.upper() - row_range_.lower(); }
            size_t size2() const { return col_range_.upper() - col_range_.lower(); }
            
          private:
            MatrixType & mat_;
            range row_range_;
            range col_range_;
        };


        //computes C = prod(A, B)
        template <typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
        void prod_AA(MatrixTypeA const & A, MatrixTypeB const & B, MatrixTypeC & C)
        {
          assert(C.size1() == A.size1());
          assert(A.size2() == B.size1());
          assert(B.size2() == C.size2());
          
          typedef typename MatrixTypeC::value_type   ScalarType;
          
          for (size_t i=0; i<C.size1(); ++i)
          {
            for (size_t j=0; j<C.size2(); ++j)
            {
              ScalarType val = 0;
              for (size_t k=0; k<A.size2(); ++k)
                val += A(i, k) * B(k, j);
              C(i, j) = val;
            }
          }
        }
        
        //computes C = prod(A^T, B)
        template <typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
        void prod_TA(MatrixTypeA const & A, MatrixTypeB const & B, MatrixTypeC & C)
        {
          assert(C.size1() == A.size2());
          assert(A.size1() == B.size1());
          assert(B.size2() == C.size2());
          
          typedef typename MatrixTypeC::value_type   ScalarType;
          
          for (size_t i=0; i<C.size1(); ++i)
          {
            for (size_t j=0; j<C.size2(); ++j)
            {
              ScalarType val = 0;
              for (size_t k=0; k<A.size1(); ++k)
                val += A(k, i) * B(k, j);
              C(i, j) = val;
            }
          }
        }
        
        

        template<typename MatrixType>
        std::vector<typename MatrixType::value_type> inplace_qr(MatrixType & A, std::size_t block_size = 32)
        {
          typedef typename MatrixType::value_type   ScalarType;
          
          if ( A.size2() % block_size != 0 )
            std::cout << "ViennaCL: Warning in inplace_qr(): Matrix columns are not divisible by block_size!" << std::endl;
            
          std::vector<ScalarType> betas(A.size2());
          std::vector<ScalarType> v(A.size1());

          //size_t block_size = 90;
          MatrixType Y(A.size1(), block_size); Y.clear();
          MatrixType W(A.size1(), block_size); W.clear();
            
          //run over A in a block-wise manner:
          for (size_t j = 0; j < std::min(A.size1(), A.size2()); j += block_size)
          {
            //determine Householder vectors:
            for (size_t k = 0; k < block_size; ++k)
            {
              betas[j+k] = setup_householder_vector(A, v, j+k);
              for (size_t l = k; l < block_size; ++l)
                householder_reflect(A, v, betas[j+k], j+k, j+l);

              write_householder_to_A(A, v, j+k);
            }

            //
            // Setup Y:
            //
            for (size_t k = 0; k < block_size; ++k)
            {
              //write Householder to Y:
              Y(k,k) = 1.0;
              for (size_t l=k+1; l<A.size1(); ++l)
                Y(l,k) = A(l, j+k);
            }
            
            //
            // Setup W:
            //
            
            //first vector:
            W(j, 0) = -betas[j];
            for (size_t l=j+1; l<A.size1(); ++l)
              W(l,0) = -betas[j] * A(l, j);
            
            //k-th column of W is given by -beta * (Id + W*Y^T) v_k, where W and Y have k-1 columns
            for (size_t k = 1; k < block_size; ++k)
            {
              //compute Y^T v_k:
              std::vector<ScalarType> temp(k);  //actually of size (k \times 1)
              for (size_t l=0; l<k; ++l)
                for (size_t n=j; n<A.size1(); ++n)
                  temp[l] += Y(n, l) * Y(n, k);
                
              //compute W * temp and add to z, which is directly written to W:
              for (size_t n=0; n<A.size1(); ++n)
              {
                ScalarType val = 0;
                for (size_t l=0; l<k; ++l)
                  val += temp[l] * W(n, l);
                W(n, k) = -1.0 * betas[j+k] * (Y(n, k) + val);
              }
            }

            //
            //apply (I+WY^T)^T = I + Y W^T to the remaining columns of A:
            //
            
            if (A.size2() - j - block_size > 0)
            {
              //temp = prod(W^T, A)
              
              MatrixType temp(block_size, A.size2() - j - block_size);
              
              boost::numeric::ublas::range A_rows(j, A.size1());
              boost::numeric::ublas::range A_cols(j+block_size, A.size2());
              boost::numeric::ublas::matrix_range<MatrixType> A_part(A, A_rows, A_cols);

              viennacl::matrix<ScalarType, viennacl::column_major> gpu_A_part(A_part.size1(), A_part.size2());
              viennacl::copy(A_part, gpu_A_part);

              //transfer W
              boost::numeric::ublas::range W_cols(0, block_size);
              boost::numeric::ublas::matrix_range<MatrixType> W_part(W, A_rows, W_cols);
              viennacl::matrix<ScalarType, viennacl::column_major> gpu_W(W_part.size1(), W_part.size2());
              viennacl::copy(W_part, gpu_W);
              
              viennacl::matrix<ScalarType, viennacl::column_major> gpu_temp(gpu_W.size2(), gpu_A_part.size2());
              gpu_temp = viennacl::linalg::prod(trans(gpu_W), gpu_A_part);
              
              
              
              //A += Y * temp:
              boost::numeric::ublas::range Y_cols(0, Y.size2());
              boost::numeric::ublas::matrix_range<MatrixType> Y_part(Y, A_rows, Y_cols);
              
              viennacl::matrix<ScalarType, viennacl::column_major> gpu_Y(Y_part.size1(), Y_part.size2());
              viennacl::copy(Y_part, gpu_Y);

              //A_part += prod(Y_part, temp);
              gpu_A_part += prod(gpu_Y, gpu_temp);
              
              MatrixType A_part_back(A_part.size1(), A_part.size2());
              viennacl::copy(gpu_A_part, A_part_back);
                
              A_part = A_part_back;
              //A_part += prod(Y_part, temp);
            }
          }
          
          return betas;
        }


        template<typename MatrixType>
        std::vector<typename MatrixType::value_type> inplace_qr_ublas(MatrixType & A)
        {
          typedef typename MatrixType::value_type   ScalarType;
          
          std::vector<ScalarType> betas(A.size2());
          std::vector<ScalarType> v(A.size1());

          size_t block_size = 3;
          MatrixType Y(A.size1(), block_size); Y.clear();
          MatrixType W(A.size1(), block_size); W.clear();
            
          //run over A in a block-wise manner:
          for (size_t j = 0; j < std::min(A.size1(), A.size2()); j += block_size)
          {
            //determine Householder vectors:
            for (size_t k = 0; k < block_size; ++k)
            {
              betas[j+k] = setup_householder_vector(A, v, j+k);
              for (size_t l = k; l < block_size; ++l)
                householder_reflect(A, v, betas[j+k], j+k, j+l);

              write_householder_to_A(A, v, j+k);
            }

            //
            // Setup Y:
            //
            for (size_t k = 0; k < block_size; ++k)
            {
              //write Householder to Y:
              Y(k,k) = 1.0;
              for (size_t l=k+1; l<A.size1(); ++l)
                Y(l,k) = A(l, j+k);
            }
            
            //
            // Setup W:
            //
            
            //first vector:
            W(j, 0) = -betas[j];
            for (size_t l=j+1; l<A.size1(); ++l)
              W(l,0) = -betas[j] * A(l, j);
            
            //k-th column of W is given by -beta * (Id + W*Y^T) v_k, where W and Y have k-1 columns
            for (size_t k = 1; k < block_size; ++k)
            {
              //compute Y^T v_k:
              std::vector<ScalarType> temp(k);  //actually of size (k \times 1)
              for (size_t l=0; l<k; ++l)
                for (size_t n=j; n<A.size1(); ++n)
                  temp[l] += Y(n, l) * Y(n, k);
                
              //compute W * temp and add to z, which is directly written to W:
              for (size_t n=0; n<A.size1(); ++n)
              {
                ScalarType val = 0;
                for (size_t l=0; l<k; ++l)
                  val += temp[l] * W(n, l);
                W(n, k) = -1.0 * betas[j+k] * (Y(n, k) + val);
              }
            }

            //
            //apply (I+WY^T)^T = I + Y W^T to the remaining columns of A:
            //
            
            if (A.size2() - j - block_size > 0)
            {
              //temp = prod(W^T, A)
              MatrixType temp(block_size, A.size2() - j - block_size);
              
              boost::numeric::ublas::range A_rows(j, A.size1());
              boost::numeric::ublas::range A_cols(j+block_size, A.size2());
              boost::numeric::ublas::matrix_range<MatrixType> A_part(A, A_rows, A_cols);

              boost::numeric::ublas::range W_cols(0, block_size);
              boost::numeric::ublas::matrix_range<MatrixType> W_part(W, A_rows, W_cols);
              
              temp = boost::numeric::ublas::prod(trans(W_part), A_part);
              
              
              //A += Y * temp:
              boost::numeric::ublas::range Y_cols(0, Y.size2());
              boost::numeric::ublas::matrix_range<MatrixType> Y_part(Y, A_rows, Y_cols);
              
              A_part += prod(Y_part, temp);
            }
          }
          
          return betas;
        }


        template<typename MatrixType>
        std::vector<typename MatrixType::value_type> inplace_qr_pure(MatrixType & A)
        {
          typedef typename MatrixType::value_type   ScalarType;
          
          std::vector<ScalarType> betas(A.size2());
          std::vector<ScalarType> v(A.size1());

          size_t block_size = 5;
          MatrixType Y(A.size1(), block_size); Y.clear();
          MatrixType W(A.size1(), block_size); W.clear();
            
          //run over A in a block-wise manner:
          for (size_t j = 0; j < std::min(A.size1(), A.size2()); j += block_size)
          {
            //determine Householder vectors:
            for (size_t k = 0; k < block_size; ++k)
            {
              betas[j+k] = setup_householder_vector(A, v, j+k);
              for (size_t l = k; l < block_size; ++l)
                householder_reflect(A, v, betas[j+k], j+k, j+l);

              write_householder_to_A(A, v, j+k);
            }

            //
            // Setup Y:
            //
            for (size_t k = 0; k < block_size; ++k)
            {
              //write Householder to Y:
              Y(k,k) = 1.0;
              for (size_t l=k+1; l<A.size1(); ++l)
                Y(l,k) = A(l, j+k);
            }
            
            //
            // Setup W:
            //
            
            //first vector:
            W(j, 0) = -betas[j];
            for (size_t l=j+1; l<A.size1(); ++l)
              W(l,0) = -betas[j] * A(l, j);
            
            //k-th column of W is given by -beta * (Id + W*Y^T) v_k, where W and Y have k-1 columns
            for (size_t k = 1; k < block_size; ++k)
            {
              //compute Y^T v_k:
              std::vector<ScalarType> temp(k);  //actually of size (k \times 1)
              for (size_t l=0; l<k; ++l)
                for (size_t n=j; n<A.size1(); ++n)
                  temp[l] += Y(n, l) * Y(n, k);
                
              //compute W * temp and add to z, which is directly written to W:
              for (size_t n=0; n<A.size1(); ++n)
              {
                ScalarType val = 0;
                for (size_t l=0; l<k; ++l)
                  val += temp[l] * W(n, l);
                W(n, k) = -1.0 * betas[j+k] * (Y(n, k) + val);
              }
            }

            //
            //apply (I+WY^T)^T = I + Y W^T to the remaining columns of A:
            //
            
            if (A.size2() - j - block_size > 0)
            {
              //temp = prod(W^T, A)
              MatrixType temp(block_size, A.size2() - j - block_size);
              ScalarType entry = 0;
              for (size_t l = 0; l < temp.size2(); ++l)
              {
                for (size_t k = 0; k < temp.size1(); ++k)
                {
                  entry = 0;
                  for (size_t n = j; n < A.size1(); ++n)
                    entry += W(n, k) * A(n, j + block_size + l);
                  temp(k,l) = entry;
                }
              }
              
              //A += Y * temp:
              for (size_t l = j+block_size; l < A.size2(); ++l)
              {
                for (size_t k = j; k<A.size1(); ++k)
                {
                  ScalarType val = 0;
                  for (size_t n=0; n<block_size; ++n)
                    val += Y(k, n) * temp(n, l-j-block_size);
                  A(k, l) += val;
                }
              }
            }
          }
          
          return betas;
        }
        
    } //linalg
} //viennacl


#endif
