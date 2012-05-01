#ifndef VIENNACL_VECTOR_PROXY_HPP_
#define VIENNACL_VECTOR_PROXY_HPP_

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
   Institute for Analysis and Scientific Computing,
   TU Wien.

   -----------------
   ViennaCL - The Vienna Computing Library
   -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file vector_proxy.hpp
  @brief Proxy classes for vectors.
  */

#include "viennacl/forwards.h"
#include "viennacl/range.hpp"
#include "viennacl/vector.hpp"

namespace viennacl
{

    template <typename VectorType>
        class vector_range
        {
            typedef vector_range<VectorType>            self_type;

            public:
            typedef typename VectorType::value_type     value_type;
            typedef range::size_type                    size_type;
            typedef range::difference_type              difference_type;
            typedef value_type                          reference;
            typedef const value_type &                  const_reference;

            static const int alignment = VectorType::alignment;

            vector_range(VectorType & v, 
                    range const & entry_range) : v_(v), entry_range_(entry_range) {}

            size_type start() const { return entry_range_.start(); }
            size_type size() const { return entry_range_.size(); }


            /** @brief Operator overload for v1 = A * v2, where v1 and v2 are vector ranges and A is a dense matrix.
             *
             * @param proxy An expression template proxy class
             */
            template <typename MatrixType>
                typename viennacl::enable_if< viennacl::is_matrix<MatrixType>::value, self_type &>::type
                operator=(const vector_expression< const MatrixType,
                        const self_type,
                        op_prod> & proxy);


            // EVAN BOLLIG
            // added support for copying vector_range into vector_range
            self_type & operator=(self_type & vec) 
            {
                assert( this->size() == vec.size() ); 
                if (size() != 0)
                {
                    cl_int err;
                    err = clEnqueueCopyBuffer(viennacl::ocl::get_queue().handle().get(), vec.get().handle().get(), get().handle().get(), 0, 0, sizeof(value_type)*size(), 0, NULL, NULL);
                    //assert(err == CL_SUCCESS);
                    VIENNACL_ERR_CHECK(err);
                }

                //assert( false && "Not implemented!");
                return *this;
            } 

            // EVAN BOLLIG
            // added support for copying vector into vector_range
            self_type & operator=(const VectorType & vec) 
            {
                assert( this->size() == vec.size() ); 
                if (size() != 0)
                {
                    cl_int err;
                    err = clEnqueueCopyBuffer(viennacl::ocl::get_queue().handle().get(), vec.handle().get(), get().handle().get(), 0, 0, sizeof(value_type)*size(), 0, NULL, NULL);
                    //assert(err == CL_SUCCESS);
                    VIENNACL_ERR_CHECK(err);
                }

                //assert( false && "Not implemented!");
                return *this;
            }      

            self_type & operator += (self_type const & other)
            {
                viennacl::linalg::inplace_add(*this, other);
                return *this;
            }

            // EVAN BOLLIG 
            // Added support to += a vector
            self_type & operator += (VectorType const & other)
            {
                viennacl::linalg::inplace_add(*this, other);
                return *this;
            }


            self_type & operator -= (self_type const & other)
            {
                viennacl::linalg::inplace_sub(*this, other);
                return *this;
            }

            // EVAN BOLLIG 
            // Added support to -= a vector
            self_type & operator -= (VectorType const & other)
            {
                viennacl::linalg::inplace_sub(*this, other);
                return *this;
            }

            // Evan Bollig
            // Added support for *=  using routine from vector.hpp
            // Also needed additional template type SCALARTYPE here. 
            // -----------------------------------------------------------------------------
            /** @brief Scales this vector by a CPU scalar value
            */
            template <typename SCALARTYPE>
                self_type & operator *= (SCALARTYPE val)
                {
                    viennacl::linalg::inplace_mult(*this, val);
                    return *this;
                }

            /** @brief Scales this vector by a GPU scalar value
            */
            template <typename SCALARTYPE>
                self_type & operator *= (scalar<SCALARTYPE> const & gpu_val)
                {
                    viennacl::linalg::inplace_mult(*this, gpu_val);
                    return *this;
                }

            /** @brief Scales this vector by a CPU scalar value
            */
            template <typename SCALARTYPE>
                self_type & operator /= (SCALARTYPE val)
                {
                    viennacl::linalg::inplace_mult(*this, static_cast<SCALARTYPE>(1) / val);
                    return *this;
                }

            /** @brief Scales this vector by a CPU scalar value
            */
            template <typename SCALARTYPE>
                self_type & operator /= (scalar<SCALARTYPE> const & gpu_val)
                {
                    viennacl::linalg::inplace_divide(*this, gpu_val);
                    return *this;
                }
            // -----------------------------------------------------------------------------

            //const_reference operator()(size_type i, size_type j) const { return A_(start1() + i, start2() + i); }
            //reference operator()(size_type i, size_type j) { return A_(start1() + i, start2() + i); }

            VectorType & get() { return v_; }
            const VectorType & get() const { return v_; }

            private:
            VectorType & v_;
            range entry_range_;
        };


    template<typename VectorType>
        std::ostream & operator<<(std::ostream & s, vector_range<VectorType> const & proxy)
        {
            typedef typename VectorType::value_type   ScalarType;
            std::vector<ScalarType> temp(proxy.size());
            viennacl::copy(proxy, temp);

            //instead of printing 'temp' directly, let's reuse the existing
            //functionality for viennacl::vector. It certainly adds overhead, but
            //printing a vector is typically not about performance...
            VectorType temp2(temp.size());
            viennacl::copy(temp, temp2);
            s << temp2;
            return s;
        }




    /////////////////////////////////////////////////////////////
    ///////////////////////// CPU to GPU ////////////////////////
    /////////////////////////////////////////////////////////////

    //row_major:
    template <typename VectorType, typename SCALARTYPE>
        void copy(const VectorType & cpu_vector,
                vector_range<vector<SCALARTYPE> > & gpu_vector_range )
        {
            assert(cpu_vector.end() - cpu_vector.begin() >= 0);

            if (cpu_vector.end() - cpu_vector.begin() > 0)
            {
                //we require that the size of the gpu_vector is larger or equal to the cpu-size
                std::vector<SCALARTYPE> temp_buffer(cpu_vector.end() - cpu_vector.begin());
                std::copy(cpu_vector.begin(), cpu_vector.end(), temp_buffer.begin());
                cl_int err = clEnqueueWriteBuffer(viennacl::ocl::get_queue().handle().get(),
                        gpu_vector_range.get().handle().get(), CL_TRUE, sizeof(SCALARTYPE)*gpu_vector_range.start(),
                        sizeof(SCALARTYPE)*temp_buffer.size(),
                        &(temp_buffer[0]), 0, NULL, NULL);
                VIENNACL_ERR_CHECK(err);
            }
        }

    // EVAN BOLLIG:
    // Added support to copy directly from a double array. Ignores vector length checks
    //row_major:
    template <typename SCALARTYPE>
        void copy(const double* cpu_vector,
                vector_range< vector<SCALARTYPE> > & gpu_vector_range, 
                unsigned int copy_size)
        {
            if (copy_size) 
            {
                //we require that the size of the gpu_vector is larger or equal to the cpu-size
                cl_int err = clEnqueueWriteBuffer(viennacl::ocl::get_queue().handle().get(),
                        gpu_vector_range.get().handle().get(), CL_TRUE, sizeof(SCALARTYPE)*gpu_vector_range.start(),
                        sizeof(SCALARTYPE)*copy_size,
                        &(cpu_vector[0]), 0, NULL, NULL);
                VIENNACL_ERR_CHECK(err);
            }
        }


    /////////////////////////////////////////////////////////////
    ///////////////////////// GPU to CPU ////////////////////////
    /////////////////////////////////////////////////////////////


    template <typename VectorType, typename SCALARTYPE>
        void copy(vector_range<vector<SCALARTYPE> > const & gpu_vector_range,
                VectorType & cpu_vector)
        {
            assert(cpu_vector.end() - cpu_vector.begin() >= 0);

            if (cpu_vector.end() > cpu_vector.begin())
            {
                std::vector<SCALARTYPE> temp_buffer(cpu_vector.end() - cpu_vector.begin());
                cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle().get(),
                        gpu_vector_range.get().handle().get(), CL_TRUE, sizeof(SCALARTYPE)*gpu_vector_range.start(), 
                        sizeof(SCALARTYPE)*temp_buffer.size(),
                        &(temp_buffer[0]), 0, NULL, NULL);
                VIENNACL_ERR_CHECK(err);
                viennacl::ocl::get_queue().finish();

                //now copy entries to cpu_vec:
                std::copy(temp_buffer.begin(), temp_buffer.end(), cpu_vector.begin());
            }
        }

    // EVAN BOLLIG:
    // Added support to copy directly to a double array. Ignores vector length checks
    template <typename SCALARTYPE>
        void copy(vector_range<vector<SCALARTYPE> > const & gpu_vector_range,
                double* cpu_vector, unsigned int copy_size)
        {
            // LEt the assertions rest
            // assert(cpu_vector.end() - cpu_vector.begin() >= 0);

            if (copy_size)
            {
                cl_int err = clEnqueueReadBuffer(viennacl::ocl::get_queue().handle().get(),
                        gpu_vector_range.get().handle().get(), CL_TRUE, sizeof(SCALARTYPE)*gpu_vector_range.start(), 
                        sizeof(SCALARTYPE)*copy_size,
                        &(cpu_vector[0]), 0, NULL, NULL);
                VIENNACL_ERR_CHECK(err);
                viennacl::ocl::get_queue().finish();
            }
        }


    ////////// operations /////////////
    // EVAN BOLLIG Added support for oeprator * required for GMRES
    /** @brief Operator overload for the expression alpha * v1, where alpha is a host scalar (float or double) and v1 is a ViennaCL vector.
     *
     * @param value   The host scalar (float or double)
     * @param vec     A ViennaCL vector
     */
    template <typename SCALARTYPE, unsigned int A>
        //    vector_expression< const vector<SCALARTYPE, A>, const SCALARTYPE, op_prod>
        vector<SCALARTYPE, A> operator * (SCALARTYPE const & value, vector_range< vector<SCALARTYPE, A> > const & vec)
        {
            vector<SCALARTYPE, A> temp1 = vec; 
            return vector_expression< const vector<SCALARTYPE, A>, const SCALARTYPE, op_prod>(temp1, value);
        }


}

#endif
