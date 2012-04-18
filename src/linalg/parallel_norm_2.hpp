#ifndef VIENNACL_LINALG_PARALLEL_NORM_2_HPP_
#define VIENNACL_LINALG_PARALLEL_NORM_2_HPP_

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

/** @file norm_2.hpp
    @brief Generic interface for the l^2-norm. See viennacl/linalg/vector_operations.hpp for implementations.
*/

#include <math.h>    //for sqrt()
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/tag_of.hpp"

#include "viennacl/linalg/norm_2.hpp"
#include "utils/comm/communicator.h"

namespace viennacl
{
  //
  // generic norm_2 function
  //   uses tag dispatch to identify which algorithm
  //   should be called 
  //
  namespace linalg 
  {

#if 0
    // Generic norm_2 should work for all types.
    // Might require overloading for different input types (i.e., if we compute
    // norm of something other than a double or float array
    template <typename VectorType, typename ScalarType=double>
    ScalarType norm_2(VectorType< ScalarType > &v, Communicator const& comm_unit) 
    {
        ScalarType norm = viennacl::linalg::norm_2(v);
        norm*=norm; 
        ScalarType r; 
        MPI_Allreduce(&norm, &r, 1, MPI_DOUBLE, MPI_SUM, comm_unit.getComm()); 
        return r; 
    }
#endif 

    // Generic for GPU (compute norm on gpu, transfer scalar to cpu, reduce and return to the gpu)
    template <typename VectorType>
    typename VectorType::value_type
    parallel_norm_2(VectorType const& v, Communicator const& comm_unit, typename VectorType::value_type& dummy) 
    {
        // Force the GPU norm to the cpu. 
        double norm = viennacl::linalg::norm_2(v);
        norm *= norm;
        double r[1]; 
        
        MPI_Allreduce(&norm, r, 1, MPI_DOUBLE, MPI_SUM, comm_unit.getComm()); 
       
        return (typename VectorType::value_type)sqrt(r[0]); 
    }


    template <typename VectorType>
    //typename VectorType::value_type
    double
    parallel_norm_2(VectorType const& v, Communicator const& comm_unit, double& dummy) 
    {
        // MPI will segfault if we dont declare norm as an array type
        //typename VectorType::value_type norm = viennacl::linalg::norm_2(v);
        double norm = viennacl::linalg::norm_2(v);
        norm *= norm;
        double r[1]; 
        
        MPI_Allreduce(&norm, r, 1, MPI_DOUBLE, MPI_SUM, comm_unit.getComm()); 
       
        return sqrt(r[0]); 
    }

    template <typename VectorType>
    float 
    parallel_norm_2(VectorType const& v, Communicator const& comm_unit, float& dummy) 
    {
        float norm = viennacl::linalg::norm_2(v);
        norm *= norm;
        float r[1]; 
        
        MPI_Allreduce(&norm, r, 1, MPI_FLOAT, MPI_SUM, comm_unit.getComm()); 
       
        return sqrt(r[0]); 
    }

    template <typename VectorType>
    typename VectorType::value_type
    norm_2(VectorType const& v, Communicator const& comm_unit) 
    {
        // Redirect the call to appropriate float or double routines (Type size matters with MPI)
        typename VectorType::value_type dummy; 
        return parallel_norm_2(v, comm_unit, dummy ); 
    }
#if 0
    template <typename VectorType, typename ScalarType>
    ScalarType norm_2(VectorType< ScalarType > &v, Communicator& comm_unit) 
    {
        ScalarType norm = viennacl::linalg::norm_2(v);
        norm*=norm; 
        ScalarType r; 
        MPI_Allreduce(&norm, &r, 1, MPI_FLOAT, MPI_SUM, comm_unit.getComm()); 
        return r; 
    }
#endif 
    
#if 0
    // ----------------------------------------------------
    // STL
    //
    template< typename VectorT>
    typename VectorT::value_type
    norm_2(VectorT const& v1,
         typename viennacl::enable_if< viennacl::is_stl< typename viennacl::traits::tag_of< VectorT >::type >::value
                                     >::type* dummy = 0)
    {
      //std::cout << "stl .. " << std::endl;
      typename VectorT::value_type result = 0;
      for (typename VectorT::size_type i=0; i<v1.size(); ++i)
        result += v1[i] * v1[i];
      
      return sqrt(result);
    }
#endif 
  } // end namespace linalg
} // end namespace viennacl
#endif





