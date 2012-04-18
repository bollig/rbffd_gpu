// Evan Bollig 2012
//
// Performs GMRES in parallel using an assumed additive schwarz decomposition. 
// The comm_unit provides details of the MPI environment (rank, size, communicator, etc.) 
// The domain provides the details of additive schwarz decomposition (implicit restriction operator)
//      - Implicit because we work directly on solution nodes, not on the differentiation matrix.
//      - if R is a restriction operator (eye only where stencils are part of
//      subdomain), then we have 
//          \sum_{p=1}^{nproc}(R_p' A R_p)u = \sum_{p=1}^{nproc}(R_p'R_p)F = Au = F. 
//      - For now we assume that R can be constructed with domain->Q; in the
//      future we might generalize this so the code is not specific to my
//      decomposition class

#ifndef VIENNACL_PARALLEL_GMRES_HPP_
#define VIENNACL_PARALLEL_GMRES_HPP_

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

/** @file parallel_gmres.hpp
  @brief Implementations of the generalized minimum residual method are in this file.
  */

#include <vector>
#include <cmath>
#include <limits>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/traits/clear.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/meta/result_of.hpp"

// We'll extend the original gmres class: 
#include "viennacl/linalg/gmres.hpp"

// But we need our own parallel norms
#include "linalg/parallel_norm_2.hpp"

#include "utils/comm/communicator.h"
#include "grids/domain.h"

namespace viennacl
{
    namespace linalg
    {

        /** @brief A tag for the solver GMRES. Used for supplying solver parameters and for dispatching the solve() function
        */
        class parallel_gmres_tag : public gmres_tag      //generalized minimum residual
        {
            public:
                /** @brief The constructor
                 *
                 * @param tol            Relative tolerance for the residual (solver quits if ||r|| < tol * ||r_initial||)
                 * @param max_iterations The maximum number of iterations (including restarts
                 * @param krylov_dim     The maximum dimension of the Krylov space before restart (number of restarts is found by max_iterations / krylov_dim)
                 */
                parallel_gmres_tag(Communicator& comm_unit, Domain& decomposition, double tol = 1e-10, unsigned int max_iterations = 300, unsigned int krylov_dim = 20) 
                    : gmres_tag(tol, max_iterations, krylov_dim), 
                    comm_ref(comm_unit), domain_ref(decomposition)
            {};

                Communicator const& comm() const { return this->comm_ref; } 

            protected: 
                mutable Communicator& comm_ref; 
                Domain& domain_ref; 
        };

        /** @brief Implementation of the GMRES solver.
         *
         * Following the algorithm proposed by Walker in "A Simpler GMRES" (1988)
         *
         * @param matrix     The system matrix
         * @param rhs        The load vector
         * @param tag        Solver configuration tag
         * @param precond    A preconditioner. Precondition operation is done via member function apply()
         * @return The result vector
         */
        template <typename MatrixType, typename VectorType, typename PreconditionerType>
            VectorType solve(const MatrixType & matrix, VectorType const & rhs, parallel_gmres_tag const & tag, PreconditionerType const & precond)
            {
                typedef typename viennacl::result_of::value_type<VectorType>::type        ScalarType;
                typedef typename viennacl::result_of::cpu_value_type<ScalarType>::type    CPU_ScalarType;
                unsigned int problem_size = viennacl::traits::size(rhs);
                // TODO: allow user specified initial guess
                // TODO: when allowed MPI_Alltoallv sync here
                VectorType result(problem_size);
                viennacl::traits::clear(result);
                unsigned int krylov_dim = tag.krylov_dim();
                if (problem_size < tag.krylov_dim())
                    krylov_dim = problem_size; //A Krylov space larger than the matrix would lead to seg-faults (mathematically, error is certain to be zero already)

                VectorType res(problem_size);
                VectorType v_k_tilde(problem_size);
                VectorType v_k_tilde_temp(problem_size);

                std::vector< std::vector<CPU_ScalarType> > R(krylov_dim);
                std::vector<CPU_ScalarType> projection_rhs(krylov_dim);
                std::vector<VectorType> U(krylov_dim);

                const CPU_ScalarType gpu_scalar_minus_1 = static_cast<CPU_ScalarType>(-1);    //representing the scalar '-1' on the GPU. Prevents blocking write operations
                const CPU_ScalarType gpu_scalar_1 = static_cast<CPU_ScalarType>(1);    //representing the scalar '1' on the GPU. Prevents blocking write operations
                const CPU_ScalarType gpu_scalar_2 = static_cast<CPU_ScalarType>(2);    //representing the scalar '2' on the GPU. Prevents blocking write operations

                // TODO: MPI_Reduce on norm_2
                CPU_ScalarType norm_rhs = viennacl::linalg::norm_2(rhs);

                unsigned int k;
                for (k = 0; k < krylov_dim; ++k)
                {
                    R[k].resize(tag.krylov_dim()); 
                    viennacl::traits::resize(U[k], problem_size);
                }

                std::cout << "Starting Parallel GMRES..." << std::endl;
                tag.iters(0);

                for (unsigned int it = 0; it <= tag.max_restarts(); ++it)
                {
                    std::cout << "-- GMRES Start " << it << " -- " << std::endl;

                    res = rhs;
                    //TODO: MPI_Alltoallv on result
                    res -= viennacl::linalg::prod(matrix, result);  //initial guess zero
                    precond.apply(res);
                    //std::cout << "Residual: " << res << std::endl;

                    // TODO: MPI_Reduce on norm_2
                    CPU_ScalarType rho_0 = viennacl::linalg::norm_2(res, tag.comm()); 
                    CPU_ScalarType rho = static_cast<CPU_ScalarType>(1.0);
                    std::cout << "rho_0: " << rho_0 << std::endl;

                    if (rho_0 / norm_rhs < tag.tolerance() || (norm_rhs == CPU_ScalarType(0.0)) )
                    {
                        std::cout << "Allowed Error reached at begin of loop" << std::endl;
                        tag.error(rho_0 / norm_rhs);
                        return result;
                    }

                    res /= rho_0;
                    //std::cout << "Normalized Residual: " << res << std::endl;

                    for (k=0; k<krylov_dim; ++k)
                    {
                        viennacl::traits::clear(R[k]);
                        viennacl::traits::clear(U[k]);
                        R[k].resize(krylov_dim); 
                        viennacl::traits::resize(U[k], problem_size);
                    }

                    for (k = 0; k < krylov_dim; ++k)
                    {
                        tag.iters( tag.iters() + 1 ); //increase iteration counter

                        //compute v_k = A * v_{k-1} via Householder matrices
                        if (k == 0)
                        {
                            // TODO: MPI_Alltoallv on res
                            //this->alltoallv(res); 
                            v_k_tilde = viennacl::linalg::prod(matrix, res);
                            precond.apply(v_k_tilde);
                        }
                        else
                        {
                            viennacl::traits::clear(v_k_tilde);
                            v_k_tilde[k-1] = gpu_scalar_1;
                            //Householder rotations part 1
                            for (int i = k-1; i > -1; --i)
                                v_k_tilde -= U[i] * (viennacl::linalg::inner_prod(U[i], v_k_tilde) * gpu_scalar_2);

                            // TODO: MPI_Alltoallv on v_k_tilde
                            //this->alltoallv(v_k_tilde); 
                            v_k_tilde_temp = viennacl::linalg::prod(matrix, v_k_tilde);
                            precond.apply(v_k_tilde_temp);
                            v_k_tilde = v_k_tilde_temp;

                            //Householder rotations part 2
                            for (unsigned int i = 0; i < k; ++i)
                                v_k_tilde -= U[i] * (viennacl::linalg::inner_prod(U[i], v_k_tilde) * gpu_scalar_2);
                        }

                        //std::cout << "v_k_tilde: " << v_k_tilde << std::endl;

                        viennacl::traits::clear(U[k]);
                        viennacl::traits::resize(U[k], problem_size);
                        //copy first k entries from v_k_tilde to U[k]:
                        gmres_copy_helper(v_k_tilde, U[k], k);

                        U[k][k] = std::sqrt( viennacl::linalg::inner_prod(v_k_tilde, v_k_tilde) - viennacl::linalg::inner_prod(U[k], U[k]) );

                        if (fabs(U[k][k]) < CPU_ScalarType(10 * std::numeric_limits<CPU_ScalarType>::epsilon()))
                            break; //Note: Solution is essentially (up to round-off error) already in Krylov space. No need to proceed.

                        //copy first k+1 entries from U[k] to R[k]
                        gmres_copy_helper(U[k], R[k], k+1);

                        U[k] -= v_k_tilde;
                        //std::cout << "U[k] before normalization: " << U[k] << std::endl;
                        U[k] *= gpu_scalar_minus_1 / viennacl::linalg::norm_2( U[k] );
                        //std::cout << "Householder vector U[k]: " << U[k] << std::endl;

                        //DEBUG: Make sure that P_k v_k_tilde equals (rho_{1,k}, ... , rho_{k,k}, 0, 0 )
#ifdef VIENNACL_GMRES_DEBUG
                        std::cout << "P_k v_k_tilde: " << (v_k_tilde - 2.0 * U[k] * inner_prod(U[k], v_k_tilde)) << std::endl;
                        std::cout << "R[k]: [" << R[k].size() << "](";
                        for (size_t i=0; i<R[k].size(); ++i)
                            std::cout << R[k][i] << ",";
                        std::cout << ")" << std::endl;
#endif
                        //std::cout << "P_k res: " << (res - 2.0 * U[k] * inner_prod(U[k], res)) << std::endl;
                        res -= U[k] * (viennacl::linalg::inner_prod( U[k], res ) * gpu_scalar_2);
                        //std::cout << "zeta_k: " << viennacl::linalg::inner_prod( U[k], res ) * gpu_scalar_2 << std::endl;
                        //std::cout << "Updated res: " << res << std::endl;

#ifdef VIENNACL_GMRES_DEBUG
                        VectorType v1(U[k].size()); v1.clear(); v1.resize(U[k].size());
                        v1(0) = 1.0;
                        v1 -= U[k] * (viennacl::linalg::inner_prod( U[k], v1 ) * gpu_scalar_2);
                        std::cout << "v1: " << v1 << std::endl;
                        boost::numeric::ublas::matrix<ScalarType> P = -2.0 * outer_prod(U[k], U[k]);
                        P(0,0) += 1.0; P(1,1) += 1.0; P(2,2) += 1.0;
                        std::cout << "P: " << P << std::endl;
#endif

                        if (res[k] > rho) //machine precision reached
                            res[k] = rho;

                        if (res[k] < -1.0 * rho) //machine precision reached
                            res[k] = -1.0 * rho;

                        projection_rhs[k] = res[k];

                        rho *= std::sin( std::acos(projection_rhs[k] / rho) );

#ifdef VIENNACL_GMRES_DEBUG
                        std::cout << "k-th component of r: " << res[k] << std::endl;
                        std::cout << "New rho (norm of res): " << rho << std::endl;
#endif        

                        if (std::fabs(rho * rho_0 / norm_rhs) < tag.tolerance())
                        {
                            //std::cout << "Krylov space big enough" << endl;
                            tag.error( std::fabs(rho*rho_0 / norm_rhs) );
                            ++k;
                            std::cout << "--- GMRES converged in " << tag.iters() << " iterations (" << it << " restarts)\n"; 
                            break;
                        }

                        //std::cout << "Current residual: " << rho * rho_0 << std::endl;
                        //std::cout << " - End of Krylov space setup - " << std::endl;
                    } // for k

#ifdef VIENNACL_GMRES_DEBUG
                    //inplace solution of the upper triangular matrix:
                    std::cout << "Upper triangular system:" << std::endl;
                    std::cout << "Size of Krylov space: " << k << std::endl;
                    for (size_t i=0; i<k; ++i)
                    {
                        for (size_t j=0; j<k; ++j)
                        {
                            std::cout << R[j][i] << ", ";
                        }
                        std::cout << " | " << projection_rhs[i] << std::endl;
                    }
#endif        

                    for (int i=k-1; i>-1; --i)
                    {
                        for (unsigned int j=i+1; j<k; ++j)
                            //temp_rhs[i] -= R[i][j] * temp_rhs[j];   //if R is not transposed
                            projection_rhs[i] -= R[j][i] * projection_rhs[j];     //R is transposed

                        projection_rhs[i] /= R[i][i];
                    }

#ifdef VIENNACL_GMRES_DEBUG
                    std::cout << "Result of triangular solver: ";
                    for (size_t i=0; i<k; ++i)
                        std::cout << projection_rhs[i] << ", ";
                    std::cout << std::endl;
#endif        
                    res *= projection_rhs[0];

                    if (k > 0)
                    {
                        for (unsigned int i = 0; i < k-1; ++i)
                        {
                            res[i] += projection_rhs[i+1];
                        }
                    }

                    for (int i = k-1; i > -1; --i)
                        res -= U[i] * (viennacl::linalg::inner_prod(U[i], res) * gpu_scalar_2);

                    res *= rho_0;
                    result += res;

                    if ( std::fabs(rho*rho_0 / norm_rhs) < tag.tolerance() )
                    {
                        //std::cout << "Allowed Error reached at end of loop" << std::endl;
                        tag.error(std::fabs(rho*rho_0 / norm_rhs));
                        return result;
                    }

                    //res = rhs;
                    //res -= viennacl::linalg::prod(matrix, result);
                    //std::cout << "norm_2(r)=" << norm_2(r) << std::endl;
                    //std::cout << "std::abs(rho*rho_0)=" << std::abs(rho*rho_0) << std::endl;
                    //std::cout << r << std::endl; 

                    tag.error(std::fabs(rho*rho_0));
                }

                return result;
            }

        /** @brief Convenience overload of the solve() function using GMRES. Per default, no preconditioner is used
        */ 
        template <typename MatrixType, typename VectorType>
            VectorType solve(const MatrixType & matrix, VectorType const & rhs, parallel_gmres_tag const & tag)
            {
                return solve(matrix, rhs, tag, no_precond());
            }


    }
}

#endif
