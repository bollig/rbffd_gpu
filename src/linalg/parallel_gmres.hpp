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
                Communicator& comm_ref; 
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

                VectorType r(problem_size);
                VectorType v_k_tilde(problem_size);
                VectorType v_k_tilde_temp(problem_size);

                std::vector< std::vector<CPU_ScalarType> > R(krylov_dim);
                std::vector<CPU_ScalarType> xi_k(krylov_dim);
                std::vector<VectorType> P(krylov_dim);

                const CPU_ScalarType gpu_scalar_minus_1 = static_cast<CPU_ScalarType>(-1);    //representing the scalar '-1' on the GPU. Prevents blocking write operations
                const CPU_ScalarType gpu_scalar_1 = static_cast<CPU_ScalarType>(1);    //representing the scalar '1' on the GPU. Prevents blocking write operations
                const CPU_ScalarType gpu_scalar_2 = static_cast<CPU_ScalarType>(2);    //representing the scalar '2' on the GPU. Prevents blocking write operations

                // TODO: MPI_Reduce on norm_2
                CPU_ScalarType norm_rhs = viennacl::linalg::norm_2(rhs);

                unsigned int k;
                for (k = 0; k < krylov_dim; ++k)
                {
                    R[k].resize(tag.krylov_dim()); 
                    viennacl::traits::resize(P[k], problem_size);
                }

                std::cout << "Starting Parallel GMRES..." << std::endl;
                tag.iters(0);

                for (unsigned int it = 0; it <= tag.max_restarts(); ++it)
                {
                    std::cout << "-- GMRES Start " << it << " -- " << std::endl;

                    r = rhs;
                    //TODO: MPI_Alltoallv on result
                    r -= viennacl::linalg::prod(matrix, result);  //initial guess zero
                    precond.apply(r);
                    //std::cout << "Residual: " << res << std::endl;

                    // TODO: MPI_Reduce on norm_2
                    CPU_ScalarType rho_0 = viennacl::linalg::norm_2(r, tag.comm()); 
                    CPU_ScalarType rho = static_cast<CPU_ScalarType>(1.0);
                    std::cout << "rho_0: " << rho_0 << std::endl;

                    if (rho_0 / norm_rhs < tag.tolerance() || (norm_rhs == CPU_ScalarType(0.0)) )
                    {
                        std::cout << "Allowed Error reached at begin of loop" << std::endl;
                        tag.error(rho_0 / norm_rhs);
                        return result;
                    }

                    r /= rho_0;
                    //std::cout << "Normalized Residual: " << r << std::endl;

                    for (k=0; k<krylov_dim; ++k)
                    {
                        viennacl::traits::clear(R[k]);
                        viennacl::traits::clear(P[k]);
                        R[k].resize(krylov_dim); 
                        viennacl::traits::resize(P[k], problem_size);
                    }

                    for (k = 0; k < krylov_dim; ++k)
                    {
                        tag.iters( tag.iters() + 1 ); //increase iteration counter

                        // -------------------------  Step 1 -----------------------
                        // Evaluate v_k_tilde = P_{k-1} P_{k-2} ... P_{1} A P_{1} P_{2} ... P_{k-1} e_{k-1}  (NOTE: v_1_tilde = A r)
                        // ---------------------------------------------------------
                        //compute v_k = A * v_{k-1} via Householder matrices
                        if (k == 0)
                        {
                            // TODO: MPI_Alltoallv on r
                            //this->alltoallv(r); 
                            v_k_tilde = viennacl::linalg::prod(matrix, r);
                            precond.apply(v_k_tilde);
                        }
                        else
                        {
                            viennacl::traits::clear(v_k_tilde);
                            v_k_tilde[k-1] = gpu_scalar_1;
                            //Householder rotations part 1
                            for (int i = k-1; i > -1; --i)
                                v_k_tilde -= P[i] * (viennacl::linalg::inner_prod(P[i], v_k_tilde) * gpu_scalar_2);

                            // TODO: MPI_Alltoallv on v_k_tilde
                            //this->alltoallv(v_k_tilde); 
                            v_k_tilde_temp = viennacl::linalg::prod(matrix, v_k_tilde);
                            precond.apply(v_k_tilde_temp);
                            v_k_tilde = v_k_tilde_temp;

                            //Householder rotations part 2
                            for (unsigned int i = 0; i < k; ++i)
                                v_k_tilde -= P[i] * (viennacl::linalg::inner_prod(P[i], v_k_tilde) * gpu_scalar_2);
                        }

                        // -------------------------  Step 2 -----------------------
                        // Determine P_{k} such that P_{k} v_k_tilde = (p_{1,k}, p_{2,k}, ... p_{k,k}, 0, ..., 0)^T
                        // ---------------------------------------------------------

                        //std::cout << "v_k_tilde: " << v_k_tilde << std::endl;

                        viennacl::traits::clear(P[k]);
                        viennacl::traits::resize(P[k], problem_size);
                        //copy first k entries from v_k_tilde to P[k]:
                        gmres_copy_helper(v_k_tilde, P[k], k);

                        P[k][k] = std::sqrt( viennacl::linalg::inner_prod(v_k_tilde, v_k_tilde) - viennacl::linalg::inner_prod(P[k], P[k]) );

                        if (fabs(P[k][k]) < CPU_ScalarType(10 * std::numeric_limits<CPU_ScalarType>::epsilon()))
                            break; //Note: Solution is essentially (up to round-off error) already in Krylov space. No need to proceed.

                        // -------------------------  Step 3 -----------------------
                        // Set R_k = ( R_{k-1}  p_{1,k} )
                        //           (            ...   )   with (R_1 = (p_(1,1))
                        //           (   0...0  p_{k,k} )
                        //copy first k+1 entries from P[k] to R[k]
                        gmres_copy_helper(P[k], R[k], k+1);

                        P[k] -= v_k_tilde;
                        //std::cout << "P[k] before normalization: " << P[k] << std::endl;
                        P[k] *= gpu_scalar_minus_1 / viennacl::linalg::norm_2( P[k] );
                        //std::cout << "Householder vector P[k]: " << P[k] << std::endl;


                        // -------------------------  Step 4 -----------------------
                        //
                        //std::cout << "P_k r: " << (r - 2.0 * P[k] * inner_prod(P[k], r)) << std::endl;
                        r -= P[k] * (viennacl::linalg::inner_prod( P[k], r) * gpu_scalar_2);
                        //std::cout << "zeta_k: " << viennacl::linalg::inner_prod( P[k], r) * gpu_scalar_2 << std::endl;
                        //std::cout << "Updated r: " << r << std::endl;

                        if (r[k] > rho) //machine precision reached
                            r[k] = rho;

                        if (r[k] < -1.0 * rho) //machine precision reached
                            r[k] = -1.0 * rho;

                        xi_k[k] = r[k];

                        rho *= std::sin( std::acos(xi_k[k] / rho) );

                        if (std::fabs(rho * rho_0 / norm_rhs) < tag.tolerance())
                        {
                            //std::cout << "Krylov space big enough" << endl;
                            tag.error( std::fabs(rho*rho_0 / norm_rhs) );
                            ++k;
                            std::cout << "--- GMRES converged in " << tag.iters() << " iterations (" << it << " restarts)\n"; 
                            break;
                        }
44
                        //std::cout << "Current residual: " << rho * rho_0 << std::endl;
                        //std::cout << " - End of Krylov space setup - " << std::endl;
                    } // for k

                    // -------------------------  BEGIN SOLVE:  -----------------------
                    // Let k be the final iteration number from Iterate
                    // 1. Solve R_k y = (xi_1, xi_2, ... , xi_k)^T for y = (eta_1, eta_2, ... eta_k)^T
                    for (int i=k-1; i>-1; --i)
                    {
                        for (unsigned int j=i+1; j<k; ++j)
                            //temp_rhs[i] -= R[i][j] * temp_rhs[j];   //if R is not transposed
                            xi_k[i] -= R[j][i] * xi_k[j];     //R is transposed

                        xi_k[i] /= R[i][i];
                    }

                    // 2. Form z = P_1 (eta_1 r)                        if k = 1 
                    //             P_1 P_2 ... P_k [ eta_1 r + (eta_2, ... , eta_k, 0, ..., 0)^T]  if k > 1
                    r *= xi_k[0];

                    if (k > 0)
                    {
                        for (unsigned int i = 0; i < k-1; ++i)
                        {
                            r[i] += xi_k[i+1];
                        }
                    }

                    for (int i = k-1; i > -1; --i)
                        r -= P[i] * (viennacl::linalg::inner_prod(P[i], r) * gpu_scalar_2);

                    r *= rho_0;
                    result += r;

                    if ( std::fabs(rho*rho_0 / norm_rhs) < tag.tolerance() )
                    {
                        //std::cout << "Allowed Error reached at end of loop" << std::endl;
                        tag.error(std::fabs(rho*rho_0 / norm_rhs));
                        return result;
                    }

                    //r = rhs;
                    //r -= viennacl::linalg::prod(matrix, result);
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
