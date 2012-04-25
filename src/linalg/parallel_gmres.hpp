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
#include "linalg/parallel_inner_prod.hpp"

#include "utils/comm/communicator.h"
#include "grids/domain.h"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/filesystem.hpp>

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
                parallel_gmres_tag(Communicator& comm_unit, Domain& decomposition, double tol = 1e-10, unsigned int max_iterations = 300, unsigned int krylov_dim = 20, unsigned int solution_dim_per_node = 1) 
                    : gmres_tag(tol, max_iterations, krylov_dim), 
                    comm_ref(comm_unit), grid_ref(decomposition), 
                    sol_dim(solution_dim_per_node)
            {
                allocateCommBuffers();            
            };

                ~parallel_gmres_tag() {
                    delete [] sbuf; 
                    delete [] sendcounts; 
                    delete [] sdispls; 
                    delete [] rdispls; 
                    delete [] recvcounts; 
                    delete [] rbuf; 
                }

                Communicator const& comm() const { return this->comm_ref; } 

                void allocateCommBuffers() {
#if 0
                    double* sbuf; 
                    int* sendcounts; 
                    int* sdispls; 
                    int* rdispls; 
                    int* recvcounts; 
                    double* rbuf; 
#endif 
                    this->sdispls = new int[sol_dim*grid_ref.O_by_rank.size()]; 
                    this->sendcounts = new int[sol_dim*grid_ref.O_by_rank.size()]; 

                    unsigned int O_tot = sol_dim*grid_ref.O_by_rank[0].size(); 
                    sdispls[0] = 0;
                    sendcounts[0] = sol_dim*grid_ref.O_by_rank[0].size(); 
                    for (size_t i = 1; i < grid_ref.O_by_rank.size(); i++) {
                        sdispls[i] = sdispls[i-1] + sendcounts[i-1];
                        sendcounts[i] = sol_dim*grid_ref.O_by_rank[i].size(); 
                        O_tot += sendcounts[i]; 
                    }
                    this->rdispls = new int[sol_dim*grid_ref.R_by_rank.size()]; 
                    this->recvcounts = new int[sol_dim*grid_ref.R_by_rank.size()]; 

                    unsigned int R_tot = sol_dim*grid_ref.R_by_rank[0].size(); 
                    rdispls[0] = 0; 
                    recvcounts[0] = sol_dim*grid_ref.R_by_rank[0].size(); 
                    for (size_t i = 1; i < grid_ref.R_by_rank.size(); i++) {
                        recvcounts[i] = sol_dim*grid_ref.R_by_rank[i].size(); 
                        rdispls[i] = rdispls[i-1] + recvcounts[i-1];   
                        R_tot += recvcounts[i]; 
                    }

                    // Not sure if we need to use malloc to guarantee contiguous?
                    this->sbuf = new double[O_tot];  
                    this->rbuf = new double[R_tot];
                }


                // Generic for GPU (transfer GPU subset to cpu buffer, alltoallv
                // and return to the gpu)
                template <typename VectorType>
                    void
                    alltoall_subset(VectorType& vec, typename VectorType::value_type& dummy) const
                    {

                        // Share data in vector with all other processors. Will only transfer
                        // data associated with nodes in overlap between domains. 
                        // Uses MPI_Alltoallv and MPI_Barrier. 
                        // Copies data from vec to transfer, then writes received data into vec
                        // before returning. 
                        if (comm_ref.getSize() > 1) {

                            std::cout << "TRANSFER " << grid_ref.O_by_rank.size() << "\n";
                            // Copy elements of set to sbuf
                            unsigned int k = 0; 
                            for (size_t i = 0; i < grid_ref.O_by_rank.size(); i++) {
                                k = this->sdispls[i]; 
                                for (size_t j = 0; j < grid_ref.O_by_rank[i].size(); j++) {
                                    // this->sbuf[k] = grid_ref.O_by_rank[i][j];
                                    this->sbuf[k] = vec[grid_ref.g2l(grid_ref.O_by_rank[i][j])]; 
                                    k++; 
                                }
                            }

                            MPI_Alltoallv(this->sbuf, this->sendcounts, this->sdispls, MPI_DOUBLE, this->rbuf, this->recvcounts, this->rdispls, MPI_DOUBLE, comm_ref.getComm()); 

                            comm_ref.barrier();
                            k = 0; 
                            for (size_t i = 0; i < grid_ref.R_by_rank.size(); i++) {
                                k = this->rdispls[i]; 
                                for (size_t j = 0; j < grid_ref.R_by_rank[i].size(); j++) {
                                    // TODO: need to translate to local indexing
                                    vec[grid_ref.g2l(grid_ref.R_by_rank[i][j])] = this->rbuf[k];  
                                    k++; 
                                }
                            }
                        }
                    }


                // For CPU (copy subset to cpu buffer, alltoallv
                // and return to the gpu)
                template <typename VectorType>
                    void
                    alltoall_subset(VectorType& vec, double& dummy) const
                    {

                        // Share data in vector with all other processors. Will only transfer
                        // data associated with nodes in overlap between domains. 
                        // Uses MPI_Alltoallv and MPI_Barrier. 
                        // Copies data from vec to transfer, then writes received data into vec
                        // before returning. 
                        if (comm_ref.getSize() > 1) {

                            // Copy elements of set to sbuf
                            unsigned int k = 0; 
                            for (size_t i = 0; i < grid_ref.O_by_rank.size(); i++) {
                                k = this->sdispls[i]; 
                                for (size_t j = 0; j < grid_ref.O_by_rank[i].size(); j++) {
                                    unsigned int s_indx = grid_ref.g2l(grid_ref.O_by_rank[i][j]) - grid_ref.getBoundaryIndicesSize();
                                    std::cout << "Sending " << s_indx << "\n";
                                    this->sbuf[k] = vec[s_indx];
                                    k++; 
                                }
                            }

                            MPI_Alltoallv(this->sbuf, this->sendcounts, this->sdispls, MPI_DOUBLE, this->rbuf, this->recvcounts, this->rdispls, MPI_DOUBLE, comm_ref.getComm()); 

                            comm_ref.barrier();
                            k = 0; 
                            for (size_t i = 0; i < grid_ref.R_by_rank.size(); i++) {
                                k = this->rdispls[i]; 
                                for (size_t j = 0; j < grid_ref.R_by_rank[i].size(); j++) {
                                    unsigned int r_indx = grid_ref.g2l(grid_ref.R_by_rank[i][j]) - grid_ref.getBoundaryIndicesSize();
                                    std::cout << "Receiving " << r_indx << "\n";
                                    // TODO: need to translate to local indexing properly. This hack assumes all boundary are dirichlet and appear first in the list
                                    vec[r_indx] = this->rbuf[k];  
                                    k++; 
                                }
                            }
                        }
                    }

                template <typename VectorType>
                    void 
                    alltoall_subset(VectorType& v) const
                    {
                        typename VectorType::value_type dummy; 
                        alltoall_subset(v, dummy);
                    }




            protected: 
                Communicator& comm_ref; 
                Domain& grid_ref; 
                unsigned int sol_dim; 
                mutable double* sbuf; 
                mutable int* sendcounts; 
                mutable int* sdispls; 
                mutable int* rdispls; 
                mutable int* recvcounts; 
                mutable double* rbuf; 
        };


        namespace ublas = boost::numeric::ublas;

        /** @brief Implementation of the GMRES solver.
         *
         * Following the algorithm 2.1 proposed by Walker in "A Simpler GMRES" (1988)
         * (Evan Bollig): I changed variable names to be consistent with the paper and other literature
         *
         * @param matrix     The system matrix
         * @param rhs        The load vector
         * @param tag        Solver configuration tag
         * @param precond    A preconditioner. Precondition operation is done via member function apply()
         * @return The result vector
         */
        template <typename MatrixType, /*typename VectorType=ublas::vector<double>,*/ typename PreconditionerType>
                          ublas::vector<double> solve(const MatrixType & matrix, ublas::vector<double> const & b, parallel_gmres_tag const & tag, PreconditionerType const & precond)
        {
            typedef ublas::vector<double> VectorType;
            // RHS is size(M,1)
            typedef typename viennacl::result_of::value_type<VectorType>::type        ScalarType;
            typedef typename viennacl::result_of::cpu_value_type<ScalarType>::type    CPU_ScalarType;
            // In parallel these dimensions are VERY IMPORTANT to get right
            //unsigned int problem_size = viennacl::traits::size(b);
            unsigned int NN = matrix.size1(); //viennacl::traits::size1(matrix);
            unsigned int MM = matrix.size2(); //viennacl::traits::size2(matrix);
            // TODO: allow user specified initial guess
            // TODO: when allowed MPI_Alltoallv sync here
            VectorType x(MM);

            viennacl::traits::clear(x);

            unsigned int krylov_dim = tag.krylov_dim();

            VectorType r_full(MM);
            ublas::vector_range<VectorType> r(r_full, ublas::range(0,NN)); 

            // The step directions
            std::vector< VectorType > v_full(krylov_dim+1); 
            std::vector< ublas::vector_range<VectorType> * > v(krylov_dim+1);

            // The set of Givens rotations
            std::vector<double> g(krylov_dim+1);
            // Hessenberg matrix (if we do the givens rotations properly this ends as upper triangular)
            std::vector< std::vector<CPU_ScalarType> > H(krylov_dim+1);

            //representing the scalar '-1' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_minus_1 = static_cast<CPU_ScalarType>(-1);    
            //representing the scalar '1' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_1 = static_cast<CPU_ScalarType>(1);    
            //representing the scalar '2' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_2 = static_cast<CPU_ScalarType>(2);    

            // DONE: MPI_Reduce on norm_2 (note: we only apply norm_2 to the
            // subset controlled by this machine)
            CPU_ScalarType b_norm = viennacl::linalg::norm_2(b, tag.comm());

            for (unsigned int k = 0; k < krylov_dim+1; ++k)
            {
                H[k].resize(tag.krylov_dim()+1); 
                v_full[k].resize(MM);
                v[k] = new ublas::vector_range<VectorType>(v_full[k], ublas::range(0,NN)); 
            }

            MPI_Barrier(MPI_COMM_WORLD);

            //                if (tag.comm().isMaster()) 
            std::cout << "Starting Parallel GMRES..." << std::endl;
            tag.iters(0);

            for (unsigned int it = 0; it <= tag.max_restarts(); ++it)
            {
                //                    if (tag.comm().isMaster()) 
                std::cout << "-- GMRES Start " << it << " -- " << std::endl;

                // Synchronize x(0) 
                tag.alltoall_subset(x); 

                // r = b - A x
                r = b - viennacl::linalg::prod(matrix, x);  //initial guess zero

                // DONE: MPI_Alltoallv before preconditioning
                //tag.alltoall_subset(r_full); 
                //                    precond.apply(r_full);
                //std::cout << "Residual: " << r << std::endl;

                // DONE: MPI_Reduce on norm_2
                // beta = norm(V(0)))
                CPU_ScalarType rho_0 = static_cast<CPU_ScalarType>(viennacl::linalg::norm_2(r, tag.comm())); 
                CPU_ScalarType rho = static_cast<CPU_ScalarType>(1.0);
                //                    if (tag.comm().isMaster()) 
                std::cout << "rho_0: " << rho_0 << std::endl;

                if (rho_0 / b_norm < tag.tolerance() || (b_norm == CPU_ScalarType(0.0)) )
                {
                    //                        if (tag.comm().isMaster()) 
                    std::cout << "Allowed Error reached at begin of loop" << std::endl;
                    tag.error(rho_0 / b_norm);
                    return x;
                }

                // v_1 = r / rho_0
                *(v[0]) = r / rho_0;


                // Should be 1: 
                // std::cout << "norm_2(v0) = " << viennacl::linalg::norm_2(*(v[0]),tag.comm()) << std::endl;

                // First givens rotation 
                g[0] = rho_0; 
                //std::cout << "Normalized Residual: " << res << std::endl;

                for (unsigned int k = 0; k < krylov_dim; ++k)
                {
                    tag.iters( tag.iters() + 1 ); //increase iteration counter


                    // Synchronize v_i with Alltoallv collective
                    tag.alltoall_subset(v_full[k]); 

                    // v_k+1 = A v_k
                    *(v[k+1]) = viennacl::linalg::prod(matrix, v_full[k]);
                    //precond.apply(v_k_tilde);

                    // Begin modified Gram-Schmidt (may require reorthogonalization)
                    for (int j = 0; j < k; j++) { 
                        H[j][k] = viennacl::linalg::inner_prod(*(v[j]), *(v[k+1]), tag.comm());
                        *(v[k+1]) -= H[j][k] * *(v[j]); 
                    }

                    H[k+1][k] = viennacl::linalg::norm_2(*(v[k+1]), tag.comm());

                    // Safety check
                    if ((H[k+1][k] > 0.) || (H[k+1][k] < 0.)) {
                        std::cout << "SAFETY FIRST\n";
                        *(v[k+1]) /= H[k+1][k];
                    } else { 
                        std::cout << "MADE IT\n";
                    }

#if 0
                    // -------------------------  Step 1 ----------------------
                    // Evaluate v_k_tilde = P_{k-1} P_{k-2} ... P_{1} A P_{1} P_{2} ... P_{k-1} e_{k-1}  (NOTE: v_1_tilde = A r)
                    // ---------------------------------------------------------
                    //compute v_k = A * v_{k-1} via Householder matrices
                    if (k == 0)
                    {

                    }
                    else
                    {
                        // Below the U[i] is H(k,i) in Sosonkina et al. and P_j in Walker and Zhou 
                        //
                        // V(i+1) = A*w = M*A*V(i)
                        // H(k,i) = <V(i+1, V(k)>  (need a reduce here)
                        // V(i+1) -= H(k,i) * V(k) 

                        // Need to parallelize here: 
                        // i goes from k down to 1. 
                        // if we have two processors one will go from k
                        // down to k/2 the other will go k/2 to 1 so we need to
                        viennacl::traits::clear(v_k_tilde_full);
                        v_k_tilde_full[k-1] = gpu_scalar_1;
                        //Householder rotations part 1
                        for (int j = k-1; j > -1; --j)
                            v_k_tilde_full -= P[j] * (viennacl::linalg::inner_prod(P[j], v_k_tilde_full) * gpu_scalar_2);

                        // TODO: MPI_Alltoallv on v_k_tilde_full
                        tag.alltoall_subset(v_k_tilde_full); 
                        v_k_tilde_temp = viennacl::linalg::prod(matrix, v_k_tilde_full);
                        tag.alltoall_subset(v_k_tilde_temp_full); 
                        //precond.apply(v_k_tilde_temp_full);
                        v_k_tilde_full = v_k_tilde_temp_full;

                        //Householder rotations part 2
                        for (unsigned int j = 0; j < k; ++j)
                            v_k_tilde_full -= P[j] * (viennacl::linalg::inner_prod(P[j], v_k_tilde_full) * gpu_scalar_2);
                    }

                    viennacl::traits::clear(P[k]);
                    viennacl::traits::resize(P[k], MM);
                    //copy first k entries from v_k_tilde to P[k]:

                    // -------------------------  Step 2 -----------------------
                    // Determine P_{k} such that P_{k} v_k_tilde = (p_{1,k}, p_{2,k}, ... p_{k,k}, 0, ..., 0)^T
                    // ---------------------------------------------------------

                    gmres_copy_helper(v_k_tilde_full, P[k], k);

                    P[k][k] = std::sqrt( viennacl::linalg::inner_prod(v_k_tilde_full, v_k_tilde_full) - viennacl::linalg::inner_prod(P[k], P[k]) );

                    if (fabs(P[k][k]) < CPU_ScalarType(10 * std::numeric_limits<CPU_ScalarType>::epsilon()))
                        break; //Note: Solution is essentially (up to round-off error) already in Krylov space. No need to proceed.


                    // -------------------------  Step 3 -----------------------
                    // Set R_k = ( R_{k-1}  p_{1,k} )
                    //           (            ...   )   with (R_1 = (p_(1,1))
                    //           (   0...0  p_{k,k} )
                    //copy first k+1 entries from P[k] to R[k]
                    gmres_copy_helper(P[k], R[k], k+1);

                    P[k] -= v_k_tilde_full;
                    //std::cout << "P[k] before normalization: " << P[k] << std::endl;
                    P[k] *= gpu_scalar_minus_1 / viennacl::linalg::norm_2( P[k] , tag.comm() );
                    //std::cout << "Householder vector P[k]: " << P[k] << std::endl;

                    //DEBUG: Make sure that P_k v_k_tilde_full equals (rho_{1,k}, ... , rho_{k,k}, 0, 0 )
                    //std::cout << "P_k res: " << (r - 2.0 * P[k] * inner_prod(P[k], r)) << std::endl;
                    r_full -= P[k] * (viennacl::linalg::inner_prod( P[k], r_full ) * gpu_scalar_2);
                    //std::cout << "zeta_k: " << viennacl::linalg::inner_prod( P[k], r ) * gpu_scalar_2 << std::endl;
                    //std::cout << "Updated r: " << r << std::endl;

                    if (r_full[k] > rho) //machine precision reached
                        r_full[k] = rho;

                    if (r_full[k] < -1.0 * rho) //machine precision reached
                        r_full[k] = -1.0 * rho;

                    xi_k[k] = r_full[k];

                    rho *= std::sin( std::acos(xi_k[k] / rho) );
                    if (std::fabs(rho * rho_0 / b_norm ) < tag.tolerance())
                    {
                        //std::cout << "Krylov space big enough" << endl;
                        tag.error( std::fabs(rho*rho_0 / b_norm ) );
                        ++k;
                        //                            if (tag.comm().isMaster()) 
                        std::cout << "--- GMRES converged in " << tag.iters() << " iterations (" << it << " restarts)\n"; 
                        break;
                    }
                    //std::cout << "Current residual: " << rho * rho_0 << std::endl;
                    //std::cout << " - End of Krylov space setup - " << std::endl;
#endif 
                } // for k
 std::cout << " H = \n" ;
                    for (int i = 0; i < krylov_dim+1; i++ ) { 
                        for (int l = 0; l < krylov_dim+1; l++ ) { 
                            std::cout << H[i][l] << " "; 
                        }
                        std::cout << "\n";
                    }


                exit(-1);
#if 0
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
                r_full *= xi_k[0];

                if (k > 0)
                {
                    for (unsigned int i = 0; i < k-1; ++i)
                    {
                        r_full[i] += xi_k[i+1];
                    }
                }

                for (int i = k-1; i > -1; --i)
                    r_full -= P[i] * (viennacl::linalg::inner_prod(P[i], r) * gpu_scalar_2);

                r_full *= rho_0;
                x += r_full;

                // TODO: MPI_Alltoallv on the latest solution available
                tag.alltoall_subset(r_full); 

                if ( std::fabs(rho*rho_0 / b_norm ) < tag.tolerance() )
                {
                    //std::cout << "Allowed Error reached at end of loop" << std::endl;
                    tag.error(std::fabs(rho*rho_0 / b_norm ));
                    return x;
                }
#endif 
                //r = b;
                //r -= viennacl::linalg::prod(matrix, x);
                //std::cout << "norm_2(r)=" << norm_2(r) << std::endl;
                //std::cout << "std::abs(rho*rho_0)=" << std::abs(rho*rho_0) << std::endl;
                //std::cout << r << std::endl; 

                tag.error(std::fabs(rho*rho_0));
            }

            return x;
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
