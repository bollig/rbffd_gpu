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

 //                           MPI_Alltoallv(this->sbuf, this->sendcounts, this->sdispls, MPI_DOUBLE, this->rbuf, this->recvcounts, this->rdispls, MPI_DOUBLE, comm_ref.getComm()); 

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
                                    // TODO: need to translate to local indexing properly. This hack assumes all boundary are dirichlet and appear first in the list
                                    vec[grid_ref.g2l(grid_ref.R_by_rank[i][j]) - grid_ref.getBoundaryIndicesSize()] = this->rbuf[k];  
                                    k++; 
                                }
                            }
                        }
                    }

                template <typename VectorType>
                    void 
                    alltoall_subset(VectorType& v) const
                    {
                        // Redirect the call to appropriate float or double routines (Type size matters with MPI)
                        typename VectorType::value_type dummy; 
//                        std::cout << "Rank " << comm_ref.getRank() << " v.size = " << v.size() << std::endl;
                        alltoall_subset(v, dummy);
                    }




            protected: 
                mutable Communicator& comm_ref; 
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
         * Following the algorithm proposed by Walker in "A Simpler GMRES" (1988)
         *
         * @param matrix     The system matrix
         * @param rhs        The load vector
         * @param tag        Solver configuration tag
         * @param precond    A preconditioner. Precondition operation is done via member function apply()
         * @return The result vector
         */
        template <typename MatrixType, /*typename VectorType=ublas::vector<double>,*/ typename PreconditionerType>
            ublas::vector<double> solve(const MatrixType & matrix, ublas::vector<double> const & rhs, parallel_gmres_tag const & tag, PreconditionerType const & precond)
            {
                typedef ublas::vector<double> VectorType;
                // RHS is size(M,1)
                typedef typename viennacl::result_of::value_type<VectorType>::type        ScalarType;
                typedef typename viennacl::result_of::cpu_value_type<ScalarType>::type    CPU_ScalarType;
                // In parallel these dimensions are VERY IMPORTANT to get right
                //unsigned int problem_size = viennacl::traits::size(rhs);
                unsigned int NN = matrix.size1(); //viennacl::traits::size1(matrix);
                unsigned int MM = matrix.size2(); //viennacl::traits::size2(matrix);
                // TODO: allow user specified initial guess
                // TODO: when allowed MPI_Alltoallv sync here
                VectorType result(MM);
                viennacl::traits::clear(result);

                // Make sure all procs have the full result (if we pass something in)
                tag.alltoall_subset(result); 


                unsigned int krylov_dim = tag.krylov_dim();
                if (NN < tag.krylov_dim())
                    krylov_dim = NN; //A Krylov space larger than the matrix would lead to seg-faults (mathematically, error is certain to be zero already)

                VectorType res_full(MM);
                ublas::vector_range<VectorType> res(res_full, ublas::range(0,NN)); 
                VectorType v_k_tilde_full(MM);
                ublas::vector_range<VectorType> v_k_tilde(v_k_tilde_full, ublas::range(0,NN)); 
                VectorType v_k_tilde_temp_full(MM);
                ublas::vector_range<VectorType> v_k_tilde_temp(v_k_tilde_temp_full, ublas::range(0,NN)); 

                // k x k
                std::vector< std::vector<CPU_ScalarType> > R(krylov_dim);
                // k x 1
                std::vector<CPU_ScalarType> projection_rhs(krylov_dim);
                // k x M
                std::vector<VectorType> U(krylov_dim);

                const CPU_ScalarType gpu_scalar_minus_1 = static_cast<CPU_ScalarType>(-1);    //representing the scalar '-1' on the GPU. Prevents blocking write operations
                const CPU_ScalarType gpu_scalar_1 = static_cast<CPU_ScalarType>(1);    //representing the scalar '1' on the GPU. Prevents blocking write operations
                const CPU_ScalarType gpu_scalar_2 = static_cast<CPU_ScalarType>(2);    //representing the scalar '2' on the GPU. Prevents blocking write operations

                // DONE: MPI_Reduce on norm_2 (note: we only apply norm_2 to the subset controlled by this machine
                CPU_ScalarType norm_rhs = viennacl::linalg::norm_2(rhs, tag.comm());

                unsigned int k;
                for (k = 0; k < krylov_dim; ++k)
                {
                    R[k].resize(tag.krylov_dim()); 
                    viennacl::traits::resize(U[k], MM);
                }

//                if (tag.comm().isMaster()) 
                    std::cout << "Starting Parallel GMRES..." << std::endl;
                tag.iters(0);

                for (unsigned int it = 0; it <= 1; /*tag.max_restarts();*/ ++it)
                {
//                    if (tag.comm().isMaster()) 
                        std::cout << "-- GMRES Start " << it << " -- " << std::endl;

                        // V(0) = b - A*x
                    res = rhs;
                    res -= viennacl::linalg::prod(matrix, result);  //initial guess zero
                    // DONE: MPI_Alltoallv before preconditioning
                    tag.alltoall_subset(res_full); 
//                    precond.apply(res_full);
                    //std::cout << "Residual: " << res << std::endl;

                    // DONE: MPI_Reduce on norm_2
                    // beta = norm(V(0)))
                    CPU_ScalarType rho_0 = static_cast<CPU_ScalarType>(viennacl::linalg::norm_2(res, tag.comm())); 
                    CPU_ScalarType rho = static_cast<CPU_ScalarType>(1.0);
//                    if (tag.comm().isMaster()) 
                        std::cout << "rho_0: " << rho_0 << std::endl;

                    if (rho_0 / norm_rhs < tag.tolerance() || (norm_rhs == CPU_ScalarType(0.0)) )
                    {
//                        if (tag.comm().isMaster()) 
                            std::cout << "Allowed Error reached at begin of loop" << std::endl;
                        tag.error(rho_0 / norm_rhs);
                        return result;
                    }

                    // V(0) = -V(0)/beta
                    res_full /= rho_0;
                    //std::cout << "Normalized Residual: " << res << std::endl;

                    for (k=0; k<krylov_dim; ++k)
                    {
                        viennacl::traits::clear(R[k]);
                        viennacl::traits::clear(U[k]);
                        R[k].resize(krylov_dim); 
                        //TODO: is this right? 
                        viennacl::traits::resize(U[k], MM);
                    }
#if 1
                    for (k = 0; k < krylov_dim; ++k)
                    {
                        tag.iters( tag.iters() + 1 ); //increase iteration counter

                        //compute v_k = A * v_{k-1} via Householder matrices
                        if (k == 0)
                        {
                            // TODO: MPI_Alltoallv on res, but then we also need to sync on v_k_tilde
                            tag.alltoall_subset(res_full); 
                            v_k_tilde = viennacl::linalg::prod(matrix, res_full);
                            tag.alltoall_subset(v_k_tilde_full); 
                            //precond.apply(v_k_tilde);
                        }
                        else
                        {
                            // Below the U[i] is H(k,i)
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
                            for (int i = k-1; i > -1; --i)
                                v_k_tilde_full -= U[i] * (viennacl::linalg::inner_prod(U[i], v_k_tilde_full) * gpu_scalar_2);

                            // TODO: MPI_Alltoallv on v_k_tilde_full
                            tag.alltoall_subset(v_k_tilde_full); 
                            v_k_tilde_temp = viennacl::linalg::prod(matrix, v_k_tilde_full);
                            tag.alltoall_subset(v_k_tilde_temp_full); 
                            //precond.apply(v_k_tilde_temp_full);
                            v_k_tilde_full = v_k_tilde_temp_full;

                            //Householder rotations part 2
                            for (unsigned int i = 0; i < k; ++i)
                                v_k_tilde_full -= U[i] * (viennacl::linalg::inner_prod(U[i], v_k_tilde_full) * gpu_scalar_2);
                        }

#if 1

                        viennacl::traits::clear(U[k]);
                        viennacl::traits::resize(U[k], MM);
                        //copy first k entries from v_k_tilde to U[k]:

                        // ??? EVAN ??? Here we build a krylov subspace using k dimensions just from 
                        gmres_copy_helper(v_k_tilde_full, U[k], k);

                        U[k][k] = std::sqrt( viennacl::linalg::inner_prod(v_k_tilde_full, v_k_tilde_full) - viennacl::linalg::inner_prod(U[k], U[k]) );

                        if (fabs(U[k][k]) < CPU_ScalarType(10 * std::numeric_limits<CPU_ScalarType>::epsilon()))
                            break; //Note: Solution is essentially (up to round-off error) already in Krylov space. No need to proceed.

                        //copy first k+1 entries from U[k] to R[k]
                        gmres_copy_helper(U[k], R[k], k+1);

                        U[k] -= v_k_tilde_full;
                        //std::cout << "U[k] before normalization: " << U[k] << std::endl;
                        CPU_ScalarType U_k_norm = viennacl::linalg::norm_2( U[k] , tag.comm() );
                        U[k] *= gpu_scalar_minus_1 / U_k_norm;
                        //std::cout << "Householder vector U[k]: " << U[k] << std::endl;

                        //DEBUG: Make sure that P_k v_k_tilde_full equals (rho_{1,k}, ... , rho_{k,k}, 0, 0 )
#ifdef VIENNACL_GMRES_DEBUG
                        std::cout << "P_k v_k_tilde: " << (v_k_tilde_full - 2.0 * U[k] * inner_prod(U[k], v_k_tilde_full)) << std::endl;
                        std::cout << "R[k]: [" << R[k].size() << "](";
                        for (size_t i=0; i<R[k].size(); ++i)
                            std::cout << R[k][i] << ",";
                        std::cout << ")" << std::endl;
#endif
                        //std::cout << "P_k res: " << (res - 2.0 * U[k] * inner_prod(U[k], res)) << std::endl;
                        res_full -= U[k] * (viennacl::linalg::inner_prod( U[k], res_full ) * gpu_scalar_2);
                        //std::cout << "zeta_k: " << viennacl::linalg::inner_prod( U[k], res ) * gpu_scalar_2 << std::endl;
                        //std::cout << "Updated res: " << res << std::endl;
#endif 

#ifdef VIENNACL_GMRES_DEBUG
                        VectorType v1(U[k].size()); v1.clear(); v1.resize(U[k].size());
                        v1(0) = 1.0;
                        v1 -= U[k] * (viennacl::linalg::inner_prod( U[k], v1 ) * gpu_scalar_2);
                        std::cout << "v1: " << v1 << std::endl;
                        boost::numeric::ublas::matrix<ScalarType> P = -2.0 * outer_prod(U[k], U[k]);
                        P(0,0) += 1.0; P(1,1) += 1.0; P(2,2) += 1.0;
                        std::cout << "P: " << P << std::endl;
#endif

                        if (res_full[k] > rho) //machine precision reached
                            res_full[k] = rho;

                        if (res_full[k] < -1.0 * rho) //machine precision reached
                            res_full[k] = -1.0 * rho;

                        projection_rhs[k] = res_full[k];

                        rho *= std::sin( std::acos(projection_rhs[k] / rho) );

#ifdef VIENNACL_GMRES_DEBUG
                        std::cout << "k-th component of r: " << res_full[k] << std::endl;
                        std::cout << "New rho (norm of res): " << rho << std::endl;
#endif        

                        if (std::fabs(rho * rho_0 / norm_rhs) < tag.tolerance())
                        {
                            //std::cout << "Krylov space big enough" << endl;
                            tag.error( std::fabs(rho*rho_0 / norm_rhs) );
                            ++k;
//                            if (tag.comm().isMaster()) 
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
                    res_full *= projection_rhs[0];

                    if (k > 0)
                    {
                        for (unsigned int i = 0; i < k-1; ++i)
                        {
                            res_full[i] += projection_rhs[i+1];
                        }
                    }

                    for (int i = k-1; i > -1; --i)
                        res_full -= U[i] * (viennacl::linalg::inner_prod(U[i], res_full) * gpu_scalar_2);

                    res_full *= rho_0;
                    result += res_full;

                    // TODO: MPI_Alltoallv on the latest solution available
                    tag.alltoall_subset(res_full); 

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
#endif 
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
