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
                                    //                                    std::cout << "Sending " << s_indx << "\n";
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
                                    //                                    std::cout << "Receiving " << r_indx << "\n";
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
            VectorType x_full(MM);
            ublas::vector_range<VectorType> x(x_full, ublas::range(0,NN));

            viennacl::traits::clear(x_full);

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
            std::vector<double> c(krylov_dim); 
            std::vector<double> s(krylov_dim); 


            //representing the scalar '-1' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_minus_1 = static_cast<CPU_ScalarType>(-1);    
            //representing the scalar '1' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_1 = static_cast<CPU_ScalarType>(1);    
            //representing the scalar '2' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_2 = static_cast<CPU_ScalarType>(2);    

            for (unsigned int k = 0; k < krylov_dim+1; ++k)
            {
                H[k].resize(tag.krylov_dim()); 
                v_full[k].resize(MM);
                v[k] = new ublas::vector_range<VectorType>(v_full[k], ublas::range(0,NN)); 
            }

            MPI_Barrier(MPI_COMM_WORLD);

            if (tag.comm().isMaster()) 
                std::cout << "Starting Parallel GMRES..." << std::endl;
            tag.iters(0);

            // Save very first residual norm so we know when to stop
            CPU_ScalarType b_norm = viennacl::linalg::norm_2(b, tag.comm());
            v_full[0] = b; 
            precond.apply(*(v[0])); 
            CPU_ScalarType resid0 = viennacl::linalg::norm_2(*(v[0]), tag.comm()) / b_norm;
            std::cout << "B_norm = " << b_norm << ", Resid0 = " << resid0 << std::endl;

            // -------------------- OUTER LOOP ----------------------------------
            for (unsigned int it = 0; it <= tag.max_restarts(); ++it)
            {
                if (tag.comm().isMaster()) 
                    std::cout << "-- GMRES Start " << it << " -- " << std::endl;

                tag.alltoall_subset(x_full); 

                // r = b - A x
                r = b - viennacl::linalg::prod(matrix, x_full);  //initial guess zero
                tag.alltoall_subset(r_full); 
#if 0
                precond.apply(r);
#else 
                precond.apply(r_full);
#endif 

                CPU_ScalarType rho_0 = static_cast<CPU_ScalarType>(viennacl::linalg::norm_2(r, tag.comm())); 
                CPU_ScalarType rho = static_cast<CPU_ScalarType>(1.0);
                if (tag.comm().isMaster()) 
                    std::cout << "rho_0: " << rho_0 << " , rho_0 / b_norm: " << rho_0/b_norm << std::endl;

                if (rho_0 / b_norm < tag.tolerance() || (b_norm == CPU_ScalarType(0.0)) )
                {
                    if (tag.comm().isMaster()) 
                        std::cout << "Allowed Error reached at begin of loop" << std::endl;
                    tag.error(rho_0 / b_norm);
                    return x_full;
                }

                // v_1 = r / rho_0
                *(v[0]) = r / rho_0;

                // First givens rotation 
                g[0] = rho_0; 

                // -------------------- INNER ARNOLDI PROCESS ----------------------------------
                // we declare k here so we can iterate partially through krylov
                // dims and still solve the partial system
                unsigned int k; 
                for (k = 0; k < krylov_dim; ++k)
                {
                    tag.iters( tag.iters() + 1 ); //increase iteration counter

                    tag.alltoall_subset(v_full[k]); 

                    // v_k+1 = A v_k
                    *(v[k+1]) = viennacl::linalg::prod(matrix, v_full[k]);
                    tag.alltoall_subset(v_full[k+1]); 
#if 0
                    precond.apply((*v[k+1]));
#else 
                    precond.apply(v_full[k+1]);
#endif 
                    // Begin modified Gram-Schmidt (may require reorthogonalization)
                    for (int j = 0; j < k; j++) { 
                        H[j][k] = viennacl::linalg::inner_prod(*(v[j]), *(v[k+1]), tag.comm());
                        *(v[k+1]) -= H[j][k] * *(v[j]); 
                    }

                    H[k+1][k] = viennacl::linalg::norm_2(*(v[k+1]), tag.comm());

                    // Safety check
                    if ((H[k+1][k] > 0.) || (H[k+1][k] < 0.)) {
                        *(v[k+1]) /= H[k+1][k];
                    } 


                    if (k > 0) {
                        // Apply previous rotations
                        for (int i = 0; i < k; i++) {
                            // Need additional 2*k storage for c and s
                            double hik = c[i]*H[i][k] + s[i]*H[i+1][k]; 
                            double hipk = -s[i]*H[i][k] + c[i]*H[i+1][k]; 
                            H[i][k] = hik; 
                            H[i+1][k] = hipk;
                        }
                    }

#if 0
                    double nu = sqrt(H[k][k]*H[k][k] + H[k+1][k]*H[k+1][k]);
                    if (nu > 0.) {
                        c[k] = H[k][k] / nu; 
                        s[k] = -H[k+1][k] / nu; 

                        H[k][k] = c[k]*H[k][k] + s[k]*H[k+1][k]; 
                        H[k+1][k] = 0;
                    }
#else 
                        // Generate rotation (Borrowed from CUSP v 0.3.1)
                        if (H[k+1][k] == 0.) { 
                            s[k] = 0.0; 
                            c[k] = 1.0; 
                        } else if (abs(H[k+1][k]) > abs(H[k][k])) {
                            double tmp = H[k][k]/H[k+1][k]; 
                            s[k] = 1./sqrt(1 + tmp*tmp); 
                            c[k] = tmp * s[k]; 
                        } else { 
                            double tmp = H[k+1][k]/H[k][k]; 
                            c[k] = 1./sqrt(1 + tmp*tmp); 
                            s[k] = tmp * c[k]; 
                        }

                        double hik = c[k]*H[k][k] + s[k]*H[k+1][k]; 
                        double hipk = -s[k]*H[k][k] + c[k]*H[k+1][k]; 
                        H[k][k] = hik; 
                        H[k+1][k] = hipk;
#endif 

                        double gk = c[k]*g[k] + s[k]*g[k+1]; 
                        double gkp = -s[k]*g[k] + c[k]*g[k+1]; 

                        g[k] = gk; 
                        g[k+1] = gkp; 
                    rho = fabs(g[k+1]); 
                    std::cout << "rho = " << rho << " , rho / resid0 = " << rho / resid0 << std::endl;

                    // We could add absolute tolerance here as well: 
                    if ( rho / resid0 <= b_norm * tag.tolerance() ) {
                        tag.error(rho / resid0);
                        break;
                    }
                } // for k

                // -------------------- SOLVE PROCESS ----------------------------------

                // After the Givens rotations, we have H is an upper triangular matrix 
                std::vector<double> y(k+1); 
                y[k]  = g[k] / H[k][k]; 
                for (int i = k-1; i > -1; i--) {
                    y[i]  = g[i]; 
                    for (int j = i+1; j < k; j++)  {
                        y[i] -= H[i][j] * y[j]; 
                    }
                    y[i] = y[i] / H[i][i]; 
                }

                // Update our solution
                for (int i = 0 ; i < k; i++) {
                    x += *(v[i]) * y[i]; 
                }

                tag.alltoall_subset(x_full); 

                if ( rho / resid0 <= b_norm * tag.tolerance() ) {
                    //if ( std::fabs(rho*rho_0 / b_norm ) < tag.tolerance() )
                    //               {
                    tag.error(std::fabs(rho*rho_0 / b_norm ));
                    tag.alltoall_subset(x_full); 
                    return x_full;
                }
                tag.error(std::fabs(rho*rho_0));
                }

                return x_full;
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
