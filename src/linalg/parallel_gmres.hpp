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
                 * @param R     The maximum dimension of the Krylov space before restart (number of restarts is found by max_iterations / R)
                 */
                parallel_gmres_tag(Communicator& comm_unit, Domain& decomposition, double tol = 1e-10, unsigned int max_iterations = 300, unsigned int R = 20, unsigned int solution_dim_per_node = 1) 
                    : gmres_tag(tol, max_iterations, R), 
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
        namespace vcl = viennacl;

#if 1
        template <typename ValueType>
            void ApplyPlaneRotation(ValueType& dx,
                    ValueType& dy,
                    ValueType& cs,
                    ValueType& sn)
            {
                ValueType temp = cs * dx + sn *dy;
                dy = -sn*dx+cs*dy;
                dx = temp;
            }

        template <typename ValueType> 
            void GeneratePlaneRotation(ValueType& dx,
                    ValueType& dy,
                    ValueType& cs,
                    ValueType& sn)
            {
                if(dy == ValueType(0.0)){
                    cs = 1.0;
                    sn = 0.0;
                }else if (abs(dy) > abs(dx)) {
                    ValueType tmp = dx / dy;
                    sn = ValueType(1.0) / sqrt(ValueType(1.0) + tmp*tmp);
                    cs = tmp*sn;            
                }else {
                    ValueType tmp = dy / dx;
                    cs = ValueType(1.0) / sqrt(ValueType(1.0) + tmp*tmp);
                    sn = tmp*cs;
                }
            }


        template <typename MatrixType, typename VectorType> 
            void PlaneRotation(MatrixType& H,
                    VectorType& cs,
                    VectorType& sn,
                    VectorType& s,
                    int i)
            {
                // Apply previous rotations
                for (int k = 0; k < i; k++){
                    ApplyPlaneRotation(H(k,i), H(k+1,i), cs[k], sn[k]);
                }
                // Generate new rotation
                GeneratePlaneRotation(H(i,i), H(i+1,i), cs[i], sn[i]);
                ApplyPlaneRotation(H(i,i), H(i+1,i), cs[i], sn[i]);
                ApplyPlaneRotation(s[i], s[i+1], cs[i], sn[i]);
            }
#endif 

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
        template <typename MatrixType, typename PreconditionerType>
                          vcl::vector<double> 
                          solve(const MatrixType & A, vcl::vector<double> & b_full, parallel_gmres_tag const & tag, PreconditionerType const & precond)
        {
            std::cout << "INSIDE VCL PARALLEL\n";
            typedef vcl::vector<double>                                             VectorType;
            typedef typename viennacl::result_of::value_type<VectorType>::type        ScalarType;
            typedef typename viennacl::result_of::cpu_value_type<ScalarType>::type    CPU_ScalarType;

            unsigned int NN = A.size1(); //viennacl::traits::size1(matrix);
            unsigned int MM = A.size2(); //viennacl::traits::size2(matrix);
            int R  = (int)tag.krylov_dim();

            // Solution
            VectorType x_full(MM);
            vcl::vector_range<VectorType> x(x_full, vcl::range(0,NN));
            viennacl::traits::clear(x_full);
            tag.alltoall_subset(x_full); 

            vcl::vector_range<VectorType> b(b_full, vcl::range(0,NN));

            // Workspace
            VectorType w_full(MM);
            vcl::vector_range<VectorType> w(w_full, vcl::range(0,NN)); 

            // Arnoldi Matrix
            std::vector< VectorType > v_full(R+1); 
            std::vector< vcl::vector_range<VectorType> * > v(R+1);

            VectorType v0_full(MM); 
            vcl::vector_range< VectorType > v0(v0_full, vcl::range(0,NN)); 

            // Givens rotations
            ublas::vector<double> s(R+1);

            // Hessenberg matrix (if we do the givens rotations properly this ends as upper triangular)
            //std::vector< std::vector<CPU_ScalarType> > H(R+1);
            ublas::matrix<double> H(R+1,R, 0);

            // Rotations (cs = cosine; sn = sine)
            ublas::vector<double> cs(R); 
            ublas::vector<double> sn(R); 

#if 1
            //representing the scalar '-1' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_minus_1 = static_cast<CPU_ScalarType>(-1);    
            //representing the scalar '1' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_1 = static_cast<CPU_ScalarType>(1);    
            //representing the scalar '2' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_2 = static_cast<CPU_ScalarType>(2);    
#endif 

            double beta = 0;
            double rel_resid0 = 0;

            for (int k = 0; k < R+1; ++k)
            {
                //H[k].resize(tag.krylov_dim()); 
                v_full[k].resize(MM);
                v[k] = new vcl::vector_range<VectorType>(v_full[k], vcl::range(0,NN)); 
            }

            MPI_Barrier(MPI_COMM_WORLD);

            if (tag.comm().isMaster()) 
                std::cout << "Starting Parallel GMRES..." << std::endl;
            tag.iters(0);

            // Save very first residual norm so we know when to stop
            CPU_ScalarType b_norm = viennacl::linalg::norm_2(b, tag.comm());
            v0 = b_full; 
            precond.apply(v0);
            CPU_ScalarType resid0 = viennacl::linalg::norm_2(v0, tag.comm()) / b_norm;
            //std::cout << "B_norm = " << b_norm << ", Resid0 = " << resid0 << std::endl;

            do{
                // compute initial residual and its norm //
                w = viennacl::linalg::prod(A, x_full) - b;                  // V(0) = A*x        //
                tag.alltoall_subset(w_full); 
                precond.apply(w_full);                                  // V(0) = M*V(0)     //
                beta = viennacl::linalg::norm_2(w, tag.comm()); 
                w /= gpu_scalar_minus_1 * beta;                                         // V(0) = -V(0)/beta //

                *(v[0]) = w; 
                
                // First givens rotation 
                for (int ii = 0; ii < R+1; ii++) {
                    s[ii] = 0.;
                }
                s[0] = beta; 
                int i = -1;

                if (beta / b_norm < tag.tolerance() || (b_norm == CPU_ScalarType(0.0)) )
                {
                    if (tag.comm().isMaster()) 
                        std::cout << "Allowed Error reached at begin of loop" << std::endl;
                    tag.error(beta / b_norm);
                    return x_full;
                }

                do{
                    ++i;
                    tag.iters(tag.iters() + 1); 

                    tag.alltoall_subset(w_full); 

                    //apply preconditioner
                    //can't pass in ref to column in V so need to use copy (w)
                    v0 = viennacl::linalg::prod(A,w_full); 
                    tag.alltoall_subset(v0_full); 
                    //V(i+1) = A*w = M*A*V(i)    //
                    precond.apply(v0_full); 
                    w = v0; 

                   for (int k = 0; k <= i; k++){
                        //  H(k,i) = <V(i+1),V(k)>    //
                        H(k, i) = viennacl::linalg::inner_prod(w, *(v[k]), tag.comm());
                        // V(i+1) -= H(k, i) * V(k)  //
                        //std::cout << v[k]->size() << "," << w.size() << H(k,i) * *(v[k])  << std::endl;
                        w -= H(k,i) * *(v[k]);
                    }

                    H(i+1,i) = viennacl::linalg::norm_2(w, tag.comm());   


#ifdef VIENNACL_GMRES_DEBUG 
                    std::cout << "H[" << i << "] = "; 
                    for (int j = 0; j < R+1; j++) {
                        for (int k = 0; k < R; k++) {
                            std::cout << H(j,k) << ", "; 
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
#endif 

                    // V(i+1) = V(i+1) / H(i+1, i) //
                    w /= gpu_scalar_1*H(i+1,i); 
                    v_full[i+1] = w; 

                    // Rotation takes place on host
                    PlaneRotation(H,cs,sn,s,i);

                    rel_resid0 = fabs(s[i+1]) / resid0;
                    std::cout << "\t" << i << "\t" << rel_resid0 << std::endl;

                    tag.error(rel_resid0);
                    // We could add absolute tolerance here as well: 
                    if (rel_resid0 < b_norm * tag.tolerance() ) {
                        break;
                    }

                }while (i+1 < R && tag.iters()+1 <= tag.max_iterations());
                // -------------------- SOLVE PROCESS ----------------------------------


                // After the Givens rotations, we have H is an upper triangular matrix 
                for (int j = i; j >= 0; j--) {
                    s[j] /= H(j,j); 
                    for (int k = j-1; k >= 0; k--) {
                        s[k] -= H(k,j) * s[j];
                    }  
                }
#ifdef VIENNACL_GMRES_DEBUG 
                    std::cout << "H[" << i << "] = "; 
                    for (int j = 0; j < R+1; j++) {
                        for (int k = 0; k < R; k++) {
                            std::cout << H(j,k) << ", "; 
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
    
                    std::cout << "s[" << i << "] = "; 
                    for (int j =0 ; j < R+1; j++) {
                        std::cout << s[j] << ", ";
                    }
                    std::cout << std::endl;
#endif 

                // Update our solution
                for (int j = 0 ; j < i; j++) {
                    x +=  s[j] * *(v[j]); 
                }
                tag.alltoall_subset(x_full); 

            } while (rel_resid0 >= b_norm*tag.tolerance() && tag.iters()+1 <= tag.max_iterations());

            return x_full;
        }


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
        template <typename MatrixType, typename PreconditionerType>
                          ublas::vector<double> 
                          solve(const MatrixType & A, ublas::vector<double> & b_full, parallel_gmres_tag const & tag, PreconditionerType const & precond)
        {
            std::cout << "INSIDE PARALLEL\n";
            typedef ublas::vector<double>                                             VectorType;
            typedef typename viennacl::result_of::value_type<VectorType>::type        ScalarType;
            typedef typename viennacl::result_of::cpu_value_type<ScalarType>::type    CPU_ScalarType;

            unsigned int NN = A.size1(); //viennacl::traits::size1(matrix);
            unsigned int MM = A.size2(); //viennacl::traits::size2(matrix);
            int R  = (int)tag.krylov_dim();

            // Solution
            VectorType x_full(MM);
            ublas::vector_range<VectorType> x(x_full, ublas::range(0,NN));
            viennacl::traits::clear(x_full);
            tag.alltoall_subset(x_full); 

            ublas::vector_range<VectorType> b(b_full, ublas::range(0,NN));

            // Workspace
            VectorType w_full(MM);
            ublas::vector_range<VectorType> w(w_full, ublas::range(0,NN)); 

            // Arnoldi Matrix
            std::vector< VectorType > v_full(R+1); 
            std::vector< ublas::vector_range<VectorType> * > v(R+1);

            VectorType v0_full(MM); 
            ublas::vector_range< VectorType > v0(v0_full, ublas::range(0,NN)); 

            // Givens rotations
            VectorType s(R+1);

            // Hessenberg matrix (if we do the givens rotations properly this ends as upper triangular)
            //std::vector< std::vector<CPU_ScalarType> > H(R+1);
            ublas::matrix<double> H(R+1,R);

            // Rotations (cs = cosine; sn = sine)
            VectorType cs(R); 
            VectorType sn(R); 

#if 0
            //representing the scalar '-1' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_minus_1 = static_cast<CPU_ScalarType>(-1);    
            //representing the scalar '1' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_1 = static_cast<CPU_ScalarType>(1);    
            //representing the scalar '2' on the GPU. Prevents blocking write operations
            const CPU_ScalarType gpu_scalar_2 = static_cast<CPU_ScalarType>(2);    
#endif 

            double beta = 0;
            double rel_resid0 = 0;

            for (int k = 0; k < R+1; ++k)
            {
                //H[k].resize(tag.krylov_dim()); 
                v_full[k].resize(MM);
                v[k] = new ublas::vector_range<VectorType>(v_full[k], ublas::range(0,NN)); 
            }

            MPI_Barrier(MPI_COMM_WORLD);

            if (tag.comm().isMaster()) 
                std::cout << "Starting Parallel GMRES..." << std::endl;
            tag.iters(0);

            // Save very first residual norm so we know when to stop
            double b_norm = viennacl::linalg::norm_2(b, tag.comm());
            v0 = b_full; 
            precond.apply(v0);
            double resid0 = viennacl::linalg::norm_2(v0, tag.comm()) / b_norm;
            //            std::cout << "B_norm = " << b_norm << ", Resid0 = " << resid0 << std::endl;


            do{
                // compute initial residual and its norm //
                w = b - viennacl::linalg::prod(A, x_full);                  // V(0) = A*x        //
                tag.alltoall_subset(w_full); 
                precond.apply(w_full);                                  // V(0) = M*V(0)     //
                beta = viennacl::linalg::norm_2(w, tag.comm()); 
                w /= beta;                                         // V(0) = -V(0)/beta //

                *(v[0]) = w; 

                // First givens rotation 
                for (int ii = 0; ii < R+1; ii++) {
                    s[ii] = 0.;
                }
                s[0] = beta; 
                int i = -1;

                if (beta / b_norm < tag.tolerance() || (b_norm == CPU_ScalarType(0.0)) )
                {
                    if (tag.comm().isMaster()) 
                        std::cout << "Allowed Error reached at begin of loop" << std::endl;
                    tag.error(beta / b_norm);
                    return x_full;
                }

                do{
                    ++i;
                    tag.iters(tag.iters() + 1); 

                    tag.alltoall_subset(w_full); 

                    //apply preconditioner
                    //can't pass in ref to column in V so need to use copy (w)
                    v0 = viennacl::linalg::prod(A,w_full); 
                    tag.alltoall_subset(v0_full); 
                    //V(i+1) = A*w = M*A*V(i)    //
                    precond.apply(v0_full); 
                    w = v0; 

                    for (int k = 0; k <= i; k++){
                        //  H(k,i) = <V(i+1),V(k)>    //
                        H(k, i) = viennacl::linalg::inner_prod(w, *(v[k]), tag.comm());
                        // V(i+1) -= H(k, i) * V(k)  //
                        w -= H(k,i) * *(v[k]);
                    }

                    H(i+1,i) = viennacl::linalg::norm_2(w, tag.comm());   
                
#ifdef VIENNACL_GMRES_DEBUG 
                   std::cout << "H[" << i << "] = "; 
                   for (int j = 0; j < R+1; j++) {
                    for (int k = 0; k < R; k++) {
                        std::cout << H(j,k) << ", "; 
                    }
                    std::cout << std::endl;
                   }
                    std::cout << std::endl;
#endif 
                    // V(i+1) = V(i+1) / H(i+1, i) //
                    w *= 1.0/H(i+1,i); 
                    v_full[i+1] = w; 

                    PlaneRotation(H,cs,sn,s,i);

                    rel_resid0 = fabs(s[i+1]) / resid0;
                    std::cout << "\t" << i << "\t" << rel_resid0 << std::endl;

                    tag.error(rel_resid0);
                    // We could add absolute tolerance here as well: 
                    if (rel_resid0 < b_norm * tag.tolerance() ) {
                        break;
                    }

                }while (i+1 < R && tag.iters()+1 <= tag.max_iterations());

                // -------------------- SOLVE PROCESS ----------------------------------


                // After the Givens rotations, we have H is an upper triangular matrix 
                for (int j = i; j >= 0; j--) {
                    s[j] /= H(j,j); 
                    for (int k = j-1; k >= 0; k--) {
                        s[k] -= H(k,j) * s[j];
                    }  
                }

#ifdef VIENNACL_GMRES_DEBUG 
                    std::cout << "H[" << i << "] = "; 
                    for (int j = 0; j < R+1; j++) {
                        for (int k = 0; k < R; k++) {
                            std::cout << H(j,k) << ", "; 
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
    
                    std::cout << "s[" << i << "] = "; 
                    for (int j =0 ; j < R+1; j++) {
                        std::cout << s[j] << ", ";
                    }
                    std::cout << std::endl;
#endif 

                // Update our solution
                for (int j = 0 ; j < i; j++) {
                    x += *(v[j]) * s[j]; 
                }
                tag.alltoall_subset(x_full); 

            } while (rel_resid0 >= b_norm*tag.tolerance() && tag.iters()+1 <= tag.max_iterations());

            return x_full;
        }

        /** @brief Convenience overload of the solve() function using GMRES. Per default, no preconditioner is used
        */ 
        template <typename MatrixType, typename VectorType>
            VectorType solve(const MatrixType & matrix, VectorType & rhs, parallel_gmres_tag const & tag)
            {
                std::cout << "CALLING SOLVER\n";
                return solve(matrix, rhs, tag, no_precond());
            }


    }
}

#endif
