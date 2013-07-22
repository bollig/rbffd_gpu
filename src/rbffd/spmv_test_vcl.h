#ifndef __SPMV_TEST_H__
#define __SPMV_TEST_H__

#ifndef USE_MPI
#define USE_MPI 1
#endif 

#if USE_MPI
#include <mpi.h>
#endif 
#include <iostream>

#include "rbffd.h"
#include "grids/domain.h"
#include "timer_eb.h"
#include "common_typedefs.h"

#include "utils/opencl/viennacl_typedefs.h"

#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/ell_matrix.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/prod.hpp> 
#include <viennacl/io/matrix_market.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp> 
#include <viennacl/vector_proxy.hpp> 
#include <viennacl/matrix_proxy.hpp> 
#include <viennacl/linalg/vector_operations.hpp> 

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

#include "viennacl/linalg/parallel_norm_1.hpp"                                                                                     
#include "viennacl/linalg/parallel_norm_2.hpp"
#include "viennacl/linalg/parallel_norm_inf.hpp"


class SpMVTest
{
    protected:
        RBFFD_VCL_OVERLAP* rbffd; 
        Domain* grid; 
        int rank, size;

        int sol_dim; 

        int o_comm_size; 
        int r_comm_size; 
        double* sbuf; 
        int* sendcounts; 
        int* sdispls; 
        int* rdispls; 
        int* recvcounts; 
        double* rbuf; 
        MPI_Request* O_reqs;
        MPI_Request* R_reqs;
        MPI_Status* O_stats;
        MPI_Status* R_stats;

        EB::TimerList tm; 

    public: 
        SpMVTest(RBFFD_VCL_OVERLAP* r, Domain* domain, int mpi_rank=0, int mpi_size=0) : rbffd(r), grid(domain), rank(mpi_rank), size(mpi_size), sol_dim(1) {
            o_comm_size=0; 
            r_comm_size=0;
            setupTimers(); 
            allocateCommBuffers(); 
        }

        ~SpMVTest() {
            delete [] sbuf; 
            delete [] sendcounts; 
            delete [] sdispls; 
            delete [] rdispls; 
            delete [] recvcounts; 
            delete [] rbuf; 

            tm.printAll();
            char buf[256]; 
            sprintf(buf, "time_log.spmvtest.%d", rank); 
            tm.writeToFile(buf);
            tm.clear();
        }

        void allocateCommBuffers() {
            tm["allocateCommBufs"]->start();
            // std::cout << "Allocating comm buffers\n";
            // std::cout << grid->O_by_rank.size() << "\n";
            this->sdispls = new int[grid->O_by_rank.size()]; 
            this->sendcounts = new int[grid->O_by_rank.size()]; 
            sdispls[0] = 0;
            sendcounts[0] = this->sol_dim*(grid->O_by_rank[0].size()); 
            unsigned int O_tot = sendcounts[0]; 

            if (sendcounts[0] > 0) {
                o_comm_size++; 
            }
            //std::cout << "sdispl[" << 0 << "] = " << sdispls[0] << ", sendcounts = " << sendcounts[0] << std::endl;
            for (size_t i = 1; i < grid->O_by_rank.size(); i++) {
                sendcounts[i] = this->sol_dim*(grid->O_by_rank[i].size()); 
                if (sendcounts[i] > 0) {
                    o_comm_size++; 
                }
                sdispls[i] = sdispls[i-1] + sendcounts[i-1];
                O_tot += sendcounts[i]; 
                //std::cout << "sdispl[" << i << "] = " << sdispls[i] << ", sendcounts = " << sendcounts[i] << std::endl;
            }

            this->rdispls = new int[grid->R_by_rank.size()]; 
            this->recvcounts = new int[grid->R_by_rank.size()]; 
            rdispls[0] = 0; 
            recvcounts[0] = this->sol_dim*grid->R_by_rank[0].size(); 
            unsigned int R_tot = recvcounts[0];
            if (recvcounts[0] > 0) {
                r_comm_size++; 
            }
            for (size_t i = 1; i < grid->R_by_rank.size(); i++) {
                rdispls[i] = rdispls[i-1] + recvcounts[i-1];
                recvcounts[i] = this->sol_dim*grid->R_by_rank[i].size(); 
                if (recvcounts[i] > 0) {
                    r_comm_size++; 
                }
                R_tot += recvcounts[i]; 
            }

            // std::cout << "O_tot = " << O_tot << std::endl;
            // std::cout << "R_tot = " << R_tot << std::endl;
            std::cout << "O_COMM_SIZE = " << o_comm_size << std::endl;
            std::cout << "R_COMM_SIZE = " << r_comm_size << std::endl;

            // Not sure if we need to use malloc to guarantee contiguous?
            this->sbuf = new double[O_tot];  
            this->rbuf = new double[R_tot];
            this->O_reqs = new MPI_Request[o_comm_size];
            this->O_stats = new MPI_Status[o_comm_size];
            this->R_reqs = new MPI_Request[r_comm_size];
            this->R_stats = new MPI_Status[r_comm_size];
            tm["allocateCommBufs"]->stop();
        }



        void setupTimers() {
            tm["synchronize"] = new EB::Timer("[SpMVTest] Synchronize (MPI_Isend/MPI_Irecv");
            tm["spmv"] = new EB::Timer("[SpMVTest] perform SpMV");
            tm["spmv_w_comm"] = new EB::Timer("[SpMVTest] SpMV + Communication");
            tm["irecv"] = new EB::Timer("[SpMVTest] MPI_Irecv"); 
            tm["encode_send"] = new EB::Timer("[SpMVTest] Encode sendbuf (collect all O_by_rank into one vector)"); 
            tm["decode_recv"] = new EB::Timer("[SpMVTest] Decode recvbuf (scatter R_by_rank into memory)");
            tm["alltoallv"] = new EB::Timer("[SpMVTest] MPI_Alltoallv Only (no memcpy)"); 
            tm["sendrecv_wait"] = new EB::Timer("[SpMVTest] Barrier before MPI_Isend"); 
            tm["allocateCommBufs"] = new EB::Timer("[SpMVTest] Allocate buffers for MPI_Isend/MPI_Irecv");
        }

        // Does a simple CPU CSR SpMV
        // can apply to subset of problem 
        void SpMV(RBFFD::DerType which, VCL_VEC_t& u_gpu, VCL_VEC_t& out_deriv) {
            // TODO: 
            // GPU Matrix
            // GPU Vector
            //
            //  done - post irecv
            //  queue Q\B
            //  copy O down
            //  assemble sendbuf
            //  done - post isends
            //  done - waitall(irecv)
            //  copy R up
            //  queue B
            tm["spmv_w_comm"]->start();

            //------------
            // Post irecv
            //------------
            tm["synchronize"]->start();
            if (size > 1) {
                // I found 8+ processors comm best with Isend/Irecv. Alltoallv
                // for < 8 
                //if (size > 8) { 
                    this->postIrecvs(); 
                //}
                // Else we use Alltoallv and dont need to worry
            }

            //------------
            // Queue Q\B
            //------------

            //std::vector<double*> DM = rbffd->getWeights(which);
            VCL_ELL_MAT_t& DM_qmb = *(rbffd->getGPUWeightsSetQmB(which));
            VCL_ELL_MAT_t& DM_b = *(rbffd->getGPUWeightsSetB(which));

            int nb_stencils = grid->getStencilsSize();
            int nb_nodes = grid->getNodeListSize();
            int nb_qmb_rows = grid->QmB_size;

            viennacl::range r1(0, nb_stencils);
            viennacl::range r2(0, nb_nodes);

            viennacl::range r3(0, nb_qmb_rows);
            viennacl::range r4(nb_qmb_rows, nb_stencils);

            //------------
            // Start Async copy O Down
            //------------
 
            tm["spmv"]->start();
            // SpMV on first QmB rows
            project(out_deriv, r3) = (VCL_VEC_t) viennacl::linalg::prod(DM_qmb, u_gpu); 
           
            //------------
            // Fill Sendbuf 
            //------------

            this->encodeSendBuf(u_gpu);

            //------------
            // Send O
            //------------

            if (size > 1) {
                // I found 8+ processors comm best with Isend/Irecv. Alltoallv
                // for < 8 
          //      if (size > 8) { 
                    // NOTE: this includes waitall on irecvs
                    this->postIsends(); 
           //     }
            //    this->postAlltoallv();
            }
            tm["synchronize"]->stop();

            //------------
            // Copy R up 
            //------------

            this->decodeRecvBuf(u_gpu);

            //TODO: send to GPU;

            project(out_deriv, r4) = (VCL_VEC_t) viennacl::linalg::prod(DM_b, u_gpu); 

            tm["spmv"]->stop();

            tm["spmv_w_comm"]->stop();
        }


        void postIrecvs() {
            // Buffer recvs
            tm["irecv"]->start();
            int r_count = 0;
            for (int i = 0; i < this->size; i++) { 
                if (this->recvcounts[i] > 0) {
                    MPI_Irecv(this->rbuf + this->rdispls[i], this->recvcounts[i], MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &R_reqs[r_count]);
                    r_count++;
                }
            }
        }

        void encodeSendBuf(VCL_VEC_t& gpu_vec) {
            unsigned int set_Q_size = grid->Q_size;
            unsigned int set_O_size = grid->O_size;
            unsigned int nb_bnd = grid->getBoundaryIndicesSize();

            //std::cout << "set_Q_size = " << set_Q_size << ", set_O_size = " << set_O_size << ", nb_bnd = " << nb_bnd << std::endl;

            // OUR SOLUTION IS ARRANGED IN THIS FASHION:
            //  { Q\B D O R } where B = union(D, O) and Q = union(Q\B D O)
            //  Minus 1 because we start indexing at 0

            // TODO: fix this. We have to maintain an additional index
            // map to convert from local node indices to the linear
            // system indices (i.e. filter off the dirichlet boundary
            // node indices
            unsigned int offset_to_interior = nb_bnd;
            unsigned int offset_to_set_O = (set_Q_size - set_O_size);

            // std::cout << "set_Q_size = " << set_Q_size << ", set_O_size = " << set_O_size << ", nb_bnd = " << nb_bnd << std::endl;

            viennacl::vector_range< VCL_VEC_t > setO(gpu_vec, viennacl::range((offset_to_set_O - offset_to_interior) * sol_dim, ((offset_to_set_O-offset_to_interior)+set_O_size) * sol_dim));

            //double* vec = new double[set_O_size * sol_dim];
            std::vector<double> vec(set_O_size * sol_dim);

            viennacl::copy(setO, vec); //, set_O_size * sol_dim);

            tm["encode_send"]->start();
            // Prep-Send: Copy elements of set to sbuf
            unsigned int k = 0; 
            for (size_t i = 0; i < grid->O_by_rank.size(); i++) {
                k = this->sdispls[i]; 
                for (size_t j = 0; j < grid->O_by_rank[i].size(); j++) {
                    int s_indx = grid->g2l(grid->O_by_rank[i][j]);
                    s_indx -= offset_to_set_O;
                    s_indx *= this->sol_dim; 
                    for (unsigned int d=0; d < this->sol_dim; d++) {
                        this->sbuf[k] = vec[s_indx+d];
                        k++; 
                    }
                }
            }
            tm["encode_send"]->stop();
        }

        void postIsends() {
            // Send
            int o_count = 0;
            for (int i = 0; i < this->size; i++) { 
                if (this->sendcounts[i] > 0) {
                    MPI_Isend(this->sbuf + this->sdispls[i], this->sendcounts[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &O_reqs[o_count]);
                    o_count++;
                }
            }

            // Barrier: wait for recvs to finish
            MPI_Waitall(r_comm_size, R_reqs, R_stats); 
            tm["irecv"]->stop();
        }

        void decodeRecvBuf(VCL_VEC_t &gpu_vec) {
            unsigned int set_Q_size = grid->Q_size;
            unsigned int set_R_size = grid->R_size;
            unsigned int nb_bnd = grid->getBoundaryIndicesSize();

            // OUR SOLUTION IS ARRANGED IN THIS FASHION:
            //  { Q\B B\O O R } where B = union(D, O) and Q = union(Q\B B\O O)

            // TODO: fix this. We have to maintain an additional index
            // map to convert from local node indices to the linear
            // system indices (i.e. filter off the dirichlet boundary
            // node indices
            unsigned int offset_to_interior = nb_bnd;
            unsigned int offset_to_set_R = set_Q_size;

            viennacl::vector_range< VCL_VEC_t > setR(gpu_vec, viennacl::range((offset_to_set_R-offset_to_interior) * sol_dim, ((offset_to_set_R-offset_to_interior)+set_R_size) * sol_dim));

            //double* vec = new double[set_R_size * sol_dim];
            std::vector<double> vec(set_R_size * sol_dim);

            // Post-Recv: copy elements out
            tm["decode_recv"]->start();
            unsigned int k = 0;
            for (size_t i = 0; i < grid->R_by_rank.size(); i++) {
                k = this->rdispls[i];
                for (size_t j = 0; j < grid->R_by_rank[i].size(); j++) {
                    unsigned int r_indx = grid->g2l(grid->R_by_rank[i][j]);
                    // Offset the r_indx to 0 and we'll copy the subset
                    // into the proper range
                    r_indx -= offset_to_set_R;

                    r_indx *= sol_dim;

                    // TODO: need to translate to local
                    // indexing properly. This hack assumes all
                    // boundary are dirichlet and appear first
                    // in the list
                    for (unsigned int d = 0; d < sol_dim; d++) {
                        vec[r_indx+d] = this->rbuf[k];
                        k++;
                    }
                }
            }

            tm["decode_recv"]->stop();
            viennacl::copy(vec, setR);
        }


        void postAlltoallv() {
            tm["alltoallv"]->start(); 
            MPI_Alltoallv(this->sbuf, this->sendcounts, this->sdispls, MPI_DOUBLE, this->rbuf, this->recvcounts, this->rdispls, MPI_DOUBLE, MPI_COMM_WORLD); 
            tm["alltoallv"]->stop(); 
        }
};

#endif
