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

class SpMVTest
{
    protected:
        RBFFD* rbffd; 
        Domain* grid; 
        int rank, size;

        unsigned int O_tot, R_tot;
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
        SpMVTest(RBFFD* r, Domain* domain, int mpi_rank=0, int mpi_size=0) : rbffd(r), grid(domain), rank(mpi_rank), size(mpi_size), sol_dim(1) {
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
            O_tot = sendcounts[0]; 

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
            R_tot = recvcounts[0];
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

            unsigned int *O_tot_all = NULL;
            if (rank == 0) {
                O_tot_all = new unsigned int[this->size*5];
            }
            unsigned int s[5];
            s[0] = O_tot; 
            s[1] = R_tot;
            s[2] = o_comm_size;
            s[3] = r_comm_size;
            s[4] = grid->Q_size; 
            MPI_Gather(s, 5, MPI_UNSIGNED, O_tot_all, 5, MPI_UNSIGNED, 0, MPI_COMM_WORLD); 

            if (rank ==0 ) {
                std::ofstream ftot("O_and_R_totals.log"); 
                ftot << "O_total, R_total, O_comm_size, R_comm_size, Q_size\n";
                for (int iii = 0; iii < this->size*5; iii+=5) 
                    ftot << O_tot_all[iii] << ", " << O_tot_all[iii+1] << ", " << O_tot_all[iii+2] << ", " << O_tot_all[iii+3] << ", " << O_tot_all[iii+4] <<   std::endl;
                ftot.close();
            }

        }



        void setupTimers() {
            tm["synchronize"] = new EB::Timer("[SpMVTest] Synchronize (MPI_Isend/MPI_Irecv");
            tm["spmv"] = new EB::Timer("[SpMVTest] perform SpMV");
            tm["irecv"] = new EB::Timer("[SpMVTest] MPI_Irecv"); 
            tm["pre_isend"] = new EB::Timer("[SpMVTest] Memcpy input to MPI_Isend"); 
            tm["post_irecv"] = new EB::Timer("[SpMVTest] Memcpy output from MPI_Irecv");
            tm["alltoallv"] = new EB::Timer("[SpMVTest] MPI_Alltoallv Only (no memcpy)"); 
            tm["pre_alltoallv"] = new EB::Timer("[SpMVTest] Memcpy input to MPI_Alltoallv"); 
            tm["post_alltoallv"] = new EB::Timer("[SpMVTest] Memcpy output from Alltoallv");
            tm["sendrecv_wait"] = new EB::Timer("[SpMVTest] Barrier before MPI_Isend"); 
            tm["allocateCommBufs"] = new EB::Timer("[SpMVTest] Allocate buffers for MPI_Isend/MPI_Irecv");
        }

        // Does a simple CPU CSR SpMV
        // can apply to subset of problem 
        void SpMV(RBFFD::DerType which, std::vector<double>& u, std::vector<double>& out_deriv) {
            tm["spmv"]->start();
            std::vector<double*> DM = rbffd->getWeights(which);
            std::vector<StencilType>& stencils = grid->getStencils();
            //int nb_nodes = grid->getNodeListSize();
            int nb_stencils = grid->getStencilsSize();

            for (unsigned int i=0; i < nb_stencils; i++) {
                double* w = DM[i];
                StencilType& st = stencils[i];
                double der = 0.0;
                unsigned int n = st.size();
                for (unsigned int s=0; s < n; s++) {
                    der += w[s] * u[st[s]];
                }
                out_deriv[i] = der;
            }
            tm["spmv"]->stop();

        }

        // Perform sendrecv
        void synchronize(std::vector<double>& vec) {
            // Share data in vector with all other processors. Will only transfer
            // data associated with nodes in overlap between domains. 
            // Uses MPI_Alltoallv and MPI_Barrier. 
            // Copies data from vec to transfer, then writes received data into vec
            // before returning. 

            tm["synchronize"]->start();
            if (size > 1) {
                // I found 8+ processors comm best with Isend/Irecv. Alltoallv
                // for < 8 
                if (size > 16) { 
                    // This is equivalent to: 
                    // 
                    // MPI_Alltoallv(this->sbuf, this->sendcounts, this->sdispls, MPI_DOUBLE, this->rbuf, this->recvcounts, this->rdispls, MPI_DOUBLE, MPI_COMM_WORLD); 
                    //

                    // Buffer recvs
                    tm["irecv"]->start();
                    int r_count = 0;
                    for (int i = 0; i < this->size; i++) { 
                        if (this->recvcounts[i] > 0) {
                            MPI_Irecv(this->rbuf + this->rdispls[i], this->recvcounts[i], MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &R_reqs[r_count]);
                            r_count++;
                        }
                    }


                    tm["pre_isend"]->start();
                    // Prep-Send: Copy elements of set to sbuf
                    unsigned int k = 0; 
                    for (size_t i = 0; i < grid->O_by_rank.size(); i++) {
                        k = this->sdispls[i]; 
                        for (size_t j = 0; j < grid->O_by_rank[i].size(); j++) {
                            unsigned int s_indx = grid->g2l(grid->O_by_rank[i][j]);
                            s_indx *= this->sol_dim; 
                            for (unsigned int d=0; d < this->sol_dim; d++) {
                                this->sbuf[k] = vec[s_indx+d];
                                k++; 
                            }
                        }
                    }
                    tm["pre_isend"]->stop();

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

                    // Post-Recv: copy elements out
                    tm["post_irecv"]->start();
                    k = 0; 
                    for (size_t i = 0; i < grid->R_by_rank.size(); i++) {
                        k = this->rdispls[i]; 
                        for (size_t j = 0; j < grid->R_by_rank[i].size(); j++) {
                            unsigned int r_indx = grid->g2l(grid->R_by_rank[i][j]);
                            r_indx *= this->sol_dim;
                            //std::cout << "r_indx = " << r_indx << ", k = " << k << std::endl;
                            //                                    std::cout << "Receiving " << r_indx << "\n";
                            // TODO: need to translate to local
                            // indexing properly. This hack assumes all
                            // boundary are dirichlet and appear first
                            // in the list
                            for (unsigned int d=0; d < this->sol_dim; d++) { 
                                vec[r_indx+d] = this->rbuf[k];  
                                k++; 
                            }
                        }
                    }

                    tm["post_irecv"]->stop();
                } else {
                    // I found that <= 8 procs we have better comm patterns with
                    // MPI_Alltoallv
                    tm["pre_alltoallv"]->start();
                    // TODO: the barrier can happen after this memcpy that preceeds the Alltoall 
                    // Copy elements of set to sbuf
                    unsigned int k = 0; 
                    for (size_t i = 0; i < grid->O_by_rank.size(); i++) {
                        k = this->sdispls[i]; 
                        for (size_t j = 0; j < grid->O_by_rank[i].size(); j++) {
                            unsigned int s_indx = grid->g2l(grid->O_by_rank[i][j]);
                            s_indx *= this->sol_dim; 
                            for (unsigned int d=0; d < this->sol_dim; d++) {
                                this->sbuf[k] = vec[s_indx+d];
                                k++; 
                            }
                        }
                    }
                    tm["pre_alltoallv"]->stop();

                    tm["alltoallv"]->start(); 
                    MPI_Alltoallv(this->sbuf, this->sendcounts, this->sdispls, MPI_DOUBLE, this->rbuf, this->recvcounts, this->rdispls, MPI_DOUBLE, MPI_COMM_WORLD); 
                    tm["alltoallv"]->stop(); 

                    tm["post_alltoallv"]->start();
                    // IF we need this barrier then our results will vary as #proc increases
                    // but I dont think alltoall requires a barrier. internally
                    // it can be isend/irecv but there should be a barrier
                    // internally
                    //comm_ref.barrier();

                    k = 0; 
                    for (size_t i = 0; i < grid->R_by_rank.size(); i++) {
                        k = this->rdispls[i]; 
                        for (size_t j = 0; j < grid->R_by_rank[i].size(); j++) {
                            unsigned int r_indx = grid->g2l(grid->R_by_rank[i][j]);
                            r_indx *= this->sol_dim;
                            //std::cout << "r_indx = " << r_indx << ", k = " << k << std::endl;
                            //                                    std::cout << "Receiving " << r_indx << "\n";
                            // TODO: need to translate to local
                            // indexing properly. This hack assumes all
                            // boundary are dirichlet and appear first
                            // in the list
                            for (unsigned int d=0; d < this->sol_dim; d++) { 
                                vec[r_indx+d] = this->rbuf[k];  
                                k++; 
                            }
                        }
                    }

                    tm["post_alltoallv"]->stop();
                }
            }
            tm["synchronize"]->stop();
        }
};

#endif
