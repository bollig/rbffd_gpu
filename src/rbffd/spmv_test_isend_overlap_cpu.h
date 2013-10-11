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

        int disable_timers;

    public: 
        SpMVTest(RBFFD* r, Domain* domain, int mpi_rank=0, int mpi_size=0) : rbffd(r), grid(domain), rank(mpi_rank), size(mpi_size), sol_dim(1), disable_timers(0) {
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

	void enableTimers() { disable_timers = 0; }
	void disableTimers() { disable_timers = 1; }

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
            tm["synchronize"] = new EB::Timer("[SpMVTest] Synchronize (Irecv->Waitall)"); 
            tm["spmv"] = new EB::Timer("[SpMVTest] perform SpMV (Q\\B)");
            tm["spmv2"] = new EB::Timer("[SpMVTest] perform SpMV (B)");
            tm["spmv_w_comm"] = new EB::Timer("[SpMVTest] SpMV + Communication");
            tm["irecv"] = new EB::Timer("[SpMVTest] MPI_Irecv"); 
            tm["encode_send"] = new EB::Timer("[SpMVTest] Encode sendbuf (collect all O_by_rank into one vector)"); 
            tm["encode_copy"] = new EB::Timer("[SpMVTest] Copy Device to Host (O)");
            tm["decode_recv"] = new EB::Timer("[SpMVTest] Decode recvbuf (scatter R_by_rank into memory)");
            tm["decode_copy"] = new EB::Timer("[SpMVTest] Copy Host to Device (R)");
            tm["alltoallv"] = new EB::Timer("[SpMVTest] MPI_Alltoallv Only (no memcpy)"); 
            tm["sendrecv_wait"] = new EB::Timer("[SpMVTest] Barrier before MPI_Isend"); 
            tm["allocateCommBufs"] = new EB::Timer("[SpMVTest] Allocate buffers for MPI_Isend/MPI_Irecv");
        }

        // Does a simple CPU CSR SpMV
        // can apply to subset of problem 
	
	// EVAN TOOD: finish overlap of CPU. 
        void SpMV(RBFFD::DerType which, std::vector<double>& u, std::vector<double>& out_deriv) {
            std::vector<double*> DM = rbffd->getWeights(which);
            std::vector<StencilType>& stencils = grid->getStencils();
            //int nb_nodes = grid->getNodeListSize();

	    int nb_stencils = grid->getStencilsSize();
	    int nb_nodes = grid->getNodeListSize();
	    int nb_qmb_rows = grid->QmB_size;

	    if (!disable_timers) tm["synchronize"]->start();
            //------------
            // Post irecv
            //------------
            if (size > 1) {
                // I found 8+ processors comm best with Isend/Irecv. Alltoallv
                // for < 8  on itasca. For cascade lets assume this unless we
                // see something horrible happen
                if (size > 1) { 
                    this->postIrecvs(); 
                }
                // Else we use Alltoallv and dont need to worry
            }

	    //------------
	    // Queue Q\B
	    //------------


            if (!disable_timers) tm["spmv"]->start();
	    this->encodeSendBuf(u);

            for (unsigned int i=0; i < nb_qmb_rows; i++) {
                double* w = DM[i];
                StencilType& st = stencils[i];
                double der = 0.0;
                unsigned int n = st.size();
                for (unsigned int s=0; s < n; s++) {
                    der += w[s] * u[st[s]];
                }
                out_deriv[i] = der;
            }
	    this->encodeSendBuf(u);

	    //------------
	    // Send O
	    //------------

	    if (size > 1) {
		    // I found 8+ processors comm best with Isend/Irecv. Alltoallv
                // for < 8 
                if (size > 1) { 
                    // NOTE: this includes waitall on irecvs
                    this->postIsends(); 
                } else {
                    this->postAlltoallv();
                }
            }
            if (!disable_timers) tm["synchronize"]->stop();


            this->decodeRecvBuf(u);

            if (!disable_timers) tm["spmv2"]->start();
	    for (unsigned int i=nb_qmb_rows; i < nb_stencils; i++) {
                double* w = DM[i];
                StencilType& st = stencils[i];
                double der = 0.0;
                unsigned int n = st.size();
                for (unsigned int s=0; s < n; s++) {
                    der += w[s] * u[st[s]];
                }
                out_deriv[i] = der;
            }
            if (!disable_timers) tm["spmv"]->stop();
            if (!disable_timers) tm["spmv2"]->stop();

            if (!disable_timers) tm["spmv_w_comm"]->stop();

        }


	void postIrecvs() {
		// Buffer recvs
		if (!disable_timers) tm["irecv"]->start();
		int r_count = 0;
		for (int i = 0; i < this->size; i++) { 
			if (this->recvcounts[i] > 0) {
				MPI_Irecv(this->rbuf + this->rdispls[i], this->recvcounts[i], MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &R_reqs[r_count]);
				r_count++;
			}
		}
	}

        void encodeSendBuf(std::vector<double>& cpu_vec) {
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

            if (!disable_timers) tm["encode_send"]->start();
            // Prep-Send: Copy elements of set to sbuf
            unsigned int k = 0; 
            for (size_t i = 0; i < grid->O_by_rank.size(); i++) {
                k = this->sdispls[i]; 
                for (size_t j = 0; j < grid->O_by_rank[i].size(); j++) {
#ifdef SOLDIM_GT_1
                    unsigned int s_indx = grid->g2l(grid->O_by_rank[i][j]);
                    s_indx *= this->sol_dim; 
                    for (unsigned int d=0; d < this->sol_dim; d++) {
                        this->sbuf[k] = cpu_vec[s_indx+d];
                        k++; 
                    }
#else 
                    int s_indx = grid->g2l(grid->O_by_rank[i][j]); 
                    this->sbuf[k] = cpu_vec[s_indx]; 
                    k++;
#endif 
                }
            }
            if (!disable_timers) tm["encode_send"]->stop();
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
            if (!disable_timers) tm["irecv"]->stop();
        }

        void decodeRecvBuf(std::vector<double>& cpu_vec) {
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


            if (!disable_timers) tm["decode_recv"]->start();
            // Post-Recv: copy elements out
            unsigned int k = 0;
            for (size_t i = 0; i < grid->R_by_rank.size(); i++) {
                k = this->rdispls[i]; 
                for (size_t j = 0; j < grid->R_by_rank[i].size(); j++) {
#ifdef SOLDIM_GT_1
                    unsigned int r_indx = grid->g2l(grid->R_by_rank[i][j]);
                    r_indx *= this->sol_dim;
                    //std::cout << "r_indx = " << r_indx << ", k = " << k << std::endl;
                    //                                    std::cout << "Receiving " << r_indx << "\n";
                    // TODO: need to translate to local
                    // indexing properly. This hack assumes all
                    // boundary are dirichlet and appear first
                    // in the list
                    for (unsigned int d=0; d < this->sol_dim; d++) { 
                        cpu_vec[r_indx+d] = this->rbuf[k];  
                        k++; 
                    }
#else 
                    int r_indx = grid->g2l(grid->R_by_rank[i][j]);
                    cpu_vec[r_indx] = this->rbuf[k];  
                    k++; 

#endif 
                }
            }
            if (!disable_timers) tm["decode_recv"]->stop();
        }


        void postAlltoallv() {
            if (!disable_timers) tm["alltoallv"]->start(); 
            MPI_Alltoallv(this->sbuf, this->sendcounts, this->sdispls, MPI_DOUBLE, this->rbuf, this->recvcounts, this->rdispls, MPI_DOUBLE, MPI_COMM_WORLD); 
            if (!disable_timers) tm["alltoallv"]->stop(); 
        }
};

#endif
