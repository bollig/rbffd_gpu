
#ifndef _METISDomain_H_
#define _METISDomain_H_

#define FAIL_ON_MISSING_PARTFILE 0

#include "grids/domain.h"
#include "common_typedefs.h"

#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <iostream> 
#include <fstream> 
#include <sstream>
#include <string>
#include <string.h>


#include <vector>
#include <set>
#include <map>


class METISDomain : public Domain
{
	protected:
		bool part_domain; 

	public: 
		std::vector<int> metis_part; 

        // When is this used? 
		METISDomain(int mpi_rank, int mpi_size, int dim, int max_stencil_size) : Domain(), part_domain(false) { 
			std::cout << "INSIDE METISDomain CONSTRUCTOR!" << mpi_size << "\n";

			max_st_size = max_stencil_size;	
			dim_num = dim;

			// Mimic setup for Domain
			id=mpi_rank;
			comm_size=mpi_size;
		}

		// Need to construct: 
		// 	Load grid from domain_rank0_of_1.ascii
		// 	Need to load stencils into domain ? Need to write domain stencils to disk
		// 		NOTE: if loading we can use grid writer routines; might
		// 		need to load grid content into the domain class instead
		// 		of just wrapping it as a proxy

		METISDomain(int mpi_rank, int mpi_size, Grid* grid_ptr, int dim, string metis_part_filename, bool part_file_loaded = true) : 
			Domain(), part_domain(part_file_loaded) 
	{
		std::cout << "INSIDE METISDomain CONSTRUCTOR!\n";

		// Mimic setup for Domain
		dim_num = dim;
		id=mpi_rank;
		comm_size=mpi_size;

		xmin = grid_ptr->xmin;
		xmax = grid_ptr->xmax;
		ymin = grid_ptr->ymin; 
		ymax = grid_ptr->ymax;
		zmin = grid_ptr->zmin;
		zmax = grid_ptr->zmax;

#if 1
		// We might need to know how many nodes are in the domain globally for things like Hyperviscosity
		this->global_num_nodes = grid_ptr->getNodeListSize();

		if (part_domain) {
			metis_part.reserve(this->global_num_nodes); 

			this->read_metis_file(metis_part_filename); 
		}

		this->max_st_size = grid_ptr->getMaxStencilSize();

        this->fillCenterSets(grid_ptr->getNodeList(), grid_ptr->getStencils());

        // R_by_rank must be full before filling local data. We sort our R
        // indices according to R_by_rank. 
        fill_R_by_rank(); 

		// Forms sets (Q,O,R) and l2g/g2l maps
		fillLocalData(grid_ptr->getNodeList(), grid_ptr->getStencils(), grid_ptr->getBoundaryIndices(), grid_ptr->getStencilRadii(), grid_ptr->getMaxStencilRadii(), grid_ptr->getMinStencilRadii()); 

        // Must happen AFTER R_by_rank is full
        fill_O_by_rank(); 


        // TODO: fill O_by_rank
        //      -- Need to identify R for each of the other subdomains: 
        //              a) MPI_Alltoallv to tell each processor of need
        //              (requires building R_by_rank first, then we scatter the
        //              info)

#endif 
	} 


		virtual bool isInsideSubdomain(NodeType& pt, int pt_indx) 
		{
			// Only partition if the a part file is provided. 
			if (part_domain) {
				// 0 = Outside
				// 1 = Inside
				return (metis_part[pt_indx] == id); 
			} 
			return 1;
		}

		int read_metis_file(string filename) {

			char fpath[PATH_MAX];
			char *home;
			const char *path = filename.c_str(); 
			if (*path=='~' && (home = getenv("HOME"))) {
				char s[PATH_MAX];
				realpath(strcat(strcpy(s, home), path+1), fpath);
			} else {
				realpath(filename.c_str(), fpath);
			}

			std::cout << "\treading file: " << fpath << std::endl;
			std::ifstream fin(fpath);

			if (fin.is_open()) {
				while (fin.good()) {
					int part_indx; 
					fin >> part_indx; 
					if (!fin.eof()) {
						metis_part.push_back(part_indx); 
					}
				}
			} else {
				printf("Error opening node file to read\n"); 
#if FAIL_ON_MISSING_PARTFILE
				exit(EXIT_FAILURE);
#else 
                printf("Assuming one processor: %d\n", this->global_num_nodes);
                for (int pp = 0; pp < this->global_num_nodes; pp++) {
                    metis_part.push_back(0);
                }
#endif 
				return -1;
			}
			fin.close(); 
			return 0;
		}

        // Iterate all nodes > nb_stencils (since they're all permuted to the
        // end)
        // If node
        void fill_R_by_rank() {
            int nb_stencils = this->getStencilsSize(); 
            int nb_nodes = this->getNodeListSize(); 

#if 0
            //std::cout << "START: " << nb_stencils << "\n";
            //std::cout << "END : " << nb_nodes << "\n";
            R_by_rank.clear(); 
            R_by_rank.resize(comm_size); 

            for (int i = nb_stencils ; i < nb_nodes; i++) { 
                int rank = metis_part[l2g(i)];
                int indx = l2g(i);
                
                // l2g(i) => global index
                // metis_part[l2g(i)] => rank
                R_by_rank[rank].push_back(indx); 
            }
#else 
            //std::cout << "START: " << nb_stencils << "\n";
            //std::cout << "END : " << nb_nodes << "\n";
            R_by_rank.clear(); 
            R_by_rank.resize(comm_size); 

            std::set<int>::iterator rit; 
            for (rit = this->R.begin(); rit != this->R.end(); rit++) {
                int rank = metis_part[*rit];
                int indx = *rit;
                
                // l2g(i) => global index
                // metis_part[l2g(i)] => rank
                R_by_rank[rank].push_back(indx); 
            }
#endif 
            
            isRfull = 1;
        } 

        // Iterate all nodes > nb_stencils (since they're all permuted to the
        // end)
        // If node
        void fill_O_by_rank() {
            int nb_stencils = this->getStencilsSize(); 
            int nb_nodes = this->getNodeListSize(); 

            // First we have each ranks send notification of the number of nodes
            // in R_by_rank for each of the other ranks. 
            int my_r_sizes[comm_size]; 

            for (int i = 0; i < comm_size; i++) { 
                my_r_sizes[i] = R_by_rank[i].size(); 
            }

            int their_r_sizes[comm_size]; 

            // Get the counts required by each of the processes first
            MPI_Alltoall(my_r_sizes, 1, MPI_INT, their_r_sizes, 1, MPI_INT, MPI_COMM_WORLD);

#if 0
            if (id == 0) { 
                for (int i = 0; i < comm_size; i++) {
                    std::cout << "Rank[" << i << "] needs " << their_r_sizes[i] << std::endl;
                }
            }
#endif 

            int sdispls[comm_size];
            int rdispls[comm_size];

            sdispls[0] = 0;

            unsigned int O_tot = my_r_sizes[0]; 
            for (int i = 1; i < comm_size; i++) {
                sdispls[i] = sdispls[i-1] + my_r_sizes[i-1];
                O_tot += my_r_sizes[i]; 
            }
            
            rdispls[0] = 0; 
            unsigned int R_tot = their_r_sizes[0];
            for (size_t i = 1; i < comm_size; i++) {
                rdispls[i] = rdispls[i-1] + their_r_sizes[i-1];
                R_tot += their_r_sizes[i]; 
            }

#if 0
            std::cout << "sending = " << O_tot << std::endl;
            std::cout << "receiving = " << R_tot << std::endl;
#endif

            int sendbuf[O_tot]; 
            int recvbuf[R_tot]; 

            std::vector< std::vector<int> >::iterator mit;
            std::vector<int>::iterator nit;
            int k = 0;
            int rbr_count = 0; 
            for (mit = R_by_rank.begin(); mit != R_by_rank.end(); mit++, k++) {
                int j = 0; 
                for (nit = (*mit).begin(); nit != (*mit).end(); nit++, j++, rbr_count++) {
                    sendbuf[rbr_count] = R_by_rank[k][j]; 
                }
            }
            if (rbr_count != O_tot) {
                std::cout << "ERROR: O_tot: " << O_tot << ", rbr_count: " << rbr_count << "\n"; 
                exit(-1);
            }

            // Send and recv all indices in global terms
            MPI_Alltoallv(sendbuf, my_r_sizes, sdispls, MPI_INT, recvbuf, their_r_sizes, rdispls, MPI_INT, MPI_COMM_WORLD);

            O_by_rank.resize(comm_size); 
            for (int i = 0; i < comm_size; i++) { 
                O_by_rank[i].resize(their_r_sizes[i]); 
                for (int j = 0 ; j < their_r_sizes[i]; j++) { 
                    O_by_rank[i][j] = recvbuf[rdispls[i]+j]; 
                }
            } 
        } 
}; 
#endif 
