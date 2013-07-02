
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

		// Forms sets (Q,O,R) and l2g/g2l maps
		fillLocalData(grid_ptr->getNodeList(), grid_ptr->getStencils(), grid_ptr->getBoundaryIndices(), grid_ptr->getStencilRadii(), grid_ptr->getMaxStencilRadii(), grid_ptr->getMinStencilRadii()); 
		this->max_st_size = grid_ptr->getMaxStencilSize();
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
}; 
#endif 
