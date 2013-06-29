#ifndef _METISDomain_H_
#define _METISDomain_H_

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

#include "grids/domain.h"
#include "common_typedefs.h"

class METISDomain : public Domain
{

	public: 
		std::vector<int> metis_part; 

		METISDomain(int mpi_rank, int mpi_size, Grid* grid_ptr, string metis_part_filename) : Domain()
	{
		std::cout << "INSIDE METISDomain CONSTRUCTOR!\n";

		// Mimic setup for Domain
		dim_num=3;
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

		metis_part.reserve(this->global_num_nodes); 

		this->read_metis_file(metis_part_filename); 

		// Forms sets (Q,O,R) and l2g/g2l maps
		fillLocalData(grid_ptr->getNodeList(), grid_ptr->getStencils(), grid_ptr->getBoundaryIndices(), grid_ptr->getStencilRadii(), grid_ptr->getMaxStencilRadii(), grid_ptr->getMinStencilRadii()); 
		this->max_st_size = grid_ptr->getMaxStencilSize();
#endif 
	} 


		virtual bool isInsideSubdomain(NodeType& pt, int pt_indx) 
		{
			// 0 = Outside
			// 1 = Inside
			return (metis_part[pt_indx] == id); 
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
				exit(EXIT_FAILURE);
				return -1;
			}
			fin.close(); 
			return 0;
		}
}; 
#endif 
