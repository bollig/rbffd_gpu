#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <iostream> 
#include <fstream> 
#include <sstream>
#include <string>

#include "utils/io/realpathext.h"

#include "grid_reader.h"

using namespace std;


/*----------------------------------------------------------------------*/
GridReader::GridReader(std::string filename_to_read, int numCols, unsigned int n_nodes_to_read)
    : Grid(n_nodes), 
    numExtraCols(numCols - 3),
    n_nodes(n_nodes_to_read),
    filename(filename_to_read),
    file_loaded(false)
{
    node_list.clear();
    boundary_normals.clear();
    boundary_indices.clear();
}

/*----------------------------------------------------------------------*/
GridReader::~GridReader() {

}

/*----------------------------------------------------------------------*/
int GridReader::readNodeList(int expect_num_extra_dbls_per_line) {

    char fpath[PATH_MAX];
    realpathExt(filename.c_str(), fpath); 

    std::cout << "[" << this->className() << "] \treading file: " << fpath << std::endl;
    std::ifstream fin(fpath);
       
    if (expect_num_extra_dbls_per_line < 0) {
        std::cout << "ERROR: FILES MUST CONTAIN AT LEAST 3 COLUMNS PER NODE" <<  std::endl;
        exit(EXIT_FAILURE);
    }

    unsigned int i = 0; 

    if (fin.is_open()) {
        while (fin.good()) {
            NodeType node; 
            fin >> node; 
            // Parse off extra per line
            double junk; 
            for (int j = 0; j < expect_num_extra_dbls_per_line; j++) {
                fin >> junk; 
            }
            if (!fin.eof()) {
                if (i < n_nodes) {
                    node_list.push_back(node); 
                    i++; 
                } else {
                    // Kill the loop 
                    break; 
                }
            }
        }
    } else {
        printf("Error opening node file to read\n"); 
        exit(EXIT_FAILURE);
        return -1;
    }
    fin.close(); 

//    std::cout << "-1 = " << (unsigned int) -1 << "\t UINT_MAX = " << UINT_MAX << std::endl;
    nb_nodes = node_list.size(); 
    // If we have read in a number of nodes and if the number is less than desired, and our desired is NOT -1 (max unsigned int) 
    if ((n_nodes > 0) && (nb_nodes < n_nodes) && (n_nodes != (unsigned int)-1)) {
        std::cout << "[" << this->className() << "] \tERROR: Found only " << i << " nodes \t" << filename << ". Check your configuration." << std::endl;
        exit(EXIT_FAILURE); 
        return -2; 
        
    } else {
        std::cout << "[" << this->className() << "] \tLoaded " << nb_nodes << " nodes from \t" << filename << std::endl;
        n_nodes = nb_nodes;
        global_num_nodes = n_nodes; 
    }
 
    // By default we dont try to load stencil files
    return 0;
}

/*----------------------------------------------------------------------*/
void GridReader::generate() {

    // Read 3 doubles per line and expect 1 extra (as junk for now);
    // FIXME: handle this better
    this->readNodeList(numExtraCols); 
    /*
     * FIXME: for now we assume everything is boundary
     *
     if (this->bnd_file_specified) {
     this->readBoundaryList(); 
     } else {
     this->markAllAsBoundary(); 
     }

    //TODO: Sorting nodes could be done more intelligently than just putting the boundary nodes at the front of the list
    if (boundary_nodes_first) {
    this->sortNodes();
    }

    */
    this->refreshExtents();
}


std::string GridReader::getFileDetailString() {
    std::stringstream ss(std::stringstream::out);
    ss << n_nodes << "nodes";  
    return ss.str();
}


