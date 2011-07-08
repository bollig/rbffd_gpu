#include <stdlib.h>
#include <stdio.h>
#include <iostream> 
#include <fstream> 
#include <sstream>
#include <string>

#include "grid_reader.h"

using namespace std;


/*----------------------------------------------------------------------*/
GridReader::GridReader(std::string filename_to_read, unsigned int n_nodes_to_read)
    : Grid(n_nodes), 
    n_nodes(n_nodes_to_read),
    filename(filename_to_read),
    file_loaded(false)
{
    node_list.reserve(n_nodes);
    boundary_normals.clear();
    boundary_indices.clear();
}

/*----------------------------------------------------------------------*/
GridReader::~GridReader() {

}

/*----------------------------------------------------------------------*/
int GridReader::readNodeList(int expect_num_extra_dbls_per_line) {

    std::cout << "[" << this->className() << "] \treading file: " << filename << std::endl;
    std::ifstream fin(filename.c_str());
        
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

    nb_nodes = node_list.size(); 
    if (nb_nodes < n_nodes) {
        std::cout << "[" << this->className() << "] \tERROR: Found only " << i << " nodes \t" << filename << ". Check your configuration." << std::endl;
        exit(EXIT_FAILURE); 
        return -2; 
        
    } else {
        std::cout << "[" << this->className() << "] \tLoaded " << nb_nodes << " nodes from \t" << filename << std::endl;
    }
 
    // By default we dont try to load stencil files
    return 0;
}

/*----------------------------------------------------------------------*/
void GridReader::generate() {

    // Read 3 doubles per line and expect 1 extra (as junk for now);
    // FIXME: handle this better
    this->readNodeList(1); 
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


