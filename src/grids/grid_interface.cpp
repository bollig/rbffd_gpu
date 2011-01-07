#include <iostream> 
#include <fstream> 
#include <sstream>
#include <string>

#include "grid_interface.h"
#include "utils/random.h"

void Grid::generateGrid() {
	node_list.clear(); 
	boundary_indices.clear(); 
	boundary_normals.clear(); 

	node_list.resize(nb_nodes); 
}


void Grid::loadFromFile(std::string filename) {
	std::ifstream fin(filename.c_str());
       		
	node_list.clear(); 
	boundary_indices.clear(); 
	boundary_normals.clear();

	if (fin.is_open()) {
		while (fin.good()) {
			Node node; 
			fin >> node; 
			if (!fin.eof()) {
				node_list.push_back(node); 
			}
		}
	} else {
		perror ("Error opening file to write"); 
		exit(EXIT_FAILURE);
	}

	fin.close(); 
	nb_nodes = node_list.size(); 

	std::cout << "[Grid] \tLoaded " << nb_nodes << " nodes from \t" << filename << std::endl;
}


void Grid::writeToFile(std::string filename) {
	std::ofstream fout(filename.c_str()); 
	if (fout.is_open()) {
		for (unsigned int i = 0; i < node_list.size(); i++) {
			fout << node_list[i] << std::endl; 
		}
	} else {
		perror ("Error opening file to write"); 
		exit(EXIT_FAILURE); 
	}
	fout.close();
	std::cout << "[Grid] \tWrote " << node_list.size() << " nodes to \t" << filename << std::endl;
}


void Grid::sortNodes() {
    for (int i = 0; i < this->boundary_indices.size(); i++) {
        // We only need to roughly sort the nodes so the boundary is first and the
        // interior is second

        // Run through all boundary nodes. If the node is in the boundary set (which should be ordered),
        if (boundary_indices[i] != i) {
            // backup interior
            Node interior_node = node_list[i];
	    node_list[i] = node_list[boundary_indices[i]];
	    node_list[boundary_indices[i]] = interior_node;
            
	    // Update the boundary index into coords
            boundary_indices[i] = i;
	
   	    // Normals on boundary are numbered the same as boundary indices	    
	    // : no change. 
        }
    }
}



std::string Grid::getFullName(std::string base_filename, int iter) {
	std::stringstream ss(std::stringstream::out);
	// Setup filename for NetCDF file
	ss << base_filename << "_" << nb_nodes << "nodes_" << iter << "iters.ascii";  
	std::string filename = ss.str();
	return filename;
}


void Grid::perturbNodes(double perturb_amount) {
	pert = perturb_amount; 
	for (unsigned int i = 0 ; i < node_list.size(); i ++) {
		node_list[i][0] += randf(-pert, pert); 
		node_list[i][1] += randf(-pert, pert); 
		node_list[i][2] += randf(-pert, pert); 
	}
}
