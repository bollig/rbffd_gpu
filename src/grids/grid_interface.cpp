#include <iostream> 
#include <fstream> 
#include <sstream>
#include <string>

#include "grid_interface.h"
#include "utils/random.h"

void Grid::generate() {
	node_list.clear(); 
	boundary_indices.clear(); 
	boundary_normals.clear(); 

	node_list.resize(nb_nodes); 
}

void Grid::generateStencils(StencilGenerator* stencil_generator) {
	this->stencil_map.resize(node_list.size());
	this->avg_stencil_radii.resize(node_list.size()); 
	
	// TODO: generate stencils for a *** SUBSET *** of nodes
	stencil_generator->computeStencils(this->node_list, this->boundary_indices, this->stencil_map, this->avg_stencil_radii);
}



void Grid::writeToFile() {
	this->writeToFile(this->getFilename());
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



void Grid::loadFromFile(int iter) {
	this->loadFromFile(this->getFilename(iter)); 
}



void Grid::loadFromFile(std::string filename) {
	std::ifstream fin(filename.c_str());
       		
	node_list.clear(); 
	boundary_indices.clear(); 
	boundary_normals.clear();

	if (fin.is_open()) {
		while (fin.good()) {
			NodeType node; 
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


void Grid::sortNodes() {
    for (int i = 0; i < this->boundary_indices.size(); i++) {
        // We only need to roughly sort the nodes so the boundary is first and the
        // interior is second

        // Run through all boundary nodes. If the node is in the boundary set (which should be ordered),
        if (boundary_indices[i] != i) {
            // backup interior
            NodeType interior_node = node_list[i];
	    node_list[i] = node_list[boundary_indices[i]];
	    node_list[boundary_indices[i]] = interior_node;
            
	    // Update the boundary index into coords
            boundary_indices[i] = i;
	
   	    // Normals on boundary are numbered the same as boundary indices	    
	    // : no change. 
        }
    }
}


std::string Grid::getFileDetailString() {
	std::stringstream ss(std::stringstream::out); 
	ss << nb_nodes << "nodes"; 
	return ss.str();
}

std::string Grid::getFilename(std::string base_filename, int iter) {
	std::stringstream ss(std::stringstream::out);
	if (iter < 0) {
		ss << base_filename << "_" << this->getFileDetailString() << "_final.ascii";  
	} else if (iter == 0) {
		ss << base_filename << "_" << this->getFileDetailString() << "_initial.ascii";  
	} else {
		ss << base_filename << "_" << this->getFileDetailString() << "_" << iter << "iters.ascii";  
	}
	std::string filename = ss.str();
	return filename;
}

std::string Grid::getFilename(int iter) {
	return this->getFilename(this->className(), iter); 
}


void Grid::perturbNodes(double perturb_amount) {
	pert = perturb_amount; 
	for (unsigned int i = 0 ; i < node_list.size(); i ++) {
		node_list[i][0] += randf(-pert, pert); 
		node_list[i][1] += randf(-pert, pert); 
		node_list[i][2] += randf(-pert, pert); 
	}
}
