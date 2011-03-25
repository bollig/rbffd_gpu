#include <iostream> 
#include <fstream> 
#include <sstream>
#include <string>

#include "grid_interface.h"
#include "utils/random.h"
#include "mmio.h"

//----------------------------------------------------------------------------
void Grid::generate() {
    node_list.clear(); 
    boundary_indices.clear(); 
    boundary_normals.clear(); 

    node_list.resize(nb_nodes); 
}
//----------------------------------------------------------------------------

void Grid::generateStencils(StencilGenerator* stencil_generator) {
    this->stencil_map.resize(node_list.size());
    this->avg_stencil_radii.resize(node_list.size()); 

    // TODO: generate stencils for a *** SUBSET *** of nodes
    stencil_generator->computeStencils(this->node_list, this->boundary_indices, this->stencil_map, this->max_st_size, this->avg_stencil_radii);
}

//----------------------------------------------------------------------------


void Grid::writeToFile() {
    this->writeToFile(this->getFilename());
}
//----------------------------------------------------------------------------

void Grid::writeToFile(std::string filename) {
    std::ofstream fout(filename.c_str()); 
    if (fout.is_open()) {
        for (size_t i = 0; i < node_list.size(); i++) {
            fout << node_list[i] << std::endl; 
        }
    } else {
        printf("Error opening file to write\n"); 
        exit(EXIT_FAILURE); 
    }
    fout.close();
    //std::cout << "[Grid] \tWrote " << node_list.size() << " nodes to \t" << filename << std::endl;
    std::cout << "[" << this->className() << "] \tWrote " << node_list.size() << " nodes to \t" << filename << std::endl;

    this->writeBoundaryToFile(filename); 
    this->writeNormalsToFile(filename); 
    this->writeAvgRadiiToFile(filename); 
    this->writeStencilsToFile(filename); 
    this->writeExtraToFile(filename); 
}


//----------------------------------------------------------------------------

void Grid::writeBoundaryToFile(std::string filename) {
    std::string fname = "bndry_"; 
    fname.append(filename); 
    std::ofstream fout(fname.c_str()); 
    if (fout.is_open()) {
        for (size_t i = 0; i < boundary_indices.size(); i++) {
            fout << boundary_indices[i] << std::endl; 
        }
    } else {
        printf("Error opening file to write\n"); 
        exit(EXIT_FAILURE); 
    }
    fout.close();
    std::cout << "[" << this->className() << "] \tWrote " << boundary_indices.size() << " boundary indices to \t" << fname << std::endl;
}

//----------------------------------------------------------------------------

void Grid::writeNormalsToFile(std::string filename) {
    std::string fname = "nrmls_"; 
    fname.append(filename); 
    std::ofstream fout(fname.c_str()); 
    if (fout.is_open()) {
        for (size_t i = 0; i < boundary_normals.size(); i++) {
            fout << boundary_normals[i] << std::endl; 
        }
    } else {
        printf("Error opening file to write\n"); 
        exit(EXIT_FAILURE); 
    }
    fout.close();
    std::cout << "[" << this->className() << "] \tWrote " << boundary_normals.size() << " boundary normals to \t" << fname << std::endl;
}

//----------------------------------------------------------------------------

void Grid::writeAvgRadiiToFile(std::string filename) {
    std::string fname = "avg_radii_"; 
    fname.append(filename); 
    std::ofstream fout(fname.c_str()); 
    if (fout.is_open()) {
        for (size_t i = 0; i < avg_stencil_radii.size(); i++) {
            fout << avg_stencil_radii[i] << std::endl; 
        }
    } else {
        printf("Error opening file to write\n"); 
        exit(EXIT_FAILURE); 
    }
    fout.close();
    std::cout << "[" << this->className() << "] \tWrote " << avg_stencil_radii.size() << " average stencil radii to \t" << fname << std::endl;
}

//----------------------------------------------------------------------------

void Grid::writeStencilsToFile(std::string filename) {
    if (max_st_size > 0) {
        std::ostringstream prefix; 
        prefix << "stencils_maxsz" << this->max_st_size << "_";

        std::string fname = prefix.str(); 
        fname.append(filename); 
        std::ofstream fout(fname.c_str()); 
        if (fout.is_open()) {
            for (size_t i = 0; i < stencil_map.size(); i++) {
                fout << stencil_map[i].size(); 
                for (size_t j=0; j < stencil_map[i].size(); j++) {
                    fout << " " << stencil_map[i][j];
                }
                fout << std::endl;
            }
        } else {
            printf("Error opening file to write\n"); 
            exit(EXIT_FAILURE); 
        }
        fout.close();
        std::cout << "[" << this->className() << "] \tWrote " << stencil_map.size() << " stencils to \t" << fname << std::endl;
    } else {
        std::cout << "[" << this->className() << "] \tMax stencil size not set. No stencils to write to disk" << std::endl;
    }
}

//----------------------------------------------------------------------------


void Grid::writeExtraToFile(std::string filename) {
    std::cout << "[" << this->className() << "] \tNothing extra to write" << std::endl;
}

//----------------------------------------------------------------------------



int Grid::loadFromFile(int iter) {
    return this->loadFromFile(this->getFilename(iter)); 
}


//----------------------------------------------------------------------------

int Grid::loadFromFile(std::string filename) {
    std::cout << "[" << this->className() << "] \treading file: " << filename << std::endl;
    std::ifstream fin(filename.c_str());

    node_list.clear(); 

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
        printf("Error opening node file to read\n"); 
        return -1;
        //		exit(EXIT_FAILURE);
    }
    fin.close(); 
    nb_nodes = node_list.size(); 
    std::cout << "[" << this->className() << "] \tLoaded " << nb_nodes << " nodes from \t" << filename << std::endl;

    if (this->loadBoundaryFromFile(filename)) {
        printf("Error loading boundary nodes\n"); 
        return -2;
    }

    if (this->loadNormalsFromFile(filename)) {
        printf("Error loading normals\n"); 
        return -3;
    }

    if (this->loadAvgRadiiFromFile(filename)) {
        printf("Error loading avg dists\n"); 
        return -4;
    }

    if (this->loadStencilsFromFile(filename)) {
        printf("Error loading stencils\n"); 
        return -4;
    }

    if (this->loadExtraFromFile(filename)) {
        printf("Error loading additional data\n"); 
        return -5;
    }

    return 0;
}

//----------------------------------------------------------------------------

int Grid::loadBoundaryFromFile(std::string filename) {
    std::string fname = "bndry_"; 
    fname.append(filename);
    std::cout << "[" << this->className() << "] \treading boundary file: " << fname << std::endl;    

    std::ifstream fin; 
    fin.open(fname.c_str()); 

    if (fin.is_open()) {
        boundary_indices.clear(); 
        while (fin.good()) {
            size_t indx; 
            fin >> indx; 
            if (!fin.eof()) {
                boundary_indices.push_back(indx); 
            }
        }
    } else {
        printf("Error opening boundary file to read\n"); 
        return -1;
        //		exit(EXIT_FAILURE);
    }

    fin.close(); 

    std::cout << "[" << this->className() << "] \tLoaded " << boundary_indices.size() << " boundary indices from \t" << fname << std::endl;
    return 0; 
}

//----------------------------------------------------------------------------

int Grid::loadNormalsFromFile(std::string filename) {
    std::string fname = "nrmls_"; 
    fname.append(filename);
    std::cout << "[" << this->className() << "] \treading normals file: " << fname << std::endl;    

    std::ifstream fin; 
    fin.open(fname.c_str()); 

    if (fin.is_open()) {
        boundary_normals.clear(); 
        while (fin.good()) {
            Vec3 norml; 
            fin >> norml; 
            if (!fin.eof()) {
                boundary_normals.push_back(norml); 
            }
        }
    } else {
        printf("Error opening normals file to read\n"); 
        return -1;
        //		exit(EXIT_FAILURE);
    }

    fin.close(); 

    std::cout << "[" << this->className() << "] \tLoaded " << boundary_normals.size() << " boundary normals from \t" << fname << std::endl;
    return 0; 
}

//----------------------------------------------------------------------------

int Grid::loadAvgRadiiFromFile(std::string filename) {
    std::string fname = "avg_radii_"; 
    fname.append(filename);
    std::cout << "[" << this->className() << "] \treading average stencil radii file: " << fname << std::endl;    

    std::ifstream fin; 
    fin.open(fname.c_str()); 

    if (fin.is_open()) {
        avg_stencil_radii.clear(); 
        while (fin.good()) {
            double rad; 
            fin >> rad; 
            if (!fin.eof()) {
                avg_stencil_radii.push_back(rad); 
            }
        }
    } else {
        printf("Error opening average stencil radii file to read\n"); 
        return -1;
        //		exit(EXIT_FAILURE);
    }

    fin.close(); 

    std::cout << "[" << this->className() << "] \tLoaded " << avg_stencil_radii.size() << " average stencil radii from \t" << fname << std::endl;
    return 0; 
}

//----------------------------------------------------------------------------

int Grid::loadStencilsFromFile(std::string filename) {
    if (max_st_size > 0) {
        std::ostringstream prefix; 
        prefix << "stencils_maxsz" << this->max_st_size << "_" << filename; 

        std::string fname = prefix.str();

        std::cout << "[" << this->className() << "] \treading stencil file: " << fname << std::endl;    

        std::ifstream fin; 
        fin.open(fname.c_str()); 

        size_t num_el_loaded = 0; 
        if (fin.is_open()) {
            stencil_map.clear(); 
            while (fin.good()) {
                size_t st_size; 
                fin >> st_size; 
                StencilType st; 
                for (int i = 0; i < st_size; i++) {
                    int st_el; 
                    fin >> st_el; 
                    st.push_back(st_el); 
                }
                if (!fin.eof()) {
                    stencil_map.push_back(st); 
                    num_el_loaded += st.size();
                }
            }
        } else {
            printf("Error opening stencil file to read\n"); 
            return -1;
            //		exit(EXIT_FAILURE);
        }

        fin.close(); 

        std::cout << "[" << this->className() << "] \tLoaded " << stencil_map.size() << " stencils, with a total of " << num_el_loaded << " elements from \t" << fname << std::endl;
    } else {
        std::cout << "[" << this->className() << "] \tMax stencil size not set. No stencils to read from disk" << std::endl;
    }
    return 0;
}

//----------------------------------------------------------------------------

int Grid::loadExtraFromFile(std::string filename) {
    std::cout << "No extra loads from disk required." << std::endl;
    return 0; 
}

//----------------------------------------------------------------------------

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


//----------------------------------------------------------------------------
std::string Grid::getFileDetailString() {
    std::stringstream ss(std::stringstream::out); 
    ss << nb_nodes << "nodes"; 
    return ss.str();
}

//----------------------------------------------------------------------------
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

//----------------------------------------------------------------------------
std::string Grid::getFilename(int iter) {
    return this->getFilename(this->className(), iter); 
}


//----------------------------------------------------------------------------
void Grid::perturbNodes(double perturb_amount) {
    pert = perturb_amount; 
    for (size_t i = 0 ; i < node_list.size(); i ++) {
        node_list[i][0] += randf(-pert, pert); 
        node_list[i][1] += randf(-pert, pert); 
        node_list[i][2] += randf(-pert, pert); 
    }
}
//----------------------------------------------------------------------------
void Grid::printNodeList(std::string label) {
    std::cout << label << " (NodeList) = " << std::endl;
    std::vector<NodeType>::iterator i; 
    for (i = node_list.begin(); i != node_list.end(); i++) {
        std::cout << "(" << (*i)[0] << ")" << std::endl;
    }
}
//----------------------------------------------------------------------------
void Grid::printBoundaryIndices(std::string label) {
    std::cout << label << " (BoundaryIndices) = " << std::endl;
    std::vector<size_t>::iterator i; 
    for (i = boundary_indices.begin(); i != boundary_indices.end(); i++) {
        std::cout << (*i) << std::endl;
    }
}
