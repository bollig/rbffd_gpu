#include <iostream> 
#include <fstream> 
#include <sstream>
#include <string>
#include <set>

#include "grid_interface.h"
#include "utils/random.h"
#include "mmio.h"

// ltvec declared in grid_interface.h
NodeType ltvec::xi;
vector<NodeType>* ltvec::rbf_centers;

//----------------------------------------------------------------------------
void Grid::generate() {
    node_list.clear(); 
    boundary_indices.clear(); 
    boundary_normals.clear(); 

    node_list.resize(nb_nodes); 
}
//----------------------------------------------------------------------------


void Grid::writeToFile(int iter) {
    this->writeToFile(this->getFilename(iter));
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
//----------------------------------------------------------------------------
void Grid::generateStencils(size_t st_max_size, st_generator_t generator_choice)
{
    max_st_size = st_max_size; 
    generateStencils(generator_choice); 
}

//----------------------------------------------------------------------------

    void Grid::generateStencils(st_generator_t generator_choice) {
        switch (generator_choice)
        {
            case ST_BRUTE_FORCE:
                this->generateStencilsBruteForce(); 
                break; 
            case ST_KDTREE: 
                this->generateStencilsKDTree(); 
                break; 
            case ST_HASH: 
                this->generateStencilsHash(); 
                break; 
            default: 
                std::cout << "ERROR! Invalid choice of stencil generator\n"; 
                exit(EXIT_FAILURE); 
        };
    }

//----------------------------------------------------------------------------
void Grid::computeStencilRadii() {
    // NOTE: this currently assumes that nodes are sorted [boundary; interior]
    this->avg_stencil_radii.resize(node_list.size()); 
    this->min_stencil_radii.resize(node_list.size()); 
    this->max_stencil_radii.resize(node_list.size()); 

    int nb_rbf = node_list.size();
    int nb_bnd = boundary_indices.size();
    std::vector<NodeType>& rbf_centers = node_list;

    vector<double> avg_bnd;
    vector<double> avg_int;

    avg_bnd.resize(nb_bnd);
    avg_int.resize(nb_rbf - nb_bnd);

    // O(n^2) algorithm, whose cost is independent of the number of nearest
    // sought
    for (int i = 0; i < nb_rbf; i++) {
        StencilType& st = stencil_map[i];

        //        std::cout << "st.size() = " << st.size() << std::endl;

        double dmin = (rbf_centers[st[1]] - rbf_centers[st[0]]).square(); 
        min_stencil_radii[i] = sqrt(dmin); 

        double dmax = (rbf_centers[st[st.size()-1]] - rbf_centers[st[0]]).square(); 
        max_stencil_radii[i] = sqrt(dmax); 

        //        std::cout << "st.max_dist = " << max_stencil_radii[i] << std::endl;

        avg_stencil_radii[i] = 0.;
        if (i < nb_bnd) {
            avg_bnd[i] = 0.;
        } else {
            avg_int[i - nb_bnd] = 0.;
        }

        // Now iterate over the ith stencil and query distance to neighbors
        for (int k = 1; k < st.size(); k++) {
            double d = (rbf_centers[st[k]] - rbf_centers[st[0]]).square();
            double ss = sqrt(d);

            avg_stencil_radii[i] += ss;
            if (i < nb_bnd) {
                avg_bnd[i] += ss;
            } else {
                avg_int[i - nb_bnd] += ss;
            }
            // printf("(%d, %d) dist= %f\n", i, k, ss);
        }

#if 1
        avg_stencil_radii[i] /= (st.size() - 1.); // -1. to ignore center point

        // This is not quite right. we are taking the sum of the boundary
        // nodes divided by the total number of nodes in the stencil
        if (i < nb_bnd) {
            avg_bnd[i] /= (st.size() - 1.);
        } else {
            avg_int[i - nb_bnd] /= (st.size() - 1.);
        }
#endif
        //printf("avg_dist[%d]= %f\n", i, avg_stencil_radii[i]);
        //printf("nb points in stencil: %d\n", (int) st.size());
    }

    // Global averages across ALL stencils
    double avgint = 0.;
    double avgbnd = 0.;

    printf("[Grid] avg_int.size() = %d\n", (int) avg_int.size());
    printf("[Grid] avg_bnd.size() = %d\n", (int) avg_bnd.size());

    if (nb_rbf - nb_bnd > 0) {
        for (int i = 0; i < avg_int.size(); i++) {
            avgint += avg_int[i];
        }
        avgint /= avg_int.size();
    } else {
        avgint = 0.;
    }

    if (avg_bnd.size() > 0) {
        // There should always be boundary point(s)
        for (int i = 0; i < avg_bnd.size(); i++) {
            avgbnd += avg_bnd[i];
        }
        avgbnd /= avg_bnd.size();
    } else {
        avgbnd = 0.;
    }

    printf("[Grid] mean of mean interior distances: %f (size: %d)\n", avgint, (int) avg_int.size());
    printf("[Grid] mean of mean boundary distances: %f (size: %d)\n", avgbnd, (int) avg_bnd.size());
}
//----------------------------------------------------------------------------


void Grid::generateStencilsBruteForce() {
    this->stencil_map.resize(node_list.size());

    int nb_rbf = node_list.size();
    int nb_bnd = boundary_indices.size();
    std::vector<NodeType>& rbf_centers = node_list;

    if (max_st_size < 1) {
        std::cout << "[Grid] ERROR! Stencil size must be >= 1" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (max_st_size > nb_rbf) {
        //        int new_stencil_size = (int) (0.5 * nb_rbf);
        int new_stencil_size = nb_rbf;
        new_stencil_size = (new_stencil_size > 1) ? new_stencil_size : 1;
        std::cout << "[Grid] WARNING! Not enough nodes to reach specified stencil_size (size: " << max_st_size << ")!\n"; 
        std::cout << "[Grid] WARNING! Using new stencil size: " << new_stencil_size << "!\n";
        std::cout << "[Grid] WARNING! This is the largest possible stencil give your domain resolution.\n"; 
        max_st_size = new_stencil_size;
    }

    if (nb_bnd == 0) {

        std::cout << "[Grid] WARNING! nb_bnd == 0; Did the Grid::generate() execute properly?!\n";
        //        exit(EXIT_FAILURE);
    }

    // for each node, a vector of stencil nodes (global indexing)
    if (stencil_map.size() < nb_rbf) {
        std::cout << "[Grid] WARNING! stencil_map.size() < node_list.size(). Resizing this vector and possibly corrupting memory!" << std::endl;
        stencil_map.resize(nb_rbf);
    }
    //printf("stencil size: %d, nb_rbf= %d\n", stencil.size(), nb_rbf);

    ltvec ltvec_inst;

    // O(n^2) algorithm, whose cost is independent of the number of nearest sought

    for (int i = 0; i < nb_rbf; i++) {
        Vec3& v = rbf_centers[i];
        StencilType& st = stencil_map[i];
#if 0
        if (st.size() < max_st_size) {
            //		std::cout << "[Grid] WARNING! stencil_map[" << i << "].size() < " << max_st_size << "! Resizing this vector.\n"; 	
            //            st.resize(max_st_size); 
        }
#else 
        st.clear();         // In case we have residual info
#endif 
        std::set<int, ltvec> se;
        ltvec::setRbfCenters(rbf_centers);
        ltvec::setXi(v);

        // find nearest points to center
        for (int j = 0; j < nb_rbf; j++) {
            se.insert(j);
        }

        set<int, ltvec>::iterator sei = se.begin();
        set<int, ltvec>::iterator seii = se.begin();

        // minimimum distance:
        seii++; // I now access first point

        // stencil_size = max stencil_size
        for (int k = 0; k < max_st_size; k++) {
            double d = (rbf_centers[*sei] - rbf_centers[i]).square();
            double ss = sqrt(d);

            if (ss < max_st_radius) {
                st.push_back(*sei);
                sei++;
            } else {
                break; // We can cut loop because the nodes are sorted by dist
            }
        }
    }

    this->computeStencilRadii();
}
//----------------------------------------------------------------------
void Grid::generateStencilsKDTree() 
{
    int nb_rbf = node_list.size();
    int nb_bnd = boundary_indices.size();

    // It might not be up to date, but we'll trust someone else to deal
    // with that problem
    if (node_list_kdtree == NULL) {        
        // It hasnt been constructed yet
        node_list_kdtree = new KDTree(node_list);
    }

    if (max_st_size < 1) {
        std::cout << "[Grid] ERROR! Stencil size must be >= 1" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (max_st_size > nb_rbf) {
        //        int new_stencil_size = (int) (0.5 * nb_rbf);
        int new_stencil_size = nb_rbf;
        new_stencil_size = (new_stencil_size > 1) ? new_stencil_size : 1;
        std::cout << "[Grid] WARNING! Not enough nodes to reach specified stencil_size (size: " << max_st_size << ")!\n"; 
        std::cout << "[Grid] WARNING! Using new stencil size: " << new_stencil_size << "!\n";
        std::cout << "[Grid] WARNING! This is the largest possible stencil give your domain resolution.\n"; 
        max_st_size = new_stencil_size;
    }

    if (nb_bnd == 0) {

        std::cout << "[Grid] WARNING! nb_bnd == 0; Did the Grid::generate() execute properly?!\n";
        //        exit(EXIT_FAILURE);
    }

    // for each node, a vector of stencil nodes (global indexing)
    if (stencil_map.size() < nb_rbf) {
        std::cout << "[Grid] WARNING! stencil_map.size() < node_list.size(). Resizing this vector and possibly corrupting memory!" << std::endl;
        stencil_map.resize(nb_rbf);
    }

    for (size_t i = 0; i < node_list.size(); i++) {
        StencilType& st = stencil_map[i]; 
        st.clear();     // In case of any residual stencil info
        NodeType& center = node_list[i];
        std::vector<double> nearest_dists; 
        std::vector<int> nearest_ids; 
        // Nodes are returned closest first, farthest last
        node_list_kdtree->k_closest_points(center, max_st_size, nearest_ids, nearest_dists); 

        st.resize(max_st_size);  

        for (size_t j = 0; j < max_st_size; j++) { 
            if (nearest_dists[j] < max_st_radius) {
                st[j] = nearest_ids[j]; 
            } else {
                st.resize(j); // trim off extra entries in each stencil
                break; 
            }
        }
    }
    this->computeStencilRadii();
}

//----------------------------------------------------------------------
void Grid::generateStencilsHash()
{
    // Create an overlay grid
    //      assign each node an i,j,k grid cell location
    //      search only cells within the max_radius, but go outward by 
    //              cells starting at the center cell. 
    // NOTE: does not require KDTree construction and allows immediate query

}
