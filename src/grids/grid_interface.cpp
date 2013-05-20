#include <iostream> 
#include <fstream> 
#include <sstream>
#include <string>
#include <set>
#include <limits> // for ST_HASH

#include "grid_interface.h"
#include "utils/random.h"
#include "mmio.h"

// ltvec declared in grid_interface.h
NodeType ltvec::xi;
std::vector<NodeType>* ltvec::rbf_centers;
//double* ltdist::dists;

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
        for (unsigned int i = 0; i < node_list.size(); i++) {
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
    if (stencilsComputed) {
        this->writeAvgRadiiToFile(filename); 
        this->writeMaxRadiiToFile(filename); 
        this->writeMinRadiiToFile(filename); 
        this->writeStencilsToFile(filename); 
    }
    this->writeExtraToFile(filename); 
}


//----------------------------------------------------------------------------

void Grid::writeBoundaryToFile(std::string filename) {
    std::string fname = "bndry_"; 
    fname.append(filename); 
    std::ofstream fout(fname.c_str()); 

    if (fout.is_open()) {
        for (unsigned int i = 0; i < boundary_indices.size(); i++) {
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
        for (unsigned int i = 0; i < boundary_normals.size(); i++) {
            fout << this->getBoundaryNormal(i) << std::endl; 
        }
    } else {
        printf("Error opening file to write\n"); 
        exit(EXIT_FAILURE); 
    }
    fout.close();
    std::cout << "[" << this->className() << "] \tWrote " << boundary_normals.size() << " boundary normals to \t" << fname << std::endl;
}

//----------------------------------------------------------------------------

void Grid::writeVecToFile(std::string filename_prefix, std::string filename_suffix, std::vector<double> vec_to_write) {
    std::string fname = filename_prefix; 
    fname.append(filename_suffix); 
    std::ofstream fout(fname.c_str()); 

    // NOTE: we keep the precision extension here because our radii are double precision (nodes are single and only need 8 digits)
#if 1
    // Increase our precision when writing to disk: 
    fout.setf(ios::fixed);
    fout.setf(ios::floatfield); 
//    fout.setf(ios::showpoint); 
//    fout.precision( 15 );   // Try 15 first. Double should be able to go up to 17
#endif 
    if (fout.is_open()) {
        for (unsigned int i = 0; i < avg_stencil_radii.size(); i++) {
            fout << vec_to_write[i] << std::endl; 
        }
    } else {
        printf("Error opening file to write\n"); 
        exit(EXIT_FAILURE); 
    }
    fout.close();
    std::cout << "[" << this->className() << "] \tWrote " << vec_to_write.size() << " doubles to \t" << fname << std::endl;
}
//----------------------------------------------------------------------------
void Grid::writeMaxRadiiToFile(std::string filename) {
    std::string fname = "max_radii_"; 
    this->writeVecToFile(fname, filename, max_stencil_radii); 
}
//----------------------------------------------------------------------------
void Grid::writeMinRadiiToFile(std::string filename) {
    std::string fname = "min_radii_"; 
    this->writeVecToFile(fname, filename, min_stencil_radii); 
}


//----------------------------------------------------------------------------
void Grid::writeAvgRadiiToFile(std::string filename) {
    std::string fname = "avg_radii_"; 
    this->writeVecToFile(fname, filename, avg_stencil_radii); 
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
            for (unsigned int i = 0; i < stencil_map.size(); i++) {
                fout << stencil_map[i].size(); 
                for (unsigned int j=0; j < stencil_map[i].size(); j++) {
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



Grid::GridLoadErrType Grid::loadFromFile(int iter) {
    return this->loadFromFile(this->getFilename(iter)); 
}

Grid::GridLoadErrType Grid::loadStencilsFromFile(int iter) {
    return this->loadStencilsFromFile(this->getFilename(iter)); 
}



//----------------------------------------------------------------------------

Grid::GridLoadErrType Grid::loadFromFile(std::string filename) {
    node_list.clear(); 
    boundary_normals.clear();

    if (this->loadNodesFromFile(filename)) {
        printf("Error loading nodes from file\n"); 
        return NO_GRID_FILES;
    }
    
    if (this->loadBoundaryFromFile(filename)) {
        printf("Error loading boundary nodes\n"); 
        return NO_BOUNDARY_FILES;
    }

    if (this->loadNormalsFromFile(filename)) {
        printf("Error loading normals\n"); 
        return NO_NORMAL_FILES;
    }
#if 0
    if (this->loadAvgRadiiFromFile(filename)) {
        printf("Error loading avg dists\n"); 
        stencilsComputed = false; 
        return NO_STENCIL_FILES;
    }
#endif 

    if (this->loadStencilsFromFile(filename)) {
        printf("Error loading stencils\n"); 
        stencilsComputed = false; 
        return NO_STENCIL_FILES;
    } else {
        stencilsComputed = true;
	// Since we avoid loading stencil radii from disk, lets recompute them: 
	this->computeStencilRadii();
    }

    if (this->loadExtraFromFile(filename)) {
        printf("Error loading additional data\n"); 
        return NO_EXTRA_FILES;
    }
    
    return GRID_AND_STENCILS_LOADED;
}

Grid::GridLoadErrType Grid::loadNodesFromFile(std::string filename) {
    std::cout << "[" << this->className() << "] \treading file: " << filename << std::endl;
    std::ifstream fin(filename.c_str());

    if (fin.is_open()) {
    } else {
        std::string alt_name = this->genericGridFilename();
        std::cout << "Trying generic filename: " << alt_name << std::endl;
        fin.open(alt_name.c_str());
    }

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
        return NO_GRID_FILES;
    }
    fin.close(); 
    nb_nodes = node_list.size(); 
    std::cout << "[" << this->className() << "] \tLoaded " << nb_nodes << " nodes from \t" << filename << std::endl;
    this->refreshExtents();
    return GRID_AND_STENCILS_LOADED;    
}


//----------------------------------------------------------------------------

Grid::GridLoadErrType Grid::loadBoundaryFromFile(std::string filename) {
    std::string fname = "bndry_"; 
    fname.append(filename);
    std::cout << "[" << this->className() << "] \treading boundary file: " << fname << std::endl;    

    std::ifstream fin; 
    fin.open(fname.c_str()); 

    if (fin.is_open()) {
    } else {
        std::string alt_name = this->genericBoundaryFilename();
        std::cout << "Trying generic filename: " << alt_name << std::endl;
        fin.open(alt_name.c_str());
    }

    if (fin.is_open()) {
        boundary_indices.clear(); 
        while (fin.good()) {
            unsigned int indx; 
            fin >> indx; 
            if (!fin.eof()) {
                boundary_indices.push_back(indx); 
            }
        }
    } else {
        printf("Error opening boundary file to read\n"); 
        return NO_BOUNDARY_FILES;
        //		exit(EXIT_FAILURE);
    }

    fin.close(); 

    std::cout << "[" << this->className() << "] \tLoaded " << boundary_indices.size() << " boundary indices from \t" << fname << std::endl;
    return GRID_AND_STENCILS_LOADED; 
}

//----------------------------------------------------------------------------

Grid::GridLoadErrType Grid::loadNormalsFromFile(std::string filename) {
    std::string fname = "nrmls_"; 
    fname.append(filename);
    std::cout << "[" << this->className() << "] \treading normals file: " << fname << std::endl;    

    std::ifstream fin; 
    fin.open(fname.c_str()); 

    if (fin.is_open()) {
    } else {
        std::string alt_name = this->genericNormalsFilename();
        std::cout << "Trying generic filename: " << alt_name << std::endl;
        fin.open(alt_name.c_str());
    }

    if (fin.is_open()) {
        boundary_normals.clear(); 
        while (fin.good()) {
            Vec3d norml; 
            fin >> norml; 
            if (!fin.eof()) {
                boundary_normals.push_back(norml); 
            }
        }
    } else {
        printf("Error opening normals file to read\n"); 
        return NO_NORMAL_FILES;
        //		exit(EXIT_FAILURE);
    }

    fin.close(); 

    std::cout << "[" << this->className() << "] \tLoaded " << boundary_normals.size() << " boundary normals from \t" << fname << std::endl;
    return GRID_AND_STENCILS_LOADED; 
}

//----------------------------------------------------------------------------

Grid::GridLoadErrType Grid::loadAvgRadiiFromFile(std::string filename) {
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
        return NO_RADII_FILES;
        //		exit(EXIT_FAILURE);
    }

    fin.close(); 

    std::cout << "[" << this->className() << "] \tLoaded " << avg_stencil_radii.size() << " average stencil radii from \t" << fname << std::endl;
    return GRID_AND_STENCILS_LOADED; 
}

//----------------------------------------------------------------------------

Grid::GridLoadErrType Grid::loadStencilsFromFile(std::string filename) {
    if (max_st_size > 0) {
        std::ostringstream prefix; 
        prefix << "stencils_maxsz" << this->max_st_size << "_" << filename; 

        std::string fname = prefix.str();

        std::cout << "[" << this->className() << "] \treading stencil file: " << fname << std::endl;    

        std::ifstream fin; 
        fin.open(fname.c_str()); 

        unsigned int num_el_loaded = 0; 
        if (fin.is_open()) {
            stencil_map.clear(); 
            while (fin.good()) {
                unsigned int st_size; 
                fin >> st_size; 
                StencilType st; 
                for (unsigned int i = 0; i < st_size; i++) {
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
            return NO_STENCIL_FILES;
            //		exit(EXIT_FAILURE);
        }

        fin.close(); 

        std::cout << "[" << this->className() << "] \tLoaded " << stencil_map.size() << " stencils, with a total of " << num_el_loaded << " elements from \t" << fname << std::endl;
    } else {
        std::cout << "[" << this->className() << "] \tMax stencil size not set. No stencils to read from disk" << std::endl;
    }
    return GRID_AND_STENCILS_LOADED; 
}

//----------------------------------------------------------------------------

Grid::GridLoadErrType Grid::loadExtraFromFile(std::string filename) {
    std::cout << "No extra loads from disk required." << std::endl;
    return GRID_AND_STENCILS_LOADED; 
}

//----------------------------------------------------------------------------

void Grid::sortNodes() {
    for (unsigned int i = 0; i < this->boundary_indices.size(); i++) {
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
    for (unsigned int i = 0 ; i < node_list.size(); i ++) {
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
    std::vector<unsigned int>::iterator i; 
    for (i = boundary_indices.begin(); i != boundary_indices.end(); i++) {
        std::cout << (*i) << std::endl;
    }
}
//----------------------------------------------------------------------------
void Grid::generateStencils(unsigned int st_max_size, st_generator_t generator_choice)
{
    max_st_size = st_max_size; 
    generateStencils(generator_choice); 
}

//----------------------------------------------------------------------------
void Grid::generateStencils(st_generator_t generator_choice) 
{
    switch (generator_choice)
    {
        case ST_BRUTE_FORCE:
            std::cout << "Stencil Generator: Brute Force\n"; 
            this->generateStencilsBruteForce(); 
            break; 
        case ST_KDTREE: 
            std::cout << "Stencil Generator: KDTree \n"; 
            this->generateStencilsKDTree(); 
            break; 
        case ST_HASH: 
            std::cout << "Stencil Generator: LSH Raster \n"; 
            this->generateStencilsHash(); 
            break; 
        default: 
            std::cout << "ERROR! Invalid choice of stencil generator\n"; 
            exit(EXIT_FAILURE); 
    };
    
    stencilsComputed=true;
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

        double dmin = (rbf_centers[st[1]] - rbf_centers[st[0]]).magnitude(); 
        min_stencil_radii[i] = dmin; 

        double dmax = (rbf_centers[st[st.size()-1]] - rbf_centers[st[0]]).magnitude(); 
        max_stencil_radii[i] = dmax; 

        //        std::cout << "st.max_dist = " << max_stencil_radii[i] << std::endl;

        avg_stencil_radii[i] = 0.;
        if (i < nb_bnd) {
            avg_bnd[i] = 0.;
        } else {
            avg_int[i - nb_bnd] = 0.;
        }

        // Now iterate over the ith stencil and query distance to neighbors
        for (unsigned int k = 1; k < st.size(); k++) {
            double ss = (rbf_centers[st[k]] - rbf_centers[st[0]]).magnitude();

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
        for (unsigned int i = 0; i < avg_int.size(); i++) {
            avgint += avg_int[i];
        }
        avgint /= avg_int.size();
    } else {
        avgint = 0.;
    }

    if (avg_bnd.size() > 0) {
        // There should always be boundary point(s)
        for (unsigned int i = 0; i < avg_bnd.size(); i++) {
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
    this->checkStencilSize(); 

    int nb_rbf = node_list.size();
    //int nb_bnd = boundary_indices.size();
    std::vector<NodeType>& rbf_centers = node_list;

    // O(n^2) algorithm, whose cost is independent of the number of nearest sought

    for (int i = 0; i < nb_rbf; i++) {
        Vec3d& v = rbf_centers[i];
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
        for (unsigned int k = 0; k < max_st_size; k++) {
            double ss = (rbf_centers[*sei] - rbf_centers[i]).magnitude();

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
    this->checkStencilSize(); 

    //int nb_rbf = node_list.size();
    //int nb_bnd = boundary_indices.size();

    // The KDTree might not be up to date, but we'll trust someone else to deal
    // with that problem
    if (node_list_kdtree == NULL) {        
        // The KDTree hasnt been constructed yet
        node_list_kdtree = new KDTree(node_list);
    }
    std::cout << "DONE CONSTRUCTION TREE\n"; 
    //sleep(15); 

    for (unsigned int i = 0; i < node_list.size(); i++) {
        StencilType& st = stencil_map[i]; 
        st.clear();     // In case of any residual stencil info
        NodeType& center = node_list[i];
        std::vector<double> nearest_dists; 
        std::vector<int> nearest_ids; 
        // Nodes are returned closest first, farthest last
        node_list_kdtree->k_closest_points(center, max_st_size, nearest_ids, nearest_dists); 

        st.resize(max_st_size);  

        for (unsigned int j = 0; j < max_st_size; j++) { 
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
// Create an overlay grid
//      assign each node an i,j,k grid cell location
//      search only cells within the max_radius, but go outward by 
//              cells starting at the center cell. 
// NOTE: does not require KDTree construction and allows immediate query
void Grid::generateStencilsHash()
{
    this->checkStencilSize(); 

    // Dimensions of the hash overlay grid (hnx by hny by hnz regular grid
    // spanning the full bounding box of the domain extent)
    unsigned int hnx = ns_nbx; 
    unsigned int hny = ns_nby;
    unsigned int hnz = ns_nbz;

    std::vector< std::vector<unsigned int> > cell_hash(hnx*hny*hnz);//, (unsigned int)0); 
    // list of lists 
//    cell_hash.resize(hnx * hny * hnz);  

    double cdx = (xmax - xmin) / hnx;
    //double cdx = (hnx > 1) ? (xmax - xmin) / (hnx - 1.) : (xmax - xmin) / hnx;
    double cdy = (ymax - ymin) / hny;
    double cdz = (zmax - zmin) / hnz;

    NodeType cell_start(xmin, ymin, zmin); 
    NodeType cell_end(xmax, ymax, zmax); 

    // We'll sort our nodes according to hash (roughly sorted), to try to improve caching
    std::vector<NodeType> hash_sorted_nodes(nb_nodes); 

    // Foreach node: 
    //     determine hashid (cellid)
    //          node(x,y,z) exists in cellid((x-xmin)/dx, (y-ymin)/dy, (z-zmin)/dz)
    //          linearize cellid(xc, yc, zc) = ((xc*NY) + yc)*NZ + zc
    //          append node to list contained in list[cellid]
    for (unsigned int i = 0; i < this->nb_nodes; i++) {
        NodeType node = this->getNode(i);
        // xc, yc and zc are the (x,y,z) corresponding to the cell id
        // xmin,ymin,zmin are member properties of the Grid class
        // cdx,cdy,cdz are the deltaX, deltaY, deltaZ for the cell overlays
        // TODO: we note that the xc, yc and zc can be treated at binary digits
        // to select the CELL_ID (do we really need an optimization like that
        // though?)
        unsigned int xc = (unsigned int)floor((node.x() - xmin) / cdx); 

        // This logic saves us when our nodes lie on xmax, ymax, or zmax
        // so instead of covering [n-1*dx,xmax), our cell covers [n-1*dx,xmax]
        xc = (xc == hnx) ? xc-1 : xc; 
        unsigned int yc = (unsigned int)floor((node.y() - ymin) / cdy); 
        yc = (yc == hny) ? yc-1 : yc; 
        unsigned int zc = (unsigned int)floor((node.z() - zmin) / cdz);
        zc = (zc == hnz) ? zc-1 : zc; 
        unsigned int cell_id = ((xc*hny) + yc)*hnz + zc; 

        //       std::cout << "NODE:" << node << "   in   CELL: " << cell_id << "      ( " << xc << ", " << yc << ", " << zc << " )" << std::endl;
        cell_hash[cell_id].push_back(i); 
    }

#if 1
    // Now we sort our nodes
    unsigned int indx = 0;
    unsigned int bindx = 0;
    std::vector<unsigned int>& bindices = this->getBoundaryIndices();
    for (unsigned int i =0; i < cell_hash.size(); i++) {
        for (unsigned int j = 0; j < cell_hash[i].size(); j++) {

            bool found = false; 
            unsigned int old_bindx = 0;
            hash_sorted_nodes[indx] =  this->getNode(cell_hash[i][j]); 
    
            // Update the boundary info if necessary
            // This caused a segfault on Keeneland. we'll do our own search
       //     std::vector<unsigned int>::iterator old_bindx = find(bindices.begin()+bindx, bindices.end(), cell_hash[i][j]); 
            for (unsigned int k = bindx; k < this->getBoundaryIndicesSize(); k++) {
                if (bindices[k] == cell_hash[i][j]) {
                    found = true; 
                    old_bindx = k;
                }
            }

            if (found) {
//                std::cout << "Swapping boundary[" << *old_bindx << "] for " << indx << std::endl;
                // Swap our boundary indices (not the most efficient, but...itll do the job
                this->boundary_indices[old_bindx] = this->getBoundaryIndex(bindx);
                this->boundary_indices[bindx] = indx;  

                Vec3d normal = this->getBoundaryNormal(bindx); 
                this->boundary_normals[bindx] = this->getBoundaryNormal(old_bindx); 
                this->boundary_normals[old_bindx] = normal; 
                bindx++;
            }

            // Update the cell_hash so we know about the new index;
            cell_hash[i][j] = indx;
            indx++;
        }
    }
    // Overwrite our existing node_list.
    std::copy(hash_sorted_nodes.begin(), hash_sorted_nodes.end(), this->node_list.begin());
#endif 

#if 0
    for (unsigned int i = 0; i < cell_hash.size(); i++) {
        std::cout << "CELL[" << i << "].size() = " << cell_hash[i].size() << std::endl;
    }
#endif 
    // TODO: Sort nodes according to hash for better access patterns

    // Foreach node: 
    //      Generate a stencil:
    //          append cell_hash[cellid(this->node)] list to candidate list
    //          if (stencil_size > cell_hash.length) then 
    //              append 8 (or 26 if 3D) neigboring cell_hash lists to candidate list
    //          end
    //          sort the candidate list according to distance from node
    //          select stencil_size closest matches
    for (unsigned int p = 0; p < this->nb_nodes; p++) {

        NodeType node = this->getNode(p);
        // xc, yc and zc are the (x,y,z) corresponding to the cell id
        // xmin,ymin,zmin are member properties of the Grid class
        // cdx,cdy,cdz are the deltaX, deltaY, deltaZ for the cell overlays
        unsigned int xc = (unsigned int)floor((node.x() - xmin) / cdx); 
        // This logic saves us when our nodes lie on xmax, ymax, or zmax
        // so instead of covering [n-1*dx,xmax), our cell covers [n-1*dx,xmax]
        //
        xc = (xc == hnx) ? xc-1 : xc; 
        unsigned int yc = (unsigned int)floor((node.y() - ymin) / cdy); 
        yc = (yc == hny) ? yc-1 : yc; 
        unsigned int zc = (unsigned int)floor((node.z() - zmin) / cdz);
        zc = (zc == hnz) ? zc-1 : zc; 

        //unsigned int node_cell_id = ((xc*hny) + yc)*hnz + zc; 

        // List of cell indices we will check
        // NOTE: we leverage set here because it does NOT allow duplicates,
        // so cells are not searched twice
        std::set<unsigned int> neighbor_cell_set; 

        // Generate a list of cells to check for nearest neighbors
        // For each node expand the search until the max_st_size can be satisifed
        // DO NOT check cells with 0 node inside

        unsigned int nb_neighbor_nodes_to_check = 0; 
        int level = 0; 
        // TODO: cut-off search if (max_st_radius+cdx) is execeeded
        //          (requires a working impl of max_st_radius)o
        // BUGFIX: this (level < 2) guarantees we searching neighboring cells
        // in the event that we're near the boundary of a cell and the current
        // cell has more than enough nodes to exceed max_st_size. 
        while (nb_neighbor_nodes_to_check < max_st_size || level < 2) {
            int xlevel = level;
            int ylevel = (hny > 1) ? level : 0;
            int zlevel = (hnz > 1) ? level : 0; 

            for (int xindx = 0-xlevel; xindx <= 0+xlevel; xindx++) {
                for (int yindx = 0-ylevel; yindx <= 0+ylevel; yindx++) {
                    for (int zindx = 0-zlevel; zindx <= 0+zlevel; zindx++) {
                        // Offset cell
                        int xc_o = (xc + xindx); 
                        int yc_o = (yc + yindx); 
                        int zc_o = (zc + zindx); 

#if 1
                        // If the neighbor cell is outside our overlay we ignore the task
                        if ((xc_o < 0) || ((unsigned int) xc_o >= hnx)) {
                            continue;
                        }

                        if ((yc_o < 0) || ((unsigned int) yc_o >= hny)) {
                            continue;
                        }

                        if ((zc_o < 0) || ((unsigned int) zc_o >= hny)) {
                            continue;
                        }
#endif 
                        unsigned int cell_id = ((xc_o*hny) + yc_o)*hnz + zc_o; 

                        //                        std::cout << "( " << xc_o << ", " << yc_o << ", " << zc_o << " ) = " << cell_id << std::endl;

                        if (cell_hash[cell_id].size() > 0) {
                            neighbor_cell_set.insert(cell_id);
                        }
                    }
                }
            }

            // Now count the number of nodes we'll be checking.
            // If its greater than max_st_size then we can stop expanding search

            nb_neighbor_nodes_to_check = 0;
            for (std::set<unsigned int>::iterator it = neighbor_cell_set.begin(); it != neighbor_cell_set.end(); it++) {
                unsigned int cell_id = *it; 
                //                std::cout << "Node ID: " << p << " in CELL " << node_cell_id << ", Checking NEIGHBOR CELL: " << cell_id << std::endl;
                nb_neighbor_nodes_to_check += cell_hash[cell_id].size();
            }
            //    std::cout << "NODE ID: " << p << "  LEVEL: " << level << "\t";
            //    std::cout << "NB_NEIGHBORS = " << nb_neighbor_nodes_to_check << " of " << max_st_size << std::endl;
            level ++; 
        }

        // Compute distances for each neighbor and insert them into a sorted set. 
        std::set< std::pair<double,unsigned int> , ltdist> dists;
        unsigned int d_count = 0;  
        for (std::set<unsigned int>::iterator it = neighbor_cell_set.begin(); it != neighbor_cell_set.end(); it++) {
            unsigned int cell_id = *it; 
            for (unsigned int q = 0; q < cell_hash[cell_id].size(); q++) {
                NodeType& neighbor = this->getNode(cell_hash[cell_id][q]); 
                //dists[d_count] = std::pair<double,unsigned int>( (node - neighbor).magnitude(), cell_hash[cell_id][q] ); 
                //                std::cout << "DIST (" << node << "   to   " << neighbor << ") = " << (node - neighbor).magnitude()  << std::endl;
                dists.insert(std::pair<double,unsigned int>( (node - neighbor).magnitude(), cell_hash[cell_id][q] )); 
                d_count++;
            }
        }

        // Set insertion auto-sorts by distance

        StencilType& st = stencil_map[p]; 
//        st.clear();     // In case of any residual stencil info
        st.reserve(max_st_size);  
        st.resize(max_st_size);  
        std::set< std::pair<double,unsigned int> , ltdist>::iterator sorted_ids = dists.begin(); 
        for (unsigned int j = 0; j < max_st_size; j++) { 
            if ((*sorted_ids).first < max_st_radius) {
                // std::cout << "NODE ID: " << p << "\tDIST (" << j << ") = " << (*sorted_ids).first << "\t NeighborIndx: " << (*sorted_ids).second << std::endl;
                st[j] = (*sorted_ids).second; 
                sorted_ids++;
            } else {
                st.resize(j); // trim off extra entries in each stencil
                break; 
            }
        }
    }

    this->computeStencilRadii();
}



void Grid::checkStencilSize() {
    unsigned int nb_rbf = node_list.size();
    unsigned int nb_bnd = boundary_indices.size();

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
        std::cout << "[Grid] WARNING! stencil_map.size() < node_list.size(). Resizing this vector and possibly corrupting memory! (Reserving: " << nb_rbf << " * " << max_st_size << " * 8 = " << nb_rbf * max_st_size * 8 << " bytes)" << std::endl;
        stencil_map.reserve(nb_rbf);
        stencil_map.resize(nb_rbf);
        for (unsigned int i =0 ; i < nb_rbf; i++) {
            stencil_map[i].reserve(max_st_size); 
            stencil_map[i].resize(max_st_size); 
        }
    }
}



// Partition the interior and boundary indices into sorted sets. This helps us
// when we want access only interior nodes
void Grid::partitionIndices() {

    std::cout << "CALLED PartitionIndices\n";
    std::vector<NodeType>& nodes = this->getNodeList(); 
    std::vector<unsigned int>& boundary_indx = this->getBoundaryIndices();
    unsigned int nb_bnd = boundary_indx.size();
    //unsigned int nb_stencils = this->getStencilsSize();
    unsigned int nb_nodes = nodes.size();

    b_indices.clear();
    // Use a std::set because it auto sorts as we insert
    for (unsigned int i = 0; i < nb_bnd; i++) {
        b_indices.insert(boundary_indx[i]);  
    }

    std::set<unsigned int> all_indices; 
    for (unsigned int i = 0; i < nb_nodes; i++) {
        all_indices.insert(i);  
    }

    i_indices.clear();
    std::set_difference(all_indices.begin(), all_indices.end(), b_indices.begin(), b_indices.end(), std::inserter(i_indices, i_indices.end()));

    partitioned_indices = true;
}


