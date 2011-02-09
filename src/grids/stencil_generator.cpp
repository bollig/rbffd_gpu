/*
 * File:   stencil.h
 * Author: bollig
 *
 * Created on May 11, 2010, 12:48 PM
 */
#include <stdlib.h>
#include <vector>

using namespace std;

#include "stencil_generator.h"

//----------------------------------------------------------------------
#if 1
// Class originally in grid.h sorts the grid nodes by radius

class ltvec {
public:
    static Vec3 xi;
    static vector<Vec3>* rbf_centers;

    static void setXi(Vec3& xi) {
        ltvec::xi = xi;
    }

    static void setRbfCenters(vector<Vec3>& rbf_centers_) {
        rbf_centers = &rbf_centers_;
    }

    bool operator()(const int i, const int j) {
        double d1 = ((*rbf_centers)[i] - xi).square();
        double d2 = ((*rbf_centers)[j] - xi).square();
        // allows duplicates
        return d1 <= d2;
    }
};

Vec3 ltvec::xi;
vector<Vec3>* ltvec::rbf_centers;

#endif

StencilGenerator::StencilGenerator(double st_max_radius) {
    this->st_max_radius = st_max_radius;
}
#if 0
StencilGenerator::StencilGenerator(int st_max_size, double st_max_radius) {
    this->st_max_size = st_max_size;
    this->st_max_radius = st_max_radius;
}
#endif 

StencilGenerator::~StencilGenerator() {
    printf("TODO: StencilGenerator::DESTRUCTOR\n");
}

void StencilGenerator::setRadius(double st_max_radius) {
    this->st_max_radius = st_max_radius;
}


void StencilGenerator::computeStencils(std::vector<NodeType>& node_list, std::vector<size_t>& boundary_indices, std::vector<StencilType>& stencil_map, size_t st_max_size, std::vector<double>& avg_stencil_radii) {
    int nb_rbf = node_list.size();
    int nb_bnd = boundary_indices.size();
    std::vector<NodeType>& rbf_centers = node_list;

    if (st_max_size > nb_rbf) {
        int new_stencil_size = (int) (0.5 * nb_rbf);
        new_stencil_size = (new_stencil_size > 1) ? new_stencil_size : 1;
        std::cout << "[StencilGenerator] WARNING! Not enough nodes to reach specified stencil_size (size: " << st_max_size << ")! Using new stencil size: " << new_stencil_size << "!\n";
        st_max_size = new_stencil_size;
    }

    if (nb_bnd == 0) {

        std::cout << "[StencilGenerator] WARNING! nb_bnd == 0; Did the Grid::generate() execute properly?!\n";
//        exit(EXIT_FAILURE);
    }

    // for each node, a vector of stencil nodes (global indexing)
	if (stencil_map.size() < nb_rbf) {
		std::cout << "[StencilGenerator] WARNING! stencil_map.size() < node_list.size(). Resizing this vector and possibly corrupting memory!" << std::endl;
		stencil_map.resize(nb_rbf);
    		avg_stencil_radii.resize(nb_rbf);
	}
    //printf("stencil size: %d, nb_rbf= %d\n", stencil.size(), nb_rbf);

    ltvec ltvec_inst;
    //vector<double> avg_stencil_radii;
    vector<double> avg_bnd;
    vector<double> avg_int;

    avg_bnd.resize(nb_bnd);
    avg_int.resize(nb_rbf - nb_bnd);

    //printf("nb_rbf= %d (bnd: %d)\n", nb_rbf, nb_bnd);
    //printf("nb_bnd= %d\n", nb_bnd);
    //exit(0);

    // O(n^2) algorithm, whose cost is independent of the number of nearest sought

    for (int i = 0; i < nb_rbf; i++) {
        Vec3& v = rbf_centers[i];
        StencilType& st = stencil_map[i];

	if (st.size() < st_max_size) {
//		std::cout << "[StencilGenerator] WARNING! stencil_map[" << i << "].size() < " << st_max_size << "! Resizing this vector.\n"; 	
		st.resize(st_max_size); 
	}

        set<int, ltvec> se;
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
        double min_dist = (rbf_centers[*seii] - rbf_centers[i]).square();
        min_dist = sqrt(min_dist);
        //printf("min distance: %f\n", min_dist);

#if 1
        avg_stencil_radii[i] = 0.;
        if (i < nb_bnd) {
            avg_bnd[i] = 0.;
        } else {
            avg_int[i - nb_bnd] = 0.;
        }
#endif
	
        // stencil_size = max stencil_size
        for (int k = 0; k < st_max_size; k++) {
            double d = (rbf_centers[*sei] - rbf_centers[i]).square();
            double ss = sqrt(d);

            // I am not sure code works yet
            // Errors of solution to heat equation do not stay small
            //if ((ss / min_dist) > 1.5) break;

            st[k] = *sei;

#if 1
            avg_stencil_radii[i] += ss;
            if (i < nb_bnd) {
                avg_bnd[i] += ss;
            } else {
                avg_int[i - nb_bnd] += ss;
            }
#endif

            // printf("(%d, %d) dist= %f\n", i, k, ss);

            // printf("el %d, d= %f\n", *sei, d);
            sei++;
        }

#if 1
        avg_stencil_radii[i] /= (st.size() - 1.); // ignore center point
        if (i < nb_bnd) {
            avg_bnd[i] /= (st.size() - 1.);
        } else {
            avg_int[i - nb_bnd] /= (st.size() - 1.);
        }
#endif

        //printf("avg_dist[%d]= %f\n", i, avg_stencil_radii[i]);
        //printf("nb points in stencil: %d\n", (int) st.size());
    }

#if 1
    double avgint = 0.;
    double avgbnd = 0.;

    printf("[StencilGenerator] avg_int.size() = %d\n", (int) avg_int.size());
    printf("[StencilGenerator] avg_int.size() = %d\n", (int) avg_bnd.size());

    if (nb_rbf - nb_bnd > 0) {
        for (int i = 0; i < avg_int.size(); i++) {
            avgint += avg_int[i];
        }
        avgint /= avg_int.size();
    } else {
        avgint = 0.;
    }

    // There should always be boundary point(s)
    for (int i = 0; i < avg_bnd.size(); i++) {
        avgbnd += avg_bnd[i];
    }
    avgbnd /= avg_bnd.size();

    printf("[StencilGenerator] mean of mean interior distances: %f (size: %d)\n", avgint, (int) avg_int.size());
    printf("[StencilGenerator] mean of mean boundary distances: %f (size: %d)\n", avgbnd, (int) avg_bnd.size());
#endif
}
//----------------------------------------------------------------------
