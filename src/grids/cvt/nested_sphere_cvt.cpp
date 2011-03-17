#include <stdio.h>
#include <iostream>
#include <iomanip>

#include <math.h>

using namespace std;

#include "nested_sphere_cvt.h"

#include "utils/random.h"

void NestedSphereCVT::generate() {

    // TODO: initialize boundary
#if 0
    // Inner is [0] -> [nb_inner-1]
    project_to_sphere(&r[0], nb_inner, dim_num, inner_r);

    // Outer generators are r[nb_inner] -> r[nb_inner+nb_outer-1]
    project_to_sphere(&r[nb_inner * dim_num], nb_outer, dim_num, outer_r);

    // Interior generators are r[nb_inner+nb_outer] -> r[nb_tot] and require no projection
#endif 

    if (!generatorsInitialized || (nb_nodes != node_list.size())) {
        this->node_list.resize(nb_nodes); 
        // Random in unit square
        this->cvt_sample(this->node_list, 0, nb_nodes, CVT::USER_INIT, true); 

        generatorsInitialized = true;
        std::cout << "[CVT] Done sampling initial generators\n";
    }

    // NOTE: 2D only we can assume that nodes on boundary do not need Lloyd's
    // method because it would take too long with our sampling. Just distribute
    // them uniformly
    if (dim_num == 2) {
        // Fill circles based on arclength subdivision (exact) rather than CVT
        // (saves iteraitons).
        double inner_arc_seg = (2.*M_PI) / nb_inner;
        double outer_arc_seg = (2.*M_PI) / nb_outer;

        NodeType sphere_center((xmax+xmin)/2., (ymax+ymin)/2., (zmax+zmin)/2.); 

        boundary_indices.resize(nb_inner + nb_outer);
        boundary_normals.resize(nb_inner + nb_outer);

        int j = 0; 
        //cout << "inner_r = " << inner_r << " nb_inner = " << nb_inner << " inner_arc_seg = " << inner_arc_seg << endl;
        for (int i = 0; i < nb_inner; i++) {
            // r = 0.5
            double theta = inner_arc_seg*i;
            // Convert to (x,y,z) = (r cos(theta), r sin(theta), 0.)
            node_list[i][0] = inner_r * cos(theta);
            node_list[i][1] = inner_r * sin(theta);
            node_list[i][2] = 0.;
            boundary_indices[j] = j; 
            Vec3 nrml = sphere_center - node_list[i];
            nrml.normalize();
            boundary_normals[i] = nrml; 
            j++;
        }
        for (int i = 0; i < nb_outer; i++) {
            // r = 1.0
            // add 0.001*PI to slightly shift the outer boundary in an attempt
            // to avoid alignment with the inner boundary.
            // HERE: See top of NOTES for an idea of why we shift by PI/3
            double theta = outer_arc_seg*i + (M_PI / 3.);// + 0.001 * this->PI;
            // offset i by nb_inner because we already filled those.
            node_list[(nb_inner+i)][0] = outer_r * cos(theta);
            node_list[(nb_inner+i)][1] = outer_r * sin(theta);
            node_list[(nb_inner+i)][2] = 0.;
            boundary_indices[j] = j; 
            Vec3 nrml = node_list[i] - sphere_center;
            nrml.normalize();
            boundary_normals[j] = nrml ; 
            j++;
        }


    //    std::cout << "NRMLS: " << boundary_normals.size() << std::endl;
    } else if (dim_num == 3) {

        std::cout << "TODO: 3D nested sphere cvt" << std::endl;
        exit(EXIT_FAILURE);
    } else {
        std::cout << "ONLY 2D annulus cvt supported at this time" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Should we generate a KDTree to accelerate sampling? Cost of tree
    // generation is high and should be amortized by VERY MANY samples.
    // However, too many samples argues in favor of discrete Voronoi transform

    // TODO: get CVT::GRID to generate samples in batches of subvolumes
    // otherwise batches are always the same sample sets for it. 
    this->cvt_iterate(sample_batch_size, nb_samples, CVT::USER_SAMPLE);

    this->writeToFile();

    std::cout << "CVT GENERATE NOT IMPLEMENTED" << std::endl;
}

//****************************************************************************80
void NestedSphereCVT::user_sample(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand) {
    for (int j = indx_start; j < indx_start + n_now; j++) {
        bool is_rejected = true;
        while (is_rejected) {
            // This is Gordon Erlebacher's random(,) routine from random.h
            user_node_list[j][0] = randf(xmin, xmax); 
            if (dim_num > 1) {
                user_node_list[j][1] = randf(ymin, ymax); 
                if (dim_num > 2) {
                    user_node_list[j][2] = randf(zmin, zmax); 
                }
            }

            // We have a sample point in the bounding box, so lets try to
            // reject it (e.g., if its outside the geometry contained by bounding
            // box.
            is_rejected = reject_point(user_node_list[j], dim_num);
        }
    }

    return;
}
//****************************************************************************80
// For CVT:: this samples randomly in unit circle
void NestedSphereCVT::user_init(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand) { 
    user_sample(user_node_list, indx_start, n_now, init_rand);

    return;
}

//****************************************************************************80

bool NestedSphereCVT::reject_point(NodeType& point, int ndim) {
    NodeType sphere_center((xmax+xmin)/2., (ymax+ymin)/2., (zmax+zmin)/2.); 
    double r = (point - sphere_center).magnitude();  
//    std::cout << r << ">" << outer_r << "----" << point << "----" << sphere_center << std::endl;
    // If the sample does not lie within the bounds of our geometry we
    // reject it. 
    if ((r < inner_r) || (r > outer_r)) {
        return true;
    }
    // Otherwise we keep it.
    return false;
}

#if 0
// Project k generators to the surface of the sphere of specified radius
// WARNING! Modifies the first k elements of generator[]! so pass a pointer
// to whatever element you want to start projecting from!

void NestedSphereCVT::project_to_sphere(double generator[], int k, int ndim, double radius) {

    for (int j = 0; j < k; j++) {
        // Compute vector norm
        double norm = 0.;
        for (int i = 0; i < ndim; i++) {
            norm += generator[i + j * ndim] * generator[i + j * ndim];
        }
        norm = sqrt(norm);

        // Projected point is r*(p/||p||)
        // That is, the normalized vector times radius of desired sphere.
        for (int i = 0; i < ndim; i++) {
            generator[i + j * ndim] /= norm;
            generator[i + j * ndim] *= radius;
        }
    }

    return;
}
#endif 

