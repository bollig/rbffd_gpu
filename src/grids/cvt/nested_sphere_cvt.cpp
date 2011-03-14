#include <stdio.h>
#include <iostream>
#include <iomanip>

using namespace std;

#include "nested_sphere_cvt.h"


void NestedSphereCVT::generate() {

    // TODO: initialize boundary

	if (!generatorsInitialized || (nb_nodes != node_list.size())) {
		this->node_list.resize(nb_nodes); 
		// Random in unit square
		this->cvt_sample(this->node_list, 0, nb_nodes, CVT::USER_INIT, true); 

		generatorsInitialized = true;
		std::cout << "[CVT] Done sampling initial generators\n";
	}
		
	// Should we generate a KDTree to accelerate sampling? Cost of tree generation is 
	// high and should be amortized by VERY MANY samples. However, too many samples 
	// argues in favor of discrete Voronoi transform
	
	// TODO: get CVT::GRID to generate samples in batches of subvolumes otherwise
	// 	 batches are always the same sample sets for it. 
	this->cvt_iterate(sample_batch_size, nb_samples, CVT::USER_SAMPLE);
	
	this->writeToFile();

	std::cout << "CVT GENERATE NOT IMPLEMENTED" << std::endl;
}



#if 0

//****************************************************************************80

// Inner generators are r[0] -> r[nb_inner-1]
// Outer generators are r[nb_inner] -> r[nb_inner+nb_outer-1]
// Interior generators are r[nb_inner+nb_outer] -> r[nb_tot]
void NestedSphereCVT::cvt(int *it_num, double *it_diff, double *energy) {

    t1.start();
    int i;
    bool initialize;
    int seed_base;
    int seed_init;
    double energyBefore = 0.0, energyAfter = 0.0;

    double *r = &generators[0];

    if (batch < 1) {
        cout << "\n";
        cout << "CVT - Fatal error!\n";
        cout << "  The input value BATCH < 1.\n";
        exit(1);
    }

    if (seed <= 0) {
        cout << "\n";
        cout << "CVT - Fatal error!\n";
        cout << "  The input value SEED <= 0.\n";
        exit(1);
    }

    //if ( DEBUG )
    //{
    cout << "\n";
    cout << "  Step       SEED          L2-Change        Energy(PreProject)     Energy(PostProject)\n";
    cout << "\n";
    //}

    it_num_boundary = 0;
    it_num_interior = 0;
    *it_num = 0;
    *it_diff = 0.0;
    *energy = 0.0;
    seed_init = seed;
    //
    //  Initialize the data, unless the user has already done that.
    //

    initialize = true;

    // Randomly sample all initial generators from nested spheres (note: cvt_sample is
    // overridden to achieve this sampling
    cvt_init(dim_num, nb_inner + nb_outer + nb_int, nb_inner + nb_outer + nb_int, init, initialize, &seed, r);

#if USE_KDTREE
    // KDTree is used by cvt_energy, cvt_iterate to range query generators and samples.
    kdtree = new KDTree(&r[0], nb_inner + nb_outer, dim_num);
#endif

    // BOLLIG: Added check for energy and a check for
    energyBefore = cvt_energy(dim_num, nb_inner + nb_outer, batch, sample, initialize, sample_num, &seed, r);

    // Inner is [0] -> [nb_inner-1]
    project_to_sphere(&r[0], nb_inner, dim_num, inner_r);

    // Outer generators are r[nb_inner] -> r[nb_inner+nb_outer-1]
    project_to_sphere(&r[nb_inner * dim_num], nb_outer, dim_num, outer_r);

    // Interior generators are r[nb_inner+nb_outer] -> r[nb_tot] and require no projection

    // Free kdtree and rebuild it (inefficient! We need to find a datastructure that supports moving nodes
#if USE_KDTREE
#if 0
    t5.start();
    delete(kdtree);
    kdtree = new KDTree(r, nb_inner + nb_outer, dim_num);
    //kdtree->linear_tree_print();
    //cout << "NOW AN UPDATE: \n\n";
    t5.end();
#else
    t6.start();
    kdtree->updateTree(r, nb_inner + nb_outer, dim_num);
    //kdtree->linear_tree_print();
    t6.end();
#endif
#endif


    energyAfter = cvt_energy(dim_num, nb_inner + nb_outer, batch, sample, initialize, sample_num, &seed, r);

    *energy = energyAfter;

    //if ( DEBUG )
    {
        cout << "  "
                << setw(4) << it_num_boundary << "  "
                << setw(12) << seed_init << "  "
                << setw(14) << 0 << "  "
                << setw(14) << energyBefore << "  "
                << setw(14) << energyAfter << "\n";
    }
    //
    //  If the initialization and sampling steps use the same random number
    //  scheme, then the sampling scheme does not have to be initialized.
    //
    if (init == sample) {
        initialize = false;
    } else {
        initialize = true;
    }

    // if (initialize) {
    char intermediate_file[80];

    //cout << "--> Writing intermediate files\n";
    sprintf(intermediate_file, "boundary_nodes_%.5d_inner_%.5d_outer_%.5d_iter", nb_inner, nb_outer, 0);
    cvt_write(dim_num, nb_inner + nb_outer, batch, seed_init, seed, "none",
              it_max_boundary, it_fixed, it_num_boundary, *it_diff, *energy, "none", sample_num, &r[0],
              intermediate_file, false);
    
    sprintf(intermediate_file, "interior_nodes_%.5d_interior_%.5d_iter", nb_int, 0);
    cvt_write(dim_num, nb_int, batch, seed_init, seed, "none",
              it_max_interior, it_fixed, it_num_interior, *it_diff, *energy, "none", sample_num, &r[(nb_inner + nb_outer) * dim_num],
              intermediate_file, false);

    cvt_checkpoint(-2);

    //}

    //
    //  Carry out the boundary iterations.
    //

    // The base CVT class is designed to ignore centroid updates for the first nb_bnd generators
    // so we must ensure that they are included here.
    this->nb_bnd = 0;

    if (dim_num == 2) {
        // Fill circles based on arclength subdivision (exact) rather than CVT (saves iteraitons).
        double inner_arc_seg = (2.*this->PI) / nb_inner;
        double outer_arc_seg = (2.*this->PI) / nb_outer;

        cout << "inner_r = " << inner_r << " nb_inner = " << nb_inner << " inner_arc_seg = " << inner_arc_seg << endl;
        for (int i = 0; i < nb_inner; i++) {
            // r = 0.5
            double theta = inner_arc_seg*i;
            // Convert to (x,y,z) = (r cos(theta), r sin(theta), 0.)
            r[(i * dim_num) + 0] = inner_r * cos(theta);
            r[(i * dim_num) + 1] = inner_r * sin(theta);
            // r[i * dim_num + 2] = 0.;
        }
        for (int i = 0; i < nb_outer; i++) {
            // r = 1.0
            // add 0.001*PI to slightly shift the outer boundary in an attempt to avoid
            // alignment with the inner boundary.
            // HERE: See top of NOTES for an idea of why we shift by PI/3
            double theta = outer_arc_seg*i + (this->PI / 3.);// + 0.001 * this->PI;
            // offset i by nb_inner because we already filled those.
            r[(nb_inner+i) * dim_num + 0] = outer_r * cos(theta);
            r[(nb_inner+i) * dim_num + 1] = outer_r * sin(theta);
            //r[(nb_inner+i) * dim_num + 2] = 0.;
        }

        char intermediate_file[80];
        sprintf(intermediate_file, "boundary_nodes_%.5d_inner_%.5d_outer_%.5d_iter", nb_inner, nb_outer, it_num_boundary);
        //cout << "--> Writing intermediate file:\n";
        cvt_write(dim_num, nb_inner + nb_outer, batch, seed_init, seed, "none",
                  it_max_boundary, it_fixed, it_num_boundary, *it_diff, *energy, "none", sample_num, &r[0],
                  intermediate_file, false);
    } else {

        while (it_num_boundary < it_max_boundary) {
            //
            //  If it's time to update the seed, save its current value
            //  as the starting value for all iterations in this cycle.
            //  If it's not time to update the seed, restore it to its initial
            //  value for this cycle.
            //
            if (((it_num_boundary) % it_fixed) == 0) {
                seed_base = seed;
            } else {
                seed = seed_base;
            }

            it_num_boundary++;
            (*it_num)++;
            seed_init = seed;

            // Iterate for only the first generators on the boundary (NOTE: The energy returned here is junk because it
            // considers only the boundary and no the energy of the whole system with the interior points)
            cvt_iterate(dim_num, nb_inner + nb_outer, batch, sample, initialize, sample_num, &seed, r, it_diff, &energyBefore);

            // Inner is [0] -> [nb_inner-1]
            project_to_sphere(&r[0], nb_inner, dim_num, inner_r);

            // Outer generators are r[nb_inner] -> r[nb_inner+nb_outer-1]
            project_to_sphere(&r[nb_inner * dim_num], nb_outer, dim_num, outer_r);

#if USE_KDTREE
#if 0
            t5.start();
            delete(kdtree);
            kdtree = new KDTree(r, nb_inner + nb_outer, dim_num);
            //kdtree->linear_tree_print();
            //cout << "NOW AN UPDATE: \n\n";
            t5.end();
#else

            t6.start();
            kdtree->updateTree(r, nb_inner + nb_outer, dim_num);
            //kdtree->linear_tree_print();
            t6.end();
#endif
#endif


            // Interior generators are r[nb_inner+nb_outer] -> r[nb_tot] and require no projection
            energyAfter = cvt_energy(dim_num, nb_inner + nb_outer, batch, sample, initialize, sample_num, &seed, r);

            *energy = energyAfter;

            initialize = false;

            if (DEBUG) {
                cout << "  "
                        << setw(4) << it_num_boundary << "  "
                        << setw(12) << seed_init << "  "
                        << setw(14) << *it_diff << "  "
                        << setw(14) << energyBefore << "  "
                        << setw(14) << energyAfter << "\n";
            }

            // BOLLIG:
            // TODO: only do this if a boolean is set for intermediate writes
            // 	not the same as DEBUG
            if ((it_num_boundary) % 20 == 0) {
                char intermediate_file[80];

                cout << "  "
                        << setw(4) << it_num_boundary << "  "
                        << setw(12) << seed_init << "  "
                        << setw(14) << *it_diff << "  "
                        << setw(14) << energyBefore << "  "
                        << setw(14) << energyAfter << "\n";

                sprintf(intermediate_file, "boundary_nodes_%.5d_inner_%.5d_outer_%.5d_iter", nb_inner, nb_outer, it_num_boundary);
                //cout << "--> Writing intermediate file:\n";
                cvt_write(dim_num, nb_inner + nb_outer, batch, seed_init, seed, "none",
                          it_max_boundary, it_fixed, it_num_boundary, *it_diff, *energy, "none", sample_num, &r[0],
                          intermediate_file, false);
            }
        }
    }
    cout << "--> FINISHED BOUNDARY DISTRIBUTION\n";
    // 
    // Now that the boundary is locked, we form the interior CVT using the boundary
    // as reference
    //

#if USE_KDTREE
    t5.start();
    delete(kdtree);
    kdtree = new KDTree(r, nb_inner + nb_outer + nb_int, dim_num);
    t5.end();
#endif

    // The base CVT class is designed to ignore centroid updates for the first nb_bnd generators
    // so we set that number here.
    this->nb_bnd = nb_inner + nb_outer;
    cout << "SAMPLES_NUM: " << sample_num << endl;
    while (it_num_interior < it_max_interior) {
        //
        //  If it's time to update the seed, save its current value
        //  as the starting value for all iterations in this cycle.
        //  If it's not time to update the seed, restore it to its initial
        //  value for this cycle.
        //
        if (((it_num_interior) % it_fixed) == 0) {
            seed_base = seed;
        } else {
            seed = seed_base;
        }

        it_num_interior++;
        (*it_num)++;
        seed_init = seed;
        energyBefore = *energy;

        cvt_iterate(dim_num, nb_inner + nb_outer + nb_int, batch, sample, initialize, sample_num, &seed, r, it_diff, energy);

        initialize = false;

        if (DEBUG) {
            cout << "  "
                    << setw(4) << it_num_interior << "  "
                    << setw(12) << seed_init << "  "
                    << setw(14) << *it_diff << "  "
                    << setw(14) << energyBefore << "  "
                    << setw(14) << *energy << "\n";
        }

        // BOLLIG:
        // TODO: only do this if a boolean is set for intermediate writes
        // 	not the same as DEBUG
        if ((it_num_interior) % 20 == 0) {
            char intermediate_file[80];

            cout << "  "
                    << setw(4) << it_num_interior << "  "
                    << setw(12) << seed_init << "  "
                    << setw(14) << *it_diff << "  "
                    << setw(14) << energyBefore << "  "
                    << setw(14) << *energy << "\n";

            sprintf(intermediate_file, "interior_nodes_%.5d_interior_%.5d_iter", nb_int, it_num_interior);
            //cout << "--> Writing intermediate file:\n";
            cvt_write(dim_num, nb_int, batch, seed_init, seed, "none",
                      it_max_interior, it_fixed, it_num_interior, *it_diff, *energy, "none", sample_num, &r[(nb_inner + nb_outer) * dim_num],
                      intermediate_file, false);
        }
    }

    cout << "--> FINISHED INTERIOR DISTRIBUTION\n";

    cvt_checkpoint(-1);
    
    t1.end();
    return;
}

//****************************************************************************80

void NestedSphereCVT::user_sample(int dim_num, int n, int *seed, double r[]) {
    for (int j = 0; j < n; j++) {
        bool is_rejected = true;
        while (is_rejected) {
            for (int i = 0; i < dim_num; i++) {
                // This is Gordon Erlebacher's random(,) routine (defined below)
                r[i + j * dim_num] = random(geom_extents[0 + i * 2], geom_extents[1 + i * 2]);
            }

            // We have a sample point in the bounding box, so lets try to
            // reject it (e.g., if its outside the geometry contained by bounding
            // box.
            is_rejected = reject_point(&r[j * dim_num], dim_num);
        }
    }
    *seed = (*seed) + n * dim_num;

    return;
}
//****************************************************************************80

void NestedSphereCVT::user_init(int dim_num, int n, int *seed, double r[]) {
    user_sample(dim_num, n, seed, r);

    return;
}

//****************************************************************************80

bool NestedSphereCVT::reject_point(double point[], int ndim) {
    double r = 0.0;
    // Standard sphere, so we use 2-norm
    for (int i = 0; i < ndim; i++) {
        r += point[i] * point[i];
    }
    r = sqrt(r);

    // If the sample does not lie within the bounds of our geometry we
    // reject it. 
    if ((r < inner_r) || (r > outer_r)) {
        return true;
    }
    // Otherwise we keep it.
    return false;
}

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

void NestedSphereCVT::cvt_get_file_prefix(char* filename_buffer) {
    sprintf(filename_buffer, "%s_%.5d_inner_%.5d_outer_%.5d_interior", cvt_file_name, nb_inner, nb_outer, nb_int);
}

#endif 
