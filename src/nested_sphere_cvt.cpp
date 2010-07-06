#include <iostream>
#include <iomanip>

using namespace std;

#include "nested_sphere_cvt.h"

NestedSphereCVT::NestedSphereCVT(const char* file_name, int nb_inner_bnd, int nb_outer_bnd, int nb_interior, int sample_num_, int it_max_bndry, int it_max_inter, int dimension, double inner_radius, double outer_radius, int DEBUG_)
: CVT(nb_inner_bnd + nb_outer_bnd + nb_interior, dimension, file_name, 3 /*USER INIT*/, 3 /*USER SAMPLE*/, sample_num_),
inner_r(inner_radius), outer_r(outer_radius),
nb_inner(nb_inner_bnd), nb_outer(nb_outer_bnd), nb_int(nb_interior),
it_max_interior(it_max_inter), it_max_boundary(it_max_bndry)
{
    DEBUG = DEBUG_;
    nb_bnd = 0;

#if 0
    // For testing: quarter spheres
    geom_extents = new double[2 * dim_num];
    for (int i = 0; i < dim_num; i++) {
        geom_extents[0 + i * 2] = -1;
        geom_extents[1 + i * 2] = 0;
    }
    geom_extents[0 + 0 * 2] = -1;
    geom_extents[1 + 0 * 2] = 0;
#else
    // Create bounding volume: [-1,1]^3
    geom_extents = new double[2 * dim_num];
    for (int i = 0; i < dim_num; i++) {
        geom_extents[0 + i * 2] = -1;
        geom_extents[1 + i * 2] = 1;
    }
#endif
}

//****************************************************************************80

// Inner generators are r[0] -> r[nb_inner-1]
// Outer generators are r[nb_inner] -> r[nb_inner+nb_outer-1]
// Interior generators are r[nb_inner+nb_outer] -> r[nb_tot]
void NestedSphereCVT::cvt(int *it_num, double *it_diff, double *energy) {

//void NestedSphereCVT::cvt(double r[], int *it_num_boundary, int *it_num_interior, double *it_diff, double *energy,
  //      int it_max_boundary, int it_max_interior, int sample_num) {
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
