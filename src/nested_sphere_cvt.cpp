#include <iostream>
#include <iomanip>

using namespace std;

#include "nested_sphere_cvt.h"

NestedSphereCVT::NestedSphereCVT(int nb_inner_bnd, int nb_outer_bnd, int nb_interior, double inner_radius, double outer_radius, int dimension, int DEBUG_)
: CVT(DEBUG_), inner_r(inner_radius), outer_r(outer_radius), dim_num(dimension),
nb_inner(nb_inner_bnd), nb_outer(nb_outer_bnd), nb_int(nb_interior),
seed(123456789), batch(1000), it_fixed(1), init(-1), sample(-1) {
    nb_bnd = 0;

   
#if 0
    // For testing: quarter spheres
    geom_extents = new double[2 * dim_num];
    for (int i = 0; i < dim_num; i++) {
        geom_extents[0 + i*2] = -1;
        geom_extents[1 + i*2] = 0;
    }
    geom_extents[0 + 0*2] = -1;
    geom_extents[1 + 0*2] = 0;
#else
     // Create bounding volume: [-1,1]^3
    geom_extents = new double[2 * dim_num];
    for (int i = 0; i < dim_num; i++) {
        geom_extents[0 + i*2] = -1;
        geom_extents[1 + i*2] = 1;
    }
#endif
}

//****************************************************************************80

// Inner generators are r[0] -> r[nb_inner-1]
// Outer generators are r[nb_inner] -> r[nb_inner+nb_outer-1]
// Interior generators are r[nb_inner+nb_outer] -> r[nb_tot]

void NestedSphereCVT::cvt(double r[], int *it_num_boundary, int *it_num_interior, double *it_diff, double *energy,
        int it_max, int sample_num) {
    int i;
    bool initialize;
    int seed_base;
    int seed_init;
    double energyBefore = 0.0, energyAfter = 0.0;

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

    *it_num_boundary = 0;
    *it_num_interior = 0;
    *it_diff = 0.0;
    *energy = 0.0;
    seed_init = seed;
    //
    //  Initialize the data, unless the user has already done that.
    //
    if (init != 4) {
        initialize = true;
        // Randomly sample all initial generators from nested spheres (note: cvt_sample is
        // overridden to achieve this sampling
        cvt_sample(dim_num, nb_inner + nb_outer + nb_int, nb_inner + nb_outer + nb_int, init, initialize, &seed, r);

        // BOLLIG: Added check for energy and a check for
        energyBefore = cvt_energy(dim_num, nb_inner + nb_outer, batch, sample, initialize, sample_num, &seed, r);

        // Inner is [0] -> [nb_inner-1]
        project_to_sphere(&r[0], nb_inner, dim_num, inner_r);

        // Outer generators are r[nb_inner] -> r[nb_inner+nb_outer-1]
        project_to_sphere(&r[nb_inner*dim_num], nb_outer, dim_num, outer_r);

        // Interior generators are r[nb_inner+nb_outer] -> r[nb_tot] and require no projection

        energyAfter = cvt_energy(dim_num, nb_inner + nb_outer + nb_int, batch, sample, initialize, sample_num, &seed, r);

        *energy = energyAfter;

    }
    //if ( DEBUG )
    {
        cout << "  "
                << setw(4) << *it_num_boundary << "  "
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

    cout << "Writing initial voronoi to file (from CVT::cvt())\n";

    sprintf(intermediate_file, "boundary_nodes_%.5d_inner_%.5d_outer_%.5d_iter.txt", nb_inner, nb_outer, 0);
    cvt_write(dim_num, nb_inner + nb_outer, batch, seed_init, seed, "none",
            it_max, it_fixed, *it_num_boundary, *it_diff, *energy, "none", sample_num, &r[0],
            intermediate_file, false);

    sprintf(intermediate_file, "interior_nodes_%.5d_interior_%.5d_iter.txt", nb_int, 0);
    cvt_write(dim_num, nb_int, batch, seed_init, seed, "none",
            it_max, it_fixed, *it_num_interior, *it_diff, *energy, "none", sample_num, &r[(nb_inner + nb_outer)*dim_num],
            intermediate_file, false);
    //}
    //
    //  Carry out the iteration.
    //
    while (*it_num_boundary < it_max) {
        //
        //  If it's time to update the seed, save its current value
        //  as the starting value for all iterations in this cycle.
        //  If it's not time to update the seed, restore it to its initial
        //  value for this cycle.
        //
        if (((*it_num_boundary) % it_fixed) == 0) {
            seed_base = seed;
        } else {
            seed = seed_base;
        }

        *it_num_boundary = *it_num_boundary + 1;
        seed_init = seed;

        cvt_iterate(dim_num, nb_inner + nb_outer + nb_int, batch, sample, initialize, sample_num, &seed, r, it_diff, &energyBefore);

        // Inner is [0] -> [nb_inner-1]
        project_to_sphere(&r[0], nb_inner, dim_num, inner_r);

        // Outer generators are r[nb_inner] -> r[nb_inner+nb_outer-1]
        project_to_sphere(&r[nb_inner*dim_num], nb_outer, dim_num, outer_r);


        energyAfter = cvt_energy(dim_num, nb_inner + nb_outer + nb_int, batch, sample, initialize, sample_num, &seed, r);

        *energy = energyAfter;

        initialize = false;

        if (DEBUG) {
            cout << "  "
                    << setw(4) << *it_num_boundary << "  "
                    << setw(12) << seed_init << "  "
                    << setw(14) << *it_diff << "  "
                    << setw(14) << energyBefore << "  "
                    << setw(14) << energyAfter << "\n";
        }

        // BOLLIG:
        // TODO: only do this if a boolean is set for intermediate writes
        // 	not the same as DEBUG
        if ((*it_num_boundary) % 20 == 0) {
            char intermediate_file[80];

            cout << "  "
                    << setw(4) << *it_num_boundary << "  "
                    << setw(12) << seed_init << "  "
                    << setw(14) << *it_diff << "  "
                    << setw(14) << energyBefore << "  "
                    << setw(14) << energyAfter << "\n";

            cout << "Writing intermediate voronoi to file (from CVT::cvt())\n";

            sprintf(intermediate_file, "boundary_nodes_%.5d_inner_%.5d_outer_%.5d_iter.txt", nb_inner, nb_outer, *it_num_boundary);
            cvt_write(dim_num, nb_inner + nb_outer, batch, seed_init, seed, "none",
                    it_max, it_fixed, *it_num_boundary, *it_diff, *energy, "none", sample_num, &r[0],
                    intermediate_file, false);
        }
    }

    // Wait on interior until we know boundary works
#if 0
    // Now that the boundary is locked, we form the interior CVT using the boundary
    // as reference
    while (*it_num_interior < it_max) {
        //
        //  If it's time to update the seed, save its current value
        //  as the starting value for all iterations in this cycle.
        //  If it's not time to update the seed, restore it to its initial
        //  value for this cycle.
        //
        if (((*it_num_interior) % it_fixed) == 0) {
            seed_base = *seed;
        } else {
            *seed = seed_base;
        }

        *it_num_interior = *it_num_interior + 1;
        seed_init = *seed;

        cvt_iterate(dim_num, n, batch, sample, initialize, sample_num, seed, r, it_diff, &energyBefore);

        // Inner is [0] -> [nb_inner-1]
        project_to_sphere(&r[0], nb_inner, dim_num, inner_r);

        // Outer is [nb_inner] -> [nb_inner+nb_outer-1]

        // Interior is [nb_inner+nb_outer] -> [nb_tot]

        energyAfter = cvt_energy(dim_num, n, batch, sample, initialize, sample_num, seed, r);

        *energy = energyAfter;

        initialize = false;

        if (DEBUG) {
            cout << "  "
                    << setw(4) << *it_num << "  "
                    << setw(12) << seed_init << "  "
                    << setw(14) << *it_diff << "  "
                    << setw(14) << energyBefore << "  "
                    << setw(14) << energyAfter << "\n";
        }

        // BOLLIG:
        // TODO: only do this if a boolean is set for intermediate writes
        // 	not the same as DEBUG
        if ((*it_num) % 20 == 0) {
            char intermediate_file[80];

            cout << "  "
                    << setw(4) << *it_num << "  "
                    << setw(12) << seed_init << "  "
                    << setw(14) << *it_diff << "  "
                    << setw(14) << energyBefore << "  "
                    << setw(14) << energyAfter << "\n";

            cout << "Writing intermediate voronoi to file (from CVT::cvt())\n";

            sprintf(intermediate_file, "boundary_nodes_%.5d_inner_%.5d_outer_%.5d_iter.txt", nb_inner, nb_outer, 0);
            cvt_write(dim_num, nb_inner + nb_outer, batch, seed_init, *seed, "none",
                    it_max, it_fixed, *it_num, *it_diff, *energy, "none", sample_num, &r[0],
                    intermediate_file, false);

            sprintf(intermediate_file, "interior_nodes_%.5d_interior_%.5d_iter.txt", nb_int, 0);
            cvt_write(dim_num, nb_int, batch, seed_init, *seed, "none",
                    it_max, it_fixed, *it_num, *it_diff, *energy, "none", sample_num, &r[nb_inner + nb_outer],
                    intermediate_file, false);
        }
    }
#endif
    return;
}

//****************************************************************************80

void NestedSphereCVT::cvt_sample(int dim_num, int n, int n_now, int sample, bool initialize,
        int *seed, double r[])

//****************************************************************************80
//
//  Purpose:
//
//    CVT_SAMPLE returns sample points.
//
//  Discussion:
//
//    N sample points are to be taken from the box of dimension DIM_NUM with
//    specified extent.
//
//    These sample points are usually created by a pseudorandom process
//    for which the points are essentially indexed by a quantity called
//    SEED.  To get N sample points, we generate values with indices
//    SEED through SEED+N-1.
//
//    It may not be practical to generate all the sample points in a
//    single call.  For that reason, the routine allows the user to
//    request a total of N points, but to require that only N_NOW be
//    generated now (on this call).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    16 June 2010 (Bollig)
//
//  Author:
//
//    John Burkardt and Evan Bollig
//
//  Parameters:
//
//    Input, int DIM_NUM, the spatial dimension.
//
//    Input, int N, the number of Voronoi cells.
//
//    Input, int N_NOW, the number of sample points to be generated
//    on this call.  N_NOW must be at least 1.
//
//    Input, int SAMPLE, specifies how the sampling is done.
//    -1, 'RANDOM', using C++ RANDOM function;
//
//    Input, bool INITIALIZE, is TRUE if the pseudorandom process should be
//    reinitialized.
//
//    double BOX_EXTENTS[2*NDIM], specify the extents of the bounding box to sample. Originally
//     this routine samples unit box ([0,1]^ndim). This allows us to change the extents
//     and sample for example [-1,1]^ndim. Data stored as: [xMin,xMax,yMin,yMax,zMin,zMax, ...]
//
//    Input/output, int *SEED, the random number seed.
//
//    Output, double R[DIM_NUM*N_NOW], the sample points.
//
{
    int i;
    int j;

    if (n_now < 1) {
        cout << "\n";
        cout << "CVT_SAMPLE - Fatal error!\n";
        cout << "  N_NOW < 1.\n";
        exit(1);
    }

    if (sample != -1) {

        printf("WARNING! ONLY RANDOM SAMPLING IS SUPPORTED IN %s. Exiting...\n", __FILE__);
        exit(EXIT_FAILURE);
    }
    if (initialize) {
        random_initialize(*seed);
    }

    for (j = 0; j < n_now; j++) {
        bool is_rejected = true;
        while (is_rejected) {
            for (i = 0; i < dim_num; i++) {
                // This is Gordon Erlebacher's random(,) routine (defined below)
                r[i + j * dim_num] = random(geom_extents[0 + i*2], geom_extents[1 + i*2]);
            }

            // We have a sample point in the bounding box, so lets try to
            // reject it (e.g., if its outside the geometry contained by bounding
            // box.
            is_rejected = reject_point(&r[j * dim_num], dim_num);
        }
    }
    *seed = (*seed) + n_now * dim_num;


    // print seeds
    if (DEBUG) {
        printf("Initial seed positions\n");
        for (int i = 0; i < n_now; i++) {
            printf("(%d): \n", i);
            for (int j = 0; j < dim_num; j++) {
                printf("%f ", r[j + i * dim_num]);
            }
            printf("\n");
        }
        printf("  -  end initial seeds --------------------\n");
    }
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
            generator[i + j * ndim] *= (radius/norm);
        }
    }

    return;
}