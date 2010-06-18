#include <iostream>
#include <iomanip>

using namespace std;

#include "constrained_cvt.h"

ConstrainedCVT::ConstrainedCVT(int DEBUG_) : CVT(DEBUG_) {
    nb_bnd = 0;
    npp = 1000;
}

//****************************************************************************80

void ConstrainedCVT::cvt(int dim_num, int n, int batch, int init, int sample, int sample_num,
        int it_max, int it_fixed, int *seed, double r[], int *it_num, double *it_diff,
        double *energy)

//****************************************************************************80
//
//  Purpose:
//
//    CVT computes a Centroidal Voronoi Tessellation.
//
//  Discussion:
//
//    This routine initializes the data, and carries out the
//    CVT iteration.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    23 June 2005
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Qiang Du, Vance Faber, and Max Gunzburger,
//    Centroidal Voronoi Tessellations: Applications and Algorithms,
//    SIAM Review, Volume 41, 1999, pages 637-676.
//
//  Parameters:
//
//    Input, int DIM_NUM, the spatial dimension.
//
//    Input, int N, the number of Voronoi cells.
//
//    Input, int BATCH, sets the maximum number of sample points
//    generated at one time.  It is inefficient to generate the sample
//    points 1 at a time, but memory intensive to generate them all
//    at once.  You might set BATCH to min ( SAMPLE_NUM, 10000 ), for instance.
//    BATCH must be at least 1.
//
//    Input, int INIT, specifies how the points are to be initialized.
//    -1, 'RANDOM', using C++ RANDOM function;
//     0, 'UNIFORM', using a simple uniform RNG;
//     1, 'HALTON', from a Halton sequence;
//     2, 'GRID', points from a grid;
//     3, 'USER', call "user" routine;
//     4, points are already initialized on input.
//
//    Input, int SAMPLE, specifies how the sampling is done.
//    -1, 'RANDOM', using C++ RANDOM function;
//     0, 'UNIFORM', using a simple uniform RNG;
//     1, 'HALTON', from a Halton sequence;
//     2, 'GRID', points from a grid;
//     3, 'USER', call "user" routine.
//
//    Input, int SAMPLE_NUM, the number of sample points.
//
//    Input, int IT_MAX, the maximum number of iterations.
//
//    Input, int IT_FIXED, the maximum number of iterations to take
//    with a fixed set of sample points.
//
//    Input/output, int *SEED, the random number seed.
//
//    Input/output, double R[DIM_NUM*N], the approximate CVT points.
//    If INIT = 4 on input, then it is assumed that these values have been
//    initialized.  On output, the CVT iteration has been applied to improve
//    the value of the points.
//
//    Output, int *IT_NUM, the number of iterations taken.  Generally,
//    this will be equal to IT_MAX, unless the iteration tolerance was
//    satisfied early.
//
//    Output, double *IT_DIFF, the L2 norm of the difference
//    between the iterates.
//
//    Output, double *ENERGY,  the discrete "energy", divided
//    by the number of sample points.
//
{
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

    *it_num = 0;
    *it_diff = 0.0;
    *energy = 0.0;
    seed_init = *seed;
    //
    //  Initialize the data, unless the user has already done that.
    //
    if (init != 4) {
        initialize = true;
        cvt_sample(dim_num, n, n, init, initialize, seed, r);

        // BOLLIG: Added check for energy and a check for
        energyBefore = cvt_energy(dim_num, n, batch, sample, initialize, sample_num, seed, r);

        project(dim_num, n, r, npp);

        energyAfter = cvt_energy(dim_num, n, batch, sample, initialize, sample_num, seed, r);

        *energy = energyAfter;

    }
    //if ( DEBUG )
    {
        cout << "  "
                << setw(4) << *it_num << "  "
                << setw(12) << seed_init << "  "
                << "  " << "  "
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

    if (initialize) {
        char intermediate_file[80];
        sprintf(intermediate_file, "voronoi_tmp_%.5d.txt", 0);

        cout << "Writing initial voronoi to file (from CVT::cvt())\n";

        cvt_write(dim_num, n, batch, seed_init, *seed, "none",
                it_max, it_fixed, *it_num, *it_diff, *energy, "none", sample_num, r,
                intermediate_file, false);
    }
    //
    //  Carry out the iteration.
    //
    while (*it_num < it_max) {
        //
        //  If it's time to update the seed, save its current value
        //  as the starting value for all iterations in this cycle.
        //  If it's not time to update the seed, restore it to its initial
        //  value for this cycle.
        //
        if (((*it_num) % it_fixed) == 0) {
            seed_base = *seed;
        } else {
            *seed = seed_base;
        }

        *it_num = *it_num + 1;
        seed_init = *seed;

        cvt_iterate(dim_num, n, batch, sample, initialize, sample_num, seed, r, it_diff, &energyBefore);

        project(dim_num, n, r, npp);

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
            sprintf(intermediate_file, "voronoi_tmp_%.5d.txt", *it_num);

            cout << "  "
                    << setw(4) << *it_num << "  "
                    << setw(12) << seed_init << "  "
                    << setw(14) << *it_diff << "  "
                    << setw(14) << energyBefore << "  "
                    << setw(14) << energyAfter << "\n";

            cout << "Writing intermediate voronoi to file (from CVT::cvt())\n";

            cvt_write(dim_num, n, batch, seed_init, *seed, "none",
                    it_max, it_fixed, *it_num, *it_diff, *energy, "none", sample_num, r,
                    intermediate_file, false);
            //exit(0);
        }
    }
    return;
}

//****************************************************************************80

void ConstrainedCVT::project(int ndim, int n, double generator[], int npp) {
    // if (ndim == 2) {
    //   projectSquare(ndim, n, generator, npp);
    // } else if (ndim == 3) {
    projectCube(ndim, n, generator, npp);
    // }
    return;
}


//****************************************************************************80

void ConstrainedCVT::projectSquare(int ndim, int n, double generator[], int npp)
// TODO: this projection routine ONLY supports projection within a square (not CUBE)
//      i need to add support for a 3D cube at least
// 
//****************************************************************************80
//
//  Purpose:
//
//    PROJECT projects generators onto the boundary of the region.
//
//  Discussion:
//
//    The number NPP sets the number of subintervals into which we subdivide
//    the boundary.  It does NOT specify how many points will be pulled onto
//    the boundary.  The reason for this is that, after the first boundary
//    subinterval has had a generator pulled into it, on every subsequent
//    subinterval the nearest generator is likely to be the one in the
//    previous subinterval!  Unless an interior generator is closer than
//    some small distance, this process will simply drag some unfortunate
//    generator onto the boundary, and then around from interval to interval
//    for a considerable time.
//
//    The algorithm could be changed, if desired, so that points snapped
//    to the boundary are guaranteed not to move, at least not twice in
//    one application of this routine!
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 December 2004
//
//  Author:
//
//    Lili Ju
//
//  Parameters:
//
//    Input, int NDIM, the spatial dimension.
//
//    Input, int N, the number of generators.
//
//    Input/output, double GENERATOR[NDIM*N], the coordinates of
//    the generators.  On output, some generators will have been moved.
//
//    Input, int NPP, the number of subintervals into which the
//    perimeter is divided.
//
{
    double dx;
    double dy;
    double hh;
    int i;
    int j;
    int nearest[1];
    double s;
    double u;
    double *sample;

    sample = new double[ndim];

    dx = 1.0;
    dy = 1.0;
    //
    //  HH is the length of an individual segment of the perimeter of the region.
    //
    //  U is set in such a way that on step I, it measures the distance from
    //  the lower left corner of the box to the midpoint of the I-th subinterval
    //  on the perimeter of the box.
    //
    hh = 2.0 * (dx + dy) / (double) (npp);
    // This shifts from the endpoint of the subdivision to the midpoint
    u = -0.5 * hh;

    // I could run this twice for
    // z = 0, z = 1; x=[0,1], y=[0,1]
    // Then come back and check the 4 connecting
    // lines for z=[0,1]; x = 0, x=1, y = 0, y=1
    for (i = 1; i <= npp; i++) {
        u = u + hh;
        //
        //  The portion of the bottom perimeter from (0,0) to (1,0).
        //
        if (u < dx) {
            sample[0] = u;
            sample[1] = 0.0;
            find_closest(ndim, n, 1, sample, generator, nearest);
            generator[1 + nearest[0]*2] = 0.0;
        }//
            //  The portion of the right perimeter from (1,0) to (1,1).
            //
        else if (dx < u && u < dx + dy) {
            sample[0] = 1.0;
            sample[1] = u - dx;
            find_closest(ndim, n, 1, sample, generator, nearest);
            generator[0 + nearest[0]*2] = 1.0;
        }//
            //  The portion of the top perimeter from (1,1) to (0,1).
            //
        else if (dx + dy < u && u < dx + dy + dx) {
            sample[0] = 1.0 - (u - dx - dy);
            sample[1] = 1.0;
            find_closest(ndim, n, 1, sample, generator, nearest);
            generator[1 + nearest[0]*2] = 1.0;
        }//
            //  The portion of the left perimeter from (0,1) to (0,0).
            //
        else if (dx + dy + dx < u) {
            sample[0] = 0.0;
            sample[1] = 1.0 - (u - dx - dy - dx);
            find_closest(ndim, n, 1, sample, generator, nearest);
            generator[0 + nearest[0]*2] = 0.0;
        }

    }

    delete [] sample;

    return;
}

//****************************************************************************80

void ConstrainedCVT::projectCube(int ndim, int n, double generator[], int npp)
// Evan Bollig:
// Projects points onto the surface of a cube
// Needs testing to know if it will work with the boundaries of a square (it should)
{
    double dx;
    double dy;
    double dz;
    double hh;
    int i;
    int j;
    int nearest[1];
    double s;
    double u;
    double *sample;

    // Booleans which tell us if a subvolume is exterior (i.e. on an edge/corner/cube face)
    // in specific dimensions.
    // Example: (exterior_x = false, exterior_y=true, exterior_z = true) => Edge of cube parallel to x-axis
    bool exterior_x = true;
    bool exterior_y = true;
    bool exterior_z = true;

    sample = new double[ndim];

    int nx, ny, nz;
    // We need nx, ny, nz sufficiently large to shrink the size of our boundary
    // voxels and project only a subset of points that are in the domain to the
    // surface.

    nx = ny = (int) pow(npp, 0.33);
    if (ndim > 2) {
        nz = nx; // TODO: get this related to npp
    } else {
        nz = 1;
    }

    dx = 1.0 / (double) (nx - 1);
    dy = 1.0 / (double) (ny - 1);
    dz = 1.0 / (double) (nz - 1); // Z = [0,1] (change 1.0 above to extend domain)

    for (int i = 0; i < nx; i++) {
        exterior_x = true;
        if ((i > 0) && (i < nx - 1)) {
            exterior_x = false;
        }
        for (int j = 0; j < ny; j++) {
            exterior_y = true;
            if ((j > 0) && (j < ny - 1)) {
                exterior_y = false;
            }

            for (int k = 0; k < nz; k++) {

                exterior_z = true;

                // 1) Check if voxel is on boundary
                //    (if it is not on the boundary we can ignore it)
                // 2) Find the closest point to the voxel
                // 3) Whichever dimensions of the boundary voxels are 0 and 1,
                //    are then used to overwrite the nearest point
                //    (e.g.,if i = 0, j = ny, k = nz; then generator(x,y,z) = (x, 1, 1)

                // 1)

                if ((k > 0) && (k < nz - 1)) {
                    exterior_z = false;
                }

                if (exterior_x || exterior_y || exterior_z) {
                    // 2)
#if 0
                    sample[0] = i * dx + 0.5 * dx; // include (- 0.5*dx) to get to midpoint of volume
                    sample[1] = j * dy + 0.5 * dy;
                    if (ndim > 2) {
                        sample[2] = k * dz + 0.5 * dz;
                    }
#else
                    sample[0] = i * dx; // include (- 0.5*dx) to get to midpoint of volume
                    sample[1] = j * dy;
                    if (ndim > 2) {
                        sample[2] = k * dz;
                    }
#endif 
                    // TODO: really we want the sample to be ON the boundary, not within
                    // it. Consider the case when ijk describe a corner region. Then the
                    // sample is the center of the volume.
                    // When one of the volume is an edge of the cube the sample is
                    // in the center of one side of the volume.
                    // This implies we are rounding corners of the cube to get the projection
                    // Also,

                    find_closest(ndim, n, 1, sample, generator, nearest);
                    //generator[1+nearest[0]*2] = 0.0;

                    double x1 = generator[0 + nearest[0] * ndim] - sample[0];
                    double y1 = generator[1 + nearest[0] * ndim] - sample[1];
                    double z1;
                    if (ndim > 2) {
                        z1 = generator[2 + nearest[0] * ndim] - sample[2];
                    } else {
                        z1 = 0;
                    }

                    double dist = sqrt(x1 * x1 + y1 * y1 + z1 * z1);

                    if (dist < dx) {
                        // 3) Project to x, y or z (remember: nearest contains index
                        // of the nearest point. That is index*NDIM doubles down the
                        // list of doubles in generator; then we add 0, 1, or 2 for
                        // x, y, z respectively.)
                        if (exterior_x) {
                            generator[0 + nearest[0] * ndim] = i*dx;
                            // printf("GOT X %d*%f=%f\n", i, dx, i*dx);
                        }
                        if (exterior_y) {
                            generator[1 + nearest[0] * ndim] = j*dy;
                            // printf("GOT Y %d*%f=%f\n", j, dy, j*dy);
                        }
                        if (ndim > 2 && exterior_z) {
                            generator[2 + nearest[0] * ndim] = k*dz;
                            // printf("GOT Z\n");
                        }
                    }
                }
            }
        }
    }

    delete [] sample;

    return;
}
