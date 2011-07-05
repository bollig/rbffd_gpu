//EFB052611: 
//TODO: - NEEd to add inner geometry (ie nested ellipsoid)
//      - Handle NON-uniform density with our boundary node placement

# include <cstdlib>
# include <cmath>
# include <ctime>
# include <iostream>
# include <iomanip>
# include <fstream>
#include <string.h>

#include <math.h>
#include <vector>

#include "ellipsoid_cvt.h"

using namespace std;

#if 0
//****************************************************************************80
    EllipsoidCVT::EllipsoidCVT(double major_, double minor_, double midax_, int DEBUG_)
: CVT(DEBUG_) 
{ 
    nb_bnd = 0;
}
//****************************************************************************80
#endif 

void EllipsoidCVT::displaceBoundaryNodes(int dim_num, int nb_bnd_nodes, double r_computed[], double r_updated[]) 
{
#if 0
    // FIXME: we should only project boundary nodes
    int n = nb_pts;

    // TODO: this should happen in USER_SAMPLE
    // GE: track whether Voronoi cell "intersects the boundary
    int* intersect_bnd = new int [n]; // n seeds
    //printf("n= %d\n", n); exit(0);

    for (int i=0; i < n; i++) {
        intersect_bnd[i] = 0;
    }
#endif

    for (int i = 0; i < nb_bnd_nodes; i++) {
        Vec3 pt; 
        for (int j = 0; j < dim_num; j++) {
            pt[j] = r_computed[i*dim_num + j];
        }

//EFB052611
#if 0 
        // Use iterative projection (slow)
        Vec3 projected = geom->project(pt);
#else 
        // exact projection (original was buggy, modified 052611 to wrap
        // singleProjectStep() and now it clusters in the ellipsoid end-caps 
        Vec3 projected = geom->projectToBoundary(pt);
#endif 
        for (int j = 0; j < dim_num; j++) {
            r_updated[i*dim_num + j] = projected[j];
        }
    }

    // TODO: fix
#if 0

    int* count = new int [n]; // n seeds
    int cnt = 0;
    Vec3 pt, ptc;
    for (int j=0; j < n; j++) {
        // if count == abs(intersect_bnd), then there is no intersection
        if (count[j] != abs(intersect_bnd[j])) {
            cnt++;

            //printf("intersect with bndry\n");
            ptc.setValue(r[0+j*dim_num],r[1+j*dim_num],r[2+j*dim_num]);
            //pt = geom->projectToBoundary(ptc);
            pt = geom->project(ptc);
            //ptc.print("pt off bnd (ptc)");
            //pt.print("pt on bnd");

            double howfar = geom->how_far(pt);
            Vec3 slope = (pt-ptc);
            slope.normalize();
            Vec3 grad = geom->gradient(pt.x(), pt.y(), pt.z());
            Vec3 gradptc = geom->gradient(ptc.x(), ptc.y(), ptc.z());
            grad.normalize();
            gradptc.normalize();
#if 0
            printf("===========================\n");
            ptc.print("ptc");
            pt.print("pt");
            grad.print("grad(@pt)");
            gradptc.print("grad(@ptc)");
            slope.print("slope (pt-ptc)");
            grad.print("grad(pt)");
#endif 
        }
        //printf("count: %d, intersect: %d, diff= %d\n", count[j], intersect_bnd[j],
        //count[j]-abs(intersect_bnd[j]));
        // project points with intersect == 1
    }
    printf("nb intersections: %d\n", cnt);
#endif 
}



//----------------------------------------------------------------------
void EllipsoidCVT::ellipsoid_init ( int dim_num, int& n, int *seed, double r[] )
{
    // r[] is a vector of with 3 components at each point. Make sure memory is properly allocated!

    vector<Vec3> samples;
    samples.resize(n); // total number of points

    // random seeds with rejection2d.
    printf("ellipsoid_init: rejection3d\n");
    rejection3d(n,  *rho, samples);
    printf("n= %d, nb_bnd= %d\n", n, nb_bnd);
    //exit(0);

    for (int j = 0; j < nb_bnd; j++) {
        Vec3 projected = geom->project(samples[j]); 
        for (int i=0; i < dim_num; i++) {
            r[i+j*dim_num] = projected[i];
        }
    }
#if 0 
    for ( int j = nb_bnd; j < n; j++ ) {
        for (int i = 0; i < dim_num; i++) {
            r[i+j*dim_num] = samples[j][i];
        }
    }
#endif 
#if 1
    // print initial seeds
    printf("Initial seed positions, %d seeds\n", n);
    for (int i=0; i < n; i++) {
        printf("(%d): \n", i);
        for (int j=0; j < dim_num; j++) {
            printf("%f ", r[j+i*dim_num]);
        }
        printf("\n");
    }
    printf(" -----  end initial seeds --------------------\n");
#endif 
    return;
}
//----------------------------------------------------------------------
void EllipsoidCVT::ellipsoid ( int dim_num, int& n, int *seed, double r[] )

    // return number boundary points
    // n = total number of points

    //****************************************************************************80
    //
    //  Purpose:
    //
    //    USER samples points within an ellipse, given the boundary points
    //    Grid points are uniform. For each point, compute r and theta. Remove the points outside the ellipse
    //
    //  Modified:
    //    14 March 2010
    //
    //  Author:
    //    Gordon Erlebacher
    //
    //  Parameters:
    //    Input, integer DIM_NUM, the spatial dimension.
    //    Input, integer N, the number of sample points desired.
    //    Input/output, int *SEED, the "seed" value.  On output, SEED has
    //       been updated.
    //    Output, double R[DIM_NUM*N], the sample values.
    //
    // Ellipsoid: (phi in [0,2*pi], theta in [0,pi]
    //   x= a*cos(theta)*cos(phi)
    //   y= b*sin(theta)*cos(phi)
    //   z= c*sin(phi)
{

    // na : is the number of points along the major axis
    // nb : is the number of points along the minor axis
    // nc : is the number of points along the middle axis

    vector<Vec3> samples;
    samples.resize(n);
    rejection3d(n, *rho, samples);

    for (int i=0; i < n; i++) {
        for (int j = 0; j < dim_num; j++) {
            r[j + i*dim_num] = samples[i][j];
        }
    }

    return;
}
//----------------------------------------------------------------------
void EllipsoidCVT::rejection3d(int nb_samples, Density& density, vector<Vec3>& samples)
    // Given a pdf p(x,y) (zero outside the ellipse, with unit integral over the ellipsoid domain)
    // and given a constant bound M, such that p(x,y,z) < M q(x,y,z), where q(x,y,z) is uniform distribution over the ellipse.
    // Generate a sample (x*,y*,z*), and a sample u* in [0,1]
    // x* \in [-a,a], y* \in [-b,b], z* \in [-c,c]
    // Two cases:
    // 1.   u < p(x,y,z) / M q(x,y,z) : accept (x*,y*,z*)
    // 2.   u > p(x,y,z) / M q(x,y,z)   reject (x*,y*,z*)
    //
    // What is q(x,y): simply   1. / volume(ellipsoid)
    // Choose M = max_{x,y}   p(x,y)*volume(ellipsoid)
    // Given x*,y*,z* compute r*^2 = (x*/a)^2 + (y*/b*)^2 + (z*/c*)^2
    // if r*^2 > 1., reject the point (outside the ellipsoid)

    // Used in ellipse_init (constructor helper) and ellipsoid
    // 3D version should be written though, but the ellipse is a special case
    // (no hurry)
{
    samples.resize(nb_samples);
    //	printf("nb_samples= %d\n", nb_samples);

    for (int i=0; i < nb_samples; i++) {
        samples[i] = singleRejection3d(density);
        //		samples[i].print("samples3d");
        Vec3& s = samples[i];
        //		printf("dist=%f\n", outer_geom->how_far(s.x(), s.y(), s.z()));
    }
}
//----------------------------------------------------------------------
Vec3 EllipsoidCVT::singleRejection3d(Density& density)
{
    // Apparently not used in this file
    // 3D version should be written though, but the ellipse is a special case
    // (no hurry)
    double major = axis_major;

    double xs, ys, zs;
    double u;
    double r2;
    double maxrhoi = 1. / rho->getMax();

    double dist;

    while (1) {
        // should use other geometry
        // WE sample the full extents of the domain, then reject nodes that lie outside the geometry
        xs = random(-major, major);
        ys = random(-major, major);
        zs = random(-major, major);

        // use outer geometry for seed and random point distribution
        dist = geom->how_far(xs,ys,zs);
        if (dist > 0.) continue;  // outside the ellipse

        // rejection part if non-uniform distribution
        u = random(0.,1.);
        if (u < (density(xs,ys,zs))*maxrhoi) break;
    }
    return Vec3(xs,ys,zs);
}
//----------------------------------------------------------------------

//****************************************************************************
#if 0
void EllipsoidCVT::cvt_iterate ( int dim_num, int n, int batch, int sample, bool initialize,
        int sample_num, int *seed, double r[], double *it_diff, double *energy )

//****************************************************************************80
//
//  Purpose:
//
//    EllipsoidCVT_ITERATE takes one step of the EllipsoidCVT iteration.
//
//  Discussion:
//
//    The routine is given a set of points, called "generators", which
//    define a tessellation of the region into Voronoi cells.  Each point
//    defines a cell.  Each cell, in turn, has a centroid, but it is
//    unlikely that the centroid and the generator coincide.
//
//    Each time this EllipsoidCVT iteration is carried out, an attempt is made
//    to modify the generators in such a way that they are closer and
//    closer to being the centroids of the Voronoi cells they generate.
//
//    A large number of sample points are generated, and the nearest generator
//    is determined.  A count is kept of how many points were nearest to each
//    generator.  Once the sampling is completed, the location of all the
//    generators is adjusted.  This step should decrease the discrepancy
//    between the generators and the centroids.
//
//    The centroidal Voronoi tessellation minimizes the "energy",
//    defined to be the integral, over the region, of the square of
//    the distance between each point in the region and its nearest generator.
//    The sampling technique supplies a discrete estimate of this
//    energy.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    20 September 2004
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
//    Input, int SAMPLE, specifies how the sampling is done.
//    -1, 'RANDOM', using C++ RANDOM function;
//     0, 'UNIFORM', using a simple uniform RNG;
//     1, 'HALTON', from a Halton sequence;
//     2, 'GRID', points from a grid;
//     3, 'USER', call "user" routine.
//
//    Input, bool INITIALIZE, is TRUE if the SEED must be reset to SEED_INIT
//    before computation.  Also, the pseudorandom process may need to be
//    reinitialized.
//
//    Input, int SAMPLE_NUM, the number of sample points.
//
//    Input/output, int *SEED, the random number seed.
//
//    Input/output, double R[DIM_NUM*N], the Voronoi
//    cell generators.  On output, these have been modified
//
//    Output, double *IT_DIFF, the L2 norm of the difference
//    between the iterates.
//
//    Output, double *ENERGY,  the discrete "energy", divided
//    by the number of sample points.
//
{
    int *count;
    int get;
    int have;
    int i;
    int j;
    int j2;
    int *nearest;
    double *r2;
    double *s;
    bool success;
    double term;
    //
    //  Take each generator as the first sample point for its region.
    //  This can slightly slow the convergence, but it simplifies the
    //  algorithm by guaranteeing that no region is completely missed
    //  by the sampling.
    //

    *energy = 0.0;
    r2 = new double[dim_num*n];
    count = new int[n];
    nearest = new int[sample_num];
    s = new double[dim_num*sample_num];

    for ( j = 0; j < n; j++ ) {
        for ( i = 0; i < dim_num; i++ ) {
            r2[i+j*dim_num] = r[i+j*dim_num];
        }}
    for ( j = 0; j < n; j++ ) {
        count[j] = 0;   // 3/12/10
        //count[j] = 1; // original code (wrong in GE opinion)
    }
    //
    //  Generate the sampling points S.
    //
    have = 0;

    // GE: track whether Voronoi cell "intersects the boundary
    int* intersect_bnd = new int [n]; // n seeds
    //printf("n= %d\n", n); exit(0);

    for (int i=0; i < n; i++) {
        intersect_bnd[i] = 0;
    }

    while ( have < sample_num ) {   // ***********************
        get = i4_min ( sample_num - have, batch );
        cvt_sample ( dim_num, sample_num, get, sample, initialize, seed, s );

        //printf("cvt_sample3d\n");exit(0);

        initialize = false;
        have = have + get;
        //
        //  Find the index N of the nearest cell generator to each sample point S.
        //
        // brute force (GE: MUST IMPROVE!!)
        // find nearest r point to each s point
        find_closest ( dim_num, n, get, s, r, nearest );
        //
        //  Add S to the centroid associated with generator N.
        //  Gordon Erlebacher: Build in check for including within the geometry
        //    at this point. Each time a sample point is inside the geometry, set
        //    a flag stating that the seed "intersects" the boundary
        //
        Vec3 inside;

        for ( j = 0; j < get; j++ ) {
            j2 = nearest[j];
            for ( i = 0; i < dim_num; i++ ) {
                // r2: centroid, s: sample point
                r2[i+j2*dim_num] = r2[i+j2*dim_num] + s[i+j*dim_num];
            }
            inside.setValue(s[0+j*dim_num],s[1+j*dim_num],s[2+j*dim_num]);
            double dist = geom->how_far(inside); // use actual geometry
            if (dist > 0.) { // outside
                //printf("** dist= %f, j2= %d\n", dist, j2);
                //inside.print("inside vec");
                intersect_bnd[j2] += 1; // j2 is the seed
            } else {
                intersect_bnd[j2] -= 1;
            }
            //printf("dist (< 0 if inside) = %f\n", dist);

            for ( i = 0; i < dim_num; i++ ) {
                *energy = *energy + pow ( r[i+j2*dim_num] - s[i+j*dim_num], 2 );
            }
            count[j2] = count[j2] + 1;
        }
    } //******************

    //
    //  Estimate the centroids.
    //
    for ( j = 0; j < n; j++ ) {
        for ( i = 0; i < dim_num; i++ ) {
            if (count[j] > 0) {
                r2[i+j*dim_num] = r2[i+j*dim_num] / ( double ) ( count[j] );
            } else {
                printf("count[j] should never be zero\n");
                //exit(1);
            }
        }}
    //
    //  Determine the sum of the distances between generators and centroids.
    //
    *it_diff = 0.0;

    for ( j = 0; j < n; j++ ) {
        term = 0.0;
        for ( i = 0; i < dim_num; i++ ) {
            term = term + ( r2[i+j*dim_num] - r[i+j*dim_num] )
                * ( r2[i+j*dim_num] - r[i+j*dim_num] );
        }
        *it_diff = *it_diff + sqrt ( term );
    }
    //
    //  Replace the generators by the centroids.
    //

    // Identify which cells intersect the boundary.
    // When a cell intersects a boundary (i.e., a sample point is outside the
    // ellipse), project the centroid onto the boundary.

    // Displace all the points
    // after displacement, project  "boundary centroids" to the surface

    for ( j = 0; j < n; j++ ) {
        for ( i = 0; i < dim_num; i++ ) {
            r[i+j*dim_num] = r2[i+j*dim_num];
        }}

    int cnt = 0;
    Vec3 pt, ptc;
    for (j=0; j < n; j++) {
        // if count == abs(intersect_bnd), then there is no intersection
        if (count[j] != abs(intersect_bnd[j])) {
            cnt++;
            printf("===========================\n");
            //printf("intersect with bndry\n");
            ptc.setValue(r[0+j*dim_num],r[1+j*dim_num],r[2+j*dim_num]);
            //pt = geom->projectToBoundary(ptc);
            pt = geom->project(ptc);
            //ptc.print("pt off bnd (ptc)");
            //pt.print("pt on bnd");

            double howfar = geom->how_far(pt);
            Vec3 slope = (pt-ptc);
            slope.normalize();
            Vec3 grad = geom->gradient(pt.x(), pt.y(), pt.z());
            Vec3 gradptc = geom->gradient(ptc.x(), ptc.y(), ptc.z());
            grad.normalize();
            gradptc.normalize();
            ptc.print("ptc");
            pt.print("pt");
            grad.print("grad(@pt)");
            gradptc.print("grad(@ptc)");
            slope.print("slope (pt-ptc)");
            grad.print("grad(pt)");
        }
        //printf("count: %d, intersect: %d, diff= %d\n", count[j], intersect_bnd[j],
        //count[j]-abs(intersect_bnd[j]));
        // project points with intersect == 1
    }
    printf("nb intersections: %d\n", cnt);

#if 0
    printf("(%s, Line: %d) EXIT_FAILURE\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
#endif 

    //printf("r2: %f, %f\n", r2[0+nb_bnd*dim_num], r2[1+nb_bnd*dim_num]);
    //exit(0);
    //
    //  Normalize the discrete energy estimate.
    //
    *energy = *energy / sample_num;

    printf("exit iterate_3d\n");

    delete [] count;
    delete [] nearest;
    delete [] r2;
    delete [] s;

    return;
}
#endif 


void EllipsoidCVT::user_sample(int dim_num, int n, int *seed, double r[]) {
    // The guts and glory (this is the user defined ellipse sampling)
    ellipsoid(dim_num, n, seed, r);

#if 0
    // TODO: track which samples returned by user_sample intersect the boundary geometry 
    Vec3 inside;
    for (int j = 0; j < n; j++) {
        // s is the validated sample
        inside.setValue(j[0+j*dim_num],s[1+j*dim_num],s[2+j*dim_num]);
        double dist = geom->how_far(inside); // use actual geometry
        if (dist > 0.) { // outside
            //printf("** dist= %f, j2= %d\n", dist, j2);
            //inside.print("inside vec");
            intersect_bnd[j2] += 1; // j2 is the seed
        } else {
            intersect_bnd[j2] -= 1;
        }
        //printf("dist (< 0 if inside) = %f\n", dist);
    }
#endif 


}

void EllipsoidCVT::user_init(int dim_num, int n, int *seed, double r[]) {

    this->fillBoundaryPoints(dim_num, nb_nodes, seed, r);

    // This is the user defined initialization
    //ellipsoid_init(dim_num, n, seed, r);
    this->user_sample(dim_num, n-nb_bnd, seed, &r[nb_bnd*dim_num]);
}


//----------------------------------------------------------------------

void EllipsoidCVT::fillBoundaryPoints(int dim_num, int nb_nodes, int *seed, double bndry_nodes[])
{
    std::vector<double> intg;

    unsigned int high_nb_pts = 200;
    double bnd_intg = computeBoundaryIntegral(*rho, high_nb_pts, intg);
    double dom_intg = computeDomainIntegral(high_nb_pts, *rho);

    // total nb points used to compute Voronoi mesh. 
    // Only (nb_interior_pts-nb_bnd) will be able to move freely
    unsigned int tot_nb_pts = nb_nodes;
    // number of boundary points, automatically calculated
    printf("tot_nb_pts= %d\n", tot_nb_pts);
    printf("domain integral = %f\n", dom_intg);
    printf("boundary integral = %f\n", bnd_intg);

    // These are heuristic.
    // Assume an Ellipsoid has volume 4/3 * pi * a * b *c
    // Divide that by the total number of nodes and we get an average voronoi volume 
    // Its not exact, but we can imagine the volume is approximately cube. Thus taking
    // the cube root gives us an average separation between nodes. 
    // The number of nodes on the boundary is then our surface integral divided
    // by the separation squared. 
    double avg_vol = dom_intg / nb_nodes; 
    double avg_sep = pow(avg_vol, 1./3.); 
    double nb_computed_bnd = bnd_intg / (avg_sep * avg_sep);

#if 0
    //EFB052611: This is the ellipse
    unsigned int nb_computed_bnd = bnd_intg * sqrt(tot_nb_pts / dom_intg);
    unsigned int nb_bnd_1 = 1. + 16. * tot_nb_pts * dom_intg / (bnd_intg * bnd_intg);
    nb_bnd_1 = -bnd_intg * bnd_intg / (4. * dom_intg) * (1. - sqrt(nb_bnd_1));
    nb_computed_bnd = nb_bnd_1; // more accurate formula
    printf("calculated nb boundary pts: %d\n", nb_computed_bnd);
    printf("improved nb boundary pts: %d\n", nb_bnd_1);
#endif 

    // Now that we know how many boundary nodes we want out of the total
    // number of nodes in the domain, resize to that value
    this->nb_bnd = (int)nb_computed_bnd;
    this->resizeBoundary(this->nb_bnd);
    //bndry_pts.resize(this->nb_bnd);

    // Now compute the actual boundary nodes: 
    this->computeBoundaryPointDistribution(dim_num, bnd_intg, high_nb_pts, nb_bnd, intg, bndry_nodes);

    // Verify that things match
    //		printf("nb_bnd= %d, bndry_pts.size= %d\n", nb_bnd, (int) bndry_pts.size());
    //        printf("node_list.size= %d\n" , this->node_list.size());
    //
}

//----------------------------------------------------------------------
// Compute the line integral of the boundary using npts number of samples/divisions
//
// npts: number of sample points (first and last points are the same for closed intervals)
// rho: functor computing point density (= 1 + some function(x,y))
// bnd: list of boundary points
// return: value of boundary integral
//
// NOTE: npts should be large! At least 200. 
//
// GORDONs comment: Ideally, I should be integrating with respect to theta for 
// more accuracy.
double EllipsoidCVT::computeBoundaryIntegral(Density& rho, unsigned int npts, vector<double>& intg)
{
    intg.resize(npts);
#if 0
    double major = axis_major; 
    double minor = axis_minor;

    double pi = acos(-1.);
    double dtheta = 2. * pi / (npts - 1.);

    //npts = 2000;
    //vector<double> intg;
    intg.resize(npts);

    intg[0] = 0.;

    // npts-1 is the number of intervals

    for (int i = 0; i < (npts - 1); i++) {
        double t1 = i * dtheta;
        double t2 = (i + 1) * dtheta;
        //printf("t1,t2= %f, %f\n", t1, t2);
        double tm = 0.5 * (t1 + t2);
        double x1 = major * cos(t1);
        double x2 = major * cos(t2);
        double xm = major * cos(tm);
        double y1 = minor * sin(t1);
        double y2 = minor * sin(t2);
        double ym = minor * sin(tm);
        double dl = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
        double rhom = rho(xm, ym, 0.);
        intg[i + 1] = intg[i] + pow(rhom, 0.25) * dl;
    }

    // new boundary will have n points (n << npts)
    // divide into (n-1) equal length intervals
    double tot_length = intg[npts - 1];

    printf("boundary integral: %f\n", tot_length);
#endif 

    double tot_length = geom->surfaceIntegral();
    return tot_length;
}

//----------------------------------------------------------------------

double EllipsoidCVT::computeDomainIntegral(unsigned int npts, Density& rho) {
    double major = axis_major; 
    double minor = axis_minor;
    double midax = axis_midax;

    //EFB052611 : 
    //  TODO: We should approximate the domain volume rather than use the
    //  exact. But for the sake of time...we'll cheat
#if 0 

    // what is the surface element?
    // Ellipse: 
    // y1 = b*sqrt[1. - x^2 / a^2]  
    // integration limit: -y1 to y1. x limits: [-a,a]. 
    // area element: dx*dy

    // use 500 x 500 points across the ellipse

    double dx = 2. * major / (npts - 1);
    double integ = 0.;

    for (int i = 0; i < (npts - 1); i++) {
        double xa = -1. + (i + 0.5) * dx / major;
        double y1 = -minor * sqrt(1. - xa * xa);
        double dy = 2. * fabs(y1) / (npts - 1);
        for (int j = 0; j < (npts - 1); j++) {
            double x = xa * major;
            double y = y1 + (j + 0.5) * dy;
            integ += sqrt(rho(x, y, 0.)) * dx * dy;
        }
    }

#endif 
    // FIXME: we are assuming UNIFORM density for simplicity to start placing boundary nodes. 
    // The CVT can move them around according to density later, but we might not have ENOUGH!
    double exact_integ = (4./3.)*M_PI*major*minor*midax;
    double integ = exact_integ;
    if ((integ - exact_integ) > 1e-4) {
        std::cout << "ERROR: integral should be more precise. Exact: " << exact_integ << ", Approx: " << integ << ", N subdivs: " << npts << std::endl;
    }

    return integ;
}

//----------------------------------------------------------------------
// NOTE: fills the first nb_bnd elements of bnd with boundary node information. 
//

void EllipsoidCVT::computeBoundaryPointDistribution(int dim_num, double tot_length, int npts, int nb_bnd, std::vector<double> intg, double bnd[]) {

    double major = axis_major; 
    double minor = axis_minor;
    double midax = axis_midax;

    double tot_intv = tot_length / (npts - 1.);
    vector<double> equ_dist, theta, phi;

    int n = nb_bnd + 1; // space so that first and last point are the same
    equ_dist.resize(n);
    theta.resize(n);
    phi.resize(n);

#if 0

    double pi = acos(-1.);
    double dtheta = 2. * pi / (npts - 1.);
    printf("npts= %d, n= %d, nb_bnd = %d\n", npts, n, nb_bnd);

    double intv_length = tot_length / (n - 1);
    for (int i = 0; i < (n - 1); i++) {
        equ_dist[i] = i * intv_length;
        //theta[i] = i*dtheta;
    }

    equ_dist[n - 1] = tot_length;

    printf("tot_length= %f, intg[npts-1]= %f\n", tot_length, intg[npts - 1]);

    // Compute theta distribution of new points 

    // Brute force O(n*npts)
    // Should rewrite to be O(npts)
    //double dthetaj = 2.*pi / (n-1);
    theta[0] = 0.;
    for (int i = 1; i < (n - 1); i++) {
        theta[i] = -1.;
        printf("-----i= %d------\n", i);
        for (int j = 1; j < npts; j++) { // npts >> n
            // find interval that contains equ_dist[i]
            // intg[j-1] <= equ_dist[i] <= intg[j]
            if ((equ_dist[i] <= intg[j]) && equ_dist[i] >= intg[j - 1]) {
                //printf("i=%d, j/npts= %d/%d, equ_dist[%d]= %f, intg= %f, %f\n", i, j, npts, i, equ_dist[i], intg[j-1], intg[j]);
                double th = (j - 1) * dtheta;
                double dth = dtheta * (equ_dist[i] - intg[j - 1]) / (intg[j] - intg[j - 1]);
                //printf("dtheta= %f, th= %f, dth= %f\n", dtheta, th, dth);
                theta[i] = th + dth;
                break;
            }
        }
    }
    theta[n - 1] = 2. * pi;
#endif 

    printf("print length intervals: should be equal\n");
    for (int i = 0; i < nb_bnd; i++) {
        theta[i] = geom->randomU();
        phi[i] = geom->randomV();

#if 0
        bnd[i*dim_num + 0] = major * sin(phi[i]) * cos(theta[i]);     
        bnd[i*dim_num + 1] = minor * sin(phi[i]) * sin(theta[i]); 
        bnd[i*dim_num + 2] = midax * cos(phi[i]); 
#endif 
        bnd[i*dim_num + 0] = geom->x(theta[i], phi[i]); 
        bnd[i*dim_num + 1] = geom->y(theta[i], phi[i]); 
        bnd[i*dim_num + 2] = geom->z(theta[i], phi[i]); 
        //printf("(%d) x,y= %f, %f, theta= %f\n", i, x[i], y[i], theta[i]);
    }

    printf("print length intervals: should be equal\n");

#if 0
    for (int i=0; i < (n-1); i++) {
        double dx = (x[i+1]-x[i]);
        double dy = (y[i+1]-y[i]);
        double dl = sqrt(dx*dx + dy*dy);
        //printf("dl[%d]= %f\n", i, dl);
    }
#endif

    //   printf("Weighted ellipse perimeter: %f\n", tot_length);
}

//----------------------------------------------------------------------


