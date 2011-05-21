//#include <cstdlib>
//#include <cmath>
//#include <ctime>
//#include <iostream>
//#include <iomanip>
//#include <fstream>
//#include <string.h>

//#include <math.h>
#include <vector>

#include "ellipse_cvt.h"
#include "utils/random.h" 


void EllipseCVT::generate() {

    // TODO: initialize boundary
    if (!generatorsInitialized || (nb_nodes != node_list.size())) {
        this->node_list.resize(nb_nodes); 
        // Random in ELLIPSE 
        // NOTE: also fills boundary
        this->cvt_sample(this->node_list, 0, nb_nodes, CVT::USER_INIT, true); 

        generatorsInitialized = true;
        std::cout << "[CVT] Done sampling initial generators\n";
    }

    // Should we generate a KDTree to accelerate sampling? Cost of tree
    // generation is high and should be amortized by VERY MANY samples.
    // However, too many samples argues in favor of discrete Voronoi transform

    // TODO: get CVT::GRID to generate samples in batches of subvolumes
    // otherwise batches are always the same sample sets for it. 
    this->cvt_iterate(sample_batch_size, nb_samples, CVT::USER_SAMPLE);

    this->writeToFile();
}

//----------------------------------------------------------------------
//
void EllipseCVT::user_init(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand) { 

    size_t n = n_now; 
    size_t nb_bnd = nb_locked_nodes; 
    double dtheta = 1. * M_PI / nb_bnd; // periodic in theta

    // boundary points
    this->fillBoundaryPoints(nb_bnd);

    //void EllipseCVT::rejection2d(int nb_samples, double area, double weighted_area, Density& density, vector<Vec3>& samples)
    std::vector<NodeType> samples;
    samples.resize(n - nb_bnd);

    // We lock our nb_bnd seeds in place and generate an additional set of
    // random seeds with rejection2d.
    if (rho) {
        std::cout << "[EllipseCVT] using density function to sample " << n - nb_bnd << " nodes\n";
        rejection2d(n - nb_bnd, 0., 1., *rho, samples);
    } else {
        std::cout << "[EllipseCVT] Error! Rho is not set. Exiting...\n"; 
        exit(EXIT_FAILURE);
    }

    // Fill the non-locked nodes (i.e., interior) with initial samples 
    // within the ellipse
    for (int j = nb_bnd; j < n; j++) {
        user_node_list[j][0] = samples[j - nb_bnd].x();
        user_node_list[j][1] = samples[j - nb_bnd].y();
    }

    return;
}
//----------------------------------------------------------------------

void EllipseCVT::rejection2d(int nb_samples, double area, double weighted_area, Density& density, vector<Vec3>& samples)
    // Given a pdf p(x,y) (zero outside the ellipse, with unit integral over the elliptic domain)
    // and given a constant bound M, such that p(x,y) < M q(x,y), where q(x,y) is uniform distribution
    // over the ellipse.
    // Generate a sample (x*,y*), and a sample u* in [0,1]
    // x* \in [-a,a], y* \in [-b,b]
    // Two cases:
    // 1.   u < p(x,y) / M q(x,y) : accept (x*,y*)
    // 2.   u > p(x,y) / M q(x,y)   reject (x*,y*)
    //
    // What is q(x,y): simply   1. / area(ellipse)
    // Choose M = max_{x,y}   p(x,y)*area(ellipse)
    // Given x*,y*, compute r*^2 = (x*/a)^2 + (y*/b*)^2
    // if r*^2 > 1., reject the point (outside the ellipse)

    // Used in ellipse_init (constructor helper) and ellipse
    // 3D version should be written though, but the ellipse is a special case
    // (no hurry)
{
    //printf("area= %f\n", area);
    //printf("weighted area= %f\n", weighted_area);

    samples.resize(nb_samples);

    for (int i = 0; i < nb_samples; i++) {
        samples[i] = singleRejection2d(area, weighted_area, density);
        samples[i].print("rnd");
    }
    //printf("nb_samples= %d\n", nb_samples);

    //exit(0);
}


//----------------------------------------------------------------------

NodeType EllipseCVT::singleRejection2d(double area, double weighted_area, Density& density) {
    // Apparently not used in this file
    // 3D version should be written though, but the ellipse is a special case
    // (no hurry)

    double xs, ys;
    double u;
    double r2;
    double maxrhoi = 1. / rho->getMax();
    printf("maxrhoi= %f\n", maxrhoi);
    printf("axis_major = %f\naxis_minor = %f\n", axis_major, axis_minor);
    double maj2i = 1. / axis_major / axis_major;
    double min2i = 1. / axis_minor / axis_minor;
    printf("maj2i,min2i= %f, %f\n", maj2i, min2i);

    while (1) {
        xs = randf(-axis_major, axis_major);
        ys = randf(-axis_major, axis_major); // to make sure that cells are all same size
      //  printf("xs,ys= %f, %f\n", xs, ys);
        r2 = xs * xs * maj2i + ys * ys*min2i;
    //    printf("r2= %f\n", r2);
        if (r2 >= 1.) continue; // inside the ellipse

        // rejection part if non-uniform distribution
        u = randf(0., 1.);
        //printf("rho= %f\n", density(xs,ys));
        if (u < (density(xs, ys)) * maxrhoi) break;
    }
    return NodeType(xs, ys);
}

//----------------------------------------------------------------------

void EllipseCVT::fillBoundaryPoints(int nb_boundary_nodes)
{
#if 0
		Density rho;
		//printf("rho.getMax() = %f\n", rho.getMax());
		//exit(0);

		vector<Vec3> bndry_pts;
		vector<double> intg;

		int high_nb_pts = 200;
		double bnd_intg = computeBoundaryIntegral(rho, high_nb_pts, intg);
		double dom_intg = computeDomainIntegral(high_nb_pts, rho);

		// total nb points used to compute Voronoi mesh. 
		// Only (nb_interior_pts-nb_bnd) will be able to move freely
		int tot_nb_pts = 500;
		// number of boundary points, automatically calculated
		printf("tot_nb_pts= %d\n", tot_nb_pts);
		printf("dom_intg= %f\n", dom_intg);
		printf("bnd_intg= %f\n", bnd_intg);

		int nb_bnd = bnd_intg * sqrt(tot_nb_pts / dom_intg);
		int nb_bnd_1 = 1. + 16. * tot_nb_pts * dom_intg / (bnd_intg * bnd_intg);
		nb_bnd_1 = -bnd_intg * bnd_intg / (4. * dom_intg) * (1.
				- sqrt(nb_bnd_1));
		nb_bnd = nb_bnd_1; // more accurate formula
		printf("calculated nb boundary pts: %d\n", nb_bnd);
		printf("improved nb boundary pts: %d\n", nb_bnd_1);

		computeBoundaryPointDistribution(bnd_intg, high_nb_pts, nb_bnd, intg,
				bndry_pts);
		printf("domain integral = %f\n", dom_intg);
		printf("boundary integral = %f\n", bnd_intg);

		printf("nb_bnd= %d, bndry_pts.size= %d\n", nb_bnd,
				(int) bndry_pts.size());


#endif 
}


//using namespace std;

//****************************************************************************80
#if 0
EllipseCVT::EllipseCVT(double major_, double minor_, int DEBUG_) : CVT(DEBUG_) {
    this->major = major_;
    this->minor = minor_;
    nb_bnd = 0;
}
#endif 
//****************************************************************************80

#if 0
//----------------------------------------------------------------------

void EllipseCVT::ellipse(int dim_num, size_t& n, size_t& nb_bnd, int *seed, vector<double>& r)

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
    //    1 September 2009
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
    // Ellipse:
    //   x= a*cos(theta)
    //   y= b*sin(theta)
{

    printf("SHOULD NOT GET HERE: ellipse, n= %d\n", n);
    exit(0);

    // na : is the number of points along the major axis
    // nb : is the number of points along the minor axis

    r.resize(n * dim_num);

    //double dx = 2.*major / (na - 1);
    //double dy = 2.*minor / (nb - 1);
    double dtheta = 2. * M_PI / nb_bnd; // periodic in theta

    // boundary points
    for (int i = 0; i < nb_bnd; i++) {
        double angle = i*dtheta;
        //r.push_back(major * cos(angle));
        //r.push_back(minor * sin(angle));
        r[0 + i * 2] = axis_major * cos(angle);
        r[1 + i * 2] = axis_minor * sin(angle);
    }

    //int nb_bound = r.size() / 2;
    //if (nb_bnd != nb_bound) {
    //printf("error in boundary points\n");
    //exit(0);
    //}


    for (int j = nb_bnd; j < n; j++) {
        double angle = 2.0 * M_PI * randf(0., 1.);
        double radius = sqrt(randf(0., 1.));
        //radius = radius * .95; // avoid boundary
        //r.push_back(radius*major*cos(angle));
        //r.push_back(radius*minor*sin(angle));
        r[0 + j * 2] = radius * axis_major * cos(angle);
        r[1 + j * 2] = radius * axis_minor * sin(angle);
        //printf("%f, %f\n", r[0+j*2], r[1+j*2]);
    }

    //exit(0);

    return;
}
//----------------------------------------------------------------------

void EllipseCVT::ellipse(int dim_num, size_t& n, size_t& nb_bnd, int *seed, double r[])

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
    //    1 September 2009
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
    // Ellipse:
    //   x= a*cos(theta)
    //   y= b*sin(theta)
{
    // na : is the number of points along the major axis
    // nb : is the number of points along the minor axis

    //    printf("enter ellipse\n");

    //r.resize(n*dim_num);
    //printf("***1 rhomax: %f\n", rho->getMax()); exit(0);

    //double dx = 2.*major / (na - 1);
    //double dy = 2.*minor / (nb - 1);

    double dtheta;
    if (nb_bnd > 0) {
        dtheta = 2. * M_PI / nb_bnd; // periodic in theta
    } else {
        dtheta = 1.;
    }

    //// boundary points
    /*
       for (int i=0; i < nb_bnd; i++) {
       double angle = i*dtheta;
       r[0+i*2] = bndry_pts[i].x();
       r[1+i*2] = bndry_pts[i].y();
//r.push_back(major * cos(angle));
//r.push_back(minor * sin(angle));
//r[0+i*2] = major * cos(angle);
//r[1+i*2] = minor * sin(angle);
}
printf("nb_bnd= %d\n", nb_bnd);
printf("n= %d\n", n);
*/

vector<Vec3> samples;
samples.resize(n);
// printf("ellipse: rejection2d\n");
rejection2d(n, 0., 1., *rho, samples);


// interior points
//for (int i=nb_bnd; i < n; i++) {
//printf("n= %d\n", n);
for (int i = 0; i < n; i++) {
    //double angle = 2.0 * PI * ( double ) ::random ( ) / ( double ) RAND_MAX;
    //double radius = sqrt ( ( double ) ::random ( ) / ( double ) RAND_MAX );  // in [0,1]
    //r.push_back(major * cos(angle));
    //r.push_back(minor * sin(angle));
    //r[0+i*2] = major * radius * cos(angle);
    //r[1+i*2] = minor * radius * sin(angle);
    //printf("%f, %f\n", r[0+j*2], r[1+j*2]);

    r[0 + i * 2] = samples[i].x();
    r[1 + i * 2] = samples[i].y();
}

//printf("ellipse: n = %d\n", n);

return;
}
//----------------------------------------------------------------------

void EllipseCVT::user_sample(int dim_num, size_t n, int *seed, double r[]) {
    // The guts and glory (this is the user defined ellipse sampling)
    ellipse(dim_num, n, nb_locked_nodes, seed, r);
}

void EllipseCVT::user_init(int dim_num, size_t n, int *seed, double r[]) {
    // This is the user defined initialization
    ellipse_init(dim_num, n, nb_locked_nodes, &seed[0], &r[0]);
}

#endif 
