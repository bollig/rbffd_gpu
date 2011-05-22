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

//----------------------------------------------------------------------

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

    // Take care of boundary points, then finish by sampling the same as a general iteration
    this->fillBoundaryPoints(this->getNodeListSize());
    this->user_sample(user_node_list, indx_start, n_now, init_rand);

    return;
}

//----------------------------------------------------------------------
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
void EllipseCVT::user_sample(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand) 
{
    size_t n = n_now; 
    size_t nb_bnd = nb_locked_nodes; 
    double dtheta = 1. * M_PI / nb_bnd; // periodic in theta

    //void EllipseCVT::rejection2d(int nb_samples, double area, double weighted_area, Density& density, vector<Vec3>& samples)
    std::vector<NodeType> samples;
    samples.resize(n - nb_bnd);

    // We lock our nb_bnd seeds in place and generate an additional set of
    // random seeds with rejection2d.
    if (rho) {
        //        std::cout << "[EllipseCVT] using density function to sample " << n - nb_bnd << " nodes\n";
        rejection2d(n - nb_bnd, 0., 1., *rho, samples);
    } else {
        std::cout << "[EllipseCVT] Error! Rho is not set. Exiting...\n"; 
        exit(EXIT_FAILURE);
    }

    // Fill the non-locked nodes (i.e., interior) with initial samples 
    // within the ellipse
    for (int j = nb_bnd; j < n; j++) {
        user_node_list[j] = samples[j - nb_bnd];
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
        //samples[i].print("rnd");
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
    //printf("maxrhoi= %f\n", maxrhoi);

    double maj2i = 1. / axis_major / axis_major;
    double min2i = 1. / axis_minor / axis_minor;
    //printf("maj2i,min2i= %f, %f\n", maj2i, min2i);

    while (1) {
        xs = randf(-axis_major, axis_major);
        ys = randf(-axis_major, axis_major); // to make sure that cells are all same size
        //printf("xs,ys= %f, %f\n", xs, ys);
        r2 = xs * xs * maj2i + ys * ys*min2i;
        //printf("r2= %f\n", r2);
        if (r2 >= 1.) continue; // inside the ellipse

        // rejection part if non-uniform distribution
        u = randf(0., 1.);
        //printf("rho= %f\n", density(xs,ys));
        if (u <= (density(xs, ys)) * maxrhoi) break;
    }
    return NodeType(xs, ys);
}

//----------------------------------------------------------------------

void EllipseCVT::fillBoundaryPoints(int nb_nodes)
{
    std::vector<NodeType> bndry_pts;
    std::vector<double> intg;

    size_t high_nb_pts = 200;
    double bnd_intg = computeBoundaryIntegral(*rho, high_nb_pts, intg);
    double dom_intg = computeDomainIntegral(high_nb_pts, *rho);

    // total nb points used to compute Voronoi mesh. 
    // Only (nb_interior_pts-nb_bnd) will be able to move freely
    size_t tot_nb_pts = nb_nodes;
    // number of boundary points, automatically calculated
    printf("tot_nb_pts= %d\n", tot_nb_pts);
    printf("domain integral = %f\n", dom_intg);
    printf("boundary integral = %f\n", bnd_intg);

    size_t nb_bnd = bnd_intg * sqrt(tot_nb_pts / dom_intg);
    size_t nb_bnd_1 = 1. + 16. * tot_nb_pts * dom_intg / (bnd_intg * bnd_intg);
    nb_bnd_1 = -bnd_intg * bnd_intg / (4. * dom_intg) * (1.
            - sqrt(nb_bnd_1));
    nb_bnd = nb_bnd_1; // more accurate formula
    printf("calculated nb boundary pts: %d\n", nb_bnd);
    printf("improved nb boundary pts: %d\n", nb_bnd_1);

    // Now that we know how many boundary nodes we want out of the total
    // number of nodes in the domain, resize to that value
    this->resizeBoundary(nb_bnd);
    this->nb_locked_nodes = nb_bnd;
    bndry_pts.resize(nb_bnd);

    // Now compute the actual boundary nodes: 
    this->computeBoundaryPointDistribution(bnd_intg, high_nb_pts, nb_bnd, intg, bndry_pts);

    // Verify that things match
    //		printf("nb_bnd= %d, bndry_pts.size= %d\n", nb_bnd, (int) bndry_pts.size());
    //        printf("node_list.size= %d\n" , this->node_list.size());
    //
    for (size_t i = 0; i < nb_bnd; i++) {
        this->setNode(i, bndry_pts[i]);
        this->setBoundaryIndex(i, i);
        // TODO: boundary normals
        //            this->getBoundaryNormal(i) = computeBoundaryNormal(bndry_pts[i]);
    }

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
double EllipseCVT::computeBoundaryIntegral(Density& rho, size_t npts, vector<double>& intg)
{
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
        double rhom = rho(xm, ym);
        intg[i + 1] = intg[i] + pow(rhom, 0.25) * dl;
    }

    // new boundary will have n points (n << npts)
    // divide into (n-1) equal length intervals
    double tot_length = intg[npts - 1];

    printf("boundary integral: %f\n", tot_length);

    return tot_length;
}

//----------------------------------------------------------------------

double EllipseCVT::computeDomainIntegral(size_t npts, Density& rho) {
    // what is the surface element?
    // Ellipse: 
    // y1 = b*sqrt[1. - x^2 / a^2]  
    // integration limit: -y1 to y1. x limits: [-a,a]. 
    // area element: dx*dy

    // use 500 x 500 points across the ellipse

    double major = axis_major; 
    double minor = axis_minor;

    double dx = 2. * major / (npts - 1);
    double integ = 0.;

    for (int i = 0; i < (npts - 1); i++) {
        double xa = -1. + (i + 0.5) * dx / major;
        double y1 = -minor * sqrt(1. - xa * xa);
        double dy = 2. * fabs(y1) / (npts - 1);
        for (int j = 0; j < (npts - 1); j++) {
            double x = xa * major;
            double y = y1 + (j + 0.5) * dy;
            integ += sqrt(rho(x, y)) * dx * dy;
        }
    }

    return integ;
}

//----------------------------------------------------------------------
// NOTE: fills the first nb_bnd elements of bnd with boundary node information. 
//
void EllipseCVT::computeBoundaryPointDistribution(double tot_length, int npts, int nb_bnd, std::vector<double> intg, std::vector<NodeType>& bnd) {

    double major = axis_major; 
    double minor = axis_minor;

    double tot_intv = tot_length / (npts - 1.);
    vector<double> equ_dist, theta;
    bnd.resize(0);

    int n = nb_bnd + 1; // space so that first and last point are the same
    double pi = acos(-1.);
    double dtheta = 2. * pi / (npts - 1.);
    printf("npts= %d, n= %d\n", npts, n);

    equ_dist.resize(n);
    theta.resize(n);

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
//        printf("-----i= %d------\n", i);
        for (int j = 1; j < npts; j++) { // npts >> n
            // find interval that contains equ_dist[i]
            // intg[j] <= equ_dist[i] <= intg[j]
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
    //exit(0);

    vector<double> x, y;
    x.resize(nb_bnd);
    y.resize(nb_bnd);

    for (int i = 0; i < nb_bnd; i++) {
        if (theta[i] < 0.) {
            printf("Equipartitioning of boundary is incomplete\n");
            exit(0);
        }
        x[i] = major * cos(theta[i]);
        y[i] = minor * sin(theta[i]);
        bnd[i] = Vec3(x[i], y[i]);
        //printf("(%d) x,y= %f, %f, theta= %f\n", i, x[i], y[i], theta[i]);
    }

//    printf("print length intervals: should be equal\n");

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
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

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
