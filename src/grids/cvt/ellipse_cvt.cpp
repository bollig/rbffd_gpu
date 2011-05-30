#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>

#include <math.h>
#include <vector>

#include "ellipse_cvt.h"

using namespace std;

//****************************************************************************80

Vec3 EllipseCVT::singleRejection2d(double area, double weighted_area, Density& density) {
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
        xs = random(-axis_major, axis_major);
        ys = random(-axis_major, axis_major); // to make sure that cells are all same size
        //printf("xs,ys= %f, %f\n", xs, ys);
        r2 = xs * xs * maj2i + ys * ys*min2i;
        //printf("r2= %f\n", r2);
        if (r2 >= 1.) continue; // inside the ellipse

        // rejection part if non-uniform distribution
        u = random(0., 1.);
        //printf("rho= %f\n", density(xs,ys));
        if (u < (density(xs, ys)) * maxrhoi) break;
    }
    return Vec3(xs, ys);
}

void EllipseCVT::user_init(int dim_num, int n, int *seed, double r[]) {
    // Fill our initial boundary points
    this->fillBoundaryPoints(dim_num, n, seed, &r[0]);
    std::cout << "NOW NB_BND = " << nb_bnd << std::endl;
    // Then sample the interior using our singleRejection2d defined routine
    this->user_sample(dim_num, n-nb_bnd, seed, &r[nb_bnd*dim_num]);
}

//----------------------------------------------------------------------

void EllipseCVT::fillBoundaryPoints(int dim_num, int nb_nodes, int *seed, double bndry_nodes[])
{
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

    size_t nb_computed_bnd = bnd_intg * sqrt(tot_nb_pts / dom_intg);
    size_t nb_bnd_1 = 1. + 16. * tot_nb_pts * dom_intg / (bnd_intg * bnd_intg);
    nb_bnd_1 = -bnd_intg * bnd_intg / (4. * dom_intg) * (1. - sqrt(nb_bnd_1));
    nb_computed_bnd = nb_bnd_1; // more accurate formula
    printf("calculated nb boundary pts: %d\n", nb_computed_bnd);
    printf("improved nb boundary pts: %d\n", nb_bnd_1);

    // Now that we know how many boundary nodes we want out of the total
    // number of nodes in the domain, resize to that value
    this->nb_bnd = nb_computed_bnd;
    this->resizeBoundary(this->nb_bnd);
    //bndry_pts.resize(this->nb_bnd);

    // Now compute the actual boundary nodes: 
    this->computeBoundaryPointDistribution(dim_num, bnd_intg, high_nb_pts, nb_bnd, intg, bndry_nodes);

    // Verify that things match
    //		printf("nb_bnd= %d, bndry_pts.size= %d\n", nb_bnd, (int) bndry_pts.size());
    //        printf("node_list.size= %d\n" , this->node_list.size());
    //
    for (size_t i = 0; i < nb_bnd; i++) {
        Vec3 nd;
        for (int j = 0; j < dim_num; j++) {
            nd[j] = bndry_nodes[i*dim_num + j];
        }
        this->setNode(i, nd);
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
void EllipseCVT::computeBoundaryPointDistribution(int dim_num, double tot_length, int npts, int nb_bnd, std::vector<double> intg, double bnd[]) {

    double major = axis_major; 
    double minor = axis_minor;

    double tot_intv = tot_length / (npts - 1.);
    vector<double> equ_dist, theta;
    //bnd.resize(0);

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


    for (int i = 0; i < nb_bnd; i++) {
        if (theta[i] < 0.) {
            printf("Equipartitioning of boundary is incomplete\n");
            exit(0);
        }
        bnd[i*dim_num + 0] = major * cos(theta[i]);
        bnd[i*dim_num + 1] = minor * sin(theta[i]);
        // We are limited to 2D in this class
        if (dim_num > 2) {
            bnd[i*dim_num + 2] = 0.;
        }
 //       printf("(%d) x,y= %f, %f, theta= %f\n", i, bnd[i*dim_num+0], bnd[i*dim_num + 1], theta[i]);
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


