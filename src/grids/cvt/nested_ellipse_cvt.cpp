#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>

#include <math.h>
#include <vector>

#include "nested_ellipse_cvt.h"

using namespace std;


//****************************************************************************80
// For CVT:: this samples randomly in unit circle
//void NestedEllipseCVT::user_init(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand) { 
void NestedEllipseCVT::user_init(int dim_num, int n, int *seed, double r[]) {
    
    // Fill our initial boundary points
    this->fillBoundaryPoints(dim_num, n, seed, &r[0]);
    // Then sample the interior using our singleRejection2d defined routine
    this->user_sample(dim_num, n-nb_bnd, seed, &r[nb_bnd*dim_num]);


    return;
}


#if 0
void NestedEllipseCVT::fillBoundaryPoints(int dim_num, int nb_nodes, int *seed, double bndry_nodes[])
{
    this->nb_bnd = nb_inner + nb_outer;
    this->resizeBoundary(this->nb_bnd);


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
            bndry_nodes[i*dim_num + 0] = inner_r * cos(theta);
            bndry_nodes[i*dim_num + 1] = inner_r * sin(theta);
            bndry_nodes[i*dim_num + 2] = 0.;
            j++;
        }
        for (int i = 0; i < nb_outer; i++) {
            // r = 1.0
            // add 0.001*PI to slightly shift the outer boundary in an attempt
            // to avoid alignment with the inner boundary.
            // HERE: See top of NOTES for an idea of why we shift by PI/3
            double theta = outer_arc_seg*i + (M_PI / 3.);// + 0.001 * this->PI;
            // offset i by nb_inner because we already filled those.
            bndry_nodes[(nb_inner + i)*dim_num + 0] = outer_r * cos(theta);
            bndry_nodes[(nb_inner + i)*dim_num + 1] = outer_r * sin(theta);
            bndry_nodes[(nb_inner + i)*dim_num + 2] = 0.;
        }

        //    std::cout << "NRMLS: " << boundary_normals.size() << std::endl;
    } else if (dim_num == 3) {

        std::cout << "TODO: 3D nested sphere cvt" << std::endl;
        exit(EXIT_FAILURE);
    } else {
        std::cout << "ONLY 2D annulus cvt supported at this time" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif 






//****************************************************************************80
Vec3 NestedEllipseCVT::singleRejection2d(double area, double weighted_area, Density& density) {
    // Apparently not used in this file
    // 3D version should be written though, but the ellipse is a special case
    // (no hurry)

    double xs, ys;
    double u;
    double r2_o;
    double r2_i;
    double r_inner;
    double maxrhoi = 1.;
    if (rho != NULL) {
        double maxrhoi = 1. / density.getMax();
    }
    //printf("maxrhoi= %f\n", maxrhoi);

    double maj2o = 1. / (outer_axis_major * outer_axis_major);
    double min2o = 1. / (outer_axis_minor * outer_axis_minor);
    double maj2i = 1. / (inner_axis_major * inner_axis_major);
    double min2i = 1. / (inner_axis_minor * inner_axis_minor);
    //printf("maj2i,min2i= %f, %f\n", maj2i, min2i);

    double xc = 0.1; 
    double yc = 0.1; 

    while (1) {
        xs = random(-1.5*outer_axis_major, 1.5*outer_axis_major);
        ys = random(-1.5*outer_axis_major, 1.5*outer_axis_major); // to make sure that cells are all same size

        //printf("xs,ys= %f, %f\n", xs, ys);
        //
        // We're using he implicit definition of the geometry. 
        r2_o = xs * xs * maj2o + ys * ys*min2o;
        r2_i = xs * xs * maj2i + ys * ys*min2i;
        
        // In the case of the implicit we exit geometry at 1.
        if (r2_o > 1.) continue; // outside outer boundary
        if (r2_i < 1.) continue; // inside inner boundary

        // rejection part if non-uniform distribution
        u = random(0., 1.);
        if (rho != NULL) {
            //printf("rho= %f\n", density(xs,ys));
            if (u < (density(xs, ys, 0.)) * maxrhoi) break;
        } else {
            break; 
        }
    }
    return Vec3(xs, ys);
}



//****************************************************************************80

bool NestedEllipseCVT::reject_point(NodeType& point, int ndim) {
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

std::string NestedEllipseCVT::getFileDetailString() {
    std::stringstream ss(std::stringstream::out); 
    ss << nb_inner << "_inner_" << nb_outer << "_outer_" << nb_int << "_interior_" << dim_num << "d"; 
    return ss.str();	
}


#if 1
// Project k generators to the surface of the sphere of specified radius
// WARNING! Modifies the first k elements of generator[]! so pass a pointer
// to whatever element you want to start projecting from!

void NestedEllipseCVT::project_to_sphere(double generator[], int k, int ndim, double radius) {

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



//----------------------------------------------------------------------

void NestedEllipseCVT::fillBoundaryPoints(int dim_num, int nb_nodes, int *seed, double bndry_nodes[])
{
    std::vector<double> intg1;
    std::vector<double> intg2;

    unsigned int high_nb_pts = 200;
    
    double outer_bnd_intg = computeBoundaryIntegral(*rho, high_nb_pts, intg1, outer_axis_major, outer_axis_minor);
    double inner_bnd_intg = computeBoundaryIntegral(*rho, high_nb_pts, intg2, inner_axis_major, inner_axis_minor);

    double dom_intg = computeDomainIntegral(high_nb_pts, *rho);

    // total nb points used to compute Voronoi mesh. 
    // Only (nb_interior_pts-nb_bnd) will be able to move freely
    int tot_nb_pts = nb_nodes;
    // number of boundary points, automatically calculated
    printf("tot_nb_pts= %d\n", tot_nb_pts);
    printf("domain integral = %f\n", dom_intg);
    printf("outer boundary integral = %f\n", outer_bnd_intg);
    printf("inner boundary integral = %f\n", inner_bnd_intg);

    //int nb_computed_bnd = (int)(outer_bnd_intg * sqrt(tot_nb_pts / dom_intg));
    
    int nb_bnd_1 = (int)(1. + 16. * tot_nb_pts * dom_intg / (outer_bnd_intg * outer_bnd_intg));
    nb_bnd_1 = (int)(-outer_bnd_intg * outer_bnd_intg / (4. * dom_intg) * (1. - sqrt(nb_bnd_1)));
    int nb_computed_outer_bnd = nb_bnd_1; // more accurate formula
    
    
    int nb_bnd_2 = (int)(1. + 16. * tot_nb_pts * dom_intg / (inner_bnd_intg * inner_bnd_intg));
    nb_bnd_2 = (int)(-inner_bnd_intg * inner_bnd_intg / (4. * dom_intg) * (1. - sqrt(nb_bnd_2)));
    int nb_computed_inner_bnd = nb_bnd_2; // more accurate formula

    printf("calculated nb outer boundary pts: %d\n", nb_computed_outer_bnd);
    printf("calculated nb inner boundary pts: %d\n", nb_computed_inner_bnd);

    // Now that we know how many boundary nodes we want out of the total
    // number of nodes in the domain, resize to that value
    this->nb_bnd = nb_computed_outer_bnd + nb_computed_inner_bnd;
    this->resizeBoundary(this->nb_bnd);
    //bndry_pts.resize(this->nb_bnd);

    // Now compute the actual boundary nodes: 
    this->computeBoundaryPointDistribution(dim_num, outer_bnd_intg, outer_axis_major, outer_axis_minor, high_nb_pts, nb_computed_outer_bnd, intg1, bndry_nodes);
    this->computeBoundaryPointDistribution(dim_num, inner_bnd_intg, inner_axis_major, inner_axis_minor, high_nb_pts, nb_computed_inner_bnd, intg2, &bndry_nodes[nb_computed_outer_bnd*dim_num]);

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
double NestedEllipseCVT::computeBoundaryIntegral(Density& rho, unsigned int npts, vector<double>& intg, double major, double minor)
{
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

    return tot_length;
}

//----------------------------------------------------------------------

double NestedEllipseCVT::computeDomainIntegral(unsigned int npts, Density& rho) {
    // what is the surface element?
    // Ellipse: 
    // y1 = b*sqrt[1. - x^2 / a^2]  
    // integration limit: -y1 to y1. x limits: [-a,a]. 
    // area element: dx*dy

    // use 500 x 500 points across the ellipse

    // Integrate over the full outer ellipse (assuming its a solid geom)
    double major = outer_axis_major; 
    double minor = outer_axis_minor;

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

    // And cut out the interior ellipse
    major = inner_axis_major; 
    minor = inner_axis_minor;
    dx = 2. * major / (npts - 1);
    
    for (int i = 0; i < (npts - 1); i++) {
        double xa = -1. + (i + 0.5) * dx / major;
        double y1 = -minor * sqrt(1. - xa * xa);
        double dy = 2. * fabs(y1) / (npts - 1);
        for (int j = 0; j < (npts - 1); j++) {
            double x = xa * major;
            double y = y1 + (j + 0.5) * dy;
            integ -= sqrt(rho(x, y, 0.)) * dx * dy;
        }
    }

    return integ;
}

//----------------------------------------------------------------------
// NOTE: fills the first nb_bnd elements of bnd with boundary node information. 
//
void NestedEllipseCVT::computeBoundaryPointDistribution(int dim_num, double tot_length, double major, double minor, int npts, int nb_bnd, std::vector<double> intg, double bnd[]) {

    double tot_intv = tot_length / (npts - 1.);
    vector<double> equ_dist, theta;
    //bnd.resize(0);

    unsigned int n = nb_bnd + 1; // space so that first and last point are the same
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


