#ifndef _ELLIPSE_CVT_H_
#define _ELLIPSE_CVT_H_

#include <vector>
#include "Vec3.h"
//#include "parametric_patch.h"
#include "density.h"
#include "cvt.h"

// Extend the original CVT class, but override the void function pointers such that
// initial seed placement is within an ELLIPSE, and the random samples are
// within the ELLIPSE as well.

class EllipseCVT : public CVT {
    protected:
        // better to pass a grid structure and different types of grids will be
        // represented by different subclasses, which are accessed by
        // individual samplers (that populate the domain with random points
        double axis_major;
        double axis_minor;

    public:
        //EllipseCVT(double major_ = 1., double minor_ = 1., int DEBUG = 0);
        EllipseCVT (size_t nb_generators, size_t dimension, Density* density_func, double major_axis, double minor_axis, size_t num_samples=2000, size_t max_num_iters=10, size_t write_freq=100, size_t sample_batch_size=1000)
            : CVT(nb_generators, dimension, 0, density_func, num_samples, max_num_iters, write_freq, sample_batch_size), 
            axis_major(major_axis), axis_minor(minor_axis)
    { ; }
        
        virtual void displaceBoundaryNodes(int dim_num, int nb_bnd_nodes, double r[]);

        // custom "user" initialization
        // NOTE: we should rewrite the base CVT class so there is a separate
        // custom_init and custom_sample routine (since they CAN be different
        // like in this case).
        virtual void user_init(int dim_num, int n, int *seed, double r[]);

        virtual Vec3 singleRejection2d(double area, double weighted_area, Density& density);
        virtual Vec3 singleRejection2d_EllipseMinusCircle(double area, double weighted_area, Density& density);

        void setEllipseAxes(double major_, double minor_) {
            this->axis_major = major_;
            this->axis_minor = minor_;
        }

//        void setSubtractCircle(double xc, double yc, double cir_radius, double inner_axis_1, double inner_axis_2); 

        /*** FOR THE BOUNDARY ***/ 
        void fillBoundaryPoints(int dim_num, int nb_nodes, int *seed, double bndry_nodes[]);
        double computeBoundaryIntegral(Density& rho, size_t npts, std::vector<double>& intg);
        double computeDomainIntegral(size_t npts, Density& rho);
        void computeBoundaryPointDistribution(int dim_num, double tot_length, int npts, int nb_bnd, std::vector<double> intg, double bndry_pts[]);

        /*** FOR FILE NAMES: ***/ 

        virtual std::string className() {return "ellipse_cvt";}
};

#endif //_ELLIPSE_CVT_H_
