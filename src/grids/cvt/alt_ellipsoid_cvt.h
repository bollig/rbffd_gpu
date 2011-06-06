#ifndef _ALT_ELLIPSOID_CVT_H_
#define _ALT_ELLIPSOID_CVT_H_

#include <stdlib.h>
#include <vector>
#include "Vec3.h"
#include "utils/geom/ellipsoid_patch.h"
#include "density.h"
#include "cvt.h"

// Extend the original CVT class, but override the void function pointers such that
// initial seed placement is within an ellipsoid, and the random samples are
// within the ellipsoid as well. 

class AltEllipsoidCVT : public CVT {
    protected:
        // better to pass a grid structure and different types of grids will be represented by
        // different subclasses, which are accessed by individual samplers (that populate the domain
        // with random points
        double axis_major;
        double axis_minor;
        double axis_midax; // good for 3D case

        // TODO: cut these two
       // ParametricPatch *outer_geom; // not quite the correct class. In reality, the
        ParametricPatch *geom; // not quite the correct class. In reality, the
        // correct class is Geometry = vector<ParametricPatch>

    public:
        //    AltEllipsoidCVT(double major_ = 1., double minor_ = 1., double midax_ = 1., int DEBUG_ = 0);
        AltEllipsoidCVT (size_t nb_generators, size_t dimension, Density* density_func, double major_axis, double minor_axis, double midax_axis, size_t num_samples=2000, size_t max_num_iters=10, size_t write_freq=100, size_t sample_batch_size=1000)
            : CVT(nb_generators, 3, 0, density_func, num_samples, max_num_iters, write_freq, sample_batch_size), 
            axis_major(major_axis), axis_minor(minor_axis), axis_midax(midax_axis)
    {
        // FIXME: understand the parameters here.
        int n1 = 100; // not really need here
        int n2 = 100;
        double pi = acos(-1.0);
        // Outer ellipse
        geom = new EllipsoidPatch(0., pi, 0., 2.*pi, n1, n2, axis_major, axis_minor, axis_midax);
        // Inner ellipse
      //  outer_geom = new EllipsoidPatch(0., pi, 0., 2.*pi, n1, n2, axis_major+1.0, axis_minor+1.0, axis_midax+1.);
    }


        // Custom "sample" routine (sample inside box with rejection outside of the ellipse).
        virtual void user_sample(int dim_num, int n, int *seed, double r[]);

        // custom "user" initialization
        // NOTE: we should rewrite the base CVT class so there is a separate
        // custom_init and custom_sample routine (since they CAN be different like in this case).
        virtual void user_init(int dim_num, int n, int *seed, double r[]);


        void setEllipsoidAxes(double major_, double minor_, double midax_ = 1.) {
            this->axis_major = major_;
            this->axis_minor = minor_;
            this->axis_midax = midax_;
        }

        // Gordon Erlebacher: March 14, 2010
        void ellipsoid(int dim_num, int& n, int *seed, double r[]);
        void ellipsoid_init(int dim_num, int& n, int *seed, double r[]);

        void rejection3d(int nb_samples, Density& density, std::vector<Vec3>& samples);
        Vec3 singleRejection3d(Density& density);

        // Overrdie the default behavior of cvt_iterate (includes projection of boundary points to the surface)
      //  virtual void cvt_iterate(int dim_num, int n, int batch, int sample, bool initialize, int sample_num, int *seed, double r[], double *it_diff, double *energy);

        // Use parametric points to project samples to boundary 
        virtual void displaceBoundaryNodes(int dim_num, int nb_bnd_nodes, double r_computed[], double r_updated[]);

        /*** FOR THE BOUNDARY ***/ 
        void fillBoundaryPoints(int dim_num, int nb_nodes, int *seed, double bndry_nodes[]);
        double computeBoundaryIntegral(Density& rho, size_t npts, std::vector<double>& intg);
        double computeDomainIntegral(size_t npts, Density& rho);
        void computeBoundaryPointDistribution(int dim_num, double tot_length, int npts, int nb_bnd, std::vector<double> intg, double bndry_pts[]);

        void setGeometry(ParametricPatch* geom_) {
            geom = geom_;
        }
#if 0
        void setOuterGeometry(ParametricPatch* geom_) {
            outer_geom = geom_;
        }

#endif 
        /*** FOR FILE NAMES: ***/ 

        virtual std::string className() {return "ellipsoid_cvt";}

};

#endif //_ELLIPSOID_CVT_H_
