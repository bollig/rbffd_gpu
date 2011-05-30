#ifndef _ELLIPSOID_CVT_H_
#define _ELLIPSOID_CVT_H_

#include <stdlib.h>
#include <vector>
#include "Vec3.h"
#include "utils/geom/parametric_patch.h"
#include "density.h"
#include "cvt.h"

// Extend the original CVT class, but override the void function pointers such that
// initial seed placement is within an ellipsoid, and the random samples are
// within the ellipsoid as well. 

class EllipsoidCVT : public CVT {
protected:
    // better to pass a grid structure and different types of grids will be represented by
    // different subclasses, which are accessed by individual samplers (that populate the domain
    // with random points
    double axis_major;
    double axis_minor;
    double axis_midax; // good for 3D case

    // TODO: cut these two
    ParametricPatch *outer_geom; // not quite the correct class. In reality, the
    ParametricPatch *geom; // not quite the correct class. In reality, the
    // correct class is Geometry = vector<ParametricPatch>

public:
//    EllipsoidCVT(double major_ = 1., double minor_ = 1., double midax_ = 1., int DEBUG_ = 0);
    EllipsoidCVT (size_t nb_generators, size_t dimension, Density* density_func, double major_axis, double minor_axis, double midax_axis, size_t num_samples=2000, size_t max_num_iters=10, size_t write_freq=100, size_t sample_batch_size=1000)
            : CVT(nb_generators, dimension, 0, density_func, num_samples, max_num_iters, write_freq, sample_batch_size), 
            axis_major(major_axis), axis_minor(minor_axis), axis_midax(midax_axis)
    { ; }
 

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
    virtual void cvt_iterate(int dim_num, int n, int batch, int sample, bool initialize, int sample_num, int *seed, double r[], double *it_diff, double *energy);
   
    void setGeometry(ParametricPatch* geom_) {
        geom = geom_;
    }

    void setOuterGeometry(ParametricPatch* geom_) {
        outer_geom = geom_;
    }
};

#endif //_ELLIPSOID_CVT_H_
