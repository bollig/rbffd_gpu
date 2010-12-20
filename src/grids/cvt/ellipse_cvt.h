#ifndef _ELLIPSE_CVT_H_
#define _ELLIPSE_CVT_H_

#include <vector>
#include "Vec3.h"
#include "utils/geom/parametric_patch.h"
#include "density.h"
#include "cvt.h"

// Extend the original CVT class, but override the void function pointers such that
// initial seed placement is within an ELLIPSE, and the random samples are
// within the ELLIPSE as well.

class EllipseCVT : public CVT {
protected:
    // better to pass a grid structure and different types of grids will be represented by
    // different subclasses, which are accessed by individual samplers (that populate the domain
    // with random points
    double major;
    double minor;

    // TODO: cut these two
    ParametricPatch *outer_geom; // not quite the correct class. In reality, the
    ParametricPatch *geom; // not quite the correct class. In reality, the
    // correct class is Geometry = vector<ParametricPatch>

public:
    EllipseCVT(double major_ = 1., double minor_ = 1., int DEBUG = 0);

    // Custom "sample" routine (sample inside box with rejection outside of the ellipse).
     virtual void user_sample(int dim_num, int n, int *seed, double r[]);

    // custom "user" initialization
    // NOTE: we should rewrite the base CVT class so there is a separate
    // custom_init and custom_sample routine (since they CAN be different like in this case).
    virtual void user_init(int dim_num, int n, int *seed, double r[]);

    // Gordon Erlebacher, 9/1/2009
    void ellipse(int dim_num, int& n, int& nb_bnd, int *seed, std::vector<double>& r);
    void ellipse(int dim_num, int& n, int& nb_bnd, int *seed, double r[]);
    void ellipse_init(int dim_num, int& n, int& nb_bnd, int *seed, double r[]);

    void rejection2d(int nb_samples, double area, double weighted_area, Density& density, std::vector<Vec3>& samples);
    Vec3 singleRejection2d(double area, double weighted_area, Density& density);

    void setEllipseAxes(double major_, double minor_) {
        this->major = major_;
        this->minor = minor_;
    }

    void setGeometry(ParametricPatch* geom_) {
        geom = geom_;
    }

    void setOuterGeometry(ParametricPatch* geom_) {
        outer_geom = geom_;
    }
};

#endif //_ELLIPSE_CVT_H_
