#ifndef _ELLIPSE_CVT_H_
#define _ELLIPSE_CVT_H_

// FIXME: Need density in the framework dir and not in the test dir
#include "density.h" 

#include <vector>
//#include "utils/geom/parametric_patch.h"
#include "density.h"
#include "grids/cvt/cvt.h"

// Extend the original CVT class, but override the void function pointers such that
// initial seed placement is within an ELLIPSE, and the random samples are
// within the ELLIPSE as well.

class EllipseCVT : public CVT {
protected:
    double axis_major;
    double axis_minor;

#if 0
    // TODO: cut these two
    ParametricPatch *outer_geom; // not quite the correct class. In reality, the
    ParametricPatch *geom; // not quite the correct class. In reality, the
    // correct class is Geometry = vector<ParametricPatch>
#endif 

public:
    //    EllipseCVT(double major_ = 1., double minor_ = 1., int DEBUG = 0);
    EllipseCVT (size_t nb_interior_nodes, size_t nb_boundary_nodes, size_t dimension, Density* density_func, double major_axis = 1., double minor_axis = 1., size_t num_samples=2000, size_t max_num_iters=10, size_t write_frequency=5, size_t sample_batch_size=800)
        : CVT(nb_interior_nodes+nb_boundary_nodes, dimension, nb_boundary_nodes, density_func, num_samples, max_num_iters, write_frequency, sample_batch_size), 
        axis_major(major_axis), axis_minor(minor_axis)
    { ; }

    void rejection2d(int nb_samples, double area, double weighted_area, Density& density, std::vector<NodeType>& samples);
    NodeType singleRejection2d(double area, double weighted_area, Density& density);
    void fillBoundaryPoints(int nb_boundary_nodes);


    void setEllipseAxes(double major_, double minor_) {
        this->axis_major = major_;
        this->axis_minor = minor_;
    }

    double computeBoundaryIntegral(Density& rho, size_t npts, std::vector<double>& intg);
    double computeDomainIntegral(size_t npts, Density& rho);

    // TODO: add support for parametric patch boundary
#if 0
    void setGeometry(ParametricPatch* geom_) {
        geom = geom_;
    }

    void setOuterGeometry(ParametricPatch* geom_) {
        outer_geom = geom_;
    }
#endif 

    /*******************
     * OVERRIDES GRID:: and CVT::
     *******************/
    // Overrides Grid::generate()
    virtual void generate(); 

    virtual std::string className() {return "ellipse_cvt";}


    /***********************
     * OVERRIDES CVT.h ROUTINES:
     ***********************/
    // Customized initial sampling of domain could be redirected to the user_sample
    // so both node initialization and cvt sampling are the same
    //

    // For CVT:: this samples randomly in unit circle
    virtual void user_sample(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand); 

    // For CVT:: this samples randomly in unit circle
    virtual void user_init(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand); 

};

#endif //_ELLIPSE_CVT_H_
