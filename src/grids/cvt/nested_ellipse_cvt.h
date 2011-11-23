#ifndef _NESTED_ELLIPSE_CVT_H_
#define _NESTED_ELLIPSE_CVT_H_

#include <vector>
#include "Vec3.h"
//#include "parametric_patch.h"
#include "density.h"
#include "cvt.h"

// Create a constrained CVT
// in the volume created by two nested spheres.

class NestedEllipseCVT : public CVT {
    protected:
        bool guess_nb_boundary; 
        size_t nb_inner, nb_outer, nb_int;
        double outer_axis_major; 
        double outer_axis_minor; 
        double inner_axis_major; 
        double inner_axis_minor; 
        NodeType outer_center; 
        NodeType inner_center;

    public: 

        NestedEllipseCVT (size_t total_nb_nodes, 
                size_t nb_nodes_inner_boundary, size_t nb_nodes_outer_boundary, 
                size_t dimension, Density* density_func, size_t nb_locked=0, size_t num_samples=2000, 
                size_t max_num_iters=10, size_t write_frequency=20, 
                size_t sample_batch_size=800
                )
            : CVT(total_nb_nodes, dimension,
                    nb_nodes_outer_boundary+nb_nodes_inner_boundary
                    /*nb_locked*/, density_func, num_samples,
                    max_num_iters, write_frequency, sample_batch_size),
            nb_int(total_nb_nodes - nb_nodes_outer_boundary - nb_nodes_inner_boundary),
            nb_outer(nb_nodes_outer_boundary),
            nb_inner(nb_nodes_inner_boundary), 
            guess_nb_boundary(false),
            outer_axis_major(1.0), outer_axis_minor(1.0), 
            inner_axis_major(0.5), inner_axis_minor(0.5)
    {
        if (dimension > 2) { 
            std::cout << "ERROR: 3D Nested spheres not supported yet. This code assumes direct placement of nodes on inner and outer boundary. Code needs changes to do CVT on surface of 3D sphere." << std::endl;
        }
    }

        NestedEllipseCVT (size_t total_nb_nodes, 
                size_t nb_nodes_inner_boundary, size_t nb_nodes_outer_boundary, 
                size_t dimension, 
                double o_axis_major, double o_axis_minor, double i_axis_major, double i_axis_minor,
                Density* density_func, size_t nb_locked=0, size_t num_samples=2000, 
                size_t max_num_iters=10, size_t write_frequency=20, 
                size_t sample_batch_size=800
                )
            : CVT(total_nb_nodes, dimension,
                    nb_nodes_outer_boundary+nb_nodes_inner_boundary
                    /*nb_locked*/, density_func, num_samples,
                    max_num_iters, write_frequency, sample_batch_size),
            nb_int(total_nb_nodes - nb_nodes_outer_boundary - nb_nodes_inner_boundary),
            nb_outer(nb_nodes_outer_boundary),
            nb_inner(nb_nodes_inner_boundary), 
            guess_nb_boundary(false),
            outer_axis_major(o_axis_major), outer_axis_minor(o_axis_minor), 
            inner_axis_major(i_axis_major), inner_axis_minor(i_axis_minor)
    {
        if (dimension > 2) { 
            std::cout << "ERROR: 3D Nested spheres not supported yet. This code assumes direct placement of nodes on inner and outer boundary. Code needs changes to do CVT on surface of 3D sphere." << std::endl;
        }
    }

        NestedEllipseCVT (size_t total_nb_nodes, 
                size_t dimension, Density* density_func, size_t nb_locked=0, size_t num_samples=2000, 
                size_t max_num_iters=10, size_t write_frequency=20, 
                size_t sample_batch_size=800
                )
            : CVT(total_nb_nodes, dimension,
                    0/*nb_locked*/, density_func, num_samples,
                    max_num_iters, write_frequency, sample_batch_size),
            nb_int(total_nb_nodes),
            nb_inner(0), nb_outer(0), 
            guess_nb_boundary(true),
            outer_axis_major(1.0), outer_axis_minor(1.0), 
            inner_axis_major(0.5), inner_axis_minor(0.5)
    {
        if (dimension > 2) { 
            std::cout << "ERROR: 3D Nested spheres not supported yet. This code assumes direct placement of nodes on inner and outer boundary. Code needs changes to do CVT on surface of 3D sphere." << std::endl;
        }
    }


        NestedEllipseCVT (size_t total_nb_nodes, 
                size_t dimension,
                double o_axis_major, double o_axis_minor, double i_axis_major, double i_axis_minor,
                Density* density_func, size_t nb_locked=0, size_t num_samples=2000, 
                size_t max_num_iters=10, size_t write_frequency=20, 
                size_t sample_batch_size=800
                )
            : CVT(total_nb_nodes, dimension,
                    0/*nb_locked*/, density_func, num_samples,
                    max_num_iters, write_frequency, sample_batch_size),
            nb_int(total_nb_nodes),
            nb_inner(0), nb_outer(0), 
            guess_nb_boundary(true),
            outer_axis_major(o_axis_major), outer_axis_minor(o_axis_minor), 
            inner_axis_major(i_axis_major), inner_axis_minor(i_axis_minor)
    {
        if (dimension > 2) { 
            std::cout << "ERROR: 3D Nested spheres not supported yet. This code assumes direct placement of nodes on inner and outer boundary. Code needs changes to do CVT on surface of 3D sphere." << std::endl;
        }
    }

        void setInnerRadius(double inner_r_) { 
            inner_axis_major = inner_r_; 
            inner_axis_minor = inner_r_;
        }
        void setOuterRadius(double outer_r_) { 
            outer_axis_major = outer_r_; 
            outer_axis_minor = outer_r_; 
        }

        void setInnerAxes(double inner_axis_maj, double inner_axis_min) { 
            inner_axis_major = inner_axis_maj; 
            inner_axis_minor = inner_axis_min;
        }
        void setOuterAxes(double outer_axis_maj, double outer_axis_min) { 
            outer_axis_major = outer_axis_maj; 
            outer_axis_minor = outer_axis_min;
        }


        /*******************
         * OVERRIDES GRID:: and CVT::
         *******************/
        // Overrides Grid::getFileDetailString()
        //ONLY REPLACE IF WE WANT A MORE VERBOSE FILENAME FOR # OF BOUNDARY NODES
        virtual std::string getFileDetailString(); 

        virtual std::string className() {return "nested_ellipse_cvt";}


        /***********************
         * OVERRIDES CVT.h ROUTINES:
         ***********************/
        virtual void user_init(int dim_num, int n, int *seed, double r[]);
        virtual void guessBoundaryPoints(int dim_num, int nb_nodes, int *seed, double bndry_nodes[]);
        virtual void generateBoundaryPoints(int dim_num, int nb_nodes, int *seed, double bndry_nodes[]);
        virtual Vec3 singleRejection2d(double area, double weighted_area, Density& density);

        double computeBoundaryIntegral(Density& rho, unsigned int npts, std::vector<double>& intg, double axis_major, double axis_minor);
        double computeDomainIntegral(unsigned int npts, Density& rho);
        void computeBoundaryPointDistribution(int dim_num, double tot_length, double major, double minor, int npts, int nb_bnd, std::vector<double> intg, double bndry_pts[]);

    protected: 
        void project_to_sphere(double generator[], int k, int ndim, double radius);

};

#endif //_NESTED_ELLIPSE_CVT_H_
