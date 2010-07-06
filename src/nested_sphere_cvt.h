#ifndef _NESTED_SPHERE_CVT_H_
#define _NESTED_SPHERE_CVT_H_

#include <vector>
#include "Vec3.h"
#include "density.h"
#include "cvt.h"

// Create a constrained CVT
// in the volume created by two nested spheres.

class NestedSphereCVT : public CVT {
protected:
    double inner_r, outer_r;
    double* geom_extents;
    
    int nb_inner, nb_outer, nb_int;
   
    int it_max_interior, it_max_boundary;
    int it_num_interior, it_num_boundary;

public:
    NestedSphereCVT(const char* file_name, int nb_inner_bnd, int nb_outer_bnd, int nb_interior, int sample_num_, int it_max_bndry=60, int it_max_inter = 240, int dimension = 3, double inner_radius = 0.5, double outer_radius = 1.0, int DEBUG_ = 0);

    // Generate the boundary nodes first as an independent CVT
    // Then using the boundary nodes locked into position, generate the interior
    // CVT.
    // Inner generators are r[0] -> r[nb_inner-1]
    // Outer generators are r[nb_inner] -> r[nb_inner+nb_outer-1]
    // Interior generators are r[nb_inner+nb_outer] -> r[nb_tot]
    virtual void cvt(int *it_num, double *it_diff, double *energy);

    // Redirect custom call to the default CVT generation routine
    virtual void cvt(double r[], int *it_num_boundary_, int *it_num_interior_, double *it_diff, double *energy, int it_max_boundary_, int it_max_interior_, int sample_num) {
        int it_num;
        it_max_boundary = it_max_boundary_;
        it_max_interior = it_max_interior_;
        delete [] generators;
        generators = &r[0];
        this->cvt(&it_num, it_diff, energy);
        *it_num_boundary_ = it_num_boundary;
        *it_num_interior_ = it_num_interior;
    }

    // Override the filename generation 
    virtual void cvt_get_file_prefix(char* filename_buffer);

    // Set/Get the number of iterations that the seed is fixed. That is,
    // the number of iterations during which random samples are the same.
    void SetSeedFixedIterations(int new_it_fixed) { it_fixed = new_it_fixed; }
    int GetSeedFixedIterations() { return it_fixed; }

    // Set/Get the radom number generator seed
    void SetSeed(int new_seed) { seed = new_seed; }
    int GetSeed() { return seed; }

    // Set/Get the number of samples to generate in batches
    void SetBatch(int new_batch) { batch = new_batch; }
    int GetBatch() { return batch; }

    void SetInitType(int init_type) { init = init_type; }
    int GetInitType() { return init; }
    void SetSampleType(int sample_type) { sample = sample_type; }
    int GetSampleType() { return sample; }
    

public: // Overridden Methods:
    
    // Generate n random samples within our geometry.
    // The sampling is random is the bounding box described by box_extents.
    // If a sample is found to be outside of the geometry (by calling reject_point(...)
    // then we must re-sample and until we get a point inside the geometry.
    virtual void user_sample ( int dim_num, int n, int *seed, double r[] );
    // For this class: redirect to user_sample(...)
    virtual void user_init ( int dim_num, int n, int *seed, double r[] );

protected:
    // Check if the point of dimension ndim is contained within the boundary and interior
    // of the actual geometry, not just the bounding box extents. If it is outside
    // then we reject the sample point.
    virtual bool reject_point(double point[], int ndim);

    // Project points to the surface of a sphere with specified radius
    // generator[] contains all n points of dimension ndim
    virtual void project_to_sphere(double generator[], int n, int ndim, double radius);
};

#endif //_NESTED_SPHERE_CVT_H_
