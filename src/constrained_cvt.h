#ifndef _ELLIPSE_CVT_H_
#define _ELLIPSE_CVT_H_

#include <vector>
#include "Vec3.h"
#include "density.h"
#include "cvt.h"

// This is a special case of Centroidal Voronoi Tessellation where some of the
// Generating points lie ON the boundary and not within. Since CVTs find centers
// of mass that lie within the interior, we use an orthogonal projection step to
// constrain generators that are within a prescribed distance to the boundary. 

class ConstrainedCVT : public CVT {
protected:
    // Number of Projection Partitions (From John's code: the number of
    //    subintervals into which we subdivide
    //    the boundary.  It does NOT specify how many points will be pulled onto
    //    the boundary.  The reason for this is that, after the first boundary
    //    subinterval has had a generator pulled into it, on every subsequent
    //    subinterval the nearest generator is likely to be the one in the
    //    previous subinterval!  Unless an interior generator is closer than
    //    some small distance, this process will simply drag some unfortunate
    //    generator onto the boundary, and then around from interval to interval
    //    for a considerable time).

    int npp;


public:
    ConstrainedCVT(int DEBUG_ = 0);

    // Override this to customize the initial seed point generation/sampling
    // Also, override this if you want to force projection at the start
    // of the code
    virtual void cvt(int dim_num, int n, int batch, int init, int sample, int sample_num, int it_max, int it_fixed, int *seed, double r[], int *it_num, double *it_diff, double *energy);

    // This is the projection routine
    virtual void project(int ndim, int n, double generator[], int npp);
};

#endif //_ELLIPSE_CVT_H_
