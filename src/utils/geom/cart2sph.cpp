#include "cart2sph.h"

sph_coords_type cart2sph(double x, double y, double z) {
    sph_coords_type ret; 
    // NOTE: we use atan2 to get the tan^{-1} for the right quadrant
    ret.theta = atan2(y,x); 
    ret.r     = sqrt(x*x + y*y + z*z); 
    ret.phi   = atan2(z, sqrt(x*x + y*y));

    return ret;
}


