#ifndef __CART_2_SPH_H__
#define __CART_2_SPH_H__

#include <math.h> 

typedef struct sph_coords { 
    double theta; 
    double phi; 
    double r; 
} sph_coords_type; 


sph_coords_type cart2sph(double x, double y, double z) {
    sph_coords_type ret; 
    // NOTE: we use atan2 to get the tan^{-1} for the right quadrant
    ret.theta = atan2(y,x); 
    ret.r     = sqrt(x*x + y*y + z*z); 
    ret.phi = atan2(z, sqrt(x*x + y*y));

    return ret;
}

// NOTE: sech is not provided in the c math library. Its equal to 1/cosh(x).
// We define it here for convenience. 
inline double sech(double x) {
    return 1./cosh(x);
//    return 2./(exp(x) + exp(-x));
}

#endif 
