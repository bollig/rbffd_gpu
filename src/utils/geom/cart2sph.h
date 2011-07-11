#ifndef __CART_2_SPH_H__
#define __CART_2_SPH_H__

#include <math.h> 

typedef struct sph_coords { 
    double theta; 
    double phi; 
    double r; 
} sph_coords_type; 

sph_coords_type cart2sph(double x, double y, double z);

// NOTE: sech is not provided in the c math library. Its equal to 1/cosh(x).
// We define it here for convenience. 
inline double sech(double x) {
    return 1./cosh(x);
//    return 2./(exp(x) + exp(-x));
}


#endif 
