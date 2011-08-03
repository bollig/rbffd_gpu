#ifndef __CART_2_SPH_CL__
#define __CART_2_SPH_CL__

FLOAT4 cart2sph(FLOAT4 node) {
    FLOAT4 ret;   
    double x = node.x; 
    double y = node.y; 
    double z = node.z;
    
    // NOTE: we use atan2 to get the tan^{-1} for the right quadrant
    // Theta: 
    ret.x = atan2(y,x); 
    // R
    ret.z = sqrt(x*x + y*y + z*z); 
    // Phi
    ret.y = atan2(z, sqrt(x*x + y*y)); 
    // EMPTY
    ret.w = 0.f;
    return ret;
}

#endif 

