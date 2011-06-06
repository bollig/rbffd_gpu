//FIXME: Need to make the laplacians time dependent. The grad already is.


#include <stdio.h>
#include <stdlib.h>

#include "exact_regulargrid_nonuniform.h"


//----------------------------------------------------------------------
ExactRegularGridNonUniform::ExactRegularGridNonUniform(int dimension, double freq)
    : ExactSolution(dimension)
{
    this->freq = freq;
}
//----------------------------------------------------------------------
ExactRegularGridNonUniform::~ExactRegularGridNonUniform()
{}

//----------------------------------------------------------------------
double ExactRegularGridNonUniform::operator()(double x, double y, double z, double t)
{
    // FIXME: assumes linear diffusion
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double x_contrib = x * x; 
    double y_contrib = y * y; 
    double z_contrib = z * z; 

    double r = sqrt(x_contrib + y_contrib + z_contrib); 

    // if temporal decay is too large, time step will have to decrease

    double T = cos(freq * r) * exp(-decay * t);
    return T;
}

double ExactRegularGridNonUniform::laplacian1D(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    return -exp(-decay * t) * freq * freq * cos(x); 
}

double ExactRegularGridNonUniform::laplacian2D(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double r = sqrt(x*x + y*y); 
    // WARNING! Catch 0 laplacian at the origin. 
    if (r < 1e-10) {
        /* exp(..) * ( 0 * cos(..) + 0 ) / 0 */
        return 0;
    }
    return -( exp(-decay * t) * freq * ( r * freq * cos(r * freq) + sin(r * freq) ) ) / r;
}

double ExactRegularGridNonUniform::laplacian3D(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double r = sqrt(x*x + y*y + z*z); 

    // WARNING! Catch case when r==0
    if (r < 1e-10) {
        return -( exp(-decay * t) * freq * ( freq * cos(r*freq) /* + 0/0 */) );
    }

    // Case when r!=0
    return -( exp(-decay * t) * freq * ( freq * cos(r*freq) + (2.*sin(r*freq) / r)) );
}

//----------------------------------------------------------------------
double ExactRegularGridNonUniform::laplacian(double x, double y, double z, double t)
{
    if (dim_num == 1) {
        return this->laplacian1D(x,y,z,t); 
    } else if (dim_num == 2) {
        return this->laplacian2D(x,y,z,t);
    } else if (dim_num == 3) {
        return this->laplacian3D(x,y,z,t);
    } else {
        printf("[ExactRegularGridNonUniform] ERROR: only dimensions 1, 2, and 3 are valid.\n"); 
        exit(EXIT_FAILURE);
    }
}
//----------------------------------------------------------------------
double ExactRegularGridNonUniform::xderiv(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double r = sqrt(x*x + y*y + z*z); 
    if (r < 1e-10) {
        return 0;
    }
    return - ( exp(-t * decay) * x * freq * sin(r*freq) ) / r;
}

//----------------------------------------------------------------------
double ExactRegularGridNonUniform::yderiv(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double r = sqrt(x*x + y*y + z*z); 
    if (r < 1e-10) {
        return 0;
    }
    return - ( exp(-t * decay) * y * freq * sin(r*freq) ) / r;
}

//----------------------------------------------------------------------
double ExactRegularGridNonUniform::zderiv(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double r = sqrt(x*x + y*y + z*z); 
    if (r < 1e-10) {
        return 0;
    }
    return - ( exp(-t * decay) * z * freq * sin(r*freq) ) / r;
}
//----------------------------------------------------------------------
double ExactRegularGridNonUniform::tderiv(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double x_contrib2 = x * x; 
    double y_contrib2 = y * y; 
    double z_contrib2 = z * z;

    double r2 = x_contrib2 + y_contrib2 + z_contrib2; 

    double r = sqrt(r2);

    // From mathematica time derivative of exact solution
    return -exp(-t*decay) * decay * cos(freq * r);
}

#if 0

// EFB060111 (THIS DID NOT WORK. I DONT WANT TO WASTE TIME DEBUGGING IT:)


//----------------------------------------------------------------------
double ExactRegularGridNonUniform::operator()(double x, double y, double z, double t)
{
    double x_contrib = x * x; 
    double y_contrib = y * y; 
    double z_contrib = z * z; 

    double r = sqrt(x_contrib + y_contrib + z_contrib); 

    // if temporal decay is too large, time step will have to decrease

    double decay = this->diffuseCoefficient(x,y,z,0.,t);

    double T = cos(freq * r) * exp(-decay* t);
    return T;
}

double ExactRegularGridNonUniform::laplacian1D(double x, double y, double z, double t)
{
    // NOTE: not sure this is a time dependent laplacian
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double lapl_u = -exp(-decay * t) * freq * freq * cos(x); 
    double grad_K_dot_grad_U = this->diffuse_xderiv(x,y,z,0.,t) * this->xderiv(x,y,z,t); 
    return grad_K_dot_grad_U + decay * lapl_u;
}

double ExactRegularGridNonUniform::laplacian2D(double x, double y, double z, double t)
{
    double r = sqrt(x*x + y*y); 
    // WARNING! Catch 0 laplacian at the origin. 
    if (r < 1e-10) {
        /* exp(..) * ( 0 * cos(..) + 0 ) / 0 */
        return 0;
    }
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double lapl_u = -( exp(-decay * t) * freq * ( r * freq * cos(r * freq) + sin(r * freq) ) ) / r;
    double grad_K_dot_grad_U = this->diffuse_xderiv(x,y,z,0.,t) * this->xderiv(x,y,z,0.,t) +  this->diffuse_yderiv(x,y,z,0.,t) * this->yderiv(x,y,z,0.,t); 
    return grad_K_dot_grad_U + decay * lapl_u;
}

double ExactRegularGridNonUniform::laplacian3D(double x, double y, double z, double t)
{
    // We use the identity that div(K * grad(U)) = K laplacian(U) if K is scalar and if vector: 
    // div(K * grad(U)) = grad(K) dot grad(U) + K * laplacian(U)
    // We take the lapl_u from ExactRegularGrid; 
    double r = sqrt(x*x + y*y + z*z); 

    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double grad_K_dot_grad_U = this->diffuse_xderiv(x,y,z,0.,t) * this->xderiv(x,y,z,0.,t) +  this->diffuse_yderiv(x,y,z,0.,t) * this->yderiv(x,y,z,0.,t) + this->diffuse_zderiv(x,y,z,0.,t) * this->zderiv(x,y,z,0.,t); 

    double lapl_u; 
    
    // WARNING! Catch case when r==0
    if (r < 1e-10) {
        lapl_u = -( exp(-decay * t) * freq * ( freq * cos(r*freq) /* + 0/0 */) );
    } else {
        lapl_u = -( exp(-decay * t) * freq * ( freq * cos(r*freq) + (2.*sin(r*freq) / r)) );
    }
    return grad_K_dot_grad_U + decay * lapl_u;
    // Case when r!=0
}

//----------------------------------------------------------------------
double ExactRegularGridNonUniform::laplacian(double x, double y, double z, double t)
{
    if (dim_num == 1) {
        return this->laplacian1D(x,y,z,t); 
    } else if (dim_num == 2) {
        return this->laplacian2D(x,y,z,t);
    } else if (dim_num == 3) {
        return this->laplacian3D(x,y,z,t);
    } else {
        printf("[ExactRegularGridNonUniform] ERROR: only dimensions 1, 2, and 3 are valid.\n"); 
        exit(EXIT_FAILURE);
    }
}

/****** BELOW THIS IS TIME DEPENDENT ********/
// NOTE: Above the line is not. 


//----------------------------------------------------------------------
double ExactRegularGridNonUniform::xderiv(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double r = sqrt(x*x + y*y + z*z); 
    if (r < 1e-10) {
        return 0;
    }

    // Derivative holding K constant
    double dU_with_K_const = - ( exp(-t * decay) * x * freq * sin(r*freq) ) / r;
    // Derivative of K holding U constant
    double dK_with_U_const = this->diffuse_xderiv(x,y,z,0.,t) * (*this)(x,y,z,t); 
    return -t * dK_with_U_const + dU_with_K_const;
}

//----------------------------------------------------------------------
double ExactRegularGridNonUniform::yderiv(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double r = sqrt(x*x + y*y + z*z); 
    if (r < 1e-10) {
        return 0;
    }
    // Derivative holding K constant
    double dU_with_K_const = - ( exp(-t * decay) * y * freq * sin(r*freq) ) / r;
    // Derivative of K holding U constant
    double dK_with_U_const = this->diffuse_yderiv(x,y,z,0.,t) * (*this)(x,y,z,t); 
    return -t * dK_with_U_const + dU_with_K_const;
}

//----------------------------------------------------------------------
double ExactRegularGridNonUniform::zderiv(double x, double y, double z, double t)
{
    double decay = this->diffuseCoefficient(x,y,z,0.,t);
    double r = sqrt(x*x + y*y + z*z); 
    if (r < 1e-10) {
        return 0;
    }
    // Derivative holding K constant 
    double dU_with_K_const = - ( exp(-t * decay) * z * freq * sin(r*freq) ) / r;
    // Derivative of K holding U constant (e^(-Kt) * Cos(r a)) * dk/dx
    // dk/dx = 0.3t^2;
    double dK_with_U_const = this->diffuse_zderiv(x,y,z,0.,t) * (*this)(x,y,z,t); 
    // The -t here is from our e^{-t * K}; d/dx(-t x) 
    return -t * dK_with_U_const + dU_with_K_const;
}
//----------------------------------------------------------------------
double ExactRegularGridNonUniform::tderiv(double x, double y, double z, double t)
{
    double r = sqrt(x*x + y*y + z*z); 
    double decay = this->diffuseCoefficient(x,y,z,0.,t);

    // From mathematica time derivative of exact solution
    double dudt_with_K_const = - exp(-t*decay) * cos(freq * r);  // -e^(-Kt) ( Cos(r alpha)
    double dKdt_with_U_const = this->diffuse_tderiv(x,y,z,0.,t);   // 0.6 * t * X
    return dKdt_with_U_const * dudt_with_K_const; 
}
#endif 
