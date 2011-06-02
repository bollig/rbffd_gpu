#include <stdio.h>
#include <stdlib.h>

#include "exact_regulargrid.h"

//----------------------------------------------------------------------
ExactRegularGrid::ExactRegularGrid(int dimension, double freq, double decay)
    : ExactSolution(dimension)
{
    this->freq = freq;
    this->decay = decay;
}
//----------------------------------------------------------------------
ExactRegularGrid::~ExactRegularGrid()
{}

//----------------------------------------------------------------------
double ExactRegularGrid::operator()(double x, double y, double z, double t)
{
    double x_contrib = x * x; 
    double y_contrib = y * y; 
    double z_contrib = z * z; 

    double r = sqrt(x_contrib + y_contrib + z_contrib); 

    // if temporal decay is too large, time step will have to decrease

    double T = cos(freq * r) * exp(-decay * t);
    return T;
}

double ExactRegularGrid::laplacian1D(double x, double y, double z, double t)
{
    return -exp(-decay * t) * freq * freq * cos(x); 
}

double ExactRegularGrid::laplacian2D(double x, double y, double z, double t)
{
    double r = sqrt(x*x + y*y); 
    // WARNING! Catch 0 laplacian at the origin. 
    if (r < 1e-10) {
        /* exp(..) * ( 0 * cos(..) + 0 ) / 0 */
        return 0;
    }
    return -( exp(-decay * t) * freq * ( r * freq * cos(r * freq) + sin(r * freq) ) ) / r;
}

double ExactRegularGrid::laplacian3D(double x, double y, double z, double t)
{
    double r = sqrt(x*x + y*y + z*z); 

    // WARNING! Catch case when r==0
    if (r < 1e-10) {
        return -( exp(-decay * t) * freq * ( freq * cos(r*freq) /* + 0/0 */) );
    }

    // Case when r!=0
    return -( exp(-decay * t) * freq * ( freq * cos(r*freq) + (2.*sin(r*freq) / r)) );
}

//----------------------------------------------------------------------
double ExactRegularGrid::laplacian(double x, double y, double z, double t)
{
    if (dim_num == 1) {
        return this->laplacian1D(x,y,z,t); 
    } else if (dim_num == 2) {
        return this->laplacian2D(x,y,z,t);
    } else if (dim_num == 3) {
        return this->laplacian3D(x,y,z,t);
    } else {
        printf("[ExactRegularGrid] ERROR: only dimensions 1, 2, and 3 are valid.\n"); 
        exit(EXIT_FAILURE);
    }
}
//----------------------------------------------------------------------
double ExactRegularGrid::xderiv(double x, double y, double z, double t)
{
    double r = sqrt(x*x + y*y + z*z); 
    if (r < 1e-10) {
        return 0;
    }
    return - ( exp(-t * decay) * x * freq * sin(r*freq) ) / r;
}

//----------------------------------------------------------------------
double ExactRegularGrid::yderiv(double x, double y, double z, double t)
{
    double r = sqrt(x*x + y*y + z*z); 
    if (r < 1e-10) {
        return 0;
    }
    return - ( exp(-t * decay) * y * freq * sin(r*freq) ) / r;
}

//----------------------------------------------------------------------
double ExactRegularGrid::zderiv(double x, double y, double z, double t)
{
    double r = sqrt(x*x + y*y + z*z); 
    if (r < 1e-10) {
        return 0;
    }
    return - ( exp(-t * decay) * z * freq * sin(r*freq) ) / r;
}
//----------------------------------------------------------------------
double ExactRegularGrid::tderiv(double x, double y, double z, double t)
{
    double x_contrib2 = x * x; 
    double y_contrib2 = y * y; 
    double z_contrib2 = z * z;

    double r2 = x_contrib2 + y_contrib2 + z_contrib2; 

    double r = sqrt(r2);

    // From mathematica time derivative of exact solution
    return -exp(-t*decay) * decay * cos(freq * r);
}
