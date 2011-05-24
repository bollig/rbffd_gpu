#include <stdio.h>
#include <stdlib.h>

#include "exact_regulargrid_shu2006.h"

//----------------------------------------------------------------------
double ExactRegularGrid_Shu2006::operator()(double x, double y, double z, double t)
{
    // Equation (14), page 1303
    double term1 = pow(1. - (x/2.), 6) * pow(1. - (y/2.), 6); 
    double term2 = 1000. * pow(1. - x, 3) * (x*x*x) * pow(1.-y, 3) * (y*y*y); 
    double term3 = pow(y, 6) * pow(1. - (x/2.), 6); 
    double term4 = pow(x, 6) * pow(1. - (y/2.), 6); 

    double u_2 = term1 + term2 + term3 + term4; 

    return u_2;

}
//----------------------------------------------------------------------
double ExactRegularGrid_Shu2006::laplacian(double x, double y, double z, double t)
{
    printf("NOT SUPPORTED: Exact::LAPL"); 
    exit(EXIT_FAILURE); 
   //	return simpLapl;
    return -1;
}
//----------------------------------------------------------------------
double ExactRegularGrid_Shu2006::xderiv(double x, double y, double z, double t)
{
    printf("NOT SUPPORTED: Exact::X"); 
    exit(EXIT_FAILURE); 
    return -1.;
}

//----------------------------------------------------------------------
double ExactRegularGrid_Shu2006::yderiv(double x, double y, double z, double t)
{
    printf("NOT SUPPORTED: Exact::Y"); 
    exit(EXIT_FAILURE); 
    return -1.;
}

//----------------------------------------------------------------------
double ExactRegularGrid_Shu2006::zderiv(double x, double y, double z, double t)
{
    printf("NOT SUPPORTED: Exact::Z"); 
    exit(EXIT_FAILURE); 
    return -1.;
}
//----------------------------------------------------------------------
double ExactRegularGrid_Shu2006::tderiv(double x, double y, double z, double t)
{
    // TODO: add time component to the equation
#if 0
    double x_contrib2 = x * x; 
    double y_contrib2 = y * y; 
    double z_contrib2 = z * z;

    double r2 = x_contrib2 + y_contrib2 + z_contrib2; 

    double r = sqrt(r2);

    // From mathematica time derivative of exact solution
    return -exp(-t*decay) * decay * cos(freq * r);
#endif 
    printf("NOT SUPPORTED: Exact::T"); 
    exit(EXIT_FAILURE); 
    return -1;
}
