#include <stdlib.h>
#include <stdio.h>
#include "exact_ellipse.h"

//----------------------------------------------------------------------
ExactEllipse::ExactEllipse(double freq, double decay, double axis1, double axis2, double axis3)
: ExactEllpisoid(freq, decay, axis1, axis2, 0.)
{
    // Make sure these do not influence our solution here (they are used in the 3D Ellipsoid case)
	princ_axis3_inv2 = 0.;
	princ_axis3_inv4 = 0.;
}
//----------------------------------------------------------------------
ExactEllipse::~ExactEllipse()
{}

//----------------------------------------------------------------------
double ExactEllipse::operator()(double x, double y, double z, double t)
{
	double x_contrib = x * x * princ_axis1_inv2; 
	double y_contrib = y * y * princ_axis2_inv2; 
	
	double r = sqrt(x_contrib + y_contrib);

	// if temporal decay is too large, time step will have to decrease

	double T = cos(freq * r) * exp(-decay * t);
	return T;
}
//----------------------------------------------------------------------
double ExactEllipse::laplacian(double x, double y, double z, double t)
{
	// This is based on mathematica simplified laplacian of exact solution
	// F = lapl(f)
	double x2 = x*x; 
	double y2 = y*y;

        // axis1_2 read as "Axis 1, squared."
	double axis1_2 = princ_axis1*princ_axis1; 
	double axis2_2 = princ_axis2*princ_axis2; 
	double axis1_4 = axis1_2*axis1_2; 
	double axis2_4 = axis2_2*axis2_2;
	
	double myf1 = freq * r * (y2*axis1_4 + x2*axis2_4) * cos(freq*r);
	double myf2 = (x2 + y2) * axis1_2 * axis2_2 * sin(freq*r); 
	double mydenom = axis1_2 * r * axis2_2 * (y2 * axis1_2 + x2 * axis2_2);
	double mylapl = -exp(-decay*t) * freq * (myf1 + myf2) / (mydenom); 
	
	return mylapl;
}
//----------------------------------------------------------------------
double ExactEllipse::tderiv(double x, double y, double z, double t)
{
	double x_contrib2 = x * x * princ_axis1_inv2; 
	double y_contrib2 = y * y * princ_axis2_inv2; 
		
	double r2 = x_contrib2 + y_contrib2; 

	double r = sqrt(r2);
	
	// From mathematica time derivative of exact solution
	return -exp(-t*decay) * decay * cos(freq * r);
}
