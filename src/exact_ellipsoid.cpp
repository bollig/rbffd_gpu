#include <stdlib.h>
#include <stdio.h>
#include "exact_ellipsoid.h"

//----------------------------------------------------------------------
ExactEllipsoid::ExactEllipsoid(double freq, double decay, double axis1, double axis2, double axis3) 
{
	this->freq = freq;
	this->decay = decay;

	// Perhaps these should be read from the grid?
	princ_axis1 = axis1; //1.0;   //grid.getMajor();
	princ_axis2 = axis2; //0.5;   //grid.getMinor();
	princ_axis3 = axis3; //0.5; 

	princ_axis1_inv2 = 1. / (princ_axis1 * princ_axis1);
	princ_axis1_inv4 = princ_axis1_inv2 * princ_axis1_inv2;
	
	princ_axis2_inv2 = 1. / (princ_axis2 * princ_axis2);
	princ_axis2_inv4 = princ_axis2_inv2 * princ_axis2_inv2;

	princ_axis3_inv2 = 1. / (princ_axis3 * princ_axis3);
	princ_axis3_inv4 = princ_axis3_inv2 * princ_axis3_inv2;
}
//----------------------------------------------------------------------
ExactEllipsoid::~ExactEllipsoid() 
{}

//----------------------------------------------------------------------
double ExactEllipsoid::operator()(double x, double y, double z, double t)
{
	double x_contrib = x * x * princ_axis1_inv2; 
	double y_contrib = y * y * princ_axis2_inv2; 
	double z_contrib = z * z * princ_axis3_inv2; 
	
	double r = sqrt(x_contrib + y_contrib + z_contrib); 

	// if temporal decay is too large, time step will have to decrease

	double T = cos(freq * r) * exp(-decay * t);
	return T;
}
//----------------------------------------------------------------------
double ExactEllipsoid::laplacian(double x, double y, double z, double t)
{
	double x_contrib2 = x * x * princ_axis1_inv2; 
	double y_contrib2 = y * y * princ_axis2_inv2; 
	double z_contrib2 = z * z * princ_axis3_inv2;
	
	double x_contrib4 = x * x * princ_axis1_inv4; 
	double y_contrib4 = y * y * princ_axis2_inv4;
	double z_contrib4 = z * z * princ_axis3_inv4;
	
	//double r2 = pt.x() * pt.x() * maji2 + pt.y() * pt.y() * mini2 + pt.z() * pt.z() * min2i2;
	//double r4 = pt.x() * pt.x() * maji4 + pt.y() * pt.y() * mini4 + pt.z() * pt.z() * min2i4;
	
	double r2 = x_contrib2 + y_contrib2 + z_contrib2; 
	double r4 = x_contrib4 + y_contrib4 + z_contrib4; 

	double r = sqrt(r2);
	double f1;
	double f2;

	// if temporal decay is too large, time step will have to decrease

	double nn = freq * r;
	double freq2 = freq * freq;
	
	#if 1
//Evan: This is not just the laplacian of the exact solution
// its df/dt - lapl(f)
	f1 = cos(nn) * ((freq2 / r2) * r4 - decay);
	//	f2 = freq2 * (maji2 + mini2 - r4 / r2);
	f2 = freq2 * (princ_axis1_inv2 + princ_axis2_inv2 - r4 / r2);	


	if (nn < 1.e-5) {
		f2 *= (1. - nn * nn / 6.);
		fprintf(stderr, "RUNNING LOW\n"); 
	} else {
		f2 *= sin(nn) / nn;
	}

	f1 = (f1 + f2) * exp(-decay * t);
	#endif 
	#if 1
	// This is based on mathematica simplified laplacian of exact solution
	// F = lapl(f)
	double x2 = x*x; 
	double y2 = y*y; 
	double axis1_2 = princ_axis1*princ_axis1; 
	double axis2_2 = princ_axis2*princ_axis2; 
	double axis1_4 = axis1_2*axis1_2; 
	double axis2_4 = axis2_2*axis2_2;
	
	double myf1 = freq * r * (y2*axis1_4 + x2*axis2_4) * cos(freq*r);
	double myf2 = (x2 + y2) * axis1_2 * axis2_2 * sin(freq*r); 
	double mydenom = axis1_2 * r * axis2_2 * (y2 * axis1_2 + x2 * axis2_2);
	double mylapl = -exp(-decay*t) * freq * (myf1 + myf2) / (mydenom); 
	#else 
	// F = df/dt - lapl(f)
	// same as above but no safety check for small nn
	double x2 = x*x; 
	double y2 = y*y; 
	double axis1_2 = princ_axis1*princ_axis1; 
	double axis2_2 = princ_axis2*princ_axis2; 
	double axis1_4 = axis1_2*axis1_2; 
	double axis2_4 = axis2_2*axis2_2; 
	
	double myf1 = - r * (-freq2 * (y2*axis1_4 + x2*axis2_4) + axis1_2*axis2_2*(y2*axis1_2 + x2*axis2_2)) * cos(nn);
	double myf2 = freq * (x2 + y2) * axis1_2 * axis2_2 * sin(nn); 
	double mydenom = axis1_2 * r * axis2_2 * (y2 * axis1_2 + x2 * axis2_2);
	double mylapl = exp(-decay*t) * (myf1 + myf2) / (mydenom); 
	#endif
	
	printf("COMPARE LAPLACIAN: %f, mine: %f\n", f1, mylapl);
	//printf("t= %f, alpha= %f\n", t, alpha);
	//printf("exp= %f, nn= %f, f1= %f, f2= %f\n", exp(-alpha*t), nn, f1, f2);

	return mylapl;
}
//----------------------------------------------------------------------
double ExactEllipsoid::xderiv(double x, double y, double z, double t)
{
	return -1.;
}

//----------------------------------------------------------------------
double ExactEllipsoid::yderiv(double x, double y, double z, double t)
{
	return -1.;
}

//----------------------------------------------------------------------
double ExactEllipsoid::zderiv(double x, double y, double z, double t)
{
	return -1.;
}
//----------------------------------------------------------------------
double ExactEllipsoid::tderiv(double x, double y, double z, double t)
{
	double x_contrib2 = x * x * princ_axis1_inv2; 
	double y_contrib2 = y * y * princ_axis2_inv2; 
	double z_contrib2 = z * z * princ_axis3_inv2;
		
	double r2 = x_contrib2 + y_contrib2 + z_contrib2; 

	double r = sqrt(r2);
	
	// From mathematica time derivative of exact solution
	return -exp(-t*decay) * decay * cos(freq * r);
}
