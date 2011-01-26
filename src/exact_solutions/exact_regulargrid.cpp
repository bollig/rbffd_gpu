#include <stdio.h>

#include "exact_regulargrid.h"

//----------------------------------------------------------------------
ExactRegularGrid::ExactRegularGrid(double freq, double decay)
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
//----------------------------------------------------------------------
double ExactRegularGrid::laplacian(double x, double y, double z, double t)
{
	double x_contrib2 = x * x; 
	double y_contrib2 = y * y; 
	double z_contrib2 = z * z;
	
	double x_contrib4 = x * x; 
	double y_contrib4 = y * y;
	double z_contrib4 = z * z;
	
	//double r2 = pt.x() * pt.x() * maji2 + pt.y() * pt.y() * mini2 + pt.z() * pt.z() * min2i2;
	//double r4 = pt.x() * pt.x() * maji4 + pt.y() * pt.y() * mini4 + pt.z() * pt.z() * min2i4;
	
	double r2 = x_contrib2 + y_contrib2 + z_contrib2; 
	double r4 = x_contrib4 + y_contrib4 + z_contrib4; 

	double r = sqrt(r2);
	
	// if temporal decay is too large, time step will have to decrease

	double nn = freq * r;
	double freq2 = freq * freq;
	
	#if 0
	// This is based on mathematica simplified laplacian of exact solution in 2D
	// F = lapl(f)
	double x2 = x*x; 
	double y2 = y*y; 
	double axis1_2 = 1;
	double axis2_2 = 1;
	double axis1_4 = 1;
	double axis2_4 = 1;
	
	double myf1 = freq * r * (y2 + x2) * cos(freq*r);
	double myf2 = (x2 + y2) * sin(freq*r); 
	double mydenom = r * (y2 + x2);
	double mylapl = -exp(-decay*t) * freq * (myf1 + myf2) / (mydenom); 
	#endif

        // 2D: e^{-t * decay} * freq * (-freq * cos(r * freq) - 2*sin(r * freq) / r)
        // Simplified laplacian from Mathematica (and verified by Gordon)
        double simpLapl = - exp(-t * decay) * freq * (freq * cos(r * freq) - (2*sin(r * freq) / r));

        // lapl(T) = [ - pi/2 cos(pi/2 r) -

        // Equation
        // dT/dt = lapl(T) - 2/r * sin(pi/2 r) * e^-t

       // printf("mylapl = %f (simpLapl = %f)\n", mylapl, simpLapl);
        // NOTE: after using simpLapl and mylapl I found a nan was introduced when calculaing the sin(..)/r when r=0 (it should short circuit because sin(..)==0 so its 0/0 == 0. To bypass his nan we doing the convoluted stuff below for "together"
        double part1 =  - exp(-t * decay) * freq ; 
        double part2 = (freq * cos(r * freq));
        double part3 = (- (2*sin(r * freq))); 
        double part3_and_4 = 0.;
        if (part3 > 1e-8) { 
            part3_and_4 = part3 / r; 
        }
        double together = part1*(part2+part3_and_4); 
        printf("%f * (%f + %e) = %e\n", part1, part2, part3_and_4, together);  
	//printf("t= %f, alpha= %f\n", t, alpha);
	//printf("exp= %f, nn= %f, f1= %f, f2= %f\n", exp(-alpha*t), nn, f1, f2);

//	return simpLapl;
        return together;
}
//----------------------------------------------------------------------
double ExactRegularGrid::xderiv(double x, double y, double z, double t)
{
	return -1.;
}

//----------------------------------------------------------------------
double ExactRegularGrid::yderiv(double x, double y, double z, double t)
{
	return -1.;
}

//----------------------------------------------------------------------
double ExactRegularGrid::zderiv(double x, double y, double z, double t)
{
	return -1.;
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
