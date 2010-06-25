#ifndef EXACT_ELLIPSE
#define EXACT_ELLIPSE

#include "exact_ellipsoid.h"

class ExactEllipsoid : public ExactEllipsoid
{
public:
	//ExactEllipsoid();
	ExactEllipse(double freq, double decay, double axis1, double axis2);
	~ExactEllipse();

	double operator()(double x, double y, double z, double t);
	double laplacian(double x, double y, double z, double t);

	double xderiv(double x, double y, double z, double t);
	double yderiv(double x, double y, double z, double t);
	double zderiv(double x, double y, double z, double t);

	double tderiv(double x, double y, double z, double t); 
	
	//virtual double divergence() = 0; // if vector function (not used)
};
//----------------------------------------------------------------------

#endif

