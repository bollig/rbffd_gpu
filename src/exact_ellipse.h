#ifndef _EXACT_ELLIPSE_
#define _EXACT_ELLIPSE_

#include "exact_ellipsoid.h"

class ExactEllipse : public ExactEllipsoid
{
public:
	ExactEllipse(double freq, double decay, double axis1, double axis2);
	~ExactEllipse();

        virtual double operator()(double x, double y, double z, double t);
        virtual double laplacian(double x, double y, double z, double t);
        virtual double tderiv(double x, double y, double z, double t);
	//virtual double divergence() = 0; // if vector function (not used)
};
//----------------------------------------------------------------------

#endif

