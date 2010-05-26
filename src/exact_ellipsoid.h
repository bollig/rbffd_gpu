#ifndef EXACT_ELLIPSOID
#define EXACT_ELLIPSOID

#include "exact_solution.h"

class ExactEllipsoid : public ExactSolution
{
private:
	double freq;
	double decay;
	double princ_axis1_inv2;
	double princ_axis1_inv4;
	double princ_axis2_inv2;
	double princ_axis2_inv4;
	double princ_axis3_inv2;
	double princ_axis3_inv4;
	double princ_axis1;
	double princ_axis2;
	double princ_axis3;

public:
	//ExactEllipsoid();
	ExactEllipsoid(double freq, double decay, double axis1, double axis2, double axis3);
	~ExactEllipsoid();

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

