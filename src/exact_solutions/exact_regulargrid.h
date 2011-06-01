#ifndef _EXACT_REGULAR_GRID_H_
#define _EXACT_REGULAR_GRID_H_

#include "exact_solution.h"

class ExactRegularGrid : public ExactSolution
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
	ExactRegularGrid(int dimension, double freq, double decay);
	~ExactRegularGrid();

	double operator()(double x, double y, double z, double t);
	double laplacian(double x, double y, double z, double t);

	double xderiv(double x, double y, double z, double t);
	double yderiv(double x, double y, double z, double t);
	double zderiv(double x, double y, double z, double t);

	double tderiv(double x, double y, double z, double t); 
};
//----------------------------------------------------------------------

#endif // _EXACT_REGULAR_GRID_H_

