#ifndef _EXACT_REGULAR_GRID_H_
#define _EXACT_REGULAR_GRID_H_


// Copying the exact solution from Shui, Ding, Zhao (2006) "Numerical
// Comparison of Least Square-Based Finite-Difference (LSFD) and Radial Basis
// Function-Based Finite-Difference (RBFFD) Methods"

#include "exact_solution.h"

class ExactRegularGrid_Shu2006 : public ExactSolution
{
private:

public:
	//ExactEllipsoid();
	ExactRegularGrid_Shu2006() {;}
	~ExactRegularGrid_Shu2006() {;}

	double operator()(double x, double y, double z, double t);
#if 1
	double laplacian(double x, double y, double z, double t);

	double xderiv(double x, double y, double z, double t);
	double yderiv(double x, double y, double z, double t);
	double zderiv(double x, double y, double z, double t);

	double tderiv(double x, double y, double z, double t); 
#endif 
};
//----------------------------------------------------------------------

#endif // _EXACT_REGULAR_GRID_H_

