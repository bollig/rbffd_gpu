#ifndef _EXACT_UNIFORM_LAPLACIAN_H_
#define _EXACT_UNIFORM_LAPLACIAN_H_

#include "exact_solution.h"

class ExactUniformLaplacian : public ExactSolution
{
private:
public:
	ExactUniformLaplacian(int dimensions)
        : ExactSolution(dimensions)
    {;}
	~ExactUniformLaplacian();

	double operator()(double x, double y, double z, double t) {
        // f(x, t) = 2x^2 + 3y^2 + 4z^2
        // NOTE: when dimension is less than 3 we have to exclude the extra terms in the laplacian
        //       this operator also conditionally handles the dimension just to play it safe
        double val = 2.*x*x; 
        if (dim_num > 1) {
            val += 3*y*y; 
        }
        if (dim_num > 2) {
            val+= 4.*z*z; 
        }
        return val;
    }
	double laplacian(double x, double y, double z, double t) {
        double val = 4.; 
        if (dim_num > 1) {
            val += 6.; 
        }
        if (dim_num > 2) {
            val+= 8.; 
        }  
        return val;
    }

	double xderiv(double x, double y, double z, double t) {
        return 4.*x;
    }
	double yderiv(double x, double y, double z, double t) {
        return 6.*y;
    }
	double zderiv(double x, double y, double z, double t) {
        return 8.*z;
    }

	double tderiv(double x, double y, double z, double t) {
        return 0;
    }
};
//----------------------------------------------------------------------

#endif 


