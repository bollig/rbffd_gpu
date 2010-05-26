#ifndef EXACT_SOLUTION
#define EXACT_SOLUTION

#include "Vec3.h"

//----------------------------------------------------------------------
class ExactSolution
{
private:

public:

//	ExactSolution(); 
//	~ExactSolution() {}; 
 
	virtual double operator()(double x, double y, double z, double t) = 0;
	virtual double operator()(Vec3& r, double t) {
		return (*this)(r.x(), r.y(), r.z(), t);
	}
	
	virtual double at(Vec3& r, double t) {
		return (*this)(r, t);
	}

	virtual double laplacian(Vec3& v, double t) {
		return this->laplacian(v.x(), v.y(), v.z(), t); 
	}

	// Return the value of the analytic Laplacian of the equation f(x,y,z;t)
	// NOTE: in previous versions of code this called to get a forcing term:
	// 			F(x,y,z;t) = df/dt - lapl(f)
	// instead of just:
	// 			lapl(f)
	// To get the original behavior, substitute: this->tderiv() - this->laplacian()
	virtual double laplacian(double x, double y, double z, double t) = 0; // if scalar function

	virtual double xderiv(double x, double y, double z, double t) = 0; // if scalar function
	virtual double yderiv(double x, double y, double z, double t) = 0; // if scalar function
	virtual double zderiv(double x, double y, double z, double t) = 0; // if scalar function
	virtual double tderiv(double x, double y, double z, double t) = 0; // if scalar function

	double xderiv(Vec3& r, double t) {
		return xderiv(r.x(), r.y(), r.z(), t);
	}

	double yderiv(Vec3& r, double t) {
		return yderiv(r.x(), r.y(), r.z(), t);
	}

	double zderiv(Vec3& r, double t) {
		return zderiv(r.x(), r.y(), r.z(), t);
	}

	double tderiv(Vec3& r, double t) {
		return tderiv(r.x(), r.y(), r.z(), t);
	}


	//virtual double divergence() = 0; // if vector function (not used)
};
//----------------------------------------------------------------------

#endif
