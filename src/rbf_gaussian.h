#ifndef _RBF_Gaussian_H_
#define _RBF_Gaussian_H_

#include <math.h>
#include <Vec3.h>
#include "rbf.h"

// RBF_Gaussian function. 
// I might create subclass for different rbf functions

// Gaussian RBF_Gaussian
// Theorically positive definite

class RBF_Gaussian : public RBF{
private:

public:
	RBF_Gaussian(double epsilon) : RBF(epsilon) {
	}

	RBF_Gaussian(CMPLX epsilon) : RBF(epsilon) {
	}

	~RBF_Gaussian() {};

	double operator()(const Vec3& x, const Vec3& xi) {
		return eval(x,xi);
	}

	inline double eval(const Vec3& x, const Vec3& xi) {
		const Vec3& xx = x-xi;
		double r = xx*xx;
		return exp(-eps2*r);
	}

	// added Aug. 15, 2009
	inline double eval(const Vec3& x) {
		double r2 = x.square();
		return exp(-eps*r2);
	}

	// added Aug. 15, 2009
	inline double eval(double x) {
	    //printf("eval, x= %f, eps2= %f, rbf= %21.14e\n", x, eps2, sqrt(1.+eps2*x*x));
		return exp(-eps2*x*x);
	}

	// added Aug. 16, 2009
	CMPLX eval(CMPLX x) {
		return exp(-ceps2*x*x);
	}
	
	double xderiv(const Vec3& xvec, const Vec3& xi) {
		return(-2.*eps2*(xvec.x()-xi.x())*eval(xvec,xi));
	}

	double yderiv(const Vec3& xvec, const Vec3& xi) {
		return(-2.*eps2*(xvec.y()-xi.y())*eval(xvec,xi));
	}

	double zderiv(const Vec3& xvec, const Vec3& xi) {
		return(-2.*eps2*(xvec.z()-xi.z())*eval(xvec,xi));
	}

	double xxderiv(const Vec3& xvec, const Vec3& xi) {
		return(0.);
	}

	double yyderiv(const Vec3& xvec, const Vec3& xi) {
		return(0.);
	}

	double xyderiv(const Vec3& xvec, const Vec3& xi) {
		return(0.);
	}

	//  lapl(f) = 4.*f*eps^2*(eps^2*(xvec-xi)^2 - 1)
	double lapl_deriv(const Vec3& xvec, const Vec3& xi) {
		double d = 4.*eval(xvec,xi)*eps2*(eps2*(xvec-xi).square() - 1.);
		return d;
	}

	// added Aug. 15, 2009
	double lapl_deriv(const Vec3& xvec) {
		double d = 4.*eval(xvec)*eps2*(eps2*xvec.square() - 1.);
		return d;
	}

	// added Aug. 15, 2009
	double lapl_deriv(const double x) {
		double d = 4.*eval(x)*eps2*(eps2*x*x - 1.);
		return d;
	}

	// added Aug. 16, 2009
	CMPLX lapl_deriv(const CMPLX x) {
		CMPLX d = 4.*eval(x)*ceps2*(ceps2*x*x - 1.);
		return d;
	}
};

#endif
