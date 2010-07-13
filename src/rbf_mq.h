#ifndef _RBF_MQ_H_
#define _RBF_MQ_H_

#include <math.h>
#include <Vec3.h>
#include "rbf.h"

using namespace std;

// RBF_MQ function. 
// I might create subclass for different rbf functions

// Gaussian RBF_MQ
// Theorically positive definite

class RBF_MQ : public RBF{
private:

public:
	RBF_MQ(double epsilon) : RBF(epsilon) {
		//eps2= eps*eps;
	}

	RBF_MQ(CMPLX epsilon) : RBF(epsilon) {
		//ceps2= ceps*ceps;
	}

	~RBF_MQ() {};

	// f = (1+eps2*r^2)^{1/2}
	inline double eval(const Vec3& x, const Vec3& xi) {
		double r2 = (x-xi).square();
                return sqrt(1.+(eps2*r2));
	}
	// added Aug. 15, 2009
	inline double eval(const Vec3& x) {
		double r2 = x.square();
                return sqrt(1.+(eps2*r2));
	}

	// added Sept. 11, 2009
	inline CMPLX eval(const CVec3& x) {
		//printf("inside eval CVec3\n");
		CMPLX r2 = x.square();
		return sqrt(1.+(ceps2*r2));
	}

	// added Aug. 15, 2009
	inline double eval(double x) {
	    //printf("eval, x= %f, eps2= %f, rbf= %21.14e\n", x, eps2, sqrt(1.+eps2*x*x));
                //cout << "ceps= " << ceps << endl;
                return sqrt(1.+(eps2*x*x));
	}

	// added Aug. 16, 2009
	CMPLX eval(CMPLX x) {
		CMPLX dd = sqrt(1.+(ceps2*x*x));
	    //printf("cmplx eval, x= %f,%f, eps= %f,%f, rbf= %21.14e,%21.14e\n", real(x),imag(x), real(ceps), imag(ceps), real(dd), imag(dd));
		return sqrt(1.+(ceps2*x*x));
	}

	// f  = (1+eps2*r^2)^{1/2}
	// fx = (1/2) (1+(eps*r)^2))^{-1/2} * 2*eps2 (x-xi)
	//    = eps2*(x-xi)/f 
	// fxx = eps2/f - eps2*(x-xi)*f^{-2}*eps2(x-xi)*f^{-1}
	//     = eps2*f^{-1} * (1 - eps2*(x-xi)^2*f^{-2})
	// fxx+fyy 
	//     = eps2*f^{-1} * (2 - eps2*R^2*f^{-2})
	//  where R^2 = (x-xi)^2 + (y-yi)^2
	double xderiv(const Vec3& xvec, const Vec3& xi) {
		double f = eval(xvec, xi);
		return(eps2*(xvec.x()-xi.x())/f);
	}

	double xderiv(const Vec3& xvec) {
		// not called for xderiv svd
		double f = eval(xvec);
		return(eps2*xvec.x()/f);
	}

	CMPLX xderiv(const CVec3& xvec) {
		//printf("inside xderiv CVec3\n");
                //cout << xvec.x() << "\t" << xvec.y() << "\t" << xvec.z() << endl;
                CMPLX f = eval(xvec);
                //cout << real(f) << "+" << imag(f) << "i" << endl;
		return(ceps2*xvec.x()/f);
	}

	double yderiv(const Vec3& xvec) {
		double f = eval(xvec);
		return(eps2*xvec.y()/f);
	}

	CMPLX yderiv(const CVec3& xvec) {
            //cout << xvec.x() << "\t" << xvec.y() << "\t" << xvec.z() << endl;
            CMPLX f = eval(xvec);
            //cout << real(f) << "+" << imag(f) << "i" << endl;
		return(ceps2*xvec.y()/f);
	}

	double yderiv(const Vec3& xvec, const Vec3& xi) {
		double f = eval(xvec, xi);
		return(eps2*(xvec.y()-xi.y())/f);
	}

        double zderiv(const Vec3& xvec, const Vec3& xi) {
                double f = eval(xvec, xi);
		return(eps2*(xvec.z()-xi.z())/f);
	}

        double zderiv(const Vec3& zvec) {
                // not called for zderiv svd
                double f = eval(zvec);
                return(eps2*zvec.z()/f);
        }

        CMPLX zderiv(const CVec3& xvec) {
                //printf("inside zderiv CVec3\n");
            //cout << xvec.x() << "\t" << xvec.y() << "\t" << xvec.z() << endl;
            CMPLX f = eval(xvec);
            //cout << real(f) << "+" << imag(f) << "i" << endl;
                return(ceps2*xvec.z()/f);
        }

        // xvec is the center
        double rderiv(const Vec3& xvec, const Vec3& xi) {
                return xvec.x() * xderiv(xvec, xi) + xvec.y() * yderiv(xvec, xi) + xvec.z() * zderiv(xvec, xi);
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

	// fxx+fyy 
	//     = eps2*f^{-1} * (2 - eps2*R^2*f^{-2})
	//  where R^2 = (x-xi)^2 + (y-yi)^2
	//  lapl(f) = eps2*f^{-1}*(2.-(r*eps)^2*f^{-2})

	double lapl_deriv(const Vec3& xvec, const Vec3& xi) {
		return lapl_deriv(xvec-xi);
		/*
		double f = eval(xvec, xi);
		double f2 = 1./(f);
		double f3 = 1./(f*f);
		double r2e = (xi-xvec).square() * eps2;
		double d = eps2*f2*(2.-r2e*f3);
		return d;
		*/
	}

	// added Aug. 15, 2009
	double lapl_deriv(const Vec3& xvec) {
		double f = eval(xvec);
		double f2 = 1./(f);
		double f3 = 1./(f*f);
		double r2e = xvec.square() * eps2;
		double d = eps2*f2*(2.-r2e*f3);
		return d;
	}

	// added Aug. 15, 2009
	double lapl_deriv(const double x) {
		//printf("lapl_deriv(double), x= %f\n", x);
		double f = eval(x);
		double f2 = 1./(f);
		double f3 = 1./(f*f);
		double r2e = x*x*eps2; 
		double d = eps2*f2*(2.-r2e*f3);
		return d;
	}

	// added Aug. 16, 2009
	CMPLX lapl_deriv(const CMPLX x) {
		//printf("lapl_deriv: complex, x= (%f,%f)\n", real(x), imag(x));
		CMPLX f = eval(x);
		CMPLX f2 = 1./(f);
		CMPLX f3 = 1./(f*f);
		CMPLX r2e = x*x*ceps2; 
		CMPLX d = ceps2*f2*(2.-r2e*f3);
		return d;
	}

};

#endif
