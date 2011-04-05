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
        RBF_MQ(double epsilon, int dim_num) : RBF(epsilon, dim_num) {
        }

        RBF_MQ(CMPLX epsilon, int dim_num) : RBF(epsilon, dim_num) {
        }

        ~RBF_MQ() {};

        // f = (1+eps2*r^2)^{1/2}
        inline double eval(const Vec3& x, const Vec3& x_center) {
            double r = (x-x_center).magnitude();
            //double r2 = (x-x_center).square();
            double r2 = r*r;
            //       printf("eval, r= %f, eps=%f, eps2= %f, rbf= %21.14e\n", r, eps, eps2, sqrt(1.+eps2*r2));
            return sqrt(1.+(eps2*r2));
        }
        // added Aug. 15, 2009
        inline double eval(const Vec3& x) {
            double r = x.magnitude();
            //double r2 = x.square();
            double r2 = r*r;
            //      printf("eval, r= %f, eps=%f, eps2= %f, rbf= %21.14e\n", r, eps, eps2, sqrt(1.+eps2*r2));
            return sqrt(1.+(eps2*r2));
        }

        // added Sept. 11, 2009
        inline CMPLX eval(const CVec3& x) {
            //printf("inside eval CVec3\n");
            CMPLX r = x.magnitude();
            CMPLX r2 = x.square();
            return sqrt(1.+(ceps2*r2));
        }

        // added Aug. 15, 2009
        inline double eval(double x) {
            double r = x;
            double r2 = r*r;
            // printf("eval, r= %f, eps=%f, eps2= %f, rbf= %21.14e\n", r, eps, eps2, sqrt(1.+eps2*r2));
            //cout << "ceps= " << ceps << endl;
            return sqrt(1.+(eps2*r2));
        }

        // added Aug. 16, 2009
        virtual CMPLX eval(CMPLX x) {
            CMPLX dd = sqrt(1.+(ceps2*x*x));
            //printf("cmplx eval, x= %f,%f, eps= %f,%f, rbf= %21.14e,%21.14e\n", real(x),imag(x), real(ceps), imag(ceps), real(dd), imag(dd));
            return sqrt(1.+(ceps2*x*x));
        }

        // f  = (1+eps2*r^2)^{1/2}
        // fx = (1/2) (1+(eps*r)^2))^{-1/2} * 2*eps2 (x-x_center)
        //    = eps2*(x-x_center)/f 
        // fxx = eps2/f - eps2*(x-x_center)*f^{-2}*eps2(x-x_center)*f^{-1}
        //     = eps2*f^{-1} * (1 - eps2*(x-x_center)^2*f^{-2})
        // fxx+fyy 
        //     = eps2*f^{-1} * (2 - eps2*R^2*f^{-2})
        //  where R^2 = (x-x_center)^2 + (y-yi)^2
        virtual double xderiv(const Vec3& xvec, const Vec3& x_center) {
            Vec3 r = (xvec-x_center);
            double f = eval(r);
            return(eps2*(r.x())/f);
        }

        virtual double xderiv(const Vec3& xvec) {
            // not called for xderiv svd
            double f = eval(xvec);
            return(eps2*xvec.x()/f);
        }

        virtual CMPLX xderiv(const CVec3& xvec) {
            //printf("inside xderiv CVec3\n");
            //cout << xvec.x() << "\t" << xvec.y() << "\t" << xvec.z() << endl;
            CMPLX f = eval(xvec);
            //cout << real(f) << "+" << imag(f) << "i" << endl;
            return((ceps2*xvec.x())/f);
        }

        virtual double yderiv(const Vec3& xvec) {
            double f = eval(xvec);
            return((eps2*xvec.y())/f);
        }

        virtual CMPLX yderiv(const CVec3& xvec) {
            //cout << xvec.x() << "\t" << xvec.y() << "\t" << xvec.z() << endl;
            CMPLX f = eval(xvec);
            //cout << real(f) << "+" << imag(f) << "i" << endl;
            return((ceps2*xvec.y())/f);
        }

        virtual double yderiv(const Vec3& xvec, const Vec3& x_center) {
            Vec3 r = (xvec-x_center);
            double f = eval(r);
            return(eps2*(r.y())/f);
        }

        virtual double zderiv(const Vec3& xvec, const Vec3& x_center) {
            Vec3 r = (xvec-x_center);
            double f = eval(r);
            return(eps2*(r.z())/f);
        }

        virtual double zderiv(const Vec3& zvec) {
            // not called for zderiv svd
            double f = eval(zvec);
            return(eps2*zvec.z()/f);
        }

        virtual CMPLX zderiv(const CVec3& xvec) {
            //printf("inside zderiv CVec3\n");
            //cout << xvec.x() << "\t" << xvec.y() << "\t" << xvec.z() << endl;
            CMPLX f = eval(xvec);
            //cout << real(f) << "+" << imag(f) << "i" << endl;
            return((ceps2*xvec.z())/f);
        }

#if 0
        // First derivative:  d Phi(r) / dr
        virtual double first_deriv(const Vec3& x) {
            double r = x.magnitude();

            // first derivative is: eps^2 r/sqrt(1+eps^2 r^2)
            return (eps2 * r) / eval(x);
        }

        // First derivative:  d Phi(r) / dr
        virtual double second_deriv(const Vec3& x) {
            double r = x.magnitude();

            // second derivative is: eps^2 / (1+eps^2 r^2)^3/2
            double f = eval(x);
            double ff = f*f*f;
            return eps2 / ff;
        }


        // radial derivative dPhi/dr(x,y,z) = x/r * d/dx  + y/r * d/dy + z/r * d/dz
        // xj is the center; x/r above is center.x() / center.magnitude()
        double radialderiv(const Vec3& xvec, const Vec3& xj) {

            // Allow easy swap of center in our equation below
            const Vec3& center = xvec;
            const Vec3& node = xj;

            const Vec3& r_d = xvec - xj;
            double r = center.magnitude();
            double f = sqrt(1. + eps2 * r_d.square());

            double top = (center.x()*center.x() - center.x()*node.x() + center.y()*center.y() - center.y()*node.y() + center.z()*center.z() - center.z()*node.z())*eps2;
            double bottom = sqrt(center.x()*center.x() + center.y()*center.y() + center.z()*center.z())*sqrt(1. + eps2 * (r_d.magnitude()));
            //return top/bottom;

            return eps2*(center.x()*r_d.x() + center.y()*r_d.y() + center.z()*r_d.z())/(r*f);
            //return eps2*((r*r) - (center.x()*node.x() + center.y()*node.y() + center.z()*node.z())) / (r * f);
        }
#endif 
        virtual double xxderiv(const Vec3& xvec, const Vec3& x_center) {
            return(0.);
        }

        virtual double yyderiv(const Vec3& xvec, const Vec3& x_center) {
            return(0.);
        }

        virtual double xyderiv(const Vec3& xvec, const Vec3& x_center) {
            return(0.);
        }

        // fxx+fyy 
        //     = eps2*f^{-1} * (2 - eps2*R^2*f^{-2})
        //  where R^2 = (x-x_center)^2 + (y-yi)^2
        //  lapl(f) = eps2*f^{-1}*(2.-(r*eps)^2*f^{-2})
        virtual double lapl_deriv(const Vec3& xvec, const Vec3& x_center) {
            return lapl_deriv(xvec-x_center);
        }

        // added Aug. 15, 2009
        virtual double lapl_deriv(const Vec3& xvec) {
#if 1
            //printf("lapl_deriv: x= %f %f %f %f %d\n", xvec.x(), xvec.y(), xvec.z(), eps2, dim);
            // general form: lapl = d^2 Phi / d r^2 + ((DIM-1)/r) * dPhi / dr
            // however, if r is 0 then we have issues with that and need the simplified equation.
            // This is the simplified equation:
            double r = xvec.magnitude();
            double r2 = xvec.square();
            double f = eval(xvec);
            double lapl = (dim*eps2 + (dim-1)*eps2*eps2*r*r) / (f*f*f);
#if 0
            double lapl2 = ((dim * eps2) / f) - ((eps2 * eps2 * r2) / (f*f*f));
            if (lapl != lapl2) {
                std::cout << "ERROR: " << lapl << " != " << lapl2 << std::endl;
                exit(EXIT_FAILURE);
            }
#endif
            return lapl;
#else
            double f = eval(xvec);
            double f2 = 1./(f);
            double f3 = 1./(f*f);
            double r2e = xvec.square() * eps2;
            double d = eps2*f2*(2.-r2e*f3);
            return d;
#endif
        }

        // added Aug. 15, 2009
        virtual double lapl_deriv(const double x) {
#if 1
            //printf("lapl_deriv: x= %f\n", x);
            double r = x;
            double r2 = r*r;
            // general form: lapl = d^2 Phi /s d r^2 + ((DIM-1)/r) * dPhi / dr
            // however, if r is 0 then we have issues with that and need the simplified equation.
            // This is the simplified equation:
            double f = eval(r);
            double lapl = (dim*eps2 + (dim-1)*eps2*eps2*r*r) / (f*f*f);
#if 0
            double lapl2 = ((dim * eps2) / f) - ((eps2 * eps2 * r2) / (f*f*f));
            if (lapl != lapl2) {
                std::cout << "ERROR: " << lapl << " != " << lapl2 << std::endl;
                exit(EXIT_FAILURE);
            }
#endif
            return lapl;
#else
            //printf("lapl_deriv(double), x= %f\n", x);
            double f = eval(x);
            double f2 = 1./(f);
            double f3 = 1./(f*f);
            double r2e = x*x*eps2;
            double d = eps2*f2*(2.-r2e*f3);
            return d;
#endif
        }

        // added Aug. 16, 2009
        virtual CMPLX lapl_deriv(const CMPLX x) {
            //printf("lapl_deriv: complex, x= (%f,%f)\n", real(x), imag(x));

            // general form: lapl = d^2 Phi /s d r^2 + ((DIM-1)/r) * dPhi / dr
            // however, if r is 0 then we have issues with that and need the simplified equation.
            // This is the simplified equation:
            CMPLX f = eval(x);
            double scale1 = dim;
            double scale2 = dim-1;
            // cout << "WARNING! CMPLX LAPLACIAN MAY NOT FUNCTION CORRECTLY: " << scale1 << "\t" << scale2 << endl;
            CMPLX lapl = (scale1*ceps2 + scale2*(ceps2*ceps2)*(x*x)) / (f*f*f);
#if 0	    
            CMPLX lapl2 = ((scale1*ceps2) / f) - ((ceps2 * ceps2 * x*x) / (f*f*f));
            if ( > 1e-15) {
                std::cout << eps << " != " << ceps << endl;
                std::cout << scale1*ceps2 << " + " << scale2*(ceps2*ceps2) << " * " << (x*x) << " / " << (f*f*f) << std::endl; 
                std::cout << (ceps2/f) << " - " << (ceps2*ceps2) << " * " << x*x << "  / " << f*f*f << std::endl; 
                std::cout << "ERROR: lapl:" << lapl << " != lapl2:" << lapl2 << " diff=" << lapl-lapl2 << "  f=" << f << "  x=" << x << " eps=" << eps << "  eps2=" << eps2 << " ceps=" << ceps << "  ceps2=" << ceps2 << std::endl;
                exit(EXIT_FAILURE);
            }
#endif 
            return lapl;
        }

};

#endif
