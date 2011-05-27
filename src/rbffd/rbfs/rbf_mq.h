#ifndef _RBF_MQ_H_
#define _RBF_MQ_H_

#include <math.h>
#include <Vec3.h>
#include "rbf.h"

using namespace std;

// RBF_MQ function. 
// I might create subclass for different rbf functions

// Gaussian RBF_MQ
// Theorically positive definite? 
#if 1
class RBF_MQ : public RBF{
    private:

    public:
        RBF_MQ(double epsilon, int dim_num) : RBF(epsilon, dim_num) {
        }

        RBF_MQ(CMPLX epsilon, int dim_num) : RBF(epsilon, dim_num) {
        }

        virtual ~RBF_MQ() {};

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
        //
        // fxx = eps2/f - eps2*(x-x_center)*f^{-2}*eps2(x-x_center)*f^{-1}
        //     = eps2*f^{-1} * (1 - eps2*(x-x_center)^2*f^{-2})
        //
        // fxx+fyy 
        //     = eps2*f^{-1} * (2 - eps2*R^2*f^{-2})
        //  where R^2 = (x-x_center)^2 + (y-yi)^2
        virtual double xderiv(const Vec3& xvec, const Vec3& x_center) {
            Vec3 r = (xvec-x_center);
            return this->xderiv(r); 
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
            return this->yderiv(r);
        }

        virtual double zderiv(const Vec3& xvec, const Vec3& x_center) {
            Vec3 r = (xvec-x_center);
            this->zderiv(r);
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
            double f = eval(xvec);
            double lapl = (dim*eps2 + (dim-1)*eps2*eps2*r*r) / (f*f*f);
#if 0
            double r2 = xvec.square();
            double lapl2 = ((dim * eps2) / f) - ((eps2 * eps2 * r2) / (f*f*f));
            if (lapl - lapl2  > 1e-6) {
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
#else 

// EFB052311 (from original grady conversion)
// Might not support 3D. Also, im not sure .square() is the proper way to get [(x-x0)^2 + (y-y0)^2]


class RBF_MQ : public RBF {
    private:

    public:
        RBF_MQ(double epsilon, int dim_num) : RBF(epsilon, dim_num) {
        }

        RBF_MQ(CMPLX epsilon, int dim_num) : RBF(epsilon, dim_num) {
        }

        virtual ~RBF_MQ() {};

        //double operator()(const Vec3& x, const Vec3& xi) {
        //return eval(x,xi);
        //}

        // added Aug. 15, 2009
        //double operator()(const Vec3& x) {
        //return eval(x);
        //}

        // added Aug. 15, 2009
        //double operator()(double x) {
        //return eval(x);
        //}

        // f = (1+eps2*r^2)^{1/2}
        // added Aug. 15, 2009
        virtual double eval(const Vec3& x) {
            double r2 = x.square();
            return sqrt(1+(eps2*r2));
        }
        virtual CMPLX eval(const CVec3& x) {
            CMPLX r2 = x.square();
            return sqrt(1.+(ceps2*r2));
        }



        // added Aug. 15, 2009
        virtual double eval(double x) {
            //printf("eval, x= %f, eps2= %f, rbf= %21.14e\n", x, eps2, sqrt(1.+eps2*x*x));
            return sqrt(1+(eps2*x*x));
        }

        // added Aug. 16, 2009
        virtual CMPLX eval(CMPLX x) {
            CMPLX dd = sqrt(1.+(ceps2*x*x));
            //printf("cmplx eval, x= %f,%f, eps= %f,%f, rbf= %21.14e,%21.14e\n", real(x),imag(x), real(ceps), imag(ceps), real(dd), imag(dd));
            return sqrt(1.+(ceps2*x*x));
        }


        // fx = (1/2) (1+(eps*r)^2))^{-1/2} * 2*eps2 (x-xi)
        //    = eps2*(x-xi)/f 
        // fxx = eps2/f - eps2*(x-xi)*f^{-2}*eps2(x-xi)*f^{-1}
        //     = eps2*f^{-1} * (1 - eps2*(x-xi)^2*f^{-2})
        // fxx+fyy 
        //     = eps2*f^{-1} * (2 - eps2*R^2*f^{-2})
        //  where R^2 = (x-xi)^2 + (y-yi)^2
        virtual double xderiv(const Vec3& xvec) {
            double f = eval(xvec);
            return(eps2*(xvec.x())/(f*f));
        }
        virtual CMPLX xderiv(const CVec3& xvec) {
            CMPLX f = eval(xvec);
            return((ceps2*xvec.x())/f);
        }


        virtual double yderiv(const Vec3& xvec) {
            double f = eval(xvec);
            return(eps2*(xvec.y())/(f*f));
        }
        virtual CMPLX yderiv(const CVec3& xvec) {
            CMPLX f = eval(xvec);
            return((ceps2*xvec.y())/f);
        }


        virtual double zderiv(const Vec3& xvec) {
            double f = eval(xvec);
            return(eps2*(xvec.z())/(f*f));
        }
        virtual CMPLX zderiv(const CVec3& xvec) {
            CMPLX f = eval(xvec);
            return((ceps2*xvec.z())/f);
        }


        virtual double xxderiv(const Vec3& xvec, const Vec3& xi) {
            return(0.);
        }

        virtual double yyderiv(const Vec3& xvec, const Vec3& xi) {
            return(0.);
        }

        virtual double xyderiv(const Vec3& xvec, const Vec3& xi) {
            return(0.);
        }

        // fxx+fyy 
        //     = eps2*f^{-1} * (2 - eps2*R^2*f^{-2})
        //  where R^2 = (x-xi)^2 + (y-yi)^2
        //  lapl(f) = eps2*f^{-1}*(2.-(r*eps)^2*f^{-2})

        // added Aug. 15, 2009
        virtual double lapl_deriv(const Vec3& xvec) {
            double f = eval(xvec);
            double f2 = 1./(f);
            double f3 = 1./(f*f);
            double r2e = xvec.square() * eps2;
            double d = eps2*f2*(2.-r2e*f3);
            return d;
        }

        // added Aug. 15, 2009
        virtual double lapl_deriv(const double x) {
            double f = eval(x);
            double f2 = 1./(f);
            double f3 = 1./(f*f);
            double r2e = x*x*eps2; 
            double d = eps2*f2*(2.-r2e*f3);
            return d;
        }

        // added Aug. 16, 2009
        virtual CMPLX lapl_deriv(const CMPLX x) {
            CMPLX f = eval(x);
            CMPLX f2 = 1./(f);
            CMPLX f3 = 1./(f*f);
            CMPLX r2e = x*x*ceps2; 
            CMPLX d = ceps2*f2*(2.-r2e*f3);
            return d;
        }


        // EFB052311: dont know why these are not inherited. Probably due to parameter polymorph
        // maybe because of const'ness?
        double xderiv(const Vec3& xvec, const Vec3& x_center) { this->xderiv(xvec-x_center); }
        double yderiv(const Vec3& xvec, const Vec3& x_center) { this->yderiv(xvec-x_center); }
        double zderiv(const Vec3& xvec, const Vec3& x_center) { this->zderiv(xvec-x_center); }
        double lapl_deriv(const Vec3& xvec, const Vec3& x_center) { this->zderiv(xvec-x_center); }
};

#endif 

#endif
