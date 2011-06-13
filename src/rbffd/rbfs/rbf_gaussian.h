#ifndef _RBF_Gaussian_H_
#define _RBF_Gaussian_H_

#include <math.h>
#include <Vec3.h>
#include "rbf.h"

// RBF_Gaussian function. 
// I might create subclass for different rbf functions

// Gaussian RBF_Gaussian
// Theorically positive definite

class RBF_Gaussian : public RBF
{
    private:

    public:
        RBF_Gaussian(double epsilon, int dim_num) 
            : RBF(epsilon, dim_num) 
        {
        }

        RBF_Gaussian(CMPLX epsilon, int dim_num) 
            : RBF(epsilon, dim_num) 
        {
        }

        ~RBF_Gaussian() {};


        // DONT KNOW WHY THESE ARENT AVAILABLE FROM SUPERCLASS: 
        // seems like defining the pure virtual routines below overrides the availability of these?
        virtual double xderiv(const Vec3& xvec, const Vec3& x_center) { this->xderiv(xvec-x_center); }
        virtual double yderiv(const Vec3& xvec, const Vec3& x_center) { this->yderiv(xvec-x_center); }
        virtual double zderiv(const Vec3& xvec, const Vec3& x_center) { this->zderiv(xvec-x_center); }

        //------------------------------------------------
        virtual double eval(const Vec3& x) {
            double r2 = x.square();
            return exp(-(r2*eps2));
        }

        virtual CMPLX eval(const CVec3& x) {
            CMPLX r2 = x.square();
            return exp(-(r2*ceps2));
        }

        virtual double eval(double x) { 
            double r2 = x*x; 
            return exp(-(r2*eps2)); 
        }

        virtual CMPLX eval(CMPLX x) {
            CMPLX r2 = x*x; 
            return exp(-(r2*ceps2));
        }

        //------------------------------------------------
        virtual double xderiv(const Vec3& x) { 
            double r2 = x.square();
            double xeps = x.x() * eps2;
            return -2. * xeps * this->eval(x);
        }

        virtual CMPLX xderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX xeps = x.x() * ceps2;
            return -2. * xeps * this->eval(x);
        }
        //------------------------------------------------
        virtual double yderiv(const Vec3& x) { 
            double r2 = x.square();
            double yeps = x.y() * eps2;
            return -2. * yeps * this->eval(x);
        }

        virtual CMPLX yderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX yeps = x.y() * ceps2;
            return -2. * yeps * this->eval(x);
        }
        //------------------------------------------------
        virtual double zderiv(const Vec3& x) { 
            double r2 = x.square();
            double zeps = x.z() * eps2;
            return -2. * zeps * this->eval(x);
        }

        virtual CMPLX zderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX zeps = x.z() * ceps2;
            return -2. * zeps * this->eval(x);
        }
        //------------------------------------------------
        virtual double lapl_deriv1D(const Vec3& x) {
            double x2eps2 = x.x()*x.x() * eps2; 
            return 2. * eps2 * (-1. + 2.*x2eps2) * this->eval(x); 
        }

        virtual CMPLX lapl_deriv1D(const CVec3& x) {
            CMPLX x2eps2 = x.x()*x.x() * ceps2; 
            return 2. * ceps2 * (-1. + 2.*x2eps2) * this->eval(x); 
        }
        //------------------------------------------------
        virtual double lapl_deriv2D(const Vec3& x) {
            double x2eps2 = x.x()*x.x() * eps2; 
            double y2eps2 = x.y()*x.y() * eps2; 
            return 4. * eps2 * (-1. + x2eps2 + y2eps2) * this->eval(x); 
        }

        virtual CMPLX lapl_deriv2D(const CVec3& x) {
            CMPLX x2eps2 = x.x()*x.x() * ceps2; 
            CMPLX y2eps2 = x.y()*x.y() * ceps2; 
            return 4. * ceps2 * (-1. + x2eps2 + y2eps2) * this->eval(x); 
        }
        //------------------------------------------------
        virtual double lapl_deriv3D(const Vec3& x) {
            double r2 = x.square();
            double r2eps4 = r2 * (eps2 * eps2); 
            return (-6. * eps2 + 4.*r2eps4) * this->eval(x);
        }

        virtual CMPLX lapl_deriv3D(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX r2eps4 = r2 * (ceps2 * ceps2); 
            return (-6. * ceps2 + 4.*r2eps4) * this->eval(x);
        }

};

#endif
