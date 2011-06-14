#ifndef _RBF_Multiquadric_H_
#define _RBF_Multiquadric_H_

#include <math.h>
#include <Vec3.h>
#include "rbf.h"

// RBF_Multiquadric function. 
// I might create subclass for different rbf functions

// Multiquadric RBF_Multiquadric
// NOT theoretically positive definite. 

class RBF_Multiquadric : public RBF
{
    private:

    public:
        RBF_Multiquadric(double epsilon, int dim_num) 
            : RBF(epsilon, dim_num) 
        {
        }

        RBF_Multiquadric(CMPLX epsilon, int dim_num) 
            : RBF(epsilon, dim_num) 
        {
        }

        ~RBF_Multiquadric() {};


        // DONT KNOW WHY THESE ARENT AVAILABLE FROM SUPERCLASS: 
        // seems like defining the pure virtual routines below overrides the availability of these?
        virtual double xderiv(const Vec3& xvec, const Vec3& x_center) { this->xderiv(xvec-x_center); }
        virtual double yderiv(const Vec3& xvec, const Vec3& x_center) { this->yderiv(xvec-x_center); }
        virtual double zderiv(const Vec3& xvec, const Vec3& x_center) { this->zderiv(xvec-x_center); }

        //------------------------------------------------
        virtual double eval(const Vec3& x) {
            double r2 = (double) x.square();
            return sqrt(1.+(r2*eps2));
        }

        virtual CMPLX eval(const CVec3& x) {
            CMPLX r2 = x.square();
            return sqrt(1.+(r2*ceps2));
        }

        virtual double eval(double x) { 
            double r2 = x*x; 
            return sqrt(1.+(r2*eps2));
        }

        virtual CMPLX eval(CMPLX x) {
            CMPLX r2 = x*x; 
            return sqrt(1.+(r2*ceps2));
        }

        //------------------------------------------------
        virtual double xderiv(const Vec3& x) { 
            double r2 = x.square();
            double xeps = x.x() * eps2;
            return xeps / this->eval(x);
        }

        virtual CMPLX xderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX xeps = x.x() * ceps2;
            return xeps / this->eval(x);
        }
        //------------------------------------------------
        virtual double yderiv(const Vec3& x) { 
            double r2 = x.square();
            double yeps = x.y() * eps2;
            return yeps / this->eval(x);
        }

        virtual CMPLX yderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX yeps = x.y() * ceps2;
            return yeps / this->eval(x);
        }
        //------------------------------------------------
        virtual double zderiv(const Vec3& x) { 
            double r2 = x.square();
            double zeps = x.z() * eps2;
            return zeps / this->eval(x);
        }

        virtual CMPLX zderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX zeps = x.z() * ceps2;
            return zeps / this->eval(x);
        }
        //------------------------------------------------
        virtual double lapl_deriv1D(const Vec3& x) {
            double x2eps2 = x.x()*x.x() * eps2; 
            return eps2 / pow(this->eval(x), 3); 
        }

        virtual CMPLX lapl_deriv1D(const CVec3& x) {
            CMPLX x2eps2 = x.x()*x.x() * ceps2; 
            return ceps2 / pow(this->eval(x), 3); 
        }
        //------------------------------------------------
        virtual double lapl_deriv2D(const Vec3& x) {
            double x2eps2 = x.x()*x.x() * eps2; 
            double y2eps2 = x.y()*x.y() * eps2; 
            return eps2 * (2. + x2eps2 + y2eps2) / pow(this->eval(x), 3); 
        }

        virtual CMPLX lapl_deriv2D(const CVec3& x) {
            CMPLX x2eps2 = x.x()*x.x() * ceps2; 
            CMPLX y2eps2 = x.y()*x.y() * ceps2; 
            return ceps2 * (2. + x2eps2 + y2eps2) / pow(this->eval(x), 3); 
        }
        //------------------------------------------------
        virtual double lapl_deriv3D(const Vec3& x) {
            double x2eps2 = x.x()*x.x() * eps2; 
            double y2eps2 = x.y()*x.y() * eps2; 
            double z2eps2 = x.z()*x.z() * eps2; 
            return eps2 * (3. + 2.*x2eps2 + 2.*y2eps2 +2.*z2eps2) / pow(this->eval(x), 3); 
        }

        virtual CMPLX lapl_deriv3D(const CVec3& x) {
            CMPLX x2eps2 = x.x()*x.x() * ceps2; 
            CMPLX y2eps2 = x.y()*x.y() * ceps2; 
            CMPLX z2eps2 = x.z()*x.z() * ceps2; 
            return ceps2 * (3. + 2.*x2eps2 + 2.*y2eps2 +2.*z2eps2) / pow(this->eval(x), 3); 
        }

};

#endif 
