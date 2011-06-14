#ifndef _RBF_InvMultiquadric_H_
#define _RBF_InvMultiquadric_H_

#include <math.h>
#include <Vec3.h>
#include "rbf.h"

class RBF_InvMultiquadric : public RBF
{
    private:

    public:
        RBF_InvMultiquadric(double epsilon, int dim_num) 
            : RBF(epsilon, dim_num) 
        {
        }

        RBF_InvMultiquadric(CMPLX epsilon, int dim_num) 
            : RBF(epsilon, dim_num) 
        {
        }

        ~RBF_InvMultiquadric() {};


        // DONT KNOW WHY THESE ARENT AVAILABLE FROM SUPERCLASS: 
        // seems like defining the pure virtual routines below overrides the availability of these?
        virtual double xderiv(const Vec3& xvec, const Vec3& x_center) { this->xderiv(xvec-x_center); }
        virtual double yderiv(const Vec3& xvec, const Vec3& x_center) { this->yderiv(xvec-x_center); }
        virtual double zderiv(const Vec3& xvec, const Vec3& x_center) { this->zderiv(xvec-x_center); }

        //------------------------------------------------
        virtual double eval(const Vec3& x) {
            double r2 = (double) x.square();
            return 1./sqrt(1.+(r2*eps2));
        }

        virtual CMPLX eval(const CVec3& x) {
            CMPLX r2 = x.square();
            return 1./sqrt(1.+(r2*ceps2));
        }

        virtual double eval(double x) { 
            double r2 = x*x; 
            return 1./sqrt(1.+(r2*eps2));
        }

        virtual CMPLX eval(CMPLX x) {
            CMPLX r2 = x*x; 
            return 1./sqrt(1.+(r2*ceps2));
        }

        //------------------------------------------------
        virtual double xderiv(const Vec3& x) { 
            double r2 = x.square();
            double xeps = x.x() * eps2;
            // -xeps / (1+(xe)^2)^(3/2)
            return -xeps * pow(this->eval(x), 3);
        }

        virtual CMPLX xderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX xeps = x.x() * ceps2;
            return -xeps * pow(this->eval(x), 3);
        }
        //------------------------------------------------
        virtual double yderiv(const Vec3& x) { 
            double r2 = x.square();
            double yeps = x.y() * eps2;
            return -yeps * pow(this->eval(x), 3);
        }

        virtual CMPLX yderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX yeps = x.y() * ceps2;
            return -yeps * pow(this->eval(x), 3);
        }
        //------------------------------------------------
        virtual double zderiv(const Vec3& x) { 
            double r2 = x.square();
            double zeps = x.z() * eps2;
            return -zeps * pow(this->eval(x), 3);
        }

        virtual CMPLX zderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX zeps = x.z() * ceps2;
            return -zeps * pow(this->eval(x), 3);
        }
        //------------------------------------------------
        virtual double lapl_deriv1D(const Vec3& x) {
            double x2eps2 = x.x()*x.x() * eps2; 
            return eps2 * (-1. + 2.*x2eps2) / pow(this->eval(x), 5);
        }

        virtual CMPLX lapl_deriv1D(const CVec3& x) {
            CMPLX x2eps2 = x.x()*x.x() * ceps2; 
            return ceps2 * (-1. + 2.*x2eps2) / pow(this->eval(x), 5);
        }
        //------------------------------------------------
        virtual double lapl_deriv2D(const Vec3& x) {
            double x2eps2 = x.x()*x.x() * eps2; 
            double y2eps2 = x.y()*x.y() * eps2; 
            return eps2 * (-2. + x2eps2 + y2eps2) / pow(this->eval(x), 5); 
        }

        virtual CMPLX lapl_deriv2D(const CVec3& x) {
            CMPLX x2eps2 = x.x()*x.x() * ceps2; 
            CMPLX y2eps2 = x.y()*x.y() * ceps2; 
            return ceps2 * (-2. + x2eps2 + y2eps2) / pow(this->eval(x), 5); 
        }
        //------------------------------------------------
        virtual double lapl_deriv3D(const Vec3& x) {
            double x2eps2 = x.x()*x.x() * eps2; 
            double y2eps2 = x.y()*x.y() * eps2; 
            double z2eps2 = x.z()*x.z() * eps2; 
            return (3.*eps2) / pow(this->eval(x), 5); 
        }

        virtual CMPLX lapl_deriv3D(const CVec3& x) {
            CMPLX x2eps2 = x.x()*x.x() * ceps2; 
            CMPLX y2eps2 = x.y()*x.y() * ceps2; 
            CMPLX z2eps2 = x.z()*x.z() * ceps2; 
            return (3.*ceps2) / pow(this->eval(x), 5); 
        }

};

#endif 

