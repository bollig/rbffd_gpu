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
        virtual double xderiv(const Vec3& xvec, const Vec3& x_center) { return this->xderiv(xvec-x_center); }
        virtual double yderiv(const Vec3& xvec, const Vec3& x_center) { return this->yderiv(xvec-x_center); }
        virtual double zderiv(const Vec3& xvec, const Vec3& x_center) { return this->zderiv(xvec-x_center); }
        virtual double rderiv(const Vec3& xvec, const Vec3& x_center) { return this->rderiv(xvec-x_center); }

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
        virtual double rderiv(const Vec3& x) { 
            double r2 = x.square();
            double reps = sqrt(r2) * eps2;
            return -2. * reps * this->eval(x);
        }

        virtual CMPLX rderiv(const CVec3& x) {
            CMPLX r2 = x.square();
            CMPLX reps = sqrt(r2) * ceps2;
            return -2. * reps * this->eval(x);
        }

         //------------------------------------------------
         //ANalytically remove r/r
        virtual double rderiv_over_r(const Vec3& x) { 
            double r2 = x.square();
            return -2. * eps2 * this->eval(x);
        }

        virtual CMPLX rderiv_over_r(const CVec3& x) {
            CMPLX r2 = x.square();
            return -2. * eps2 * this->eval(x);
        }

         //------------------------------------------------
         //ANalytically remove r/r
        virtual double rderiv2(const Vec3& x) { 
            double r2 = x.square();
            return this->rderiv_over_r(x) + 4.*(eps2*eps2)*r2 *exp(-(eps2*r2));
        }

        virtual CMPLX rderiv2(const CVec3& x) {
            CMPLX r2 = x.square();
            return this->rderiv_over_r(x) + 4.*(ceps2*ceps2)*r2 *exp(-(ceps2*r2));
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


        //------------------------------------------------
        //  HYPER VISCOSITY: k = 2 
        // From Fornberg, Lehto 2011 pg 2274
        virtual double lapl2_deriv2D(const Vec3& x) {
            double r2 = x.square(); 
            double e2r2 = r2 * eps2; 
            double e4r4 = e2r2 * e2r2;
            return eps2*eps2 * (16. * e4r4 - 64. * e2r2 + 32.) * this->eval(x); 
        }

        virtual CMPLX lapl2_deriv2D(const CVec3& x) {
            CMPLX r2 = x.square(); 
            CMPLX e2r2 = r2 * ceps2; 
            CMPLX e4r4 = e2r2 * e2r2;
            return ceps2*eps2 * (16. * e4r4 - 64. * e2r2 + 32.) * this->eval(x);
        }

        // Hyperviscosity: (lapl)^k
        // NOTE: maintain scaling separately
        // L(phi(r)) = eps^{2k} * p_k(r) * phi(r)
        // where 
        // p_k(r) = (-4)^k * k! * P_{k}^{d/2-1}( (eps*r)^2 )
        //
        // or: 
        //  p_0(r) = 1
        //  p_1(r) = 4(eps*r)^2 - 2d
        //  p_{k+1}(r) = 4( (eps*r)^2 - 2k - d/2 ) * p_k(r) - 8k(2k - 2 + d) * p_{k-1}(r)
        virtual double hyperviscosity(const Vec3& x, const int k) {
            double eps2r2 = eps2 * x.square();
            double r2 = x.square(); 
#if 0
            double hv_fact1; 
            switch (k) {
                case 0:
                    hv_fact1 = this->eval(x); 
                    break; 
                case 1:
                    hv_fact1 = eps2 * (4.*(eps2r2 - 4.)) * this->eval(x);
                    break; 
                case 2:
                    hv_fact1 = eps2*eps2 * (16.*eps2r2*eps2r2 - 64.*eps2r2 + 32) * this->eval(x); 
                    break; 
                case 3:
                    hv_fact1 = eps2*eps2*eps2 * (64.*eps2r2*eps2r2*eps2r2 - 576.*eps2r2*eps2r2 + 1152.*eps2r2 - 384.) * this->eval(x); 
                    break;
                default: 
                    break;
            };


            double hv_fact2 = pow(eps, 2*k) * this->hv_p_k(r2, k) * this->eval(x); 
            if (hv_fact1 - hv_fact2 > 1e-10) {
                std::cout << "[RBFFD] HV fact1-fact2 = " << hv_fact1 - hv_fact2 << " (" << hv_fact1 << ", " << hv_fact2 << ")" << std::endl;
            }
            return hv_fact1; 
#endif 
            return pow(eps, 2*k) * this->hv_p_k(r2, k) * this->eval(x); 
        }

        // recursive evaluation of generalized Laguerre polynomials for hyperviscosity. 
        // Should reproduce the regular Laguerre polynomials when dim==2
        virtual double hv_p_k(double r2, int k) {
            double eps2r2 = eps2*r2; 
            switch (k) {
                case 0:
                    return 1.;
                    break;
                case 1:
                    return 4.*(eps2r2) - 2.*this->dim;
                    break;
                default: 
                    // Pass: 
                    break;
            }    
            // IS this d/2 supposed to be INTEGER or FLOAT division? 
            return 4.*(eps2r2 - 2.*(k-1) - this->dim/2.) * this->hv_p_k(r2, k-1) - 8.*(k-1)*(2.*(k-1) - 2. + this->dim) * this->hv_p_k(r2,k-2); 
        }
};

#endif
