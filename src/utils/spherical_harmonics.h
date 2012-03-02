#ifndef __SPHERICAL_HARMONIC_DERIVATIVES_H__
#define __SPHERICAL_HARMONIC_DERIVATIVES_H__

#include <stdlib.h>
#include <math.h> 

// TODO: Template meta-programming expressions to complete the cartesian sph
// given any l,m. Is it possible with the derivatives? Must be, but it'll be
// difficult. 



/***** 
 * All of these classes are filled by CForm output from the SphericalHarmonics_Cartesian.nb script. 
 * To create a new spherical harmonic: 
 *  - Create a class inheriting from SphericalHarmonicBase (Template for this follows)
 *  - Inside mathematica, constructed the spherical harmonic
 *              Sph2020 = sphFullCart[20,20]
 *              {pdx2020, pdy2020, pdz2020} = sphGradCart[20,20]
 *              LSph2020 = sphLaplCart[20,20]
 *  - Get the CForm expressions:
 *              CForm[Sph2020]
 *              CForm[pdx2020]
 *              ...
 *  - Copy the output from CForm and paste it directly into the matching routines within your new class
 */ 

#if 0
    // Template for Y_l^m (replace LL and MM with l,m resp.)
    class SphLLMM : public SphericalHarmonicBase
    {
        virtual double eval(double Xx, double Yy, double Zz) {
            return 
        }
        virtual double d_dx(double Xx, double Yy, double Zz) {
            return 
        }
        virtual double d_dy(double Xx, double Yy, double Zz) {
            return 
        }
        virtual double d_dz(double Xx, double Yy, double Zz) {
            return 
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return 
        }
    };
#endif 

namespace SphericalHarmonic
{
    class SphericalHarmonicBase
    {
        public:
            virtual double operator()(double Xx, double Yy, double Zz) { return this->eval(Xx, Yy, Zz); }

            virtual double eval(double Xx, double Yy, double Zz) = 0;
            virtual double d_dx(double Xx, double Yy, double Zz) = 0;
            virtual double d_dy(double Xx, double Yy, double Zz) = 0;
            virtual double d_dz(double Xx, double Yy, double Zz) = 0;
            virtual double lapl(double Xx, double Yy, double Zz) = 0;

            /***********   MATHEMATICA mdefs.h PROVIDED MOST OF THESE ************/
            // Its ok to directly use their definitions
            
            // NOTE: I found that RHEL6 (not sure if its just this version) was
            // MUCH slower when using the double precision version of
            // pow(double,double) vs pow(float,float). For spherical harmonics,
            // we do not see a noticable difference in the harmonic
            // approximations caused by downcasting to float here. However, the
            // impact on performance is 1000x speedup for 10201 nodes. This
            // casting causes 1e-5 error in production of sph32105, no noticeable
            // error in sph32 or sph2020.  Presumably the lack of error is
            // because the pow is integer, not double/float.  
#if 0
            double Power(double x, double y) { return    (pow((double)(x), (double)(y))) ; }
#else 
            double Power(double x, double y) { return    (double)(pow((float)(x), (float)(y))) ; }
#endif 
            double Sqrt(double x)  { return (sqrt((double)(x))) ; }

            double Abs(double x)   { return (fabs((double)(x))) ; }

            double Exp(double x)   { return (exp((double)(x))) ; } 
            double Log(double x)   { return (log((double)(x))) ; }

            double Sin(double x)   { return (sin((double)(x))) ; }
            double Cos(double x)   { return (cos((double)(x))) ; }
            double Tan(double x)   { return (tan((double)(x))) ; }

            double ArcSin(double x){ return (asin((double)(x))) ; }
            double ArcCos(double x){ return (acos((double)(x))) ; }
            double ArcTan(double x){ return (atan((double)(x))) ; }
            double ArcTan(double x,double y){ return (atan2((double)(y), (double)(x))) ; }

            double Sinh(double x)  { return (sinh((double)(x))) ; }
            double Cosh(double x)  { return (cosh((double)(x))) ; }
            double Tanh(double x)  { return (tanh((double)(x))) ; }

            // I Know M_PI is defined in math.h 
            const static double Pi = M_PI;
            /***********   END MATHEMATICA mdefs.h PROVIDED ************/
    };

    // Y_3^2
    class Sph32 : public SphericalHarmonicBase
    {
        virtual double eval(double Xx, double Yy, double Zz) {
            return (Sqrt(105/Pi)*(Xx - Yy)*(Xx + Yy)*Zz)/(4.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),1.5));
        }
        virtual double d_dx(double Xx, double Yy, double Zz) {
            return -(Sqrt(105/Pi)*Xx*Zz*((Xx*Xx) - 5*(Yy*Yy) - 2*(Zz*Zz)))/(4.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),2.5));
        }
        virtual double d_dy(double Xx, double Yy, double Zz) {
            return (Sqrt(105/Pi)*Yy*Zz*(-5*(Xx*Xx) + (Yy*Yy) - 2*(Zz*Zz)))/(4.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),2.5));
        }
        virtual double d_dz(double Xx, double Yy, double Zz) {
            return (Sqrt(105/Pi)*(Xx - Yy)*(Xx + Yy)*((Xx*Xx) + (Yy*Yy) - 2*(Zz*Zz)))/(4.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),2.5));
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return (-3*Sqrt(105/Pi)*(Xx - Yy)*(Xx + Yy)*Zz)/Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),2.5);
        }
    };

    // Y_10^5
    class Sph105 : public SphericalHarmonicBase
    {
        virtual double eval(double Xx, double Yy, double Zz) {
            return (-3*Sqrt(1001/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),2.5)*Zz*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                    (15*Power((Xx*Xx) + (Yy*Yy),2) - 140*((Xx*Xx) + (Yy*Yy))*(Zz*Zz) + 168*Power(Zz,4))*Cos(5*ArcTan(Xx,Yy)))/
                (128.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),4.5)); 
        }
        virtual double d_dx(double Xx, double Yy, double Zz) {
            return  (15*Sqrt(1001/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),1.5)*Zz*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                    (Xx*(3*Power((Xx*Xx) + (Yy*Yy),3) - 111*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                         364*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) - 
                     Yy*(15*Power((Xx*Xx) + (Yy*Yy),3) - 125*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                         28*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) + 168*Power(Zz,6))*Sin(5*ArcTan(Xx,Yy))))/
                (128.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),5.5)); 
        }
        virtual double d_dy(double Xx, double Yy, double Zz) {
            return (15*Sqrt(1001/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),1.5)*Zz*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                    (Yy*(3*Power((Xx*Xx) + (Yy*Yy),3) - 111*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                         364*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) + 
                     Xx*(15*Power((Xx*Xx) + (Yy*Yy),3) - 125*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                         28*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) + 168*Power(Zz,6))*Sin(5*ArcTan(Xx,Yy))))/
                (128.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),5.5));
        }
        virtual double d_dz(double Xx, double Yy, double Zz) {
            return (-15*Sqrt(1001/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),2.5)*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                    (3*Power((Xx*Xx) + (Yy*Yy),3) - 111*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                     364*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)))/
                (128.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),5.5)); 
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return (165*Sqrt(1001/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),2.5)*Zz*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                    (15*Power((Xx*Xx) + (Yy*Yy),2) - 140*((Xx*Xx) + (Yy*Yy))*(Zz*Zz) + 168*Power(Zz,4))*Cos(5*ArcTan(Xx,Yy)))/
                (64.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),5.5));
        }
    }; 


    // Y_3^2 + Y_10^5
    class Sph32105 : public SphericalHarmonicBase
    {
        virtual double eval(double Xx, double Yy, double Zz) {
            return (Sqrt(105/Pi)*(Xx - Yy)*(Xx + Yy)*Zz)/(4.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),1.5)) - 
                (3*Sqrt(1001/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),2.5)*Zz*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                 (15*Power((Xx*Xx) + (Yy*Yy),2) - 140*((Xx*Xx) + (Yy*Yy))*(Zz*Zz) + 168*Power(Zz,4))*Cos(5*ArcTan(Xx,Yy)))/
                (128.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),4.5));
        }
        virtual double d_dx(double Xx, double Yy, double Zz) {
            return (Sqrt(7/Pi)*Zz*(-64*Sqrt(15)*Xx*((Xx*Xx) - 5*(Yy*Yy) - 2*(Zz*Zz))*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),3) + 
                        15*Sqrt(286)*Power((Xx*Xx) + (Yy*Yy),1.5)*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                        (Xx*(3*Power((Xx*Xx) + (Yy*Yy),3) - 111*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                             364*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) - 
                         Yy*(15*Power((Xx*Xx) + (Yy*Yy),3) - 125*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                             28*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) + 168*Power(Zz,6))*Sin(5*ArcTan(Xx,Yy)))))/
                (256.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),5.5)); 
        }
        virtual double d_dy(double Xx, double Yy, double Zz) {
            return (Sqrt(7/Pi)*Zz*(64*Sqrt(15)*Yy*(-5*(Xx*Xx) + (Yy*Yy) - 2*(Zz*Zz))*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),3) + 
                        15*Sqrt(286)*Power((Xx*Xx) + (Yy*Yy),1.5)*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                        (Yy*(3*Power((Xx*Xx) + (Yy*Yy),3) - 111*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                             364*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) + 
                         Xx*(15*Power((Xx*Xx) + (Yy*Yy),3) - 125*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                             28*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) + 168*Power(Zz,6))*Sin(5*ArcTan(Xx,Yy)))))/
                (256.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),5.5));
        }
        virtual double d_dz(double Xx, double Yy, double Zz) {
            return (Sqrt(7/Pi)*(64*Sqrt(15)*(Xx - Yy)*(Xx + Yy)*((Xx*Xx) + (Yy*Yy) - 2*(Zz*Zz))*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),3) - 
                        15*Sqrt(286)*Power((Xx*Xx) + (Yy*Yy),2.5)*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                        (3*Power((Xx*Xx) + (Yy*Yy),3) - 111*Power((Xx*Xx) + (Yy*Yy),2)*(Zz*Zz) + 
                         364*((Xx*Xx) + (Yy*Yy))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy))))/
                (256.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),5.5));
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return (-3*Sqrt(7/Pi)*Zz*(128*Sqrt(15)*(Xx - Yy)*(Xx + Yy)*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),3) - 
                        55*Sqrt(286)*Power((Xx*Xx) + (Yy*Yy),2.5)*Sqrt(1/((Xx*Xx) + (Yy*Yy) + (Zz*Zz)))*
                        (15*Power((Xx*Xx) + (Yy*Yy),2) - 140*((Xx*Xx) + (Yy*Yy))*(Zz*Zz) + 168*Power(Zz,4))*Cos(5*ArcTan(Xx,Yy))))/
                (128.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),5.5));
        }
    };

    // Y_20^20
    class Sph2020 : public SphericalHarmonicBase
    {
        virtual double eval(double Xx, double Yy, double Zz) {
            return (3*Sqrt(156991880045/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),10)*Cos(20*ArcTan(Xx,Yy)))/
                   (524288.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),10));
        }
        virtual double d_dx(double Xx, double Yy, double Zz) {
            return (15*Sqrt(156991880045/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),9)*
                         (Xx*(Zz*Zz)*Cos(20*ArcTan(Xx,Yy)) + Yy*((Xx*Xx) + (Yy*Yy) + (Zz*Zz))*Sin(20*ArcTan(Xx,Yy))))/
                   (131072.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),11));
        }
        virtual double d_dy(double Xx, double Yy, double Zz) {
            return (-15*Sqrt(156991880045/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),9)*
                         (-(Yy*(Zz*Zz)*Cos(20*ArcTan(Xx,Yy))) + Xx*((Xx*Xx) + (Yy*Yy) + (Zz*Zz))*Sin(20*ArcTan(Xx,Yy))))/
                   (131072.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),11));
        }
        virtual double d_dz(double Xx, double Yy, double Zz) {
            return (-15*Sqrt(156991880045/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),10)*Zz*Cos(20*ArcTan(Xx,Yy)))/
                   (131072.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),11));
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return (-315*Sqrt(156991880045/(2.*Pi))*Power((Xx*Xx) + (Yy*Yy),10)*Cos(20*ArcTan(Xx,Yy)))/
                   (131072.*Power((Xx*Xx) + (Yy*Yy) + (Zz*Zz),11));
        }
    };

};

#endif 
