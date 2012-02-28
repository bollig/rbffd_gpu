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
            double Power(double x, double y) { return    (pow((double)(x), (double)(y))) ; }
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
            return (Sqrt(105/Pi)*(Xx - Yy)*(Xx + Yy)*Zz)/(4.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),1.5));
        }
        virtual double d_dx(double Xx, double Yy, double Zz) {
            return -(Sqrt(105/Pi)*Xx*Zz*(Power(Xx,2) - 5*Power(Yy,2) - 2*Power(Zz,2)))/(4.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5));
        }
        virtual double d_dy(double Xx, double Yy, double Zz) {
            return (Sqrt(105/Pi)*Yy*Zz*(-5*Power(Xx,2) + Power(Yy,2) - 2*Power(Zz,2)))/(4.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5));
        }
        virtual double d_dz(double Xx, double Yy, double Zz) {
            return (Sqrt(105/Pi)*(Xx - Yy)*(Xx + Yy)*(Power(Xx,2) + Power(Yy,2) - 2*Power(Zz,2)))/(4.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5));
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return (-3*Sqrt(105/Pi)*(Xx - Yy)*(Xx + Yy)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5);
        }
    };

    // Y_10^5
    class Sph105 : public SphericalHarmonicBase
    {
        virtual double eval(double Xx, double Yy, double Zz) {
            return (-3*Sqrt(1001/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),2.5)*Zz*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Cos(5*ArcTan(Xx,Yy)))/
                (128.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5)); 
        }
        virtual double d_dx(double Xx, double Yy, double Zz) {
            return  (15*Sqrt(1001/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),1.5)*Zz*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (Xx*(3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                         364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) - 
                     Yy*(15*Power(Power(Xx,2) + Power(Yy,2),3) - 125*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                         28*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) + 168*Power(Zz,6))*Sin(5*ArcTan(Xx,Yy))))/
                (128.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5)); 
        }
        virtual double d_dy(double Xx, double Yy, double Zz) {
            return (15*Sqrt(1001/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),1.5)*Zz*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (Yy*(3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                         364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) + 
                     Xx*(15*Power(Power(Xx,2) + Power(Yy,2),3) - 125*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                         28*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) + 168*Power(Zz,6))*Sin(5*ArcTan(Xx,Yy))))/
                (128.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5));
        }
        virtual double d_dz(double Xx, double Yy, double Zz) {
            return (-15*Sqrt(1001/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),2.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                     364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)))/
                (128.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5)); 
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return (165*Sqrt(1001/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),2.5)*Zz*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Cos(5*ArcTan(Xx,Yy)))/
                (64.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5));
        }
    }; 


    // Y_3^2 + Y_10^5
    class Sph32105 : public SphericalHarmonicBase
    {
        virtual double eval(double Xx, double Yy, double Zz) {
            return (Sqrt(105/Pi)*(Xx - Yy)*(Xx + Yy)*Zz)/(4.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),1.5)) - 
                (3*Sqrt(1001/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),2.5)*Zz*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                 (15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Cos(5*ArcTan(Xx,Yy)))/
                (128.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5));
        }
        virtual double d_dx(double Xx, double Yy, double Zz) {
            return (Sqrt(7/Pi)*Zz*(-64*Sqrt(15)*Xx*(Power(Xx,2) - 5*Power(Yy,2) - 2*Power(Zz,2))*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) + 
                        15*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),1.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                        (Xx*(3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                             364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) - 
                         Yy*(15*Power(Power(Xx,2) + Power(Yy,2),3) - 125*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                             28*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) + 168*Power(Zz,6))*Sin(5*ArcTan(Xx,Yy)))))/
                (256.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5)); 
        }
        virtual double d_dy(double Xx, double Yy, double Zz) {
            return (Sqrt(7/Pi)*Zz*(64*Sqrt(15)*Yy*(-5*Power(Xx,2) + Power(Yy,2) - 2*Power(Zz,2))*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) + 
                        15*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),1.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                        (Yy*(3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                             364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) + 
                         Xx*(15*Power(Power(Xx,2) + Power(Yy,2),3) - 125*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                             28*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) + 168*Power(Zz,6))*Sin(5*ArcTan(Xx,Yy)))))/
                (256.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5));
        }
        virtual double d_dz(double Xx, double Yy, double Zz) {
            return (Sqrt(7/Pi)*(64*Sqrt(15)*(Xx - Yy)*(Xx + Yy)*(Power(Xx,2) + Power(Yy,2) - 2*Power(Zz,2))*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) - 
                        15*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),2.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                        (3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 
                         364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy))))/
                (256.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5));
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return (-3*Sqrt(7/Pi)*Zz*(128*Sqrt(15)*(Xx - Yy)*(Xx + Yy)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) - 
                        55*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),2.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                        (15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Cos(5*ArcTan(Xx,Yy))))/
                (128.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5));
        }
    };

    // Y_20^20
    class Sph2020 : public SphericalHarmonicBase
    {
        virtual double eval(double Xx, double Yy, double Zz) {
            return (3*Sqrt(156991880045/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),10)*Cos(20*ArcTan(Xx,Yy)))/
                   (524288.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),10));
        }
        virtual double d_dx(double Xx, double Yy, double Zz) {
            return (15*Sqrt(156991880045/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),9)*
                         (Xx*Power(Zz,2)*Cos(20*ArcTan(Xx,Yy)) + Yy*(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))*Sin(20*ArcTan(Xx,Yy))))/
                   (131072.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11));
        }
        virtual double d_dy(double Xx, double Yy, double Zz) {
            return (-15*Sqrt(156991880045/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),9)*
                         (-(Yy*Power(Zz,2)*Cos(20*ArcTan(Xx,Yy))) + Xx*(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))*Sin(20*ArcTan(Xx,Yy))))/
                   (131072.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11));
        }
        virtual double d_dz(double Xx, double Yy, double Zz) {
            return (-15*Sqrt(156991880045/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),10)*Zz*Cos(20*ArcTan(Xx,Yy)))/
                   (131072.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11));
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return (-315*Sqrt(156991880045/(2.*Pi))*Power(Power(Xx,2) + Power(Yy,2),10)*Cos(20*ArcTan(Xx,Yy)))/
                   (131072.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11));
        }
    };

};

#endif 