#include <stdio.h>

#include "exact_ncar_poisson1.h"

//----------------------------------------------------------------------
ExactNCARPoisson1::ExactNCARPoisson1() : ExactSolution()
{

}
//----------------------------------------------------------------------
ExactNCARPoisson1::~ExactNCARPoisson1()
{}

//----------------------------------------------------------------------
double ExactNCARPoisson1::operator()(double Xx, double Yy, double Zz, double t)
{
    double A = (3.*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*(-0.5 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
               (35.*Power(Zz,4) - 30.*Power(Zz,2)*(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 3.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2))*
               Sin((-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*(-0.5 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))))/
             (16.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2));

            return A;
}
//----------------------------------------------------------------------
double ExactNCARPoisson1::laplacian(double Xx, double Yy, double Zz, double t)
{
#if 0
    double simpLapl = (-3.*(3.*Power(Xx,4) + 3.*Power(Yy,4) - 24.*Power(Yy,2)*Power(Zz,2) + 8.*Power(Zz,4) + 6.*Power(Xx,2)*(Power(Yy,2) - 4.*Power(Zz,2)))*
         (-4.*(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))*(-3. - 48.*Power(Zz,2) + 24.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) +
              28.*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 4.*Power(Xx,2)*(-12. + 7.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
              4.*Power(Yy,2)*(-12. + 7.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))*
            Cos(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.) +
           (-216.*Power(Zz,2) - 51.*Power(Zz,4) - 96.*Power(Zz,6) + 80.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) +
              121.*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 106.*Power(Zz,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) +
              32.*Power(Zz,6)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 32.*Power(Xx,6)*(-3. + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
              32.*Power(Yy,6)*(-3. + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
              Power(Yy,4)*(-51. + 106.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96.*Power(Zz,2)*(-3. + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))) +
              Power(Xx,4)*(-51. + 106.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96.*Power(Yy,2)*(-3. + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
                 96.*Power(Zz,2)*(-3, + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))) +
              Power(Yy,2)*(-216, + 121,*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96.*Power(Zz,4)*(-3. + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
                 2*Power(Zz,2)*(-51. + 106.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))) +
              Power(Xx,2)*(-216. + 121.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96.*Power(Yy,4)*(-3. + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
                 96.*Power(Zz,4)*(-3. + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 2.*Power(Zz,2)*(-51. + 106.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
                 2.*Power(Yy,2)*(-51. + 106.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96.*Power(Zz,2)*(-3. + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))))*
            Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.)))/
       (128.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3.5));
#endif
    double LA = (-15*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
         (-4*(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))*(-3 - 48*Power(Zz,2) + 24*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) +
              28*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 4*Power(Xx,2)*(-12 + 7*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
              4*Power(Yy,2)*(-12 + 7*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))*
            Cos(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.) +
           (-216*Power(Zz,2) - 51*Power(Zz,4) - 96*Power(Zz,6) + 80*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) +
              121*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 106*Power(Zz,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) +
              32*Power(Zz,6)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 32*Power(Xx,6)*(-3 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
              32*Power(Yy,6)*(-3 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
              Power(Yy,4)*(-51 + 106*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96*Power(Zz,2)*(-3 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))) +
              Power(Xx,4)*(-51 + 106*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96*Power(Yy,2)*(-3 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
                 96*Power(Zz,2)*(-3 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))) +
              Power(Yy,2)*(-216 + 121*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96*Power(Zz,4)*(-3 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
                 2*Power(Zz,2)*(-51 + 106*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))) +
              Power(Xx,2)*(-216 + 121*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96*Power(Yy,4)*(-3 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
                 96*Power(Zz,4)*(-3 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 2*Power(Zz,2)*(-51 + 106*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) +
                 2*Power(Yy,2)*(-51 + 106*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 96*Power(Zz,2)*(-3 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))))*
            Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.)))/
       (128.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3.5));
    return LA;
}
//----------------------------------------------------------------------
double ExactNCARPoisson1::xderiv(double Xx, double Yy, double Zz, double t)
{
    double dardx = (3*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (2*Xx - (3*Xx)/(2.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Cos(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))
                   /(32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5)) +
                 (3*Xx*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))/
                  (16.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3)) +
                 (3*Xx*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))
                   /(32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3)) +
                 (3*(12*Power(Xx,3) + 12*Xx*(Power(Yy,2) - 4*Power(Zz,2)))*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))
                   /(32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5)) -
                 (15*Xx*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*(-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))/
                  (32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3.5));
    return dardx;
}

//----------------------------------------------------------------------
double ExactNCARPoisson1::yderiv(double Xx, double Yy, double Zz, double t)
{
    double dardy = (3*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (2*Yy - (3*Yy)/(2.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Cos(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))
                   /(32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5)) +
                 (3*Yy*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))/
                  (16.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3)) +
                 (3*Yy*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))
                   /(32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3)) +
                 (3*(12*Power(Xx,2)*Yy + 12*Power(Yy,3) - 48*Yy*Power(Zz,2))*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))
                   /(32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5)) -
                 (15*Yy*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*(-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))/
                  (32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3.5));
    return dardy;
}

//----------------------------------------------------------------------
double ExactNCARPoisson1::zderiv(double Xx, double Yy, double Zz, double t)
{
    // WARNING! COULD BE WRONG:  d/dz * ( A/r )
    double dardz = (3*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (2*Zz - (3*Zz)/(2.*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Cos(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))
                   /(32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5)) +
                 (3*Zz*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))/
                  (16.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3)) +
                 (3*Zz*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))
                   /(32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3)) +
                 (3*(-48*Power(Xx,2)*Zz - 48*Power(Yy,2)*Zz + 32*Power(Zz,3))*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    (-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))
                   /(32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5)) -
                 (15*Zz*(3*Power(Xx,4) + 3*Power(Yy,4) - 24*Power(Yy,2)*Power(Zz,2) + 8*Power(Zz,4) + 6*Power(Xx,2)*(Power(Yy,2) - 4*Power(Zz,2)))*
                    (-1 + Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*(-1 + 2*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                    Sin(0.5 + Power(Xx,2) + Power(Yy,2) + Power(Zz,2) - (3*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))/2.))/
                  (32.*Sqrt(Pi)*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3.5));
    return dardz;
}
//----------------------------------------------------------------------
double ExactNCARPoisson1::tderiv(double x, double y, double z, double t)
{
    // From mathematica time derivative of exact solution
    return 0.;
}
