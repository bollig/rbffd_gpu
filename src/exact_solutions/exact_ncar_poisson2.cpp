#include "exact_ncar_poisson2.h"

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
using namespace std;

//----------------------------------------------------------------------
ExactNCARPoisson2::ExactNCARPoisson2() : ExactSolution(), SCALE(100000.)
{

}
//----------------------------------------------------------------------
ExactNCARPoisson2::~ExactNCARPoisson2()
{}

//----------------------------------------------------------------------
double ExactNCARPoisson2::operator()(double Xx, double Yy, double Zz, double t)
{
    // Scale by 10000 to make allow easier check of relative error. (Absolute = relative)
    double A = (SCALE)*(     sin((sqrt(Xx*Xx+Yy*Yy)-1.)*(sqrt(Xx*Xx+Yy*Yy)-0.5)+Pi)
                              +   (sqrt(Xx*Xx+Yy*Yy)-1.)*(sqrt(Xx*Xx+Yy*Yy)-0.5)     );
    //cout << "EXACT: " << A << endl;

    return A;
}
//----------------------------------------------------------------------
double ExactNCARPoisson2::laplacian(double Xx, double Yy, double Zz, double t)
{
    // Scalar for equation to adjust relative error
#if 0
    // Direct from CForm[LA] in mathematica (Unedited)
    double LA = (SCALE)*((-1.5*Power(Xx,4) - 3.*Power(Xx,2)*Power(Yy,2) - 1.5*Power(Yy,4) +
                 4.*Power(Xx,4)*Sqrt(Power(Xx,2) + Power(Yy,2)) +
                 8.*Power(Xx,2)*Power(Yy,2)*Sqrt(Power(Xx,2) + Power(Yy,2)) +
                 4.*Power(Yy,4)*Sqrt(Power(Xx,2) + Power(Yy,2)) +
                 (Power(Xx,2)*Power(Yy,2)*(3. - 8.*Sqrt(Power(Xx,2) + Power(Yy,2))) +
                  Power(Xx,4)*(1.5 - 4.*Sqrt(Power(Xx,2) + Power(Yy,2))) +
                  Power(Yy,4)*(1.5 - 4.*Sqrt(Power(Xx,2) + Power(Yy,2))))*
                 Cos(0.5 + Power(Xx,2) + Power(Yy,2) - 1.5*Sqrt(Power(Xx,2) + Power(Yy,2))) +
                 (2.25*Power(Yy,4)*Sqrt(Power(Xx,2) + Power(Yy,2)) +
                  Power(Xx,6)*(-6. + 4.*Sqrt(Power(Xx,2) + Power(Yy,2))) +
                  Power(Yy,6)*(-6. + 4.*Sqrt(Power(Xx,2) + Power(Yy,2))) +
                  Power(Xx,4)*(2.25*Sqrt(Power(Xx,2) + Power(Yy,2)) +
                               Power(Yy,2)*(-18. + 12.*Sqrt(Power(Xx,2) + Power(Yy,2)))) +
                  Power(Xx,2)*(4.5*Power(Yy,2)*Sqrt(Power(Xx,2) + Power(Yy,2)) +
                               Power(Yy,4)*(-18. + 12.*Sqrt(Power(Xx,2) + Power(Yy,2)))))*
                 Sin(0.5 + Power(Xx,2) + Power(Yy,2) - 1.5*Sqrt(Power(Xx,2) + Power(Yy,2))))/
                Power(Power(Xx,2) + Power(Yy,2),2.5));
#else
    // Edited:
    double x2 = Xx*Xx;
    double x4 = x2*x2;
    double x6 = x2*x4;
    double y2 = Yy*Yy;
    double y4 = y2*y2;
    double y6 = y2*y4;

    double r = sqrt(x2 + y2);
    double r5 = r*r*r*r*r;      // (x^2 + y^2)^(5/2)

    double LA = (SCALE) * ((-1.5*x4 - 3.*x2*y2 - 1.5*y4 + 4.*x4*r + 8.*x2*y2*r + 4.*y4*r
                        + (x2*y2*(3. - 8.*r) + x4*(1.5 - 4.*r) + y4*(1.5 - 4.*r))
                 * cos(0.5 + x2 + y2 - 1.5*r)
                 + (2.25*y4*r + x6*(-6. + 4.*r) + y6*(-6. + 4.*r)
                        + x4*(2.25*r + y2*(-18. + 12.*r)) + x2*(4.5*y2*r + y4*(-18. + 12.*r)))
                 * sin(0.5 + x2 + y2 - 1.5*r)) / r5);
#endif
    //cout << "LA: " << LA << endl;
    return LA;
}
//----------------------------------------------------------------------
double ExactNCARPoisson2::xderiv(double Xx, double Yy, double Zz, double t)
{
    double h = 0.001;

    double xi = (*this)(Xx, Yy, Zz, t);
    double xip1 = (*this)(Xx+h, Yy, Zz, t);
    double xip2 = (*this)(Xx+2.*h, Yy, Zz, t);
    double xim1 = (*this)(Xx-h, Yy, Zz, t);
    double xim2 = (*this)(Xx-2.*h, Yy, Zz, t);

    // First derivative weights from: http://en.wikipedia.org/wiki/Five-point_stencil
    double firstDeriv = ( -xip2 + 8.*xip1 - 8.*xim1 + xim2 ) / (12.*h);

    double firstDerivExact =  (SCALE) * ((Xx*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2))))/Sqrt(Power(Xx,2) + Power(Yy,2)) +
                             (Xx*(-0.5 + Sqrt(Power(Xx,2) + Power(Yy,2))))/Sqrt(Power(Xx,2) + Power(Yy,2)) -
                             ((Xx*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2))))/Sqrt(Power(Xx,2) + Power(Yy,2)) +
                                (Xx*(-0.5 + Sqrt(Power(Xx,2) + Power(Yy,2))))/Sqrt(Power(Xx,2) + Power(Yy,2)))*
                              Cos((-1 + Sqrt(Power(Xx,2) + Power(Yy,2)))*(-0.5 + Sqrt(Power(Xx,2) + Power(Yy,2)))));

    double relerror = abs(firstDerivExact - firstDeriv) / abs(firstDerivExact);

   //cerr << "xderiv Approx: " << firstDeriv << "   \tExact: " << firstDerivExact << "   \tRelative Error (|exact - approx|/|exact|): " << relerror <<endl;
    //exit(EXIT_FAILURE);
    double dardx = firstDerivExact;
    return dardx;
}

//----------------------------------------------------------------------
double ExactNCARPoisson2::yderiv(double Xx, double Yy, double Zz, double t)
{
    double h = 1.e-4;

    double xi = (*this)(Xx, Yy, Zz, t);
    double xip1 = (*this)(Xx, Yy+h, Zz, t);
    double xip2 = (*this)(Xx, Yy+2.*h, Zz, t);
    double xim1 = (*this)(Xx, Yy-h, Zz, t);
    double xim2 = (*this)(Xx, Yy-2.*h, Zz, t);

    // First derivative weights from: http://en.wikipedia.org/wiki/Five-point_stencil
    double firstDeriv = ( -xip2 + 8.*xip1 - 8.*xim1 + xim2 ) / (12.*h);

    double firstDerivExact = (SCALE) * ( (Yy*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2))))/Sqrt(Power(Xx,2) + Power(Yy,2)) +
                             (Yy*(-0.5 + Sqrt(Power(Xx,2) + Power(Yy,2))))/Sqrt(Power(Xx,2) + Power(Yy,2)) -
                             ((Yy*(-1 + Sqrt(Power(Xx,2) + Power(Yy,2))))/Sqrt(Power(Xx,2) + Power(Yy,2)) +
                                (Yy*(-0.5 + Sqrt(Power(Xx,2) + Power(Yy,2))))/Sqrt(Power(Xx,2) + Power(Yy,2)))*
                              Cos((-1 + Sqrt(Power(Xx,2) + Power(Yy,2)))*(-0.5 + Sqrt(Power(Xx,2) + Power(Yy,2)))));

    double relerror = abs(firstDerivExact - firstDeriv) / abs(firstDerivExact);
    double dardy = firstDerivExact;

    //cerr << "yderiv Approx: " << firstDeriv << "\t Exact: " << firstDerivExact << "\t Relative Error (|exact - approx|/|exact|): " << relerror <<endl;
    return dardy;
}

//----------------------------------------------------------------------
double ExactNCARPoisson2::zderiv(double Xx, double Yy, double Zz, double t)
{
    double dardz = 0.;
#if 0
    cerr << "zderiv NOT IMPLEMENTED" <<endl;
    exit(EXIT_FAILURE);
#endif

    return dardz;
}
//----------------------------------------------------------------------
double ExactNCARPoisson2::tderiv(double x, double y, double z, double t)
{
    // From mathematica time derivative of exact solution
    return 0.;
}
//----------------------------------------------------------------------
Vec3* ExactNCARPoisson2::diffuseGradient(Vec3& v, double t) {
        double diffuseDDX = v.y()*Cos(v.x())*Cos(2.*v.y());
        double diffuseDDY = Cos(2.*v.y())*Sin(v.x()) - 2. * v.y()*Sin(v.x())*Sin(2.*v.y());
        double diffuseDDZ = 0.;
        return new Vec3(diffuseDDX,diffuseDDY,diffuseDDZ);
}
//----------------------------------------------------------------------
double ExactNCARPoisson2::diffuseCoefficient(Vec3& v, double t) {
        double diffuse = v.y() * Sin(v.x()) * Cos(2.*v.y());
        return diffuse;
}
//----------------------------------------------------------------------
