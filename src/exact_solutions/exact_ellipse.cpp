#include <stdlib.h>
#include <stdio.h>
#include "exact_ellipse.h"

//----------------------------------------------------------------------

ExactEllipse::ExactEllipse(double freq, double decay, double axis1, double axis2)
: ExactEllipsoid(freq, decay, axis1, axis2, 0.) {
    // Make sure these do not influence our solution here (they are used in the 3D Ellipsoid case)
    princ_axis3_inv2 = 0.;
    princ_axis3_inv4 = 0.;
}
//----------------------------------------------------------------------

ExactEllipse::~ExactEllipse() {
}

//----------------------------------------------------------------------

double ExactEllipse::operator()(double x, double y, double z, double t) {
    double x_contrib = x * x * princ_axis1_inv2;
    double y_contrib = y * y * princ_axis2_inv2;

    double r = sqrt(x_contrib + y_contrib);

    // if temporal decay is too large, time step will have to decrease
    
    
    // for 2D grid
    //  lambda = [(n*pi*x)/L]^2 + [(m*pi*y)/H]^2
    //  exp(t * decay * lambda) makes sure our decay 0's out on the square's half width.
    //
    //  if we have an ellipse with axes a,b: 
    //        lambda = [(n*pi*(x/a))/L]^2 + [(m*pi*(y/b))/H]^2
    //
    //
    double T = cos(freq * r) * exp(-decay * t);
    return T;
}
//----------------------------------------------------------------------

double ExactEllipse::laplacian(double x, double y, double z, double t) {
#if 0
    // This is based on mathematica simplified laplacian of exact solution
    // F = lapl(f)
    double x2 = x*x;
    double y2 = y*y;
    double r = sqrt(x2 + y2);

    // This is based on mathematica simplified laplacian of exact solution
    // F = lapl(f)
    double axis1_2 = princ_axis1*princ_axis1;
    double axis2_2 = princ_axis2*princ_axis2;
    double axis1_4 = axis1_2*axis1_2;
    double axis2_4 = axis2_2*axis2_2;

    double myf1 = freq * r * (y2 * axis1_4 + x2 * axis2_4) * cos(freq * r);
    double myf2 = (x2 + y2) * axis1_2 * axis2_2 * sin(freq * r);
    double mydenom = axis1_2 * r * axis2_2 * (y2 * axis1_2 + x2 * axis2_2);
    double mylapl = -exp(-decay * t) * freq * (myf1 + myf2) / (mydenom);

    printf("mydenom: %f, myf1: %f, myf2: %f, mylapl: %f, decay: %f, freq: %f, remain: %f, exp: %f\n", mydenom, myf1, myf2, mylapl, decay, freq, freq * (myf1 + myf2) / mydenom, -exp(-decay * t));


    // Simplified laplacian from Mathematica 
    double simpLapl = -((freq*(freq*(axis2_4*x2 + axis1_4*y2)*sqrt(x2/axis1_2 + y2/axis2_2)*cos(freq*sqrt(x2/axis1_2 + y2/axis2_2)) +
         axis1_2*axis2_2*x2 + y2)* sin(freq*sqrt(x2/axis1_2 + y2/axis2_2))))
        /(axis1_2*axis2_2*exp(decay*t)*(axis2_2*x2 + axis1_2*y2)*sqrt(x2/axis1_2 + y2/axis2_2));

    printf("Diff (simpLapl - mylapl) = %f\n", simpLapl - mylapl);
#if 1
    // Below is directly from Mathematica's "CForm[...]" except Power(E,Q*t) has been changed to exp(Q*t)
double Xx = x;
double Yy = y;
double Zz = z;
double A = princ_axis1;
double B = princ_axis2;
double P = freq;
double Q = decay;

    double LA = -((P*(P*(Power(B,4)*Power(Xx,2) + Power(A,4)*Power(Yy,2))*
                     Sqrt(Power(Xx,2)/Power(A,2) + Power(Yy,2)/Power(B,2))*
                     Cos(P*Sqrt(Power(Xx,2)/Power(A,2) + Power(Yy,2)/Power(B,2))) +
                    Power(A,2)*Power(B,2)*(Power(Xx,2) + Power(Yy,2))*
                     Sin(P*Sqrt(Power(Xx,2)/Power(A,2) + Power(Yy,2)/Power(B,2)))))/
                (Power(A,2)*Power(B,2)*exp(Q*t)*
                  (Power(B,2)*Power(Xx,2) + Power(A,2)*Power(Yy,2))*
                  Sqrt(Power(Xx,2)/Power(A,2) + Power(Yy,2)/Power(B,2))));

#endif
    //return mylapl;
    //return simpLapl;
    return LA;
#else
#if 0
    double r2 = x * x * princ_axis1_inv2 + y * y*princ_axis2_inv2;
    double r4 = x * x * princ_axis1_inv4 + y * y*princ_axis2_inv4;
    double r = sqrt(r2);
    double f1;
    double f2;

    // if temporal decay is too large, time step will have to decrease

    double nn = freq*r;
    double freq2 = freq*freq;

    f1 = cos(nn) * ((freq2 / r2) * r4 - decay);
    f2 = freq2 * (princ_axis1_inv2 + princ_axis2_inv2 - r4 / r2);

    if (nn < 1.e-5) {
        f2 *= (1. - nn * nn / 6.);
    } else {
        f2 *= sin(nn) / nn;
    }

    f1 = (f1 + f2) * exp(-decay * t);
    //printf("t= %f, alpha= %f\n", t, alpha);
    //printf("exp= %f, nn= %f, f1= %f, f2= %f\n", exp(-alpha*t), nn, f1, f2);

    return f1;
#else
    // Below is directly from Mathematica's "CForm[...]" except Power(E,Q*t) has been changed to exp(Q*t)
    double Xx = x;
    double Yy = y;
    double Zz = z;
    double A = princ_axis1;
    double B = princ_axis2;
    double P = freq;
    double Q = decay;
    double LA = -((P*(P*(Power(B,4)*Power(Xx,2) + Power(A,4)*Power(Yy,2))*
                     Sqrt(Power(Xx,2)/Power(A,2) + Power(Yy,2)/Power(B,2))*
                     Cos(P*Sqrt(Power(Xx,2)/Power(A,2) + Power(Yy,2)/Power(B,2))) +
                    Power(A,2)*Power(B,2)*(Power(Xx,2) + Power(Yy,2))*
                     Sin(P*Sqrt(Power(Xx,2)/Power(A,2) + Power(Yy,2)/Power(B,2)))))/
                (Power(A,2)*Power(B,2)*exp(Q*t)*
                  (Power(B,2)*Power(Xx,2) + Power(A,2)*Power(Yy,2))*
                  Sqrt(Power(Xx,2)/Power(A,2) + Power(Yy,2)/Power(B,2))));
    return LA;
#endif
#endif
}
//----------------------------------------------------------------------

double ExactEllipse::tderiv(double x, double y, double z, double t) {
    double x_contrib2 = x * x * princ_axis1_inv2;
    double y_contrib2 = y * y * princ_axis2_inv2;

    double r2 = x_contrib2 + y_contrib2;

    double r = sqrt(r2);

    // From mathematica time derivative of exact solution
    return -exp(-t * decay) * decay * cos(freq * r);
}
