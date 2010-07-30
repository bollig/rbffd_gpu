#ifndef _EXACT_NCAR_POISSON2_H_
#define _EXACT_NCAR_POISSON2_H_

#include "exact_solution.h"

class ExactNCARPoisson2 : public ExactSolution
{
private:

public:
    ExactNCARPoisson2();
    ~ExactNCARPoisson2();

    double operator()(double Xx, double Yy, double Zz, double t);
    double laplacian(double Xx, double Yy, double Zz, double t);

    double xderiv(double Xx, double Yy, double Zz, double t);
    double yderiv(double Xx, double Yy, double Zz, double t);
    double zderiv(double Xx, double Yy, double Zz, double t);

    double tderiv(double Xx, double Yy, double Zz, double t);
private:
    // Scale the problem (scalar for A and LA)
    double SCALE;
};
//----------------------------------------------------------------------

#endif
