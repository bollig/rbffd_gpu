#ifndef _EXACT_NCAR_POISSON2_H_
#define _EXACT_NCAR_POISSON2_H_
#include <stdlib.h>
#include "exact_solution.h"

class ExactNCARPoisson2 : public ExactSolution
{
private:

public:
    ExactNCARPoisson2();
    ~ExactNCARPoisson2();

    virtual double operator()(double Xx, double Yy, double Zz, double t);
    virtual double laplacian(double Xx, double Yy, double Zz, double t);

    virtual double xderiv(double Xx, double Yy, double Zz, double t);
    virtual double yderiv(double Xx, double Yy, double Zz, double t);
    virtual double zderiv(double Xx, double Yy, double Zz, double t);

    virtual double tderiv(double Xx, double Yy, double Zz, double t);

    virtual double diffuseCoefficient(Vec3& v, double t);
    virtual Vec3* diffuseGradient(Vec3& v, double t);

private:
    // Scale the problem (scalar for A and LA)
    double SCALE;
};
//----------------------------------------------------------------------

#endif
