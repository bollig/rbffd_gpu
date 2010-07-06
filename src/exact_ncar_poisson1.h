#ifndef _EXACT_NCAR_POISSON1_H_
#define _EXACT_NCAR_POISSON1_H_

#include "exact_solution.h"

class ExactNCARPoisson1 : public ExactSolution
{
private:

public:
    ExactNCARPoisson1();
    ~ExactNCARPoisson1();

    double operator()(double Xx, double Yy, double Zz, double t);
    double laplacian(double Xx, double Yy, double Zz, double t);

    double xderiv(double Xx, double Yy, double Zz, double t);
    double yderiv(double Xx, double Yy, double Zz, double t);
    double zderiv(double Xx, double Yy, double Zz, double t);

    double tderiv(double Xx, double Yy, double Zz, double t);

};
//----------------------------------------------------------------------

#endif
