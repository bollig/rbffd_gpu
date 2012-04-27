#ifndef __MANUFACTURED_SOLUTION_H__
#define __MANUFACTURED_SOLUTION_H__

#include "utils/mathematica_base.h"

class ManufacturedSolution : public MathematicaBase
{
    public: 
        virtual double operator()(double Xx, double Yy, double Zz) { return this->eval(Xx, Yy, Zz); }
        virtual double eval(double Xx, double Yy, double Zz) {
            return Sin(Pi*Xx) ;
        }
        virtual double lapl(double Xx, double Yy, double Zz) {
            return -(Power(Pi,2)*Sin(Pi*Xx)); 
        }
};


#endif 
