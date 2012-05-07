#ifndef __MANUFACTURED_SOLUTION_H__
#define __MANUFACTURED_SOLUTION_H__

#include "utils/mathematica_base.h"

class ManufacturedSolution : public MathematicaBase
{
    public: 
        virtual double U(double Xx, double Yy, double Zz) {
            return 11.;
        }

        virtual double V(double Xx, double Yy, double Zz) {
            return 22.;
        }

        virtual double W(double Xx, double Yy, double Zz) {
            return 33.;
        }

        virtual double P(double Xx, double Yy, double Zz) {
            return 44.;
        }

        virtual double RHS_U(double Xx, double Yy, double Zz) {
            return 55.;
        }

        virtual double RHS_V(double Xx, double Yy, double Zz) {
            return 66.;
        }

        virtual double RHS_W(double Xx, double Yy, double Zz) {
            return 77.;
        }
        
        virtual double RHS_P(double Xx, double Yy, double Zz) {
            return 88.; 
        }
};


#endif 
