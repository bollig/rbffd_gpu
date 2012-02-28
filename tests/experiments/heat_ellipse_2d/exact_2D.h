#ifndef _EXACT_2D_H_
#define _EXACT_2D_H_

#include "exact_solutions/exact_solution.h"


class Exact2D : public ExactSolution
{
    private:
        // See Haberman p47,48 for details
        double B;

        double a; 
        double b;

        double diffuseConst; 
    public:
        Exact2D(double axis1, double axis2, double alpha)
            // 2D
            : ExactSolution(2), 
            B(3.),
            a(axis1), b(axis2),
            diffuseConst(alpha)
    {;}
        ~Exact2D();

        virtual double operator()(double x, double y, double z, double t) {
            // See Haberman p48 for details
            // B is the coefficient we choose for the problem. It could even be another function
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double r2 = ((x*x)/(a*a)) + ((y*y)/(b*b)); 
            double spatial = -B * (r2 - 1);
            //double lambda = M_PI*M_PI / (a*a + b*b); 
            double temporal = exp(-alpha * t);// * lambda);
            double val = spatial * temporal;
            return val;
        }

        virtual double laplacian(double x, double y, double z, double t) {
            return 0.;
        }

        virtual double xderiv(double x, double y, double z, double t) {
            return 0.;
        }
        virtual double yderiv(double x, double y, double z, double t) {
            return 0.;
        }
        virtual double zderiv(double x, double y, double z, double t) {
            return 0.;
        }

        virtual double tderiv(double x, double y, double z, double t) {
            return 0.;
        }

        // Return the diffusivity at node v (K=1 by default)
        virtual double diffuseCoefficient(double x, double y, double z, double sol, double t) {
            return diffuseConst;            
        }
        virtual double diffuse_xderiv(double x, double y, double z, double sol, double t) {
            return 0.;
        }
        virtual double diffuse_yderiv(double x, double y, double z, double sol, double t) {
            return 0.;
        }
        virtual double diffuse_zderiv(double x, double y, double z, double sol, double t) {
            return 0.;
        }
        virtual double diffuse_tderiv(double x, double y, double z, double sol, double t) {
            return 0.;
        }
};
//----------------------------------------------------------------------

#endif 



