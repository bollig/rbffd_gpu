#ifndef _EXACT_1D_H_
#define _EXACT_1D_H_

#include "exact_solutions/exact_solution.h"

// These expressions were obtained from Mathematica. 
// The original idea for this solution came from
// http://web.cecs.pdx.edu/~gerry/class/ME448/codes/FDheat.pdf
class Exact1D : public ExactSolution
{
    private:
            // See Haberman p47,48 for details
        double n;
        double B;
        double decay; 
        double L;
    public:
        Exact1D(double maxX, double alpha)
            : ExactSolution(1), 
            n(8),B(1),
            L(maxX), decay(alpha)
    {;}
        ~Exact1D();

        virtual double operator()(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            // See Haberman p48 for details
            // B is the coefficient we choose for the problem. It could even be another function
            double val = B * sin((n * M_PI * x)/L) * exp(-alpha * t * (n * n * M_PI * M_PI)/(L*L)); 
            return val;
        }

        virtual double laplacian(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double val = -B*((n*n * M_PI * M_PI * sin((n * M_PI * x)/L)) * exp(-alpha * t * (n * n * M_PI * M_PI)/(L*L))) / (L*L);
            return val;
        }

        virtual double xderiv(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double val =  (B * n * M_PI * cos((n * M_PI * x)/L)/L) * exp(-alpha * t * ((n * n * M_PI * M_PI)/(L*L))); 
            return val;
        }
        virtual double yderiv(double x, double y, double z, double t) {
            return 0.;
        }
        virtual double zderiv(double x, double y, double z, double t) {
            return 0.;
        }

        virtual double tderiv(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t);             
            double val = (B * n*n * M_PI*M_PI * sin((n * M_PI * x)/L)/(L*L)) * exp(-alpha * t * ((n * n * M_PI * M_PI)/(L*L))); 

            return val; 
        }

        // Return the diffusivity at node v (K=1 by default)
        virtual double diffuseCoefficient(double x, double y, double z, double sol, double t) {
            return decay;            
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


