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
        double xmin;
        double L;
        double decay; 

    public:
        Exact1D(double minX, double maxX, double alpha)
            : ExactSolution(1), 
            n(8),B(1),xmin(minX),
            L(maxX-minX), decay(alpha)
    {;}
        ~Exact1D();

        virtual double operator()(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            // See Haberman p48 for details
            // B is the coefficient we choose for the problem. It could even be another function
            double val = B * sin((n * M_PI * (x-xmin))/L) * exp(-alpha * t * (n * n * M_PI * M_PI)/(L*L)); 
            return val;
        }

        virtual double laplacian(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double mu = this->diffuse_xderiv(x,y,z,0.,t);
            // From mathematica: 
            double val = (B * exp(-alpha * t * (n * n * M_PI * M_PI)/(L*L)) * n*n * M_PI*M_PI * ((-2.* L * n * M_PI * t * mu * cos( (n * M_PI * (x-xmin))/L )) + ((-(L*L) + (n*n)*(M_PI*M_PI) * (t*t) * (mu*mu)) * sin( (n*M_PI*(x-xmin))/L )))) / (L*L*L*L); 

            return val;
        }

        virtual double xderiv(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double deriv =  ( (B * n * M_PI * cos((n * M_PI * (x-xmin))/L)) * exp(-alpha * t * ((n * n * M_PI * M_PI)/(L*L))) )/L; 
            double alpha_deriv = - ( (B * n*n * M_PI*M_PI * t * sin((n * M_PI * (x-xmin))/L)) * exp(-alpha * t * ((n * n * M_PI * M_PI)/(L*L))) )/(L*L); 
            return deriv+alpha_deriv;
        }
        virtual double yderiv(double x, double y, double z, double t) {
            return 0.;
        }
        virtual double zderiv(double x, double y, double z, double t) {
            return 0.;
        }

        virtual double tderiv(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t);  
            //double alpha_deriv = this->diffuse_xderiv(x,y,z,0.,t);
            double val = - ( (B * n*n * M_PI*M_PI * alpha * sin((n * M_PI * (x-xmin))/L)) * exp(-alpha * t * ((n * n * M_PI * M_PI)/(L*L))) ) / (L*L); 

            return val; 
        }

        // Return the diffusivity at node v (K=1 by default)
        virtual double diffuseCoefficient(double x, double y, double z, double sol, double t) {
            // Vary the decay from 0 to the specified decay rate over the full domain.
            return decay*((x-xmin)/L)*(1.-exp(-t));            
        }
        virtual double diffuse_xderiv(double x, double y, double z, double sol, double t) {
            return decay;
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


