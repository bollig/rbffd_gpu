#ifndef _EXACT_2D_H_
#define _EXACT_2D_H_

#include "exact_solutions/exact_solution.h"

// These expressions were obtained from Mathematica. 
// The original idea for this solution came from
// http://web.cecs.pdx.edu/~gerry/class/ME448/codes/FDheat.pdf
class Exact2D : public ExactSolution
{
    private:
            // See Haberman p47,48 for details
        double n;
        double m;
        double B;
        double decay; 
        // maxX
        double L; 
        // maxY
        double H;
    public:
        Exact2D(double maxX, double maxY, double alpha)
            // 2D
            : ExactSolution(2), 
            n(2),m(3),B(10),
            L(maxX),H(maxY),
            decay(alpha)
    {;}
        ~Exact2D();

        virtual double operator()(double x, double y, double z, double t) {
            // See Haberman p48 for details
            // B is the coefficient we choose for the problem. It could even be another function
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double initialConst = B; 
            double spatial_x = sin((n * M_PI * x)/L);
            double spatial_y = sin((m * M_PI * y)/H);
            double lambda = ((n * n * M_PI * M_PI)/(L*L)) + ((m * m * M_PI * M_PI)/(H*H));
            double temporal = exp(-alpha * t * lambda);
            double val = initialConst * spatial_x * spatial_y * temporal;
            return val;
        }

        virtual double laplacian(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double initialConst = B; 
            double spatial_x = sin((n * M_PI * x)/L);
            double spatial_y = sin((m * M_PI * y)/H);
            double lambda = ((n * n * M_PI * M_PI)/(L*L)) + ((m * m * M_PI * M_PI)/(H*H));
            double temporal = exp(-alpha * t * lambda);

            //changed from operator()
            double val = (initialConst * spatial_x * spatial_y * temporal * (L*L * m*m + H*H * n*n)*M_PI*M_PI) / (H*H*L*L);
            
            return val;
        }

        virtual double xderiv(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double initialConst = B; 
            
            // Chagned from operator()
            double d_spatial_x = n*M_PI*cos((n * M_PI * x)/L) / L;
            
            double spatial_y = sin((m * M_PI * y)/H);
            double lambda = ((n * n * M_PI * M_PI)/(L*L)) + ((m * m * M_PI * M_PI)/(H*H));
            double temporal = exp(-alpha * t * lambda);
            double val = initialConst * d_spatial_x * spatial_y * temporal;
            return val;
        }
        virtual double yderiv(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double initialConst = B; 
            double spatial_x = sin((n * M_PI * x)/L);

            // Changed from operator()
            double d_spatial_y = (m * M_PI)*sin((m * M_PI * y)/H)/H;
            
            double lambda = ((n * n * M_PI * M_PI)/(L*L)) + ((m * m * M_PI * M_PI)/(H*H));
            double temporal = exp(-alpha * t * lambda);
            double val = initialConst * spatial_x * d_spatial_y * temporal;
            return val;
        }
        virtual double zderiv(double x, double y, double z, double t) {
            return 0.;
        }

        virtual double tderiv(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double initialConst = B; 
            double spatial_x = sin((n * M_PI * x)/L);
            double spatial_y = sin((m * M_PI * y)/H);
            double lambda = ((n * n * M_PI * M_PI)/(L*L)) + ((m * m * M_PI * M_PI)/(H*H));
            double temporal = exp(-alpha * t * lambda);

            // Changed from operator()
            double d_temporal = exp(-alpha * t * lambda) * -alpha * lambda; 

            double val = initialConst * spatial_x * spatial_y * d_temporal;
            return val;
        }

        // Return the diffusivity at node v (K=1 by default)
        virtual double diffuseCoefficient(double x, double y, double z, double sol, double t) {
            return 0.1;            
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



