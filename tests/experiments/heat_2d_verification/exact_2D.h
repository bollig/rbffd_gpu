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
        double diffuseConst; 
        // maxX
        double L; 
        double xmin;
        // maxY
        double H;
        double ymin;
    public:
        Exact2D(double minX, double maxX, double minY, double maxY, double alpha)
            // 2D
            : ExactSolution(2), 
            n(8),m(5),B(2.),
            xmin(minX), ymin(minY),
            L(maxX-minX),H(maxY-minY),
            diffuseConst(alpha)
    {;}
        ~Exact2D();

        virtual double operator()(double x, double y, double z, double t) {
            // See Haberman p48 for details
            // B is the coefficient we choose for the problem. It could even be another function
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double initialConst = B; 
            double spatial_x = sin((n * M_PI * (x-xmin))/L);
            double spatial_y = sin((m * M_PI * (y-ymin))/H);
            double lambda = ((n * n * M_PI * M_PI)/(L*L)) + ((m * m * M_PI * M_PI)/(H*H));
            double temporal = exp(-alpha * t * lambda);
            double val = initialConst * spatial_x * spatial_y * temporal;
            return val;
        }

        virtual double laplacian(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double initialConst = B; 
            double d2_spatial_xx = -sin((n * M_PI * (x-xmin))/L) * (n*n*M_PI*M_PI) / L*L;
            double d2_spatial_yy = -sin((m * M_PI * (y-ymin))/H) * (m*m*M_PI*M_PI) / H*H;
            double lambda = ((n * n * M_PI * M_PI)/(L*L)) + ((m * m * M_PI * M_PI)/(H*H));
            double temporal = exp(-alpha * t * lambda);

            //changed from operator()
            double val = initialConst * d2_spatial_xx * d2_spatial_yy * temporal;
            
            return val;
        }

        virtual double xderiv(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double initialConst = B; 
            
            // Chagned from operator()
            double d_spatial_x = n*M_PI*cos((n * M_PI * (x-xmin))/L) / L;
            
            double spatial_y = sin((m * M_PI * (y-ymin))/H);
            double lambda = ((n * n * M_PI * M_PI)/(L*L)) + ((m * m * M_PI * M_PI)/(H*H));
            double temporal = exp(-alpha * t * lambda);
            double val = initialConst * d_spatial_x * spatial_y * temporal;
            return val;
        }
        virtual double yderiv(double x, double y, double z, double t) {
            double alpha = this->diffuseCoefficient(x,y,z,0.,t); 
            double initialConst = B; 
            double spatial_x = sin((n * M_PI * (x-xmin))/L);

            // Changed from operator()
            double d_spatial_y = (m * M_PI)*sin((m * M_PI * (y-ymin))/H)/H;
            
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
            double spatial_x = sin((n * M_PI * (x-xmin))/L);
            double spatial_y = sin((m * M_PI * (y-ymin))/H);
            double lambda = ((n * n * M_PI * M_PI)/(L*L)) + ((m * m * M_PI * M_PI)/(H*H));
            double temporal = exp(-alpha * t * lambda);

            // Changed from operator()
            double d_temporal = exp(-alpha * t * lambda) * -alpha * lambda; 

            double val = initialConst * spatial_x * spatial_y * d_temporal;
            return val;
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



