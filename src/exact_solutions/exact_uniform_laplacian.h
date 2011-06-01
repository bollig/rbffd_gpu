#ifndef _EXACT_UNIFORM_LAPLACIAN_H_
#define _EXACT_UNIFORM_LAPLACIAN_H_

#include "exact_solution.h"

// **************
//   WARNING!
// **************
//  This is not a time dependent ExactSolution. It will not work with the
//  HeatEquation. It is only good for checking if laplacian weights can
//  adequately approximate the desired laplacian values
//



class ExactUniformLaplacian : public ExactSolution
{
    private:
    public:
        ExactUniformLaplacian(int dimensions)
            : ExactSolution(dimensions)
        {;}
        ~ExactUniformLaplacian();

        virtual double operator()(double x, double y, double z, double t) {
            // f(x, t) = 2x^2 + 3y^2 + 4z^2
            // NOTE: when dimension is less than 3 we have to exclude the extra terms in the laplacian
            //       this operator also conditionally handles the dimension just to play it safe
            double val = 2.*x*x; 
            if (dim_num > 1) {
                val += 3*y*y; 
            }
            if (dim_num > 2) {
                val+= 4.*z*z; 
            }
            return val;
        }
        virtual double laplacian(double x, double y, double z, double t) {
            double val = 4.; 
            if (dim_num > 1) {
                val += 6.; 
            }
            if (dim_num > 2) {
                val+= 8.; 
            }  
            return val;
        }

        virtual double xderiv(double x, double y, double z, double t) {
            return 4.*x;
        }
        virtual double yderiv(double x, double y, double z, double t) {
            return 6.*y;
        }
        virtual double zderiv(double x, double y, double z, double t) {
            return 8.*z;
        }

        virtual double tderiv(double x, double y, double z, double t) {
            return 0;
        }

        // Return the diffusivity at node v (K=1 by default)
        virtual double diffuseCoefficient(Vec3& v, double t=0.){
            return 0.1;
        }

        // Return the gradient of the diffusivity at node v
        virtual Vec3* diffuseGradient(Vec3& v, double t=0.){
            return new Vec3(0.,0.,0.);
        }


};
//----------------------------------------------------------------------

#endif 


