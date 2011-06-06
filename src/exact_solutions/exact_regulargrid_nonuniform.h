#ifndef _EXACT_REGULAR_GRID_NON_UNIFORM_H_
#define _EXACT_REGULAR_GRID_NON_UNIFORM_H_

#include "exact_solution.h"

class ExactRegularGridNonUniform : public ExactSolution
{
    private:
        double freq;

    public:
        ExactRegularGridNonUniform(int dimension, double freq);
        ~ExactRegularGridNonUniform();

        double operator()(double x, double y, double z, double t);
        double laplacian(double x, double y, double z, double t);

        double xderiv(double x, double y, double z, double t);
        double yderiv(double x, double y, double z, double t);
        double zderiv(double x, double y, double z, double t);

        double tderiv(double x, double y, double z, double t); 

        // Return the diffusivity at node v (K=1 by default)
        virtual double diffuseCoefficient(double x, double y, double z, double sol, double t=0.){
            double diff = (x - 0.5)*(x-0.5); 
            return diff;
        }

        virtual double diffuse_xderiv(double x, double y, double z, double sol, double t) {
            return 2.*(x-0.5);            
        }
        virtual double diffuse_yderiv(double x, double y, double z, double sol, double t) {
            return 0.;
        }
        virtual double diffuse_zderiv(double x, double y, double z, double sol, double t) {
            return 0.;
        }

    private: 

        double laplacian1D(double x, double y, double z, double t);
        double laplacian2D(double x, double y, double z, double t);
        double laplacian3D(double x, double y, double z, double t);

};

//----------------------------------------------------------------------

#endif // _EXACT_REGULAR_GRID_NON_UNIFORM_H_

