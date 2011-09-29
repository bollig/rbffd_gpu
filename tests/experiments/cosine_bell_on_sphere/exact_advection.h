#ifndef _EXACT_ADVECTION_H_
#define _EXACT_ADVECTION_H_

#include "exact_solutions/exact_solution.h"
#include "utils/geom/cart2sph.h"
#include <float.h>      // From C Standard Library (defines DBL_EPSILON as machine eps)

// These expressions were obtained from Mathematica. 
// The original idea for this solution came from
// http://web.cecs.pdx.edu/~gerry/class/ME448/codes/FDheat.pdf
class ExactAdvection : public ExactSolution
{
    private:
        // Radius of sphere/earth
        double a; 
        // Initial height of cosine bell
        double h0;
        // Radius of cosine bell
        double R; 

        // Center point of sphere (relative to Equator+GreenwhichMeanTime)
        double theta_c; 
        double lambda_c;

        // Minimum value in domain (typically 0, but try 1 if we want normalized errors).
        double base_val; 

    public:
        ExactAdvection(double sphere_radius, double initial_peak_height=1.0)
            // param 3 => 3D
            : ExactSolution(3), 
            a(sphere_radius), h0(1), base_val(0.),
            lambda_c((3.*M_PI)/2.), theta_c(0.)
    { R = a/3.; }
        ~ExactAdvection();

        virtual double operator()(double x, double y, double z, double t) {

            // NOTE: in Natasha's email they assume [phi theta r] = cart2sph(x,y,z)
            // The real return order is [theta phi r]. However, to maintain consistency
            // with their code I am swapping the phi and theta here. 
            sph_coords_type spherical_coords = cart2sph(x, y, z);
            double TI = spherical_coords.phi; 
            double LI = spherical_coords.theta; 

            double r = a * acos(sin(theta_c) * sin(TI) + cos(theta_c) * cos(TI) * cos(LI - lambda_c));

            // All bell will be > 1
            double h = (h0/2.) * (1. + cos(M_PI * (r/R))) + base_val; 
            if (r >= R) {       // Enforce the discontinuity
                h = base_val;
            }
            return h;
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
            return 0.;            
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



