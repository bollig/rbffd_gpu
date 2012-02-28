#ifndef _EXACT_ROLLUP_H_
#define _EXACT_ROLLUP_H_

#include "exact_solutions/exact_solution.h"
#include "utils/geom/cart2sph.h"
#include <float.h>      // From C Standard Library (defines DBL_EPSILON as machine eps)

// These expressions were obtained from Mathematica. 
// The original idea for this solution came from
// http://web.cecs.pdx.edu/~gerry/class/ME448/codes/FDheat.pdf
class ExactRollup : public ExactSolution
{
    private:
        double gamma; 
        double rho0; 

    public:
        ExactRollup()
            // param 3 => 3D
            : ExactSolution(3), 
            gamma(5), rho0(3)
    {;}
        ~ExactRollup();

        virtual double operator()(double x, double y, double z, double t) {

            // NOTE: in Natasha's email they assume [phi theta r] = cart2sph(x,y,z)
            // The real return order is [theta phi r]. However, to maintain consistency
            // with their code I am swapping the phi and theta here. 
            sph_coords_type spherical_coords = cart2sph(x, y, z);
            double theta_p = spherical_coords.phi; 
            double phi_p = spherical_coords.theta; 
            //double temp = spherical_coords.r; 

//            std::cout << "CART(" << x << ",  " << y << ", " << z << ") => SPH(" << theta_p << ", " << phi_p << ", " << temp << ")\n"; 

            // From Natasha's email: 
            // 7/7/11 4:46 pm
            double rho_p = rho0 * cos(theta_p); 

            // NOTE: The sqrt(2) is written as sqrt(3.) in the paper with Grady.
            // Natasha verified sqrt(3) is required
            // Also, for whatever reason sech is not defined in the C standard
            // math. I provide it in the cart2sph header. 
            double Vt = (3.* sqrt(3.) / 2.) * (sech(rho_p) * sech(rho_p)) * tanh(rho_p); 
            double w = (fabs(rho_p) < 4. * DBL_EPSILON) ? 0. : Vt / rho_p;
        
            double h = 1. - tanh((rho_p / gamma) * sin(phi_p - w * t)); 

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



