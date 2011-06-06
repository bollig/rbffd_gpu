#ifndef _EXACT_SOLUTION_
#define _EXACT_SOLUTION_

#include "Vec3.h"
#include <cmath>

//----------------------------------------------------------------------
class ExactSolution
{
    protected:
        double Pi; 

        // NOTE: our laplacian terms might depend on knowledge of the dimensionality (see e.g., ExactUniformLaplacian)
        int dim_num; 

    public:

        ExactSolution(int dimension) : dim_num(dimension), Pi(acos(-1.)) {};
        //	~ExactSolution() {}; 

        virtual double operator()(double x, double y, double z, double t=0.) = 0;
        virtual double operator()(Vec3& r, double t=0.) {
            return (*this)(r.x(), r.y(), r.z(), t);
        }

        virtual double at(Vec3& r, double t = 0.) {
            return (*this)(r, t);
        }

        virtual double laplacian(Vec3& v, double t=0.) {
            return this->laplacian(v.x(), v.y(), v.z(), t); 
        }

        // Return the value of the analytic Laplacian of the equation f(x,y,z;t)
        // NOTE: in previous versions of code this called to get a forcing term:
        // 			F(x,y,z;t) = df/dt - lapl(f)
        // instead of just:
        // 			lapl(f)
        // To get the original behavior, substitute: this->tderiv() - this->laplacian()
        virtual double laplacian(double x, double y, double z, double t=0.) = 0; // if scalar function

        virtual double xderiv(double x, double y, double z, double t=0.) = 0; // if scalar function
        virtual double yderiv(double x, double y, double z, double t=0.) = 0; // if scalar function
        virtual double zderiv(double x, double y, double z, double t=0.) = 0; // if scalar function
        virtual double tderiv(double x, double y, double z, double t=0.) = 0; // if scalar function

        double xderiv(Vec3& r, double t=0.) {
            return xderiv(r.x(), r.y(), r.z(), t);
        }

        double yderiv(Vec3& r, double t=0.) {
            return yderiv(r.x(), r.y(), r.z(), t);
        }

        double zderiv(Vec3& r, double t=0.) {
            return zderiv(r.x(), r.y(), r.z(), t);
        }

        double tderiv(Vec3& r, double t=0.) {
            return tderiv(r.x(), r.y(), r.z(), t);
        }

        Vec3& gradient(Vec3& pt, double t=0.) {
            Vec3* grad = new Vec3(this->xderiv(pt,t), this->yderiv(pt,t), this->zderiv(pt,t)); 
            return *grad;
        }

        // The param sol here should be the SOLUTION at pt(x,y,z) at time t.
        // This allows us to vary diffusion based on the current temperature of
        // the domain for example.
        virtual double diffuseCoefficient(double x, double y, double z, double sol, double t) {
            return 1.;
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

        double diffuse_xderiv(Vec3& r, double sol, double t=0.) {
            return diffuse_xderiv(r.x(), r.y(), r.z(), sol, t);
        }

        double diffuse_yderiv(Vec3& r, double sol, double t=0.) {
            return diffuse_yderiv(r.x(), r.y(), r.z(), sol, t);
        }

        double diffuse_zderiv(Vec3& r, double sol, double t=0.) {
            return diffuse_zderiv(r.x(), r.y(), r.z(), sol, t);
        }

        double diffuse_tderiv(Vec3& r, double sol, double t=0.) {
            return diffuse_zderiv(r.x(), r.y(), r.z(), sol, t);
        }


        // Return the diffusivity at node v (K=1 by default)
        virtual double diffuseCoefficient(Vec3& v, double sol=0., double t=0.){
            return this->diffuseCoefficient(v.x(), v.y(), v.z(), sol, t);
        }

        // Return the gradient of the diffusivity at node v
        virtual Vec3 diffuseGradient(Vec3& v, double sol=0., double t=0.){
            Vec3 diff_grad(this->diffuse_xderiv(v,sol,t), this->diffuse_yderiv(v,sol,t), this->diffuse_zderiv(v,sol,t));
            return diff_grad;
        }


        virtual double convectionCoefficient(double x, double y, double z, double t) {
            return 1.;
        }
        virtual double convection_xderiv(double x, double y, double z, double t) {
            return 0.;
        }
        virtual double convection_yderiv(double x, double y, double z, double t) {
            return 0.;
        }
        virtual double convection_zderiv(double x, double y, double z, double t) {
            return 0.;
        }
        virtual double convection_tderiv(double x, double y, double z, double t) {
            return 0.;
        }

        double convection_xderiv(Vec3& r, double t=0.) {
            return convection_xderiv(r.x(), r.y(), r.z(), t);
        }

        double convection_yderiv(Vec3& r, double t=0.) {
            return convection_yderiv(r.x(), r.y(), r.z(), t);
        }

        double convection_zderiv(Vec3& r, double t=0.) {
            return convection_zderiv(r.x(), r.y(), r.z(), t);
        }

        double convection_tderiv(Vec3& r, double t=0.) {
            return convection_zderiv(r.x(), r.y(), r.z(), t);
        }

        virtual double convectionCoefficient(Vec3& v, double t=0.){
            return this->convectionCoefficient(v.x(), v.y(), v.z(), t);
        }

        virtual Vec3 convectionGradient(Vec3& v, double t=0.){
            Vec3 conv_grad(this->convection_xderiv(v,t), this->convection_yderiv(v,t), this->convection_zderiv(v,t));
            return conv_grad;
        }


    protected:
        inline double Power(double a, double b) { return pow(a, b); }
        inline double Sqrt(double a) { return sqrt(a); }
        inline double Sin(double a) { return sin(a); }
        inline double Cos(double a) { return cos(a); }

        //virtual double divergence() = 0; // if vector function (not used)
};
//----------------------------------------------------------------------

#endif
