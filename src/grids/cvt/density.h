#ifndef _DENSITY_H_
#define _DENSITY_H_
#include <string>
#include <iostream>
#include <math.h>
class Density
{
    protected:
        double max_rho;
    public: 
        Density(double rho_max=1.05) : max_rho(rho_max) {;}

        virtual ~Density() { ; }

        double operator()(double x, double y, double z) { 
            return this->eval(x,y,z); 
        }

        virtual double eval(double x, double y, double z) = 0;

        double getMax()
        {
            return max_rho;
        }

       virtual std::string className() { return "density";} 
};


class UniformDensity : public Density
{
    public:
        UniformDensity(double rho=1.05)
            : Density(rho)
        {;}


        virtual double eval(double x, double y, double z)
        {
            return 1.; // maxrho = 1.
        }

        virtual std::string className() {return "uniform";}
};

class ExpDensity : public Density
{
    double xc;
    double yc;

    public: 
        ExpDensity() : Density(1.05), xc(0.), yc(0.4) {;}
        ExpDensity(double center_x, double center_y, double rho_max=1.0)
        : Density(rho_max),
        xc(center_x), yc(center_y)
    { ; }

        virtual double eval(double x, double y, double z)
        {
            double e, rho;
            e = (x-xc)*(x-xc)+(y-yc)*(y-yc);
            rho = 0.05 + exp(-15.*e); // maxrho = 1.05
            return rho;
        }
    
        virtual std::string className() {return "exp";}
};

#endif

