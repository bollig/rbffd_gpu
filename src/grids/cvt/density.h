#ifndef _DENSITY_H_
#define _DENSITY_H_
#include <string>

class Density
{
protected:
	double xc;
	double yc;
	double rho;
	double max_rho;
	double e;

public:
	Density()
    : xc(0.), yc(0.), max_rho(1.0)
    {;}

    Density(double xc_, double yc_, double max_rho_)
    : xc(xc_), yc(yc_), max_rho(max_rho_)
    {;}

    ~Density() { ; }

	virtual double operator()(double x, double y, double z=0.) {
		#if 0
		e = (x-xc)*(x-xc)+(y-yc)*(y-yc);
		rho = .05 + exp(-15.*e); // maxrho = 1.05
		return rho;
		#endif

		return 1.; // maxrho = 1.
	}

	double getMax()
	{
		return max_rho;
	}

    virtual std::string className() {return "uniform";}
};

#endif

