#ifndef _MY_DENSITY_H_
#define _MY_DENSITY_H_

#include <math.h> 
#include "grids/cvt/density.h"

class MyDensity : public Density
{
public:
	MyDensity()
        // xc, yc, max_rho
        : Density(0., 0.4, 1.0)
    {;}

	virtual double operator()(double x, double y, double z=0.) {
		e = (x-xc)*(x-xc)+(y-yc)*(y-yc);
		rho = .05 + exp(-15.*e); // maxrho = 1.05
		return rho;
	}

    virtual std::string className() { return "my"; }
};

#endif

