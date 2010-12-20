#ifndef _DENSITY_H_
#define _DENSITY_H_

class Density
{
private:
	double xc;
	double yc;
	double rho;
	double max_rho;
	double e;

public:
	Density()
	{
		xc = 0.;
		yc = .4;
		//max_rho = 1.05;
		max_rho = 1.0;
	}

	double operator()(double x, double y, double z=0.) {
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
};

#endif

