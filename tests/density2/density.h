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
		max_rho = 1.05;
	}

	double operator()(double x, double y) {
		e = (x-xc)*(x-xc)+(y-yc)*(y-yc);
		rho = .05 + exp(-15.*e);
		return rho;
	}

	double getMax()
	{
		return max_rho;
		//return 5.;
	}
};

#endif

