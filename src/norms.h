// Collection of norms 

#ifndef __NORMS_H__
#define __NORMS_H__

#include <vector>



//----------------------------------------------------------------------
double random(double a, double b)
{
        // use system version of random, not class version
        double r = ::random() / (double) RAND_MAX;
        return a + r*(b-a);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
double minimum(vector<double>& vec)
{
        double min = 1.e10;

        for (int i=0; i < vec.size(); i++) {
                if (vec[i] < min) {
                        min = vec[i];

                }
        }
        return min;
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
double l1norm(std::vector<double>& v1, std::vector<double>& v2, std::vector<double>& avgDist, int n1, int n2)
{
	double norm = 0;
	double err; 
	double elemt;

	for (int i=n1; i < n2; i++) {
		err = abs(v1[i] - v2[i]);
		// if n1 == 0, we are on the boundary
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i]*avgDist[i];
		norm += elemt * abs(err);
	}

	//return norm / (n2-n1);
	return norm; 
}
//-------
double l1norm(std::vector<double>& v1, std::vector<double>& avgDist, int n1, int n2)
{
	double norm = 0;
	double elemt;

	for (int i=n1; i < n2; i++) {
		// if n1 == 0, we are on the boundary
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i]*avgDist[i];
		norm += elemt * abs(v1[i]);
	}

	//return norm / (n2-n1);
	return norm;
}
//----------------------------------------------------------------------

double l2norm(std::vector<double>& v1, std::vector<double>& v2, std::vector<double>& avgDist, int n1, int n2)
{
	double norm = 0;
	double err;
	double elemt;

	for (int i=n1; i < n2; i++) {
		err = abs(v1[i] - v2[i]);
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i]*avgDist[i];
		norm += elemt * err * err;
	}

	//return sqrt(norm / (n2-n1));
	return sqrt(norm);
}
//----------------------------------------------------------------------
double l2norm(std::vector<double>& v1, std::vector<double>& avgDist, int n1, int n2)
{
	double norm = 0;
	double elemt;

	for (int i=n1; i < n2; i++) {
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i]*avgDist[i];
		norm += elemt * v1[i] * v1[i];
	}

	//return sqrt(norm / (n2-n1));
	return sqrt(norm);
}
//----------------------------------------------------------------------

double linfnorm(std::vector<double>& v1, std::vector<double>& v2, int n1, int n2)
{
	double norm = -1.e10;
	double err;

	for (int i=n1; i < n2; i++) {
		err = abs(v1[i] - v2[i]);
		norm = (norm < err) ? err : norm;
	}
	return norm;
}
//----------------------------------------------------------------------
double linfnorm(std::vector<double>& v1, int n1, int n2)
{
	double norm = -1.e10;

	for (int i=n1; i < n2; i++) {
		norm = (norm < abs(v1[i])) ? abs(v1[i]) : norm;
	}
	return norm;
}
//----------------------------------------------------------------------
#endif
