#include "norms.h"
#include <vector>
#include <iostream>
#include <cmath>
#include "utils/random.h"

//----------------------------------------------------------------------
double minimum(std::vector<double>& vec)
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
// L1 Norms
//----------------------------------------------------------------------
double l1norm(std::vector<double>& v1, std::vector<double>& v2, int n1, int n2)
{
	double norm = 0.;
	double err;

	for (int i=n1; i < n2; i++) {
		err = fabs(v1[i] - v2[i]);
		norm += err;  
	}
	return norm;
}
//----------------------------------------------------------------------
double l1norm(std::vector<double>& v1, std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        std::cout << "Error! in l1norm(...): vectors are not same length. assuming 0's for missing elements" << std::endl;
        //exit(EXIT_FAILURE);
    }

	return l1norm(v1, v2, 0, v1.size());
}
//----------------------------------------------------------------------
double l1norm(std::vector<double>& v1, int n1, int n2)
{
	double norm = 0;

	for (int i=n1; i < n2; i++) {
	    norm += fabs(v1[i]);  
	}
	return norm;

}
//----------------------------------------------------------------------
double l1norm(std::vector<double>& v1)
{
	return l1norm(v1,0,v1.size());
}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// Weighted L1 Norms (Weighted by average distances)
//----------------------------------------------------------------------
double l1normWeighted(std::vector<double>& v1, std::vector<double>& v2, std::vector<double>& avgDist, int n1, int n2)
{
	double norm = 0;
	double err; 
	double elemt;

	for (int i=n1; i < n2; i++) {
		err = fabs(v1[i] - v2[i]);
		// if n1 == 0, we are on the boundary
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i]*avgDist[i];
		norm += elemt * err;
	}

	//return norm / (n2-n1);
	return norm; 
}
//----------------------------------------------------------------------
double l1normWeighted(std::vector<double>& v1, std::vector<double>& avgDist, int n1, int n2)
{
	double norm = 0;
	double elemt;

	for (int i=n1; i < n2; i++) {
		// if n1 == 0, we are on the boundary
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i]*avgDist[i];
		norm += elemt * fabs(v1[i]);
	}

	//return norm / (n2-n1);
	return norm;
}
//----------------------------------------------------------------------






//----------------------------------------------------------------------
// L2 Norms
//----------------------------------------------------------------------
double l2norm(std::vector<double>& v1, std::vector<double>& v2, int n1, int n2)
{
	double norm = 0.;
	double err;

	for (int i=n1; i < n2; i++) {
		err = fabs(v1[i] - v2[i]);
		norm += err * err;  
	}
	return sqrt(norm);
}
//----------------------------------------------------------------------
double l2norm(std::vector<double>& v1, std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        std::cout << "Error! in l2norm(...): vectors are not same length. assuming 0's for missing elements" << std::endl;
//        exit(EXIT_FAILURE);
    }

	return l2norm(v1, v2, 0, v1.size());
}
//----------------------------------------------------------------------
double l2norm(std::vector<double>& v1, int n1, int n2)
{
	double norm = 0;

	for (int i=n1; i < n2; i++) {
	    norm += v1[i] * v1[i];  
	}
	return sqrt(norm);

}
//----------------------------------------------------------------------
double l2norm(std::vector<double>& v1)
{
	return l2norm(v1,0,v1.size());
}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// Weighted L2 Norms (weighted by average distance between nodes
//----------------------------------------------------------------------
double l2normWeighted(std::vector<double>& v1, std::vector<double>& v2, std::vector<double>& avgDist, int n1, int n2)
{
	double norm = 0;
	double err;
	double elemt;

	for (int i=n1; i < n2; i++) {
		err = fabs(v1[i] - v2[i]);
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i]*avgDist[i];
		norm += elemt * err * err;
	}
	//return sqrt(norm / (n2-n1));
	return sqrt(norm);
}
//----------------------------------------------------------------------
double l2normWeighted(std::vector<double>& v1, std::vector<double>& avgDist, int n1, int n2)
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




//----------------------------------------------------------------------
// Linf (Infinity) Norms
//----------------------------------------------------------------------
double linfnorm(std::vector<double>& v1, std::vector<double>& v2, int n1, int n2)
{
	double norm = -1.e10;
	double err;

	for (int i=n1; i < n2; i++) {
		err = fabs(v1[i] - v2[i]);
		norm = (norm < err) ? err : norm;
	}
	return norm;
}
//----------------------------------------------------------------------
double linfnorm(std::vector<double>& v1, std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        std::cout << "Error! in linfnorm(...): vectors are not same length. assuming 0's for missing elements" << std::endl;
//        exit(EXIT_FAILURE);
    }

	return linfnorm(v1, v2, 0, v1.size());
}
//----------------------------------------------------------------------
double linfnorm(std::vector<double>& v1, int n1, int n2)
{
	double norm = -1.e10;

	for (int i=n1; i < n2; i++) {
		norm = (norm < fabs(v1[i])) ? fabs(v1[i]) : norm;
	}
	return norm;
}
//----------------------------------------------------------------------
double linfnorm(std::vector<double>& v1)
{
    return linfnorm(v1,0,v1.size()); 
}
//----------------------------------------------------------------------
#
