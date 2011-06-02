// Collection of norms 

#ifndef __NORMS_H__
#define __NORMS_H__

#include <vector>
#include "utils/random.h"
#include "Vec3.h"

//----------------------------------------------------------------------
//  UTILITIES
//----------------------------------------------------------------------
double minimum(std::vector<double>& vec);


//----------------------------------------------------------------------
// L1 Norms
//----------------------------------------------------------------------
double l1norm(std::vector<double>& v1, std::vector<double>& v2, int n1, int n2);
double l1norm(std::vector<double>& v1, std::vector<double>& v2);
double l1norm(std::vector<double>& v1, int n1, int n2);
double l1norm(std::vector<double>& v1);
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// Weighted L1 Norms (Weighted by average distances)
//----------------------------------------------------------------------
double l1normWeighted(std::vector<double>& v1, std::vector<double>& v2, std::vector<double>& avgDist, int n1, int n2);
double l1normWeighted(std::vector<double>& v1, std::vector<double>& avgDist, int n1, int n2);
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// L2 Norms
//----------------------------------------------------------------------
double l2norm(std::vector<double>& v1, std::vector<double>& v2, int n1, int n2);
double l2norm(std::vector<double>& v1, std::vector<double>& v2);
double l2norm(std::vector<double>& v1, int n1, int n2);
double l2norm(std::vector<double>& v1);
double l2norm(double v1);
double l2norm(Vec3& v1);
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// Weighted L2 Norms (weighted by average distance between nodes
//----------------------------------------------------------------------
double l2normWeighted(std::vector<double>& v1, std::vector<double>& v2, std::vector<double>& avgDist, int n1, int n2);
double l2normWeighted(std::vector<double>& v1, std::vector<double>& avgDist, int n1, int n2);
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// Linf (Infinity) Norms
//----------------------------------------------------------------------
double linfnorm(std::vector<double>& v1, std::vector<double>& v2, int n1, int n2);
double linfnorm(std::vector<double>& v1, std::vector<double>& v2);
double linfnorm(std::vector<double>& v1, int n1, int n2);
double linfnorm(std::vector<double>& v1);
//----------------------------------------------------------------------
#endif
