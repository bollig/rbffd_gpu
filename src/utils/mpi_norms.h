#ifndef __MPI_NORMS_H__
#define __MPI_NORMS_H__

#include <vector>
#include "Vec3.h"

//----------------------------------------------------------------------
// L1 Norms
//----------------------------------------------------------------------
//double l1norm(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2, int n1, int n2);
double l1norm(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2); 
//double l1norm(int mpi_rank, std::vector<double>& v1, int n1, int n2);
double l1norm(int mpi_rank, std::vector<double>& v1);
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// Weighted L1 Norms (Weighted by average distances)
//----------------------------------------------------------------------
//double l1normWeighted(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2, std::vector<double>& avgDist, int n1, int n2);
//double l1normWeighted(int mpi_rank, std::vector<double>& v1, std::vector<double>& avgDist, int n1, int n2);
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// L2 Norms
//----------------------------------------------------------------------
//double l2norm(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2, int n1, int n2);
double l2norm(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2);
double l2norm(int mpi_rank, std::vector<double>& v1);
//double l2norm(int mpi_rank, std::vector<double>& v1, int n1, int n2);
//double l2norm(int mpi_rank, double v1);
//double l2norm(int mpi_rank, Vec3& v1);
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// Weighted L2 Norms (weighted by average distance between nodes
//----------------------------------------------------------------------
//double l2normWeighted(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2, std::vector<double>& avgDist, int n1, int n2);
//double l2normWeighted(int mpi_rank, std::vector<double>& v1, std::vector<double>& avgDist, int n1, int n2);
//----------------------------------------------------------------------


//----------------------------------------------------------------------
// Linf (Infinity) Norms
//----------------------------------------------------------------------
//double linfnorm(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2, int n1, int n2);
double linfnorm(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2);
double linfnorm(int mpi_rank, std::vector<double>& v1);
//double linfnorm(int mpi_rank, std::vector<double>& v1, int n1, int n2);
//----------------------------------------------------------------------

#endif 
