#include <mpi.h>

#include "mpi_norms.h"
#include "norms.h"

//----------------------------------------------------------------------

double l1norm(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2)
{
    double global_val = 0;
    // reuse the local norms from norms.h
    double local_val = l2norm(v1, v2);
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_val;
}

//----------------------------------------------------------------------

double l1norm(int mpi_rank, std::vector<double>& v1)
{
    double global_val = 0;
    double local_val = l1norm(v1);
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_val;
}

//----------------------------------------------------------------------

double l2norm(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2) {
    double global_val = 0;
    // reuse the local norms from norms.h
    double local_val = l2norm(v1, v2);
    local_val *= local_val;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(global_val);
}

//----------------------------------------------------------------------

double l2norm(int mpi_rank, std::vector<double>& v1)
{
    double global_val = 0;
    double local_val = l2norm(v1);
    local_val *= local_val;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(global_val);
}

//----------------------------------------------------------------------

double linfnorm(int mpi_rank, std::vector<double>& v1, std::vector<double>& v2) {
    double global_val = 0;
    // reuse the local norms from norms.h
    double local_val = linfnorm(v1, v2);
    // Get max of max (will bcast norm to all ranks)
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_val;
}

//----------------------------------------------------------------------

double linfnorm(int mpi_rank, std::vector<double>& v1)
{
    double global_val = 0;
    double local_val = linfnorm(v1);
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_val;
}
