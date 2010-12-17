#ifndef _DERIVATIVE_H_
#define _DERIVATIVE_H_

#include <stdlib.h>
#include <iostream>
#include <vector>
#include "ArrayT.h"
#include "Vec3.h"
//#include <armadillo>
#include "rbffd/rbfs/rbf_gaussian.h"
#include "rbffd/rbfs/rbf_mq.h"

//typedef RBF_Gaussian IRBF;
typedef RBF_MQ IRBF;

typedef std::vector<IRBF*> BasesType;
typedef std::vector< BasesType > BasesListType;

class Derivative
{
public:
    enum DerType {X, Y, Z, LAPL};

private:
    typedef ArrayT<double> AF;
    //AF arr;
    int nb_rbfs;
    double maxint;

    BasesListType rbfs;
    double* weights_p;
    // Make sure these are deleted inside the destructor
    std::vector<double*> x_weights;
    std::vector<double*> y_weights;
    std::vector<double*> z_weights;
    std::vector<double*> r_weights;
    std::vector<double*> lapl_weights;

    std::vector<Vec3>& rbf_centers;
    std::vector<std::vector<int> >& stencil;
    std::vector<double> avg_stencil_radius;

    std::vector<double> var_eps;
    double epsilon;  // RBF scaling
    int nb_bnd; // number of points on the boundary (EB: is this the boundary of subdomain or PDE?)

public:
    //Derivative(int nb_rbfs);
    Derivative(std::vector<Vec3>& rbf_centers_, std::vector<std::vector<int> >& stencil_, int nb_bnd_pts);
    ~Derivative();

    AF& cholesky_cpu(AF& arr);
    AF& cholesky(AF& arr);
    double rowDot(AF& a, int row_a, AF& b, int row_b, int ix);

    // use an SVD decomposition on the distance matrix to simplify the direct solve.
    void computeWeightsSVD_Direct(std::vector<Vec3>& rbf_centers, std::vector<int>& stencil, int irbf);

    // Use a direct solver on teh distance matrix.
    int computeWeights(std::vector<Vec3>& rbf_centers, std::vector<int>& stencil, int irbf, int dim_num);

    // Use Grady's contourSVD (my c++) version to compute the weights
    // choice: compute either "lapl", "x" or "y" derivative stencils
    void computeWeightsSVD(std::vector<Vec3>& rbf_centers, std::vector<int>&
                           stencil, int irbf, const char* choice);

    void distanceMatrix(std::vector<Vec3>& rbf_centers, std::vector<int>& stencil,int irbf, double* distance_matrix, int nrows, int ncols, int dim_num);

    AF& solve(AF& l, AF& b);
    AF& matmul(AF& arr, AF& x);

    // u : take derivative of this scalar variable (already allocated)
    // deriv : resulting derivative (already allocated)
    // which : which derivative (X, Y, LAPL)
    void computeDeriv(DerType which, std::vector<double>& u, std::vector<double>& deriv);
    void computeDeriv(DerType which, double* u, double* deriv, int npts);


    std::vector<double*>& getXWeights() { return x_weights; }
    std::vector<double*>& getYWeights() { return y_weights; }
    std::vector<double*>& getZWeights() { return z_weights; }
    std::vector<double*>& getRWeights() { return r_weights; }
    std::vector<double*>& getLaplWeights() { return lapl_weights; }

    double* getXWeights(int indx) { return x_weights[indx]; }
    double* getYWeights(int indx) { return y_weights[indx]; }
    double* getZWeights(int indx) { return z_weights[indx]; }
    double* getRWeights(int indx) { return r_weights[indx]; }
    double* getLaplWeights(int indx) { return lapl_weights[indx]; }

    BasesType& getRBFList(int stencil_indx) { return rbfs[stencil_indx]; }
    BasesListType& getRBFList() { return rbfs; }

    double computeEig();


    void setAvgStencilRadius(std::vector<double>& avg_radius_) {
        avg_stencil_radius = avg_radius_;
        var_eps.resize(avg_stencil_radius.size());
        for (int i=0; i < var_eps.size(); i++) {
            var_eps[i] = 1. / avg_stencil_radius[i];
            //printf("avg rad(%d) = %f\n", i, avg_stencil_radius[i]);
        }
    }

    void setVariableEpsilon(std::vector<double>& var_eps_) {
        this->var_eps = var_eps_;
    }

    double minimum(std::vector<double>& vec);

    double setEpsilon(double eps) { 

    		std::cout << "DERIVATIVE:: SET EPSILON = " << eps << std::endl;
	    this->epsilon = eps; return this->epsilon;
    }
};

#endif
