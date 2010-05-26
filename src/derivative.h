#ifndef _DERIVATIVE_H_
#define _DERIVATIVE_H_

#include <vector>
#include "ArrayT.h"
#include "Vec3.h"
#include <armadillo>


class Derivative
{
public:
	enum DerType {X, Y, Z, LAPL};

private:
	typedef ArrayT<double> AF;
	//AF arr;
	int nb_rbfs;
	double maxint;
	arma::mat* weights_p; // pointer (_p)
	arma::colvec bx, by, bz;
	arma::colvec bxx, bxy, bxz;
	arma::colvec byy, byz, bzz;
	arma::colvec blapl;
	arma::mat* Up;
	arma::mat* Vp;
	arma::colvec* sp;
	std::vector<arma::mat> x_weights;
	std::vector<arma::mat> y_weights;
	std::vector<arma::mat> z_weights;
	std::vector<arma::mat> lapl_weights;
//	std::vector<arma::mat> xderiv_weights;
	//std::vector<arma::mat> yderiv_weights;

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
	void computeWeights(std::vector<Vec3>& rbf_centers, std::vector<int>& stencil, int irbf);

	// Use Grady's contourSVD (my c++) version to compute the weights
	// choice: compute either "lapl", "x" or "y" derivative stencils
	void computeWeightsSVD(std::vector<Vec3>& rbf_centers, std::vector<int>&
	stencil, int irbf, const char* choice);

	int distanceMatrix(std::vector<Vec3>& rbf_centers, std::vector<int>& stencil, 
    	int irbf, int& nb_eig);
	AF& solve(AF& l, AF& b);
	AF& matmul(AF& arr, AF& x);

	// u : take derivative of this scalar variable (already allocated)
	// deriv : resulting derivative (already allocated)
	// which : which derivative (X, Y, LAPL)
	void computeDeriv(DerType which, std::vector<double>& u, std::vector<double>& deriv);
	void computeDeriv(DerType which, double* u, double* deriv, int npts);

	std::vector<arma::mat>& getXWeights() { return x_weights; }
	std::vector<arma::mat>& getYWeights() { return y_weights; }
	std::vector<arma::mat>& getZWeights() { return z_weights; }
	std::vector<arma::mat>& getLaplWeights() { return lapl_weights; }
	double computeEig();


	void setAvgStencilRadius(std::vector<double>& avg_radius_) {
		avg_stencil_radius = avg_radius_;
	}

	void setVariableEpsilon(std::vector<double>& var_eps_) {
		this->var_eps = var_eps_;
	}

	double minimum(std::vector<double>& vec);
};

#endif
