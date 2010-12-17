#ifndef _NCAR_POISSON_1_H_
#define _NCAR_POISSON_1_H_

#include <vector>
#include <ArrayT.h>
#include "grids/domain_decomposition/gpu.h"
#include "exact_solutions/exact_solution.h"
#include "rbffd/derivative.h"
#include "utils/comm/communicator.h"

class NCARPoisson1Explicit
{
private:
        std::vector<double> sol;

	std::vector<Vec3>* rbf_centers;
	std::vector<int>* boundary_set; 		// The indices of rbf_centers that correspond to global domain boundary nodes (i.e. boundaries of the PDE)


	GPU* subdomain;
	Derivative* der;
	std::vector<double> lapl_deriv;
	std::vector<double> x_deriv;
	std::vector<double> y_deriv;
	std::vector<double> xx_deriv;
	std::vector<double> yy_deriv;
	// derivate based on derivative operator
	std::vector<double> cart_laplace;
	std::vector<double> diffusion;
	std::vector<double> diff_x;
	std::vector<double> diff_y;

	double time; 

	// boundary values (in the same order as boundary_index)
	std::vector<double> bnd_sol; 

	// total number of rbfs
	int nb_rbf;
	int nb_stencils;

	ExactSolution* exactSolution; 
	
	int id; 		// Comm rank or comm id

public:
        NCARPoisson1Explicit(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank);
        ~NCARPoisson1Explicit();

        // Solve the Poisson problem
	void solve(Communicator* comm_unit);
        
	// Only update the updated_solution vector if it is non-null (i.e. we actually pass something to the routine)
	void initialConditions(std::vector<double>* solution = NULL);

        double maxNorm();
	double maxNorm(std::vector<double> sol);

        double boundaryValues(Vec3& v);

};

#endif
