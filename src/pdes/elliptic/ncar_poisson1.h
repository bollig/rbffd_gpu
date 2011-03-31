#ifndef _NCAR_POISSON_1_H_
#define _NCAR_POISSON_1_H_

#include <vector>
#include <ArrayT.h>
#include "grids/domain.h"
#include "exact_solutions/exact_solution.h"
#include "utils/comm/communicator.h"
#include "rbffd/derivative.h"
#include "utils/conf/projectsettings.h"

class NCARPoisson1
{
protected:
        std::vector<double> sol[2];

	std::vector<Vec3>* rbf_centers;
	std::vector<size_t>* boundary_set; 		// The indices of rbf_centers that correspond to global domain boundary nodes (i.e. boundaries of the PDE)
	
        Grid* subdomain;
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

        int dim_num;

        EB::TimerList tm;
        EB::Timer t1, t2, t3, t4, t5;

        // FLAGS
        bool disable_sol_constraint; // Disable the solution constraint for Neumann and Robin boundary conditions
        int  boundary_condition;     // Choose boundary condition type (0 = Dirichlet; 1 = Neumann; 2 = Robin)
        bool use_discrete_rhs;       //  Compute a discrete approximation for RHS values for the Discrete Compat. Condition
        bool use_uniform_diffusivity;// Disable non-uniform diffusion coefficients provided by exact solutions
        bool test_dirichlet_lockdown;// Lockdown the boundaries by specifying one node for each boundary with its dirichlet condition?
        enum boundary_condition_type {DIRICHLET=0, NEUMANN=1, ROBIN=2};

public:
        NCARPoisson1(ExactSolution* _solution, Grid* subdomain_, Derivative* der_, int rank, int dim_num_);
        ~NCARPoisson1();

        // Solve the Poisson problem
        virtual void solve(Communicator* comm_unit=NULL);

	// Only update the updated_solution vector if it is non-null (i.e. we actually pass something to the routine)
	void initialConditions(std::vector<double>* solution = NULL);

        double maxNorm();
	double maxNorm(std::vector<double> sol);
        //double maxNorm(arma::mat sol);
        double maxNorm(double* sol, int nrows, int ncols);
        double boundaryValues(Vec3& v);

    void setBoundaryCondition(int bc_selection) { boundary_condition = bc_selection; } 
    void setUseDiscreteRHS(bool useDiscrete) { use_discrete_rhs = useDiscrete; } 
    void setUseUniformDiffusivity(bool useUniform) { use_uniform_diffusivity = useUniform; }

};

#endif
