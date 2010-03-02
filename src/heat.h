#ifndef _HEAT_H_
#define _HEAT_H_

#include <vector>
#include <ArrayT.h>
#include "communicator.h"
#include "gpu.h"

class Grid;
class Derivative;


class Heat
{
private:
	// time step
	double dt; 

	// initial solution and forcing term
	double PI;
	double freq;
	double decay;

	// solution to heat equation
	// How to create an array of vector<double> ?
	std::vector<double> sol[2];


	//Grid& grid;
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

	// Physical domain
	// elliptical boundary
	double major, maji2, maji4;
	double minor, mini2, mini4;

	int id; 		// Comm rank or comm id

public:
	Heat(std::vector<Vec3>* rb_centers_, int stencil_size, std::vector<int>* global_boundary_nodes_, Derivative* der_, int rank);
	// Constructor #2:
	Heat(GPU* subdomain_, Derivative* der_, int rank);
	Heat(Grid& grid_, Derivative& der_);
	~Heat();

	// set the time step
	void setDt(double dt) {
		this->dt = dt;
	}

	// Advance the equation one time step using the GPU class to perform communication
	// Depends on Constructor #2 to be used so that a GPU class exists within this class.
	void advanceOneStepWithComm(Communicator* comm_unit);

	// Only update the updated_solution vector if it is non-null (i.e. we actually pass something to the routine)
	void advanceOneStep(std::vector<double>* updated_solution = NULL);
	void advanceOneStepDivGrad();

// div(D grad)T =  grad(D).grad(T) + D lapl(T)
	void advanceOneStepTwoTerms();

	// Only update the updated_solution vector if it is non-null (i.e. we actually pass something to the routine)
	void initialConditions(std::vector<double>* solution = NULL);

	std::vector<Vec3>* getRbfCenters()
		{ return rbf_centers; }

	double maxNorm();
	double maxNorm(std::vector<double> sol);
	double exactSolution(double x, double y, double t);
	void computeDiffusion(std::vector<double>& sol);

	// forcing term to force an exact solution
	// we are solving: d/dt(T) = lapl(T) + force
	double force(double x, double y, double t);
};

#endif
