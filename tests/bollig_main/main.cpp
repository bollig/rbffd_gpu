// An idea for refactoring and cleaning up the code. 
// I want to have clear distinction between classes and work with factories to consume collections of objects
// The main should ONLY call other classes in the framework library. Any advanced logic here will not be avialable
// outside the test. 

#include <iostream>

#include "density.h"

//----------------------------------------------------------------------
int main (int argc, char** argv) {
	// 1) Generate a point cloud of random points in nD (param1)
	PointCloud pc = new PointCloud(2); 

	// 2) Generate a density function for the CVT (note the inheritence)
	DensityFunc density = new SubClassedDensityFunc(); 

	// 3) Reorganize point cloud using CVT based on density (param1)	
	pc->restructureCVT(density); 

	// 4) Consume point cloud and generate stencils given radius and max stencil size	
	vector<Stencil> stencils = new StencilFactory(pc).getStencilList(radius, stencil_size); 

	// 5) Consume stencils and generate distance matrix (A)
	vector<DistanceMatrix> dms = new DistanceMatrixFactory(stencils)->getDistanceMatrices()
	
	// 6) Consume distance matrices and precompute derivatives
	vector<Derivative> derivX = new DerivativeFactory(dms)->getDerivative(DerivativeFactory.X_DERIVATIVE); 
	
	// 7) Apply derivatives to get contributions
	vector<Contribution> contrib = applyDerivatives(derivX, funcValues); 
	
	// 8) Timestep
	oldFuncValues = funcValues; 
	funcValues += dt*contrib;	// Euler method 
	
	return EXIT_SUCCESS; 
}
