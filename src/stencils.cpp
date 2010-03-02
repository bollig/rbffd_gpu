#include "stencils.h"
using namespace arma;

// returns the distance matrix where each element is the square of the distance between elements of the two arguments
mat Stencils::computeDistMatrix2(mat& x1, mat& x2)
{
	int dim = x1.n_cols;
	int nb  = x1.n_rows;

	mat dist(nb, nb);

	for (int j=0; j < nb; j++) {
	for (int i=0; i < nb; i++) {
		dist(i,j) = (x1(i,0)-x2(j,0))*(x1(i,0)-x2(j,0)) + 
		            (x1(i,1)-x2(j,1))*(x1(i,1)-x2(j,1));
	}}

	return dist;
}
//----------------------------------------------------------------------
ArrayT<Vec3> Stencils::computeDistMatrixVec(arma::mat& x1, arma::mat& x2)
{
	int dim = x1.n_cols;
	int nb =  x1.n_rows;
	ArrayT<Vec3> dist(nb, nb);

// When computing a derivative, one must consider the distance matrix (first row) as
// a function dist(x-x0) . Thus the stencil node coordinates are the second column. 

	for (int j=0; j < nb; j++) {
	for (int i=0; i < nb; i++) {
		dist(i,j) = Vec3(x1(j,0)-x2(i,0), x1(j,1)-x2(i,1));
	}}

	return dist;
}
//----------------------------------------------------------------------
