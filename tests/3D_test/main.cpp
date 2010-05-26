// Generate rbfs in a spherical shell

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <getopt.h>		// for getopt
#include <Vec3i.h>
#include <Vec3.h>
#include <ArrayT.h>
#include <vector> 
#include <algorithm> 
#include <functional> 
//#include <write_lineset.h>
//#include <write_psi.h>

#include <armadillo>
#include "gpu.h"
#include "rbf_mq.h"
//#include "rbf_gaussian.h"
#include "derivative.h"
#include "grid.h"
#include "heat.h"
#include "density.h"
#include "cvt.h"
#include "communicator.h"
#include "exact_ellipsoid.h"
#include "ellipsoid_patch.h"

// used go generate random seed that changes between runs
#include <time.h> 

using namespace std;
using namespace arma;

typedef ArrayT<double> AF;

// I need a datastructure  (i,x,y)

#define OTHER 0
#define CENTER 1
#define STENCIL 2

struct Dist {
	int i; // rbf index
	double d; // distance
	double x, y, z;
	// 0: other, 1: center, 2: stencil
	int id; // center, stencil or other
};

//----------------------------------------------------------------------
// not sure what this does any longer
//----------------------------------------------------------------------

// GLOBAL VARIABLES
CVT* cvt;
vector<Vec3> rbf_centers;
void checkDerivatives(Derivative& der, Grid& grid);
void checkXDerivatives(Derivative& der, Grid& grid);
double computeBoundaryIntegral(Density& rho, int npts, vector<double>& intg);
void computeBoundaryPointDistribution(double tot_length, int npts, int nb_bnd,
		vector<double> intg, vector<Vec3>& bnd);
double minimum(vector<double>& vec);
//double major;
//double minor;

double princ_axis1; 
double princ_axis2; 
double princ_axis3; 

vector<double> avgDist;

enum TESTFUN {
	C = 0, X, Y, X2, XY, Y2, X3, X2Y, XY2, Y3
};

//----------------------------------------------------------------------
double random(double a, double b) {
	// use system version of random, not class version
	double r = ::random() / (double) RAND_MAX;
	return a + r * (b - a);
}
//----------------------------------------------------------------------
double l1norm(vector<double>& v1, vector<double>& v2, int n1, int n2) {
	double norm = 0;
	double err;
	double elemt;

	for (int i = n1; i < n2; i++) {
		err = abs(v1[i] - v2[i]);
		// if n1 == 0, we are on the boundary
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i] * avgDist[i];
		norm += elemt * abs(err);
	}

	return norm / (n2 - n1);
}
//-------
double l1norm(vector<double>& v1, int n1, int n2) {
	double norm = 0;
	double elemt;

	for (int i = n1; i < n2; i++) {
		// if n1 == 0, we are on the boundary
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i] * avgDist[i];
		norm += elemt * abs(v1[i]);
	}

	return norm / (n2 - n1);
}
//----------------------------------------------------------------------

double l2norm(vector<double>& v1, vector<double>& v2, int n1, int n2) {
	double norm = 0;
	double err;
	double elemt;

	for (int i = n1; i < n2; i++) {
		err = abs(v1[i] - v2[i]);
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i] * avgDist[i];
		norm += elemt * err * err;
	}

	return sqrt(norm / (n2 - n1));
}
//----------------------------------------------------------------------
double l2norm(vector<double>& v1, int n1, int n2) {
	double norm = 0;
	double elemt;

	for (int i = n1; i < n2; i++) {
		elemt = (n1 == 0) ? avgDist[i] : avgDist[i] * avgDist[i];
		norm += elemt * v1[i] * v1[i];
	}

	return sqrt(norm / (n2 - n1));
}
//----------------------------------------------------------------------

double linfnorm(vector<double>& v1, vector<double>& v2, int n1, int n2) {
	double norm = -1.e10;
	double err;

	for (int i = n1; i < n2; i++) {
		err = abs(v1[i] - v2[i]);
		norm = (norm < err) ? err : norm;
	}
	return norm;
}
//----------------------------------------------------------------------
double linfnorm(vector<double>& v1, int n1, int n2) {
	double norm = -1.e10;

	for (int i = n1; i < n2; i++) {
		norm = (norm < abs(v1[i])) ? abs(v1[i]) : norm;
	}
	return norm;
}
//----------------------------------------------------------------------
#if 0
void testEigen(int stencil_size, int nb_bnd, int tot_nb_pts) {
	// read input file
	// compute stencils (do this only

	double pert = 0.05;
	vector<double> u(tot_nb_pts);
	vector<double> lapl_deriv(tot_nb_pts);

	int nx = 20;
	int ny = 20;
	// need another constructor for ellipses
	Grid grid(nx, ny, stencil_size);

	grid.setMajor(major);
	grid.setMinor(minor);
	grid.setNbBnd(nb_bnd);

	// 2nd argument: known number of boundary points (stored ahead of interior points) 
	grid.generateGrid("cvt_circle.txt", nb_bnd, tot_nb_pts);

	grid.computeStencils(); // nearest nb_points
	grid.avgStencilRadius();
	vector<double> avg_stencil_radius = grid.getAvgDist(); // get average stencil radius for each point

	vector<vector<int> >& stencil = grid.getStencil();

	// global variable
	rbf_centers = grid.getRbfCenters();
	int nb_rbf = rbf_centers.size();

	Derivative der(rbf_centers, stencil, grid.getNbBnd());
	der.setAvgStencilRadius(avg_stencil_radius);

	// Set things up for variable epsilon

	int nb_rbfs = rbf_centers.size();
	vector<double> epsv(nb_rbfs);

	for (int i = 0; i < nb_rbfs; i++) {
		//epsv[i] = 1. / avg_stencil_radius[i];
		epsv[i] = 1.; // fixed epsilon
		//printf("avg rad(%d) = %f\n", i, avg_stencil_radius[i]);
	}
	double mm = minimum(avg_stencil_radius);
	printf("min avg_stencil_radius= %f\n", mm);

	der.setVariableEpsilon(epsv);

	// Laplacian weights with zero grid perturbation
	for (int irbf = 0; irbf < rbf_centers.size(); irbf++) {
		der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "lapl");
	}

	//	double max_eig = der.computeEig(); // needs lapl_weights
	//	printf("zero perturbation: max eig: %f\n", max_eig);

	vector<Vec3> rbf_centers_orig;
	rbf_centers_orig.assign(rbf_centers.begin(), rbf_centers.end());

	double percent = 0.05; // in [0,1]
	printf("percent distortion of original grid= %f\n", percent);

	// set a random seed
	srandom(time(0));

	for (int i = 0; i < 100; i++) {
		printf("---- iteration %d ------\n", i);
		//update rbf centers by random perturbations at a fixed percentage of average radius computed
		//based on the unperturbed mesh
		rbf_centers.assign(rbf_centers_orig.begin(), rbf_centers_orig.end());

		for (int j = 0; j < nb_rbfs; j++) {
			Vec3& v = rbf_centers[j];
			double vx = avg_stencil_radius[j] * percent * random(-1., 1.);
			double vy = avg_stencil_radius[j] * percent * random(-1., 1.);
			v.setValue(v.x() + vx, v.y() + vy);
		}

		//rbf_centers[10].print("rbf_centers[10]");
		//continue;

		//recompute Laplace weights
		for (int irbf = 0; irbf < rbf_centers.size(); irbf++) {
			der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "lapl");
		}
		//	double max_eig = der.computeEig(); // needs lapl_weights
		//	printf("zero perturbation: max eig: %f\n", max_eig);
	}
}
//----------------------------------------------------------------------
void testFunction(TESTFUN which, vector<double>& u, vector<double>& dux_ex,
		vector<double>& duy_ex, vector<double>& dulapl_ex) {
	u.resize(0);
	dux_ex.resize(0);
	duy_ex.resize(0);
	dulapl_ex.resize(0);

	//vector<Vec3>& rbf_centers = grid.getRbfCenters();
	int nb_rbf = rbf_centers.size();

	switch (which) {
	case C:
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(1.);
			dux_ex.push_back(0.);
			duy_ex.push_back(0.);
			dulapl_ex.push_back(0.);
		}
		break;
	case X:
		printf("nb_rbf= %d\n", nb_rbf);
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(v.x());
			dux_ex.push_back(1.);
			duy_ex.push_back(0.);
			dulapl_ex.push_back(0.);
		}
		printf("u.size= %d\n", (int) u.size());
		break;
	case Y:
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(v.y());
			dux_ex.push_back(0.);
			duy_ex.push_back(1.);
			dulapl_ex.push_back(0.);
		}
		break;
	case X2:
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(v.x() * v.x());
			dux_ex.push_back(2. * v.x());
			duy_ex.push_back(0.);
			dulapl_ex.push_back(2.);
		}
		break;
	case XY:
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(v.x() * v.y());
			dux_ex.push_back(v.y());
			duy_ex.push_back(v.x());
			dulapl_ex.push_back(0.);
		}
		break;
	case Y2:
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(v.y() * v.y());
			dux_ex.push_back(0.);
			duy_ex.push_back(2. * v.y());
			dulapl_ex.push_back(2.);
		}
		break;
	case X3:
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(v.x() * v.x() * v.x());
			dux_ex.push_back(3. * v.x() * v.x());
			duy_ex.push_back(0.);
			dulapl_ex.push_back(6. * v.x());
		}
		break;
	case X2Y:
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(v.x() * v.x() * v.y());
			dux_ex.push_back(2. * v.x() * v.y());
			duy_ex.push_back(v.x() * v.x());
			dulapl_ex.push_back(2. * v.y());
		}
		break;
	case XY2:
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(v.x() * v.y() * v.y());
			dux_ex.push_back(v.y() * v.y());
			duy_ex.push_back(2. * v.x() * v.y());
			dulapl_ex.push_back(2. * v.x());
		}
		break;
	case Y3:
		for (int i = 0; i < nb_rbf; i++) {
			Vec3& v = rbf_centers[i];
			u.push_back(v.y() * v.y() * v.y());
			dux_ex.push_back(0.);
			duy_ex.push_back(3. * v.y() * v.y());
			dulapl_ex.push_back(6. * v.y());
		}
		break;
	}
}

//----------------------------------------------------------------------
void testDeriv(TESTFUN choice, Derivative& der, Grid& grid) {
	printf("================\n");
	printf("testderiv: choice= %d\n", choice);

	//vector<Vec3>& rbf_centers = grid.getRbfCenters();
	int nb_rbf = rbf_centers.size();
	//printf("rbf_center size: %d\n", nb_rbf); exit(0);

	vector<double> u;
	vector<double> dux_ex; // exact derivative
	vector<double> duy_ex; // exact derivative
	vector<double> dulapl_ex; // exact derivative

	// change the classes the variables are located in
	vector<double> xderiv(nb_rbf);
	vector<double> yderiv(nb_rbf);
	vector<double> lapl_deriv(nb_rbf);

	testFunction(choice, u, dux_ex, duy_ex, dulapl_ex);

	avgDist = grid.getAvgDist();

	for (int n = 0; n < 1; n++) {
		// perhaps I'll need different (rad,eps) for each. To be determined. 
		der.computeDeriv(Derivative::X, u, xderiv);
		der.computeDeriv(Derivative::Y, u, yderiv);
		der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
	}

	vector<int>& boundary = grid.getBoundary();
	int nb_bnd = boundary.size();

	enum DERIV {
		X = 0, Y, LAPL
	};
	enum NORM {
		L1 = 0, L2, LINF
	};
	enum BNDRY {
		INT = 0, BNDRY
	};
	double norm[3][3][2]; // norm[DERIV][NORM][BNDRY]
	double normder[3][3][2]; // norm[DERIV][NORM][BNDRY]

	norm[X][L1][BNDRY] = l1norm(dux_ex, xderiv, 0, nb_bnd);
	norm[X][L2][BNDRY] = l2norm(dux_ex, xderiv, 0, nb_bnd);
	norm[X][LINF][BNDRY] = linfnorm(dux_ex, xderiv, 0, nb_bnd);

	norm[Y][L1][BNDRY] = l1norm(duy_ex, yderiv, 0, nb_bnd);
	norm[Y][L2][BNDRY] = l2norm(duy_ex, yderiv, 0, nb_bnd);
	norm[Y][LINF][BNDRY] = linfnorm(duy_ex, yderiv, 0, nb_bnd);

	norm[LAPL][L1][BNDRY] = l1norm(dulapl_ex, lapl_deriv, 0, nb_bnd);
	norm[LAPL][L2][BNDRY] = l2norm(dulapl_ex, lapl_deriv, 0, nb_bnd);
	norm[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex, lapl_deriv, 0, nb_bnd);

	norm[X][L1][INT] = l1norm(dux_ex, xderiv, nb_bnd, nb_rbf);
	norm[X][L2][INT] = l2norm(dux_ex, xderiv, nb_bnd, nb_rbf);
	norm[X][LINF][INT] = linfnorm(dux_ex, xderiv, nb_bnd, nb_rbf);

	norm[Y][L1][INT] = l1norm(duy_ex, yderiv, nb_bnd, nb_rbf);
	norm[Y][L2][INT] = l2norm(duy_ex, yderiv, nb_bnd, nb_rbf);
	norm[Y][LINF][INT] = linfnorm(duy_ex, yderiv, nb_bnd, nb_rbf);

	norm[LAPL][L1][INT] = l1norm(dulapl_ex, lapl_deriv, nb_bnd, nb_rbf);
	norm[LAPL][L2][INT] = l2norm(dulapl_ex, lapl_deriv, nb_bnd, nb_rbf);
	norm[LAPL][LINF][INT] = linfnorm(dulapl_ex, lapl_deriv, nb_bnd, nb_rbf);

	// --- Normalization factors

	normder[X][L1][BNDRY] = l1norm(dux_ex, 0, nb_bnd);
	normder[X][L2][BNDRY] = l2norm(dux_ex, 0, nb_bnd);
	normder[X][LINF][BNDRY] = linfnorm(dux_ex, 0, nb_bnd);

	normder[Y][L1][BNDRY] = l1norm(duy_ex, 0, nb_bnd);
	normder[Y][L2][BNDRY] = l2norm(duy_ex, 0, nb_bnd);
	normder[Y][LINF][BNDRY] = linfnorm(duy_ex, 0, nb_bnd);

	normder[LAPL][L1][BNDRY] = l1norm(dulapl_ex, 0, nb_bnd);
	normder[LAPL][L2][BNDRY] = l2norm(dulapl_ex, 0, nb_bnd);
	normder[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex, 0, nb_bnd);

	normder[X][L1][INT] = l1norm(dux_ex, nb_bnd, nb_rbf);
	normder[X][L2][INT] = l2norm(dux_ex, nb_bnd, nb_rbf);
	normder[X][LINF][INT] = linfnorm(dux_ex, nb_bnd, nb_rbf);

	normder[Y][L1][INT] = l1norm(duy_ex, nb_bnd, nb_rbf);
	normder[Y][L2][INT] = l2norm(duy_ex, nb_bnd, nb_rbf);
	normder[Y][LINF][INT] = linfnorm(duy_ex, nb_bnd, nb_rbf);

	normder[LAPL][L1][INT] = l1norm(dulapl_ex, nb_bnd, nb_rbf);
	normder[LAPL][L2][INT] = l2norm(dulapl_ex, nb_bnd, nb_rbf);
	normder[LAPL][LINF][INT] = linfnorm(dulapl_ex, nb_bnd, nb_rbf);

	printf("--------\n");
	printf("norm[x/y/lapl][L1,L2,LINF][interior/bndry]\n");
	for (int k = 0; k < 2; k++) {
		for (int i = 0; i < 3; i++) {
			if (abs(normder[i][1][k]) < 1.e-9) {
				printf("(abs err): norm[%d][][%d]= %10.3e, %10.3e, %10.3e\n",
						i, k, norm[i][0][k], norm[i][1][k], norm[i][2][k]);
			} else {
				printf("(rel err): norm[%d][][%d]= %10.3e, %10.3e, %10.3e\n",
						i, k, norm[i][0][k] / normder[i][0][k], norm[i][1][k]
								/ normder[i][1][k], norm[i][2][k]
								/ normder[i][2][k]);
				//printf("   normder[%d][][%d]= %10.3e, %10.3e, %10.3e\n", i, k,
				//normder[i][0][k],
				//normder[i][1][k],
				//normder[i][2][k]);
			}
		}
	}

	double inter_error = 0.;
	//vector<int>& boundary = grid.getBoundary();

	for (int i = (int) boundary.size(); i < nb_rbf; i++) {
		inter_error += (dulapl_ex[i] - lapl_deriv[i]) * (dulapl_ex[i]
				- lapl_deriv[i]);
		//printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
	}
	inter_error /= (nb_rbf - boundary.size());

	double bnd_error = 0.;
	for (int ib = 0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		bnd_error += (dulapl_ex[i] - lapl_deriv[i]) * (dulapl_ex[i]
				- lapl_deriv[i]);
		//printf("bnd error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
	}
	bnd_error /= boundary.size();

	printf("avg l2_bnd_error= %14.7e\n", sqrt(bnd_error));
	printf("avg l2_interior_error= %14.7e\n", sqrt(inter_error));
}
#endif 
//----------------------------------------------------------------------
double minimum(vector<double>& vec) {
	double min = 1.e10;

	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] < min) {
			min = vec[i];

		}
	}
	return min;
}
//----------------------------------------------------------------------
struct mySort {
	bool operator()(int i, int j) {
		return i < j;
	}
	bool operator()(const Dist& d1, const Dist& d2) {
		return d1.d < d2.d;
	}
} myObject;
//----------------------------------------------------------------------
void createSubgrid(int nx, int ny, int nz,
		ArrayT<vector<int> >** subgrid_index_list)
// overlay a 2D grid of size (nx,ny). RBFs are thus in one and 
// only one subgrid. Neighbors of a points will be located in 
// its subcell and those surrounding it. 
{
	// each element represents one subgrid. List of nodes in subgrid. 

	*subgrid_index_list = new ArrayT<vector<int> > (nx, ny, nz);
}
//----------------------------------------------------------------------
GPU* distributeNodesAcrossGPUs(Grid *grid, Communicator* comm_unit, double dt) {
#if 1
	// Assume a grid of (gx, gy) GPUs. 
	// MPI can create a Cartesian grid for us (TODO)
	// MPI_Cart_create()
	// 
	int gx = comm_unit->getSize();
	int gy = 1; // Update this for Cart MPI
	vector<GPU*> gpus;
	gpus.resize(gx * gy);

	// The GPU class partitions the points as if they are in [-1,1]x[-1,1]
	// We should generalize this. 
	int xmin = -1;
	int xmax = 1;
	int ymin = -1;
	int ymax = 1;

	double deltax = (double) (xmax - xmin) / (double) gx;
	double deltay = (double) (ymax - ymin) / (double) gy;

	printf("delta gpu x, y= %f, %f\n", deltax, deltay);

	// Initialize GPU datastructures
	for (int id = 0; id < gx * gy; id++) {
		int igy = id / gx; // Row
		int igx = id - igy * gx;
		printf("igx = %d, igy = %d\n", igx, igy);
		double xm = xmin + igx * deltax;
		double ym = ymin + igy * deltay;
		printf("xm= %f, ym= %f\n", xm, ym);
		gpus[id] = new GPU(xm, xm + deltax, ym, ym + deltay, /* zmin */ 0.0, /* zmax */ 0.0, dt, id,
				comm_unit->getSize());
	}

	// Figure out the sets Bi, Oi Qi

	printf("nb gpus: %d\n", (int) gpus.size());
	for (int i = 0; i < gpus.size(); i++) {
		printf("\n ***************** CPU %d ***************** \n", i);
		gpus[i]->fillLocalData(grid->getRbfCenters(), grid->getStencil(),
				grid->getBoundary(), grid->getAvgDist()); // Forms sets (Q,O,R) and l2g/g2l maps
		gpus[i]->fillVarData(grid->getRbfCenters()); // Sets function values in U
	}

	for (int i = 0; i < gpus.size(); i++) {
		printf(
				"\n ***************** FILLING O_by_rank for CPU%d ***************** \n",
				i);
		for (int j = 0; j < gpus.size(); j++) {
			gpus[i]->fillDependencyList(gpus[j]->R, j); // appends to O_by_rank	any nodes required by gpu[j]
		}

	}

	printf("gpu structures (Q\\O,O,R) are initialized\n");
	printf("initialized on scalar variable to linear function\n");

	// Distribute nodes to each GPU. 
	for (int i = 1; i < gpus.size(); i++) {
		printf("Distributing to GPU[%d]\n", i);
		comm_unit->sendObject(gpus[i], i);
	}

	return gpus[0];

	// Compute derivative on a single GPU. Check against analytical result
	// du/dx=1, du/dx=2, du/dx=3
#if 0
	RBF_Gaussian rbf(1.);
	const Vec3 xi(.5,0.,0.);
	for (int i=0; i < 10; i++) {
		const Vec3 xvec(i*.1, 0.,0.);
		printf("%d, phi=%f, phi'=%f\n", i, rbf.eval(xvec, xi), rbf.xderiv(xvec,xi));
	}
#endif
#endif
}

//----------------------------------------------------------------------
// EVAN : ADD princ_axis3
double computeDomainIntegral(int npts, Density& rho) {
	// what is the surface element?
	// Ellipse: 
	// y1 = b*sqrt[1. - x^2 / a^2]  
	// integration limit: -y1 to y1. x limits: [-a,a]. 
	// area element: dx*dy

	// We choose dx proportional to the principal axis. Assuming princ_axis1 
	// is <= 1.0 and the maximum principal axis == 1.0
	// The factor 2 in front here is the number of dimensions npts needs to fill
	double dx = 2. * princ_axis1 / (npts - 1);
	double integ = 0.;
	
	// Midpoint method over xa = [-1, 1]
	for (int i = 0; i < (npts - 1); i++) {
		// If dx = 2 * alpha / (n-1) 
		// Then xa = -1 + (2i + 1) / (n - 1)
		double xa = -1. + (i + 0.5) * dx / princ_axis1;	
		// axis2 * (% of axis2 at xa) 		{% meaning we assume hypotenus of 1 and xa on one side, what is }
		double y1 = -princ_axis2 * sqrt(1. - xa * xa);
		double dy = 2. * fabs(y1) / (npts - 1);
		for (int j = 0; j < (npts - 1); j++) {
			// Get physical point (x,y) so we can use density at that location
			double x = xa * princ_axis1;
			double y = y1 + (j + 0.5) * dy;
			integ += sqrt(rho(x, y)) * dx * dy;
		}
		printf("Checking Integral: xa=%f\t y1=%f\t dy=%f\n", xa, y1, dy); 
	}
	
	int npts_x = (npts-1) / 2; 
	int npts_y = (npts-1) / 2;
	
	printf("NPTS_X = %d %f\n", npts_x, (npts-1)/(2.*princ_axis2));
	 
	// Iterate over all points in x plane
	for (int i = 0; i < (npts - 1); i++) {
		// i + 0.5 gets us to midpoint
		double x = -1. + (i + 0.5) * dx;
		double x_div_a = x / princ_axis1; 
		
		// Iterate over all points in y plane
		for (int j = 0; j < (npts-1); j++) {
			
			
		}
		
	}
	
	
	printf("Checking Integral: %f \t PI*Major*Minor = %f\n", integ, M_PI * princ_axis1 * princ_axis2);

	return integ;
}
//----------------------------------------------------------------------
// EVAN : ADD princ_axis3
double computeBoundaryIntegral(Density& rho, int npts, vector<double>& intg)
// npts: number of boundary points (first and last points are the same for closed intervals)
// rho: functor computing point density (= 1 + some function(x,y))
// bnd: list of boundary points
// return: value of boundary integral
{

	// number of points should be large. Ideally, I should be integrating with respect to theta for 
	// more accuracy. 

	double pi = acos(-1.);
	double dtheta = 2. * pi / (npts - 1.);

	//npts = 2000;

	//vector<double> intg;
	intg.resize(npts);

	intg[0] = 0.;

	// npts-1 is the number of intervals

	for (int i = 0; i < (npts - 1); i++) {
		double t1 = i * dtheta;
		double t2 = (i + 1) * dtheta;
		//printf("t1,t2= %f, %f\n", t1, t2);
		double tm = 0.5 * (t1 + t2);
		double x1 = princ_axis1 * cos(t1);
		double x2 = princ_axis1 * cos(t2);
		double xm = princ_axis1 * cos(tm);
		double y1 = princ_axis2 * sin(t1);
		double y2 = princ_axis2 * sin(t2);
		double ym = princ_axis2 * sin(tm);
		double dl = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
		double rhom = rho(xm, ym);
		intg[i + 1] = intg[i] + pow(rhom, 0.25) * dl;
	}

	// new boundary will have n points (n << npts)
	// divide into (n-1) equal length intervals
	double tot_length = intg[npts - 1];

	printf("boundary integral: %f\n", tot_length);

	return tot_length;
}


//----------------------------------------------------------------------
// Evan: ADD SUPPORT FOR PHI
void computeBoundaryPointDistribution(double tot_length, int npts, int nb_bnd,
		vector<double> intg, vector<Vec3>& bnd) {
	double tot_intv = tot_length / (npts - 1.);
	vector<double> equ_dist, theta, phi;
	bnd.resize(0);

	int n = nb_bnd + 1; // space so that first and last point are the same
	double pi = acos(-1.);
	double dtheta = 2. * pi / (npts - 1.);
	printf("npts= %d, n= %d\n", npts, n);

	equ_dist.resize(n);
	theta.resize(n);
	phi.resize(n);

	double intv_length = tot_length / (n - 1);
	for (int i = 0; i < (n - 1); i++) {
		equ_dist[i] = i * intv_length;
		//theta[i] = i*dtheta;
	}
	equ_dist[n - 1] = tot_length;

	printf("tot_length= %f, intg[npts-1]= %f\n", tot_length, intg[npts - 1]);

	// Compute theta distribution of new points 

	// Brute force O(n*npts)
	// Should rewrite to be O(npts)
	//double dthetaj = 2.*pi / (n-1);

	// EVAN : fill phi
	theta[0] = 0.;
	for (int i = 1; i < (n - 1); i++) {
		theta[i] = -1.;
		printf("-----i= %d------\n", i);
		for (int j = 1; j < npts; j++) { // npts >> n
			// find interval that contains equ_dist[i]
			// intg[j] <= equ_dist[i] <= intg[j]
			if ((equ_dist[i] <= intg[j]) && equ_dist[i] >= intg[j - 1]) {
				//printf("i=%d, j/npts= %d/%d, equ_dist[%d]= %f, intg= %f, %f\n", i, j, npts, i, equ_dist[i], intg[j-1], intg[j]);
				double th = (j - 1) * dtheta;
				double dth = dtheta * (equ_dist[i] - intg[j - 1]) / (intg[j]
						- intg[j - 1]);
				//printf("dtheta= %f, th= %f, dth= %f\n", dtheta, th, dth);
				theta[i] = th + dth;
				break;
			}
		}
	}
	theta[n - 1] = 2. * pi;
	//exit(0);

	vector<double> x, y, z;
	x.resize(nb_bnd);
	y.resize(nb_bnd);
	z.resize(nb_bnd);

	for (int i = 0; i < nb_bnd; i++) {
		if (theta[i] < 0.) {
			printf("Equipartitioning of boundary is incomplete\n");
			exit(0);
		}
		#if 1
		// 2D: 
		x[i] = princ_axis1 * cos(theta[i]);
		y[i] = princ_axis2 * sin(theta[i]);
		#else 
		// Evan (DONE)
		x[i] = princ_axis1 * sin(phi[i]) * cos(theta[i]); 
		y[i] = princ_axis2 * sin(phi[i]) * sin(theta[i]); 
		z[i] = princ_axis3 * cos(phi[i]); 
		#endif 
		
		
		bnd.push_back(Vec3(x[i], y[i], z[i]));
		//printf("(%d) x,y= %f, %f, theta= %f\n", i, x[i], y[i], theta[i]);
	}

	printf("print length intervals: should be equal\n");

#if 0
	for (int i=0; i < (n-1); i++) {
		double dx = (x[i+1]-x[i]);
		double dy = (y[i+1]-y[i]);
		double dl = sqrt(dx*dx + dy*dy);
		//printf("dl[%d]= %f\n", i, dl);
	}
#endif

	printf("Weighted ellipse perimeter: %f\n", tot_length);
}
//----------------------------------------------------------------------
// Distribute points along the boundary
//----------------------------------------------------------------------
void createCVT(int N, int nb_bnd, Density& rho, vector<Vec3>& bndry_pts,
		double dom_intg) {
	//# define N 200
# define DIM_NUM 2
	int batch;
	bool comment;
	double energy;
	char file_out_name[80] = "cvt_circle.txt";
	int init;
	char init_string[80];
	double it_diff;
	int it_fixed;
	int it_max;
	int it_num;
	double r[DIM_NUM * N];
	int sample;
	int sample_num;
	char sample_string[80];
	int seed;
	int seed_init;

	batch = 1000;
	init = 4; // user initialize boundary


	strcpy(init_string, "user");
	it_max = 1000;
	it_fixed = 1;
	sample = 3;
	sample_num = 10000; // 10k for 300 pts, 30k for 1000 pts)
	strcpy(sample_string, "user");
	seed = 123456789;

	seed_init = seed;
	cvt->setNbBnd(nb_bnd);
	cvt->setNbPts(N);
	cvt->setDensity(&rho);
	
	// EVAN : ADD SUPPORT FOR princ_axis3
	cvt->setEllipseAxes(princ_axis1, princ_axis2);
	cvt->setBoundaryPts(bndry_pts);

#if 0
	for (int i=0; i < nb_bnd; i++) {
		bndry_pts[i].print("bndry");
	}
	bndry_pts[nb_bnd].print("bndry");
	//exit(0);
#endif

	// Initialize the seeds in the interior of the 2D Ellipse
	// EVAN: (should do 3D as well)
	cvt->ellipse_init(DIM_NUM, N, nb_bnd, &seed, r);
	cvt->cvt(DIM_NUM, N, batch, init, sample, sample_num, it_max, it_fixed,
			&seed, r, &it_num, &it_diff, &energy);

	comment = false; // comment lines at the top of the output file

	cvt->cvt_write(DIM_NUM, N, batch, seed_init, seed, init_string, it_max,
			it_fixed, it_num, it_diff, energy, sample_string, sample_num, r,
			file_out_name, comment);

	return;

# undef DIM_NUM
# undef N
}
//----------------------------------------------------------------------
void computeDistances() {
#if 0
	// step 1: create code on a single CPU
	// step 2: create code on a single GPU
	// step 3: create partitioning of RBFs
	// step 3a: calculate domains Q, O, B for each GPU

	exit(0);

	printf("nb_rbf= %d\n", nb_rbf);
	//exit(0);

	vector<Dist> dist; //  1000 rbfs.
	double distance = 1.e10;
	double min_dist = 1.e10;
	int ix=-1;

	// compute point closest to R=fmax-2.;
	double rfind = (rmax-2.);
	printf("rfind= %f\n", rfind);

	for (int i=0; i < nb_rbf; i++) {
		distance = coord(0,i)*coord(0,i)+coord(1,i)*coord(1,i)+coord(2,i)*coord(2,i) - rfind*rfind;
		if (distance < 0.) distance *= -1.;
		if (distance < min_dist) {
			min_dist = distance;
			ix = i;
		}
	}
	//printf("min_dist= %f, ix= %d\n", min_dist, ix); exit(0);
	printf("ix= %f, %f, %f\n", coord(0,ix), coord(1,ix), coord(2,ix));
	printf("dist= %f\n", coord(0,ix)*coord(0,ix)+coord(1,ix)*coord(1,ix)+coord(2,ix)*coord(2,ix));

	Dist tmp;

	// Compute all distances to point ix;
	for (int i=0; i < nb_rbf; i++) {
		double cx = coord(0,i) - coord(0,ix);
		double cy = coord(1,i) - coord(1,ix);
		double cz = coord(2,i) - coord(2,ix);
		tmp.i = i;
		tmp.d = cx*cx + cy*cy + cz*cz;
		tmp.x = coord(0,i);
		tmp.y = coord(1,i);
		tmp.z = coord(2,i);
		tmp.id = OTHER;
		dist.push_back(tmp);
	}
	printf("ix= %d\n", ix);

	// does not work
	//sort(dist.begin(), dist.end(), greater<int>());

	//works
	sort(dist.begin(), dist.end(), myObject);

	for (int i=0; i < 100; i++) {
		printf("dist[%d] = %f, %f, %f, %f\n", dist[i].i, dist[i].d, dist[i].x, dist[i].y, dist[i].z);
	}

	// Pick 30 nearest rbfs
	int nb_stencil = 30;
#endif
}
//----------------------------------------------------------------------
void checkDerivatives(Derivative& der, Grid& grid) {
	vector<Vec3>& rbf_centers = grid.getRbfCenters();
	int nb_rbf = rbf_centers.size();

	// change the classes the variables are located in
	vector<double> xderiv(rbf_centers.size());
	vector<double> yderiv(rbf_centers.size());
	vector<double> lapl_deriv(rbf_centers.size());

	printf("deriv size: %d\n", (int) rbf_centers.size());
	printf("xderiv size: %d\n", (int) xderiv.size());

	// function to differentiate
	vector<double> u;
	vector<double> du_ex; // exact Laplacian

	vector<vector<int> >& stencil = grid.getStencil();

#if 0
	for (int i=0; i < 1600; i++) {
		vector<int>& v = stencil[i];
		printf("stencil[%d]\n", i);
		for (int s=0; s < v.size(); s++) {
			printf("%d ", v[s]);
		}
		printf("\n");
	}
	exit(0);
#endif

	double s;

	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = rbf_centers[i];
		//s = 3.*v.x() + 2.*v.y(); //   + 3.*v.z();

		s = v.x() * v.y() + 0.5 * v.x() * v.x() + 0.5 * v.y() * v.y(); //   + 3.*v.z();
		du_ex.push_back(2.);

		//s = v.x()+v.y() +v.x()+v.y();
		//du_ex.push_back(0.);

		//s = v.x()*v.x()*v.x();
		//du_ex.push_back(6.*v.x());

		u.push_back(s);
	}

	printf("main, u[0]= %f\n", u[0]);

	for (int n = 0; n < 1; n++) {
		//der.computeDeriv(Derivative::X, u, xderiv);
		//der.computeDeriv(Derivative::Y, u, yderiv);
		der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
	}

#if 0
	for (int i=0; i < xderiv.size(); i++) {
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("(%d), (%f,%f), xder[%d]= %f, yderiv= %f\n", st.size(), v.x(), v.y(), i, xderiv[i], yderiv[i]);
	}
#endif

	// interior points

	for (int i = 0; i < lapl_deriv.size(); i++) {
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("(%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n",
				(int) st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
	}

	// boundary points
	vector<int>& boundary = grid.getBoundary();
	for (int ib = 0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		//printf("bnd (%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
	}

	double inter_error = 0.;
	for (int i = boundary.size(); i < nb_rbf; i++) {
		inter_error += (du_ex[i] - lapl_deriv[i]) * (du_ex[i] - lapl_deriv[i]);
		printf("inter error[%d] = %14.7g\n", i, du_ex[i] - lapl_deriv[i]);
	}
	inter_error /= (nb_rbf - boundary.size());

	double bnd_error = 0.;
	for (int ib = 0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		bnd_error += (du_ex[i] - lapl_deriv[i]) * (du_ex[i] - lapl_deriv[i]);
		printf("bnd error[%d] = %14.7g\n", i, du_ex[i] - lapl_deriv[i]);
	}
	bnd_error /= boundary.size();

	printf("avg l2_bnd_error= %14.7e\n", sqrt(bnd_error));
	printf("avg l2_interior_error= %14.7e\n", sqrt(inter_error));
}

//----------------------------------------------------------------------
void checkXDerivatives(Derivative& der, Grid& grid) {
	vector<Vec3>& rbf_centers = grid.getRbfCenters();
	int nb_rbf = rbf_centers.size();

	// change the classes the variables are located in
	vector<double> xderiv(rbf_centers.size());
	vector<double> yderiv(rbf_centers.size());
	vector<double> lapl_deriv(rbf_centers.size());

	printf("deriv size: %d\n", (int) rbf_centers.size());
	printf("xderiv size: %d\n", (int) xderiv.size());

	// function to differentiate
	vector<double> u;
	vector<double> du_ex; // exact derivative
	vector<double> dux_ex; // exact derivative
	vector<double> duy_ex; // exact derivative

	vector<vector<int> >& stencil = grid.getStencil();

#if 0
	for (int i=0; i < 1600; i++) {
		vector<int>& v = stencil[i];
		printf("stencil[%d]\n", i);
		for (int s=0; s < v.size(); s++) {
			printf("%d ", v[s]);
		}
		printf("\n");
	}
	exit(0);
#endif

	double s;

	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = rbf_centers[i];
		//s = 3.*v.x() + 2.*v.y(); //   + 3.*v.z();

		//s = v.x()*v.y() + 0.5*v.x()*v.x() + 0.5*v.y()*v.y(); //   + 3.*v.z();
		//du_ex.push_back(2.);

		s = v.x() * v.y();
		dux_ex.push_back(v.y());
		duy_ex.push_back(v.x());

		//s = v.y();
		du_ex.push_back(1.);

		u.push_back(s);
	}

	printf("main, u[0]= %f\n", u[0]);
	printf("main, u= %ld\n", (long int) &u[0]);

	for (int n = 0; n < 1; n++) {
		// perhaps I'll need different (rad,eps) for each. To be determined. 
		der.computeDeriv(Derivative::X, u, xderiv);
		der.computeDeriv(Derivative::Y, u, yderiv);
		der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
	}
	//	der.computeEig(); // needs lapl_weights (full global set), analyzes stability of Laplace operator

#if 1
	for (int i = 0; i < xderiv.size(); i++) {
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", (int) st.size(),
				v.x(), v.y(), i, xderiv[i], dux_ex[i]);
		printf("(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", (int) st.size(),
				v.x(), v.y(), i, yderiv[i], duy_ex[i]);
	}
#endif

	//exit(0);

	// interior points

#if 0
	for (int i=0; i < xderiv.size(); i++) {
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("(%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", (int) st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
		printf("(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
		printf("(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
	}
#endif

	// boundary points
	vector<int>& boundary = grid.getBoundary();
	for (int ib = 0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("bnd(%d) sz(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", ib,
				(int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
		printf("bnd(%d) sz(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", ib,
				(int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
		//printf("bnd (%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
	}
	exit(0);

	double inter_error = 0.;
	for (int i = boundary.size(); i < nb_rbf; i++) {
		inter_error += (du_ex[i] - lapl_deriv[i]) * (du_ex[i] - lapl_deriv[i]);
		printf("inter error[%d] = %14.7g\n", i, du_ex[i] - lapl_deriv[i]);
	}
	inter_error /= (nb_rbf - boundary.size());

	double bnd_error = 0.;
	for (int ib = 0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		bnd_error += (du_ex[i] - lapl_deriv[i]) * (du_ex[i] - lapl_deriv[i]);
		printf("bnd error[%d] = %14.7g\n", i, du_ex[i] - lapl_deriv[i]);
	}
	bnd_error /= boundary.size();

	printf("avg l2_bnd_error= %14.7e\n", sqrt(bnd_error));
	printf("avg l2_interior_error= %14.7e\n", sqrt(inter_error));

	exit(0);
}

void closeLogFile(void) {
	fprintf(stderr, "Closing STDOUT file\n");
	fclose(stdout);
}

void debugExit(void) {
	fprintf(stderr, "EXIT CALLED\n");
}

int parseCommandLineArgs(int argc, char** argv, int my_rank) {
	// Borrowed from Getopt-Long Example in GNU LibC manual
	int verbose_flag = 0; /* Flag set by '--verbose' and '--brief'. */
	int c;
	int hostname_flag = 0; // Non-zero if hostname is set;

	while (1) {
		// Struct order: { Long_name, Arg_required?, FLAG_TO_SET, Short_name}
		// NOTE: if an option sets a flag, it is set to the Short_name value
		static struct option long_options[] = {
		// NOTE: following are defined in getopt.h
				// 		no_argument			--> no arg to the option is expected
				//		required_argument 	--> arg is mandatory
				// 		optional_argument	--> arg is not necessary
				/* These options set a flag. */
				{ "verbose", no_argument, &verbose_flag, 1 }, { "brief",
						no_argument, &verbose_flag, 0 },
				/* These options don't set a flag.
				 We distinguish them by their indices. */
				{ "hostname", required_argument, &hostname_flag, 'h' }, {
						"output-file", required_argument, 0, 'o' }, { "file",
						required_argument, 0, 'o' }, { "dir",
						required_argument, 0, 'd' }, { "help", no_argument, 0,
						'?' }, { 0, 0, 0, 0 } };

		/* getopt_long stores the option index here. */
		int option_index = 0;

		// the : here indicates a required argument
		c = getopt_long(argc, argv, "?o:", long_options, &option_index);

		/* Detect the end of the options. */
		if (c == -1)
			break;

		switch (c) {
		case 0: // Any long option that sets a flag
			printf("option %s", long_options[option_index].name);
			if (optarg)
				printf(" with arg %s", optarg);
			printf("\n");
			break;

		case 'o':
			char logname[256];
			sprintf(logname, "%s.%d", optarg, my_rank);
			printf("Redirecting STDOUT to file: `%s'\n", logname);
			freopen(logname, "w", stdout);
			atexit(closeLogFile);
			break;

		case 'h':
			printf("[DISABLED] option -h with value `%s'\n", optarg);
			break;

		case 'd':
			// 1) copy name into logdir (global) variable = argv/RANK
			// 2) mkdir logdir if not already made
			// 3) set logdir in all classes that write files (Heat, Derivative)
			break;

		case '?':
			printf("\nUsage: %s [options=arguments]\n", argv[0]);
			printf("Options:\n");
			printf(
					"\t-o (--output-file, --file) \tSpecify the filename for process to redirect STDOUT to.\n");
			printf("\t-? (--help) \t\t\tPrint this message\n\n");
			exit(EXIT_FAILURE);
			break;

		default:
			printf("IN DEFAULT ARG OPTION (WHY?)\n");
			abort(); // abort loop when nothing is left
			break;
		}
	} // END WHILE

	/* Instead of reporting ‘--verbose’
	 and ‘--brief’ as they are encountered,
	 we report the final status resulting from them. */
	if (verbose_flag)
		printf("verbose flag is set\n");

	/* Print any remaining command line arguments (not options). */
	if (optind < argc) {
		printf("non-option ARGV-elements: ");
		while (optind < argc)
			printf("%s ", argv[optind++]);
		putchar('\n');
	}

	// Tell us whenever program terminates with an exit is called
	atexit(debugExit);

	return 0;
}

//----------------------------------------------------------------------
int main(int argc, char** argv) {
	Communicator* comm_unit = new Communicator(argc, argv);

	// An empty GPU object for each CPU. CPU0 (master) will
	// return this from distributeNodesAcrossGPUs while all
	// other CPUs get it from comm_unit->receiveObject()
	GPU* subdomain = NULL;

	Grid* grid = NULL;

	parseCommandLineArgs(argc, argv, comm_unit->getRank());

	int my_rank = comm_unit->getRank();

	cout << " Got Rank: " << comm_unit->getRank() << endl;
	cout << " Got Size: " << comm_unit->getSize() << endl;

	if (!my_rank) { // Master thread 0

		// domain dimensions
		double pi = acos(-1.);

#if 0
		major = 1.0;
		minor = 0.5;
		int minor2 = 0.5;
#else 
		princ_axis1 = 1.0; 
		princ_axis2 = 0.5; 
		princ_axis3 = 0.5;
#endif 

		Density rho;
		//printf("rho.getMax() = %f\n", rho.getMax());
		//exit(0);

		vector<Vec3> bndry_pts;
		vector<double> intg;

		int high_nb_pts = 200;
		double bnd_intg = computeBoundaryIntegral(rho, high_nb_pts, intg);
		double dom_intg = computeDomainIntegral(high_nb_pts, rho);

		// total nb points used to compute Voronoi mesh. 
		// Only (nb_interior_pts-nb_bnd) will be able to move freely
		int tot_nb_pts = 50;
		// number of boundary points, automatically calculated
		printf("tot_nb_pts= %d\n", tot_nb_pts);
		printf("dom_intg= %f\n", dom_intg);
		printf("bnd_intg= %f\n", bnd_intg);

		int nb_bnd = bnd_intg * sqrt(tot_nb_pts / dom_intg);
		int nb_bnd_1 = 1. + 16. * tot_nb_pts * dom_intg / (bnd_intg * bnd_intg);
		nb_bnd_1 = -bnd_intg * bnd_intg / (4. * dom_intg) * (1.
				- sqrt(nb_bnd_1));
		nb_bnd = nb_bnd_1; // more accurate formula
		printf("calculated nb boundary pts: %d\n", nb_bnd);
		printf("improved nb boundary pts: %d\n", nb_bnd_1);

		computeBoundaryPointDistribution(bnd_intg, high_nb_pts, nb_bnd, intg,
				bndry_pts);
		printf("domain integral = %f\n", dom_intg);
		printf("boundary integral = %f\n", bnd_intg);

		printf("nb_bnd= %d, bndry_pts.size= %d\n", nb_bnd,
				(int) bndry_pts.size());

		// If no cvt_circle.txt exists in the current directory we need to generate a CVT. 
		// After this point we always read the cvt_circle.txt as input. 
		struct stat fileInfo;
		if (stat("cvt_circle.txt", &fileInfo)) {
			cout << "No cvt_circle.txt in current directory. Generating...."
					<< endl;
			cvt = new CVT();
			createCVT(tot_nb_pts, nb_bnd, rho, bndry_pts, dom_intg);
		}

		int stencil_size = 9;
		int nx = 20;
		int ny = 20;

#if 0
		// disable if not running tests
		testEigen(stencil_size, nb_bnd, tot_nb_pts);
		exit(0);
#endif
		grid = new Grid(nx, ny, stencil_size);

		// EVAN (done) : ADD SUPPORT FOR princ_axis3
//		grid->setMajor(princ_axis1);
//		grid->setMinor(princ_axis2);
		grid->setPrincipalAxes(princ_axis1, princ_axis2, princ_axis3); 
		
		grid->setNbBnd(nb_bnd);

		// 2nd argument: known number of boundary points (stored ahead of interior points) 
		grid->generateGrid("cvt_circle.txt", nb_bnd, tot_nb_pts);
		//grid.generateGrid();
		//grid.generateSubGrid();

		grid->computeStencils(); // nearest nb_points
		grid->avgStencilRadius(); // Fill variable grid.avg_distance (std::vector<double>)
		vector<double> avg_dist = grid->getAvgDist(); // get average stencil radius for each point

		//grid.computeStencilsRegular();   // regular 4 point stencil
		vector<vector<int> >& stencil = grid->getStencil();

		// global variable
		rbf_centers = grid->getRbfCenters();
		//vector<Vec3>& rbf_centers = grid.getRbfCenters();
		//		int nb_rbf = rbf_centers.size(); //grid.getRbfCenters().size();

		// EVAN (DONE) : ADD SUPPORT FOR princ_axis3
		double avgarea = pi * princ_axis1 * princ_axis2 * princ_axis3 / tot_nb_pts; // 200 points
		
		double avgdx = sqrt(avgarea);
		avgdx = 0.02;
		printf("avgdx= %f\n", avgdx);
		double dt;
		dt = 0.2 * avgdx * avgdx;
		printf("dt (0.2*avgdx^2 = %f\n", dt);
		//	dt = 2. / maxEig;
		//	printf("dt (2/lambda_max)= %f\n", dt);


		// Perform domain decomposition and distribute the subcollection of nodes to each CPU
		// parameter to distributeStencils should be a vector of vectors (list of node lists)
		// comm_unit handles message passing in blocking/non-blocking fashion but hides details
		// from this routine. 
		//comm_unit->distributeStencils(grid.decomposeDomain(comm_unit->getSize()));
		subdomain = distributeNodesAcrossGPUs(grid, comm_unit, dt);

	} // endif (!my_rank)
	else {
		cout << "MPI RANK " << my_rank << ": waiting to receive subdomain"
				<< endl;
		subdomain = new GPU(); // EMPTY object that will be filled by MPI

		int status = comm_unit->receiveObject(subdomain, 0); // Receive from CPU (0)

	}

#if 0
	char buf[32];
	sprintf(&buf[0], "Set AVG_DISTS", my_rank);
	subdomain->printVector(subdomain->Q_avg_dists, buf);
#endif 

	comm_unit->barrier(); // Called by all CPUs regardless of rank


	//if (subdomain->Q_stencils.size() > 0) {		// Make sure we have something to compute on

	// To construct the derivative now we need to play tricks:
	// 1) "Re"-build a Grid object
	// 2) Build a Derivative object
	//	Grid grid(nx, ny, stencil_size);

	// NOTE
	//EB 1	(COUNTING GRID OCCURRENCES; POTENTIALLY OUTDATED ONCE GPU OBJECTS ARE CONSTRUCTED)
	//	Derivative der(grid.getRbfCenters(), grid.getStencil(), grid.getNbBnd());
	//	der.setAvgStencilRadius(grid.getAvgDist());

	// EB NEED ALL STENCILS TO BE LOCAL INDEX BY THIS POINT
	Derivative der(subdomain->G_centers, subdomain->Q_stencils,
			subdomain->global_boundary_nodes.size());
	der.setAvgStencilRadius(subdomain->Q_avg_dists);

	//printf("rbf_center size: %d\n", rbf_centers.size());
	//printf("main::computing weights, stencil= %d\n", &stencil);

	// Set things up for variable epsilon

	int nb_rbfs = subdomain->Q_stencils.size();
	//EB 2
	//vector<double> avg_stencil_radius = grid.getAvgDist();
	vector<double> avg_stencil_radius = subdomain->Q_avg_dists;
	vector<double> epsv(nb_rbfs);

	for (int i = 0; i < nb_rbfs; i++) {
		epsv[i] = 1. / avg_stencil_radius[i];
		printf("avg rad(%d) = %f\n", i, avg_stencil_radius[i]);
	}
	double mm = minimum(avg_stencil_radius);
	printf("min avg_stencil_radius= %f\n", mm);

	der.setVariableEpsilon(epsv); // has no effect (vareps was disabled in derivative.cpp)
	//exit(0);

	subdomain->printCenters(subdomain->G_centers,
			"All Centers Needed by this CPU");

	printf("start computing weights\n");
	for (int irbf = 0; irbf < subdomain->Q_stencils.size(); irbf++) {
		//		der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "x");
		//		der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "y");
		//		der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "lapl");
		// 0 here implies the first element of the stencil is the center
		char label[256];
		sprintf(label, "Stencil[%d]", irbf);
		subdomain->printStencil(subdomain->Q_stencils[irbf], label);

		// Centers are vec<Vec3> where Vec3=(double,double,double)
		// Stencils are vec<int> where int=index into centers vec<Vec3>
		der.computeWeightsSVD(subdomain->G_centers,
				subdomain->Q_stencils[irbf], irbf, "x");
		der.computeWeightsSVD(subdomain->G_centers,
				subdomain->Q_stencils[irbf], irbf, "y");
		der.computeWeightsSVD(subdomain->G_centers,
				subdomain->Q_stencils[irbf], irbf, "lapl");
	}
	printf("after all computeWeights\n");
	// MY BUG IS INSIDE HERE: 
	//double maxEig = der.computeEig(subdomain->Q_stencils); // needs lapl_weights
	printf("after all computeEig\n");
	//exit(0);

	// Before solving the PDE, we will do several checks (each time)
	// Accuracy of (1st, 2nd) derivative of (constant, linear, quadratic, cubic functions)
	// Accuracy of lapl() or of (div.grad) on these three functions
	// Eigenvalues of lapl and grad^2

#if 0
	// I must compute the average error of the derivatives, with and without the boundary points
	//EB -1
	//checkDerivatives(der, grid);
	checkXDerivatives(der, grid);
	exit(0);
#endif

	comm_unit->barrier(); // Called by all CPUs regardless of rank

	// Loop this: 
#if 0	
	for (int iter = 0; iter < 5; iter++) { // practice iterating the derivatives
		cout << "*********** COMPUTE DERIVATIVES (Iteration: " << iter << ") *************" << endl;
		subdomain->printVector(subdomain->U_G, "U_G");

		// Test computing derivatives
		vector<double> lapl_deriv(subdomain->Q_stencils.size());
		der.computeDeriv(Derivative::LAPL, subdomain->U_G, lapl_deriv);

		subdomain->printVector(lapl_deriv, "LAPL_U_G");

		// Imitation timestep: 
		for (int j = 0; j < lapl_deriv.size(); j++) {
			subdomain->U_G[j] += lapl_deriv[j];
		}

		// Send updates according to MPISendable object.
		comm_unit->broadcastObjectUpdates(subdomain);
		comm_unit->barrier();
	}
#endif 

	//EB 3
#if 0
	// Need to add to GPU class: notion of global domain boundary. If the global domain
	// is represented by a vector of points where the first nb_bnd points in the
	// vector are boundary points, then we can specify a global number of boundary
	// elements and have routines like this work on "while(l2g(index) < nb_bnd)"

	printf("test Derivatives\n");
	//printf("rbf_center size: %d\n", nb_rbf); exit(0);
	testDeriv(C, der, grid);
	testDeriv(X, der, grid);
	testDeriv(Y, der, grid);
	testDeriv(X2, der, grid);
	testDeriv(XY, der, grid);
	testDeriv(Y2, der, grid);
	testDeriv(X3, der, grid);
	testDeriv(X2Y, der, grid);
	testDeriv(XY2, der, grid);
	testDeriv(Y3, der, grid);
#endif 

	//exit(0);

	//----------------------------------------------------------------------
#if 0
	vector<mat>& we = der.getLaplWeights();
	printf("Laplacian weights\n");
	for (int i=0; i < we.size(); i++) {
		mat& w = we[i];
		//printf("weight %d, nb pts in stencil: %d\n", i, w.size());
		for (int j=0; j < w.n_elem; j++) {
			printf("%f ", w(j));
		}
		printf("\n");
	}
	exit(0);
#endif
	//----------------------------------------------------------------------

	// box dimensions: hardcoded!!! [-1,1] x [-1,1]

	// 5 point laplacian (FD stencil)
	//dt = 0.24*dx*dx; // works for Cartesian mesh with 5 points stencil (non-rbf)
	// 9 point laplacian (rbf)
	// area of th ellipsoid = pi*a*b = 3.14*1*0.8 = 
	// average area of voronoi cell: 3.14
	//exit(0);

	// SOLVE HEAT EQUATION

	//EB 4
#if 1
	//	Setup an analytic solution for the PDE 	(frequency, decay rate, principal axis 1, principal axis 2, principal axis 3)
	ExactSolution* solution = new ExactEllipsoid(acos(-1.)/2., 1., 1.0, 0.5, 0.5); 
	
	Heat heat(solution, subdomain, &der, comm_unit->getRank());
	heat.initialConditions(&subdomain->U_G);

	// Send updates according to MPISendable object.
	comm_unit->broadcastObjectUpdates(subdomain);
	comm_unit->barrier();

	// This is HARDCODED because we dont have the ability currently to call 
	// maxEig = der.computeEig() and therefore we have a different timestep than 
	// the original code. I will address this next. 
	//heat.setDt(0.011122);
	heat.setDt(subdomain->dt);
	subdomain->printVector(subdomain->global_boundary_nodes,
			"GLOBAL BOUNDARY NODES: ");
	// Even with Cartesian, the max norm stays at one. Strange
	int iter;
	for (iter = 0; iter < 1000; iter++) {
		cout << "*********** COMPUTE DERIVATIVES (Iteration: " << iter
				<< ") *************" << endl;
		subdomain->printVector(subdomain->U_G, "INPUT_TO_HEAT_ADVANCE");

		heat.advanceOneStepWithComm(comm_unit);
		subdomain->printVector(subdomain->U_G, "AFTER HEAT");

		double nrm = heat.maxNorm();

		// TODO : Need to add a "comm_unit->sendTerminate()" to
		// break all processes when problem is encountered
		if (nrm > 5.)
			break;
		//if (iter > 0) break;
	}

	printf("after heat\n");
	//	exit(0);
#endif 
	//}

	comm_unit->consolidateObjects(subdomain);

	if (comm_unit->getRank() == 0) {
		// TODO assemble final solution
		subdomain->writeFinal(grid->getRbfCenters(), "FINAL_SOLUTION.txt");
		// TODO print solution to file
		cout << "FINAL ITER: " << iter << endl;
	}
	printf("REACHED THE END OF MAIN\n");
}
//----------------------------------------------------------------------
