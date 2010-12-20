#include "derivative_tests.h"
#include "utils/norms.h"
#include <vector>

#include "exact_solutions/exact_ncar_poisson2.h"
using namespace std;

//----------------------------------------------------------------------
void DerivativeTests::checkDerivatives(Derivative& der, Grid& grid)
{
	vector<Vec3>& rbf_centers = grid.getRbfCenters();
	int nb_rbf = rbf_centers.size();

	// change the classes the variables are located in
	vector<double> xderiv(rbf_centers.size());
	vector<double> yderiv(rbf_centers.size());
	vector<double> lapl_deriv(rbf_centers.size());

        // Warning! assume DIM = 2
        this->computeAllWeights(der, rbf_centers, grid.getStencil(), nb_rbf);

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

	for (int i=0; i < nb_rbf; i++) {
		Vec3& v = rbf_centers[i];
		//s = 3.*v.x() + 2.*v.y(); //   + 3.*v.z();

		s = v.x()*v.y() + 0.5*v.x()*v.x() + 0.5*v.y()*v.y(); //   + 3.*v.z();
		du_ex.push_back(2.);

		//s = v.x()+v.y() +v.x()+v.y();
		//du_ex.push_back(0.);

		//s = v.x()*v.x()*v.x();
		//du_ex.push_back(6.*v.x());

		u.push_back(s);
	}

	printf("main, u[0]= %f\n", u[0]);

	for (int n=0; n < 1; n++) {
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

	for (int i=0; i < lapl_deriv.size(); i++) {
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("(%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", (int) st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
	}

	// boundary points
	vector<int>& boundary = grid.getBoundary();
	for (int ib=0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		//printf("bnd (%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
	}

	double inter_error=0.;
	for (int i=boundary.size(); i < nb_rbf; i++) {
		inter_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
		printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
	}
	inter_error /= (nb_rbf-boundary.size());

	double bnd_error=0.;
	for (int ib=0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		bnd_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
		printf("bnd error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
	}
	bnd_error /= boundary.size();

	printf("avg l2_bnd_error= %14.7e\n", sqrt(bnd_error));
	printf("avg l2_interior_error= %14.7e\n", sqrt(inter_error));
}
//----------------------------------------------------------------------
void DerivativeTests::checkXDerivatives(Derivative& der, Grid& grid)
{
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

        this->computeAllWeights(der, rbf_centers, grid.getStencil(), nb_rbf);

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

	for (int i=0; i < nb_rbf; i++) {
		Vec3& v = rbf_centers[i];
		//s = 3.*v.x() + 2.*v.y(); //   + 3.*v.z();

		//s = v.x()*v.y() + 0.5*v.x()*v.x() + 0.5*v.y()*v.y(); //   + 3.*v.z();
		//du_ex.push_back(2.);

		s = v.x()*v.y();
		dux_ex.push_back(v.y());
		duy_ex.push_back(v.x());

		//s = v.y();
		du_ex.push_back(1.);

		u.push_back(s);
	}

	printf("main, u[0]= %f\n", u[0]);
	printf("main, u= %ld\n", (long int) &u[0]);

	for (int n=0; n < 1; n++) {
		// perhaps I'll need different (rad,eps) for each. To be determined. 
		der.computeDeriv(Derivative::X, u, xderiv);
		der.computeDeriv(Derivative::Y, u, yderiv);
		der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
	}
	der.computeEig(); // needs lapl_weights, analyzes stability of Laplace operator

	#if 1
	for (int i=0; i < xderiv.size(); i++) {
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
		printf("(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
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
	for (int ib=0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("bnd(%d) sz(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", ib, (int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
		printf("bnd(%d) sz(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", ib, (int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
		//printf("bnd (%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
	}
	exit(0);

	double inter_error=0.;
	for (int i=boundary.size(); i < nb_rbf; i++) {
		inter_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
		printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
	}
	inter_error /= (nb_rbf-boundary.size());

	double bnd_error=0.;
	for (int ib=0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		bnd_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
		printf("bnd error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
	}
	bnd_error /= boundary.size();

	printf("avg l2_bnd_error= %14.7e\n", sqrt(bnd_error));
	printf("avg l2_interior_error= %14.7e\n", sqrt(inter_error));

	exit(0);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void DerivativeTests::testEigen(Grid& grid, Derivative& der, int stencil_size, int nb_bnd, int tot_nb_pts)
{
// read input file
// compute stencils (do this only 

	double pert = 0.05;
	vector<double> u(tot_nb_pts);
	vector<double> lapl_deriv(tot_nb_pts);

	int nx = 20;
	int ny = 20;
#if 0
	// need another constructor for ellipses
	Grid grid(nx, ny, stencil_size);

//	grid.setMajor(major);
//	grid.setMinor(minor);
        grid.setPrincipalAxes(major, minor, 0.);
	grid.setNbBnd(nb_bnd);

	// 2nd argument: known number of boundary points (stored ahead of interior points) 
	grid.generateGrid("cvt_circle.txt", nb_bnd, tot_nb_pts);

	grid.computeStencils();   // nearest nb_points
	grid.avgStencilRadius(); 
#endif
	vector<double> avg_stencil_radius = grid.getAvgDist(); // get average stencil radius for each point

	vector<vector<int> >& stencil = grid.getStencil();

	// global variable
        vector<Vec3> rbf_centers = grid.getRbfCenters();
	int nb_rbf = rbf_centers.size();

       // Derivative der(rbf_centers, stencil, grid.getNbBnd());
	der.setAvgStencilRadius(avg_stencil_radius);

// Set things up for variable epsilon

	int nb_rbfs = rbf_centers.size();
	vector<double> epsv(nb_rbfs);

	for (int i=0; i < nb_rbfs; i++) {
		//epsv[i] = 1. / avg_stencil_radius[i];
		epsv[i] = 1.; // fixed epsilon
		//printf("avg rad(%d) = %f\n", i, avg_stencil_radius[i]);
	}
	double mm = minimum(avg_stencil_radius);
	printf("min avg_stencil_radius= %f\n", mm);

	der.setVariableEpsilon(epsv);

	// Laplacian weights with zero grid perturbation
	for (int irbf=0; irbf < rbf_centers.size(); irbf++) {
		der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "lapl");
	}

	double max_eig = der.computeEig(); // needs lapl_weights
	printf("zero perturbation: max eig: %f\n", max_eig);

	vector<Vec3> rbf_centers_orig;
	rbf_centers_orig.assign(rbf_centers.begin(), rbf_centers.end());

	double percent = 0.05; // in [0,1]
	printf("percent distortion of original grid= %f\n", percent);

	// set a random seed
	srandom(time(0));

	for (int i=0; i < 100; i++) {
		printf("---- iteration %d ------\n", i);
		//update rbf centers by random perturbations at a fixed percentage of average radius computed
		//based on the unperturbed mesh
		rbf_centers.assign(rbf_centers_orig.begin(), rbf_centers_orig.end());

		for (int j=0; j < nb_rbfs; j++) {
			Vec3& v = rbf_centers[j];
			double vx = avg_stencil_radius[j]*percent*random(-1.,1.);
			double vy = avg_stencil_radius[j]*percent*random(-1.,1.);
			v.setValue(v.x()+vx, v.y()+vy);
		}

		//rbf_centers[10].print("rbf_centers[10]");
		//continue;

		//recompute Laplace weights
		for (int irbf=0; irbf < rbf_centers.size(); irbf++) {
			der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "lapl");
		}
		double max_eig = der.computeEig(); // needs lapl_weights
		printf("zero perturbation: max eig: %f\n", max_eig);
	}
}
//----------------------------------------------------------------------
void DerivativeTests::testFunction(DerivativeTests::TESTFUN which, Grid& grid, vector<double>& u, vector<double>& dux_ex, vector<double>& duy_ex,
			vector<double>& dulapl_ex)
{
	u.resize(0);
	dux_ex.resize(0);
	duy_ex.resize(0);
	dulapl_ex.resize(0);

        vector<Vec3>& rbf_centers = grid.getRbfCenters();
	int nb_rbf = rbf_centers.size();

	switch(which) {
		case C:
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(1.);
				dux_ex.push_back(0.);
				duy_ex.push_back(0.);
				dulapl_ex.push_back(0.);
			}
			break;
		case X:
			printf("nb_rbf= %d\n", nb_rbf);
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(v.x());
				dux_ex.push_back(1.);
				duy_ex.push_back(0.);
				dulapl_ex.push_back(0.);
			}
			printf("u.size= %d\n", (int) u.size());
			break;
		case Y:
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(v.y());
				dux_ex.push_back(0.);
				duy_ex.push_back(1.);
				dulapl_ex.push_back(0.);
			}
			break;
		case X2:
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(v.x()*v.x());
				dux_ex.push_back(2.*v.x());
				duy_ex.push_back(0.);
				dulapl_ex.push_back(2.);
			}
			break;
		case XY:
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(v.x()*v.y());
				dux_ex.push_back(v.y());
				duy_ex.push_back(v.x());
				dulapl_ex.push_back(0.);
			}
			break;
		case Y2:
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(v.y()*v.y());
				dux_ex.push_back(0.);
				duy_ex.push_back(2.*v.y());
				dulapl_ex.push_back(2.);
			}
			break;
		case X3:
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(v.x()*v.x()*v.x());
				dux_ex.push_back(3.*v.x()*v.x());
				duy_ex.push_back(0.);
				dulapl_ex.push_back(6.*v.x());
			}
			break;
		case X2Y:
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(v.x()*v.x()*v.y());
				dux_ex.push_back(2.*v.x()*v.y());
				duy_ex.push_back(v.x()*v.x());
				dulapl_ex.push_back(2.*v.y());
			}
			break;
		case XY2:
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(v.x()*v.y()*v.y());
				dux_ex.push_back(v.y()*v.y());
				duy_ex.push_back(2.*v.x()*v.y());
				dulapl_ex.push_back(2.*v.x());
			}
			break;
		case Y3:
			for (int i=0; i < nb_rbf; i++) {
				Vec3& v = rbf_centers[i];
				u.push_back(v.y()*v.y()*v.y());
				dux_ex.push_back(0.);
				duy_ex.push_back(3.*v.y()*v.y());
				dulapl_ex.push_back(6.*v.y());
			}
			break;
               case CUSTOM:
                   for (int i=0; i < nb_rbf; i++) {
                           Vec3& v = rbf_centers[i];
                           ExactSolution* exact = new ExactNCARPoisson2();
                           u.push_back(exact->at(v));
                           dux_ex.push_back(exact->xderiv(v));
                           duy_ex.push_back(exact->yderiv(v));
                           dulapl_ex.push_back(exact->laplacian(v));
                   }
                   break;
	}
}
		
//----------------------------------------------------------------------
void DerivativeTests::testDeriv(DerivativeTests::TESTFUN choice, Derivative& der, Grid& grid, std::vector<double> avgDist)
{
	printf("================\n");
	printf("testderiv: choice= %d\n", choice);

        vector<Vec3>& rbf_centers = grid.getRbfCenters();
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

        this->computeAllWeights(der, rbf_centers, grid.getStencil(), nb_rbf);

        testFunction(choice, grid, u, dux_ex, duy_ex, dulapl_ex);


	avgDist = grid.getAvgDist();

	for (int n=0; n < 1; n++) {
		// perhaps I'll need different (rad,eps) for each. To be determined. 
		der.computeDeriv(Derivative::X, u, xderiv);
		der.computeDeriv(Derivative::Y, u, yderiv);
		der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
	}

	vector<int>& boundary = grid.getBoundary();
	int nb_bnd = boundary.size();

	enum DERIV {X=0, Y, LAPL};
	enum NORM {L1=0, L2, LINF};
	enum BNDRY {INT=0, BNDRY};
	double norm[3][3][2]; // norm[DERIV][NORM][BNDRY]
	double normder[3][3][2]; // norm[DERIV][NORM][BNDRY]

	norm[X][L1][BNDRY]   = l1norm(dux_ex,   xderiv, avgDist, 0, nb_bnd);
	norm[X][L2][BNDRY]   = l2norm(dux_ex,   xderiv, avgDist, 0, nb_bnd);
	norm[X][LINF][BNDRY] = linfnorm(dux_ex, xderiv, 0, nb_bnd);

	norm[Y][L1][BNDRY]   = l1norm(duy_ex,   yderiv, avgDist, 0, nb_bnd);
	norm[Y][L2][BNDRY]   = l2norm(duy_ex,   yderiv, avgDist, 0, nb_bnd);
	norm[Y][LINF][BNDRY] = linfnorm(duy_ex, yderiv, 0, nb_bnd);

	norm[LAPL][L1][BNDRY]   = l1norm(dulapl_ex,   lapl_deriv, avgDist, 0, nb_bnd);
	norm[LAPL][L2][BNDRY]   = l2norm(dulapl_ex,   lapl_deriv, avgDist, 0, nb_bnd);
	norm[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex, lapl_deriv, 0, nb_bnd);

	norm[X][L1][INT]   = l1norm(dux_ex,   xderiv, avgDist, nb_bnd, nb_rbf);
	norm[X][L2][INT]   = l2norm(dux_ex,   xderiv, avgDist, nb_bnd, nb_rbf);
	norm[X][LINF][INT] = linfnorm(dux_ex, xderiv, nb_bnd, nb_rbf);

	norm[Y][L1][INT]   = l1norm(duy_ex,   yderiv, avgDist, nb_bnd, nb_rbf);
	norm[Y][L2][INT]   = l2norm(duy_ex,   yderiv, avgDist, nb_bnd, nb_rbf);
	norm[Y][LINF][INT] = linfnorm(duy_ex, yderiv, nb_bnd, nb_rbf);

	norm[LAPL][L1][INT]   = l1norm(dulapl_ex,   lapl_deriv, avgDist, nb_bnd, nb_rbf);
	norm[LAPL][L2][INT]   = l2norm(dulapl_ex,   lapl_deriv, avgDist, nb_bnd, nb_rbf);
	norm[LAPL][LINF][INT] = linfnorm(dulapl_ex, lapl_deriv, nb_bnd, nb_rbf);

	// --- Normalization factors

        normder[X][L1][BNDRY]   = l1norm(dux_ex, avgDist,  0, nb_bnd);
        normder[X][L2][BNDRY]   = l2norm(dux_ex, avgDist,  0, nb_bnd);
	normder[X][LINF][BNDRY] = linfnorm(dux_ex, 0, nb_bnd);

        normder[Y][L1][BNDRY]   = l1norm(duy_ex, avgDist,  0, nb_bnd);
        normder[Y][L2][BNDRY]   = l2norm(duy_ex, avgDist,  0, nb_bnd);
	normder[Y][LINF][BNDRY] = linfnorm(duy_ex, 0, nb_bnd);

        normder[LAPL][L1][BNDRY]   = l1norm(dulapl_ex, avgDist,  0, nb_bnd);
        normder[LAPL][L2][BNDRY]   = l2norm(dulapl_ex, avgDist,  0, nb_bnd);
	normder[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex, 0, nb_bnd);

        normder[X][L1][INT]   = l1norm(dux_ex, avgDist,  nb_bnd, nb_rbf);
        normder[X][L2][INT]   = l2norm(dux_ex, avgDist,  nb_bnd, nb_rbf);
	normder[X][LINF][INT] = linfnorm(dux_ex, nb_bnd, nb_rbf);

        normder[Y][L1][INT]   = l1norm(duy_ex, avgDist,  nb_bnd, nb_rbf);
        normder[Y][L2][INT]   = l2norm(duy_ex, avgDist,  nb_bnd, nb_rbf);
	normder[Y][LINF][INT] = linfnorm(duy_ex, nb_bnd, nb_rbf);

        normder[LAPL][L1][INT]   = l1norm(dulapl_ex, avgDist,  nb_bnd, nb_rbf);
        normder[LAPL][L2][INT]   = l2norm(dulapl_ex, avgDist,  nb_bnd, nb_rbf);
	normder[LAPL][LINF][INT] = linfnorm(dulapl_ex, nb_bnd, nb_rbf);



        printf("----- RESULTS: testDeriv( %d ) ---\n", choice);
        printf("{C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3,CUSTOM=10}\n");
	printf("norm[x/y/lapl][L1,L2,LINF][interior/bndry]\n");
	for (int k=0; k < 2; k++) {
	for (int i=0; i < 3; i++) {
                // If our L2 absolute norm is > 1e-9 we show relative.
		if (abs(normder[i][1][k]) < 1.e-9) {
			printf("(abs err): norm[%d][][%d]= %10.3e, %10.3e, %10.3e\n", i, k, norm[i][0][k], norm[i][1][k], norm[i][2][k]);
		} else {
			printf("(rel err): norm[%d][][%d]= %10.3e, %10.3e, %10.3e, normder= %10.3e\n", i, k, 
			    norm[i][0][k]/normder[i][0][k], 
			    norm[i][1][k]/normder[i][1][k], 
			    norm[i][2][k]/normder[i][2][k], normder[i][1][k]); 
			//printf("   normder[%d][][%d]= %10.3e, %10.3e, %10.3e\n", i, k, 
			    //normder[i][0][k], 
			    //normder[i][1][k], 
			    //normder[i][2][k]); 
		}
	}}

	double inter_error=0.;
	//vector<int>& boundary = grid.getBoundary();

	for (int i=(int) boundary.size(); i < nb_rbf; i++) {
		inter_error += (dulapl_ex[i]-lapl_deriv[i])*(dulapl_ex[i]-lapl_deriv[i]);
		//printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
	}
	inter_error /= (nb_rbf-boundary.size());

	double bnd_error=0.;
	for (int ib=0; ib < boundary.size(); ib++) {
		int i = boundary[ib];
		bnd_error += (dulapl_ex[i]-lapl_deriv[i])*(dulapl_ex[i]-lapl_deriv[i]);
		//printf("bnd error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
	}
	bnd_error /= boundary.size();

	printf("avg l2_bnd_error= %14.7e\n", sqrt(bnd_error));
	printf("avg l2_interior_error= %14.7e\n", sqrt(inter_error));
}
//----------------------------------------------------------------------
void DerivativeTests::computeAllWeights(Derivative& der, std::vector<Vec3> rbf_centers, std::vector<std::vector<int> > stencils, int nb_stencils) {
//#define USE_CONTOURSVD 1
#undef USE_CONTOURSVD

#if USE_CONTOURSVD
    // Laplacian weights with zero grid perturbation
    for (int irbf=0; irbf < rbf_centers.size(); irbf++) {
        der.computeWeightsSVD(rbf_centers, stencils[irbf], irbf, "x");
        der.computeWeightsSVD(rbf_centers, stencils[irbf], irbf, "y");
        der.computeWeightsSVD(rbf_centers, stencils[irbf], irbf, "lapl");
    }
#else
    for (int i = 0; i < nb_stencils; i++) {
        der.computeWeights(rbf_centers, stencils[i], i);
    }
#endif
}
//----------------------------------------------------------------------
void DerivativeTests::testAllFunctions(Derivative& der, Grid& grid) {
    this->computeAllWeights(der, grid.getRbfCenters(), grid.getStencil(), grid.getStencil().size());

    // Test all: C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3,CUSTOM
#if 1
    this->testDeriv(DerivativeTests::C, der, grid, grid.getAvgDist());
    this->testDeriv(DerivativeTests::X, der, grid, grid.getAvgDist());
    this->testDeriv(DerivativeTests::Y, der, grid, grid.getAvgDist());
    this->testDeriv(DerivativeTests::X2, der, grid, grid.getAvgDist());
    this->testDeriv(DerivativeTests::XY, der, grid, grid.getAvgDist());
    this->testDeriv(DerivativeTests::Y2, der, grid, grid.getAvgDist());
    this->testDeriv(DerivativeTests::X3, der, grid, grid.getAvgDist());
    this->testDeriv(DerivativeTests::X2Y, der, grid, grid.getAvgDist());
    this->testDeriv(DerivativeTests::XY2, der, grid, grid.getAvgDist());
    #endif
    this->testDeriv(DerivativeTests::Y3, der, grid, grid.getAvgDist());
    this->testDeriv(DerivativeTests::CUSTOM, der, grid, grid.getAvgDist());
//    exit(EXIT_FAILURE);
}
