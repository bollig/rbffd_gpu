#include <math.h>
#include "grid.h"
#include "derivative.h"
#include "heat.h"

using namespace std;

Heat::Heat(std::vector<Vec3>* rbf_centers_, int stencil_size,
		std::vector<int>* global_boundary_nodes_, Derivative* der_, int rank) :
	rbf_centers(rbf_centers_), boundary_set(global_boundary_nodes_), der(der_),
			id(rank), subdomain(NULL) {
	nb_stencils = stencil_size;
	nb_rbf = rbf_centers->size();

	PI = acos(-1.);
	freq = PI / 2.;
	decay = 1.;

	// should be placed elsewhere, perhaps in grid

	// HARDCODED EB FIX THIS!
	major = 1.0;//grid.getMajor();
	minor = 0.5;//grid.getMinor();

	time = 0.0; // physical time

	maji2 = 1. / (major * major);
	mini2 = 1. / (minor * minor);
	maji4 = maji2 * maji2;
	mini4 = mini2 * mini2;

	// solution + temporary array (for time advancement)
	sol[0].resize(nb_rbf);
	sol[1].resize(nb_rbf);
	lapl_deriv.resize(nb_stencils);

	// could resize inside the advancement function
	x_deriv.resize(nb_stencils);
	y_deriv.resize(nb_stencils);
	xx_deriv.resize(nb_stencils);
	yy_deriv.resize(nb_stencils);
	diffusion.resize(nb_stencils);
	diff_x.resize(nb_stencils);
	diff_y.resize(nb_stencils);

	// Cartesian-based Laplacian
	//grid.laplace();
}

Heat::Heat(GPU* subdomain_, Derivative* der_, int rank) :
	rbf_centers(&subdomain_->G_centers), boundary_set(
			&subdomain_->global_boundary_nodes), der(der_), id(rank),
			subdomain(subdomain_) {
	nb_stencils = subdomain->Q_stencils.size();
	nb_rbf = subdomain->G_centers.size();

	PI = acos(-1.);
	freq = PI / 2.;
	decay = 1.;

	// should be placed elsewhere, perhaps in grid

	// HARDCODED EB FIX THIS!
	major = 1.0;//grid.getMajor();
	minor = 0.5;//grid.getMinor();

	time = 0.0; // physical time

	maji2 = 1. / (major * major);
	mini2 = 1. / (minor * minor);
	maji4 = maji2 * maji2;
	mini4 = mini2 * mini2;

	// solution + temporary array (for time advancement)
	sol[0].resize(nb_rbf);
	sol[1].resize(nb_rbf);
	lapl_deriv.resize(nb_stencils);

	// could resize inside the advancement function
	x_deriv.resize(nb_stencils);
	y_deriv.resize(nb_stencils);
	xx_deriv.resize(nb_stencils);
	yy_deriv.resize(nb_stencils);
	diffusion.resize(nb_stencils);
	diff_x.resize(nb_stencils);
	diff_y.resize(nb_stencils);

	// Cartesian-based Laplacian
	//grid.laplace();
}

Heat::Heat(Grid& grid_, Derivative& der_) :
	rbf_centers(&(grid_.getRbfCenters())),
			boundary_set(&(grid_.getBoundary())), der(&der_), id(0), subdomain(
					NULL) {
	nb_stencils = grid_.getStencil().size();
	nb_rbf = grid_.getRbfCenters().size();
	PI = acos(-1.);
	freq = PI / 2.;
	decay = 1.;

	// should be placed elsewhere, perhaps in grid
	major = grid_.getMajor();
	minor = grid_.getMinor();

	time = 0.0; // physical time

	maji2 = 1. / (major * major);
	mini2 = 1. / (minor * minor);
	maji4 = maji2 * maji2;
	mini4 = mini2 * mini2;

	// solution + temporary array (for time advancement)
	sol[0].resize(nb_rbf);
	sol[1].resize(nb_rbf);
	lapl_deriv.resize(nb_rbf);

	// could resize inside the advancement function
	x_deriv.resize(nb_rbf);
	y_deriv.resize(nb_rbf);
	xx_deriv.resize(nb_rbf);
	yy_deriv.resize(nb_rbf);
	diffusion.resize(nb_rbf);
	diff_x.resize(nb_rbf);
	diff_y.resize(nb_rbf);

	// Cartesian-based Laplacian
	//grid.laplace();
}
//----------------------------------------------------------------------
Heat::~Heat() {
}
//----------------------------------------------------------------------


// Advance the equation one time step using the GPU class to perform communication
// Depends on Constructor #2 to be used so that a GPU class exists within this class.
void Heat::advanceOneStepWithComm(Communicator* comm_unit) {

	if (subdomain == NULL) {
		cerr
				<< "In HEAT.CPP: Wrong advanceOneStep* routine called! No GPU class passed to Constructor. Cannot perform intermediate communication/updates."
				<< endl;
		exit(-10);
	} else {

		// This time advancement is second order.
		// It first updates s1 := s + 0.5 * dt * (lapl(s) + f)
		// Then it updates s := s + dt * (lapl(s1) + f)
		// However, in parallel our lapl(s) is size Q but depends on size Q+R
		// the lapl(s1) is size Q but depends on size Q+R; the key difference between
		// lapl(s) and lapl(s1) is that we have the set R for lapl(s) when we enter this
		// routine. Thus, we must call for the communicator comm_unit to update
		// the GPU object at the beginning/end and in the middle of this routine.
		// I think the best approach is to make the Heat class inherit from a ParallelPDE
		// type. ParallelPDE types will have pure virtual API for executing updates
		// on MPISendable types. Also I think ParallelPDE types should have internal
		// looping rather than require the main code control the loop. Mostly Im
		// thinking of the GLUT structure: glutInitFunc(), glutDisplayFunc(), ...
		// such that when glutMainLoop is called, a specific workflow is executed
		// on the registered callbacks (i.e. MPISendable update functions). This
		// would allow the main.cpp to initialize and forget, and allow more fine
		// grained control by registering custom routines.

		// compute laplace u^n
		// 2nd argument is vector<double>


		vector<double>& s = sol[0];
		vector<double>& s1 = sol[1];

		comm_unit->broadcastObjectUpdates(subdomain);

		// Do NOT use GPU as buffer for computation
		// Only go up to the number of stencils since we solve for a subset of the values in U_G
		// Since U_G in R is at end of U_G vector we can ignore those.
		for (int i = 0; i < s.size(); i++) {
			s[i] = subdomain->U_G[i];
		}

		//der->computeDeriv(Derivative::LAPL, s, lapl_deriv);
		der->computeDeriv(Derivative::LAPL, s, lapl_deriv);

		// Use 5 point Cartesian formula for the Laplacian
		//lapl_deriv = grid.computeCartLaplacian(s);
#if 1
		for (int i = 0; i < lapl_deriv.size(); i++) {
			Vec3& v = (*rbf_centers)[i];
			printf("(local: %d), lapl(%f,%f)= %f\t%f\n", i, v.x(), v.y(),
					lapl_deriv[i], s[i]);
		}
		//exit(0);
#endif

		// compute u^* = u^n + dt*lapl(u^n)

		printf("heat, dt= %f, time= %f\n", dt, time);

		// second order time advancement if SECOND is defined
#define SECOND

		// explicit scheme
		for (int i = 0; i < lapl_deriv.size(); i++) {
			// first order
			Vec3& v = (*rbf_centers)[i];
			//printf("dt= %f, time= %f\n", dt, time);
			double f = force(v[0], v[1], time);
			printf("force (local: %d) = %f\n", i, f);
#ifdef SECOND
			s1[i] = s[i] + 0.5 * dt * (lapl_deriv[i] + f);
#else
			s[i] = s[i] + dt* ( lapl_deriv[i] + f);
#endif
		}

		time = time + dt;

		// reset boundary solution

		// assert bnd_sol.size() == bnd_index.size()
		vector<int>& bnd_index = *boundary_set; //grid.getBoundary();
		int sz = bnd_sol.size();

		printf("nb bnd pts: %d\n", sz);

		for (int i = 0; i < bnd_sol.size(); i++) {
			// first order
			Vec3& v = (*rbf_centers)[bnd_index[i]];
			printf("bnd[%d] = {%d} %f, %f\n", i, bnd_index[i], v.x(), v.y());
#ifdef SECOND
			s1[bnd_index[i]] = bnd_sol[i];
#else
			s[bnd_index[i]] = bnd_sol[i];
#endif
		}

		// Now we need to make sure all CPUs have R2 (intermediate part of timestep)
		// Do NOT use GPU as buffer for computation
		for (int i = 0; i < s1.size(); i++) {
			subdomain->U_G[i] = s1[i];
		}
		comm_unit->broadcastObjectUpdates(subdomain);
		// Do NOT use GPU as buffer for computation
		for (int i = 0; i < s1.size(); i++) {
			s1[i] = subdomain->U_G[i];
		}

#ifdef SECOND
		// compute laplace u^*
		der->computeDeriv(Derivative::LAPL, s1, lapl_deriv);
		// compute u^{n+1} = u^n + dt*lapl(u^*)
		for (int i = 0; i < lapl_deriv.size(); i++) {
			Vec3& v = (*rbf_centers)[i];
			//printf("i= %d, nb_rbf= %d\n", i, nb_rbf);
			//v.print("v");
			double f = force(v[0], v[1], time + 0.5 * dt);
			s[i] = s[i] + dt * (lapl_deriv[i] + f); // RHS at time+0.5*dt
		}

		// reset boundary solution

		for (int i = 0; i < sz; i++) {
			s[bnd_index[i]] = bnd_sol[i];
		}
#endif

#if 0
		for (int i=0; i < s.size(); i++) {
			Vec3* v = rbf_centers[i];
			printf("(%f,%f): T(%d)=%f\n", v.x(), v.y(), i, s[i]);
		}
#endif

		// solution analysis

		vector<double> sol_ex;
		vector<double> sol_error;

		sol_ex.resize(nb_stencils);
		sol_error.resize(nb_stencils);

		// exact solution
		for (int i = 0; i < nb_stencils; i++) {
			Vec3& v = (*rbf_centers)[i];
			sol_ex[i] = exactSolution(v[0], v[1], time);
			sol_error[i] = sol_ex[i] - s[i];
		}

		// print error to a file
		//	printf("nb_rbf= %d\n", nb_rbf);
		//	exit(0);

		char filename[256];
		sprintf(filename, "error.out.%d", id);

		FILE* fderr = fopen(filename, "w");
		for (int i = 0; i < nb_stencils; i++) {
			Vec3& v = (*rbf_centers)[i];
			fprintf(fderr, "%f %f %f\n", v[0], v[1], sol_error[i]);
		}
		fclose(fderr);

		sprintf(filename, "solution.out.%d", id);

		// print solution to a file
		FILE* fdsol = fopen(filename, "w");
		for (int i = 0; i < nb_stencils; i++) {
			Vec3& v = (*rbf_centers)[i];
			fprintf(fdsol, "%f %f %f\n", v.x(), v.y(), s[i]);
		}
		fclose(fdsol);

		double nrm_ex = maxNorm(sol_ex);
		printf("exact max norm: %f\n", nrm_ex);
		double nrm_sol0 = maxNorm(s);
		printf("max norm(s[0]): %f\n", nrm_sol0);
		double nrm_error = maxNorm(sol_error);
		printf("nrm_error= %f\n", nrm_error);

		// And now we have full derivative calculated so we need to overwrite U_G
		for (int i = 0; i < s.size(); i++) {
			subdomain->U_G[i] = s[i];
		}
	}
	return;
}

//----------------------------------------------------------------------
void Heat::advanceOneStep(std::vector<double>* updated_solution) {
	// This time advancement is second order. 
	// It first updates s1 := s + 0.5 * dt * (lapl(s) + f)
	// Then it updates s := s + dt * (lapl(s1) + f) 
	// However, in parallel our lapl(s) is size Q but depends on size Q+R
	// the lapl(s1) is size Q but depends on size Q+R; the key difference between 
	// lapl(s) and lapl(s1) is that we have the set R for lapl(s) when we enter this
	// routine. Thus, we must call for the communicator comm_unit to update
	// the GPU object at the beginning/end and in the middle of this routine. 
	// I think the best approach is to make the Heat class inherit from a ParallelPDE 
	// type. ParallelPDE types will have pure virtual API for executing updates
	// on MPISendable types. Also I think ParallelPDE types should have internal 
	// looping rather than require the main code control the loop. Mostly Im 
	// thinking of the GLUT structure: glutInitFunc(), glutDisplayFunc(), ... 
	// such that when glutMainLoop is called, a specific workflow is executed
	// on the registered callbacks (i.e. MPISendable update functions). This 
	// would allow the main.cpp to initialize and forget, and allow more fine
	// grained control by registering custom routines. 

	// compute laplace u^n
	// 2nd argument is vector<double> 
	vector<double>& s = sol[0];
	vector<double>& s1 = sol[1];

	//	vector<Vec3>* rbf_centers = grid.getRbfCenters();

	// compute laplace u^n 
	if (updated_solution != NULL) {
		for (int i = 0; i < (*updated_solution).size(); i++) {
			s[i] = (*updated_solution)[i];
		}
		cout << "USING updated_solution" << endl;
	}
	der->computeDeriv(Derivative::LAPL, s, lapl_deriv);

	// Use 5 point Cartesian formula for the Laplacian
	//lapl_deriv = grid.computeCartLaplacian(s);
#if 1
	for (int i = 0; i < lapl_deriv.size(); i++) {
		Vec3& v = (*rbf_centers)[i];
		printf("(local: %d), lapl(%f,%f)= %f\t%f\n", i, v.x(), v.y(),
				lapl_deriv[i], s[i]);
	}
	//exit(0);
#endif

	// compute u^* = u^n + dt*lapl(u^n)

	printf("heat, dt= %f, time= %f\n", dt, time);

	// second order time advancement if SECOND is defined
#define SECOND 

	// explicit scheme
	for (int i = 0; i < nb_stencils; i++) {
		// first order
		Vec3& v = (*rbf_centers)[i];
		//printf("dt= %f, time= %f\n", dt, time);
		double f = force(v[0], v[1], time);
		printf("force (local: %d) = %f\n", i, f);
#ifdef SECOND
		s1[i] = s[i] + 0.5 * dt * (lapl_deriv[i] + f);
#else
		s[i] = s[i] + dt* ( lapl_deriv[i] + f);
#endif
	}

	time = time + dt;

	// reset boundary solution

	// assert bnd_sol.size() == bnd_index.size()
	vector<int>& bnd_index = *boundary_set; //grid.getBoundary();
	int sz = bnd_sol.size();

	printf("nb bnd pts: %d\n", sz);

	for (int i = 0; i < sz; i++) {
		// first order
		Vec3& v = (*rbf_centers)[bnd_index[i]];
		printf("bnd[%d] = {%d} %f, %f\n", i, bnd_index[i], v.x(), v.y());
#ifdef SECOND
		s1[bnd_index[i]] = bnd_sol[i];
#else
		s[bnd_index[i]] = bnd_sol[i];
#endif
	}

#ifdef SECOND
	// compute laplace u^* 
	der->computeDeriv(Derivative::LAPL, s1, lapl_deriv);
	cerr << "SECOND ORDER TIME" << endl;
	// compute u^{n+1} = u^n + dt*lapl(u^*)
	for (int i = 0; i < nb_stencils; i++) {
		Vec3& v = (*rbf_centers)[i];
		//printf("i= %d, nb_rbf= %d\n", i, nb_rbf);
		//v.print("v");
		double f = force(v[0], v[1], time + 0.5 * dt);
		s[i] = s[i] + dt * (lapl_deriv[i] + f); // RHS at time+0.5*dt
	}

	// reset boundary solution

	for (int i = 0; i < sz; i++) {
		s[bnd_index[i]] = bnd_sol[i];
	}
#endif

#if 0
	for (int i=0; i < nb_rbf; i++) {
		Vec3* v = rbf_centers[i];
		printf("(%f,%f): T(%d)=%f\n", v.x(), v.y(), i, s[i]);
	}
#endif

	// solution analysis

	vector<double> sol_ex;
	vector<double> sol_error;

	sol_ex.resize(nb_stencils);
	sol_error.resize(nb_stencils);

	// exact solution
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		sol_ex[i] = exactSolution(v[0], v[1], time);
		sol_error[i] = sol_ex[i] - s[i];
	}

	// print error to a file
	//	printf("nb_rbf= %d\n", nb_rbf);
	//	exit(0);

	char filename[256];
	sprintf(filename, "error.out.%d", id);

	FILE* fderr = fopen(filename, "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fderr, "%f %f %f\n", v[0], v[1], sol_error[i]);
	}
	fclose(fderr);

	sprintf(filename, "solution.out.%d", id);

	// print solution to a file
	FILE* fdsol = fopen(filename, "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fdsol, "%f %f %f\n", v.x(), v.y(), s[i]);
	}
	fclose(fdsol);

	double nrm_ex = maxNorm(sol_ex);
	printf("exact max norm: %f\n", nrm_ex);
	double nrm_sol0 = maxNorm(s);
	printf("max norm(s[0]): %f\n", nrm_sol0);
	double nrm_error = maxNorm(sol_error);
	printf("nrm_error= %f\n", nrm_error);

	if (updated_solution != NULL) {
		for (int i = 0; i < s.size(); i++) {
			(*updated_solution)[i] = s[i];
		}
	}

	return;

}
//----------------------------------------------------------------------
void Heat::advanceOneStepDivGrad() {
	// instead of computing the Laplacian, compute div(grad)

	// compute laplace u^n
	// 2nd argument is vector<double> 
	vector<double>& s = sol[0];
	vector<double>& s1 = sol[1];

	//	vector<Vec3>* rbf_centers = grid.getRbfCenters();

	// compute laplace u^n 
	der->computeDeriv(Derivative::X, s, x_deriv);
	der->computeDeriv(Derivative::Y, s, y_deriv);

#if 0
	for (int i=0; i < nb_rbf; i++) {
		printf("x,y deriv: %f, %f\n", x_deriv[i], y_deriv[i]);
	}
	double nxd = maxNorm(x_deriv);
	printf(" max norm of x_deriv = %f\n", nxd);
	double nyd = maxNorm(y_deriv);
	printf(" max norm of y_deriv = %f\n", nyd);
	exit(0);
#endif

	der->computeDeriv(Derivative::X, x_deriv, xx_deriv);
	der->computeDeriv(Derivative::Y, y_deriv, yy_deriv);
	//der.computeDeriv(Derivative::LAPL, s, lapl_deriv);

	for (int i = 0; i < s.size(); i++) {
		lapl_deriv[i] = xx_deriv[i] + yy_deriv[i];
	}

	// compute u^* = u^n + dt*lapl(u^n)

	printf("heat, dt= %f, time= %f\n", dt, time);

	// second order time advancement if SECOND is defined
#define SECOND

	// explicit scheme
	for (int i = 0; i < nb_stencils; i++) {
		// first order
		Vec3& v = (*rbf_centers)[i];
		//printf("dt= %f, time= %f\n", dt, time);
		double f = force(v[0], v[1], time);
		//printf("f= %f\n", f);
#ifdef SECOND
		s1[i] = s[i] + 0.5 * dt * (lapl_deriv[i] + f);
#else
		s[i] = s[i] + dt* ( lapl_deriv[i] + f);
#endif
	}

	time = time + dt;

	// reset boundary solution

	// assert bnd_sol.size() == bnd_index.size()
	vector<int>& bnd_index = *boundary_set; //grid.getBoundary();
	int sz = bnd_sol.size();

	//printf("nb bnd pts: %d\n", sz);

	for (int i = 0; i < sz; i++) {
		// first order
		Vec3& v = (*rbf_centers)[bnd_index[i]];
		//printf("bnd[%d] = %f, %f\n", i, v.x(), v.y());
#ifdef SECOND
		s1[bnd_index[i]] = bnd_sol[i];
#else
		s[bnd_index[i]] = bnd_sol[i];
#endif
	}

	double ns1 = maxNorm(s1);
	//printf(" max norm of s1 = %f\n", ns1);
	//exit(0);

#ifdef SECOND
	// compute laplace u^* 
	//der.computeDeriv(Derivative::LAPL, s1, lapl_deriv);
	// compute laplace u^n 
	der->computeDeriv(Derivative::X, s1, x_deriv);
	der->computeDeriv(Derivative::Y, s1, y_deriv);
	der->computeDeriv(Derivative::X, x_deriv, xx_deriv);
	der->computeDeriv(Derivative::Y, y_deriv, yy_deriv);

	for (int i = 0; i < s.size(); i++) {
		lapl_deriv[i] = xx_deriv[i] + yy_deriv[i];
	}

	// compute u^{n+1} = u^n + dt*lapl(u^*)
	for (int i = 0; i < nb_stencils; i++) {
		Vec3& v = (*rbf_centers)[i];
		//printf("i= %d, nb_rbf= %d\n", i, nb_rbf);
		//v.print("v");
		double f = force(v[0], v[1], time + 0.5 * dt);
		s[i] = s[i] + dt * (lapl_deriv[i] + f); // RHS at time+0.5*dt
	}

	// reset boundary solution

	for (int i = 0; i < sz; i++) {
		s[bnd_index[i]] = bnd_sol[i];
	}
#endif

#if 0
	for (int i=0; i < nb_rbf; i++) {
		Vec3* v = rbf_centers[i];
		printf("(%f,%f): T(%d)=%f\n", v.x(), v.y(), i, s[i]);
	}
#endif

	// solution analysis

	vector<double> sol_ex;
	vector<double> sol_error;

	sol_ex.resize(nb_stencils);
	sol_error.resize(nb_stencils);

	// exact solution
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		sol_ex[i] = exactSolution(v[0], v[1], time);
		sol_error[i] = sol_ex[i] - s[i];
	}

	// print error to a file
	//printf("nb_rbf= %d\n", nb_rbf);
	//exit(0);

	FILE* fderr = fopen("error.out", "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fderr, "%f %f %f\n", v[0], v[1], sol_error[i]);
	}
	fclose(fderr);

	// print solution to a file
	FILE* fdsol = fopen("solution.out", "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fdsol, "%f %f %f\n", v.x(), v.y(), s[i]);
	}
	fclose(fdsol);

	double nrm_ex = maxNorm(sol_ex);
	printf("exact max norm: %f\n", nrm_ex);
	double nrm_sol0 = maxNorm(s);
	printf("max norm(s[0]): %f\n", nrm_sol0);
	double nrm_error = maxNorm(sol_error);
	printf("nrm_error= %f\n", nrm_error);

	return;

#undef SECOND
}
//----------------------------------------------------------------------
void Heat::advanceOneStepTwoTerms() {
	// instead of computing the div(D grad)T, compute
	//     grad(D).grad(T) + D lapl(T)

	// compute laplace u^n
	// 2nd argument is vector<double> 
	vector<double>& s = sol[0];
	vector<double>& s1 = sol[1];

	//	vector<Vec3>* rbf_centers = grid.getRbfCenters();

	// compute laplace u^n 
	computeDiffusion(s);
	der->computeDeriv(Derivative::X, s, x_deriv);
	der->computeDeriv(Derivative::Y, s, y_deriv);
	der->computeDeriv(Derivative::X, diffusion, diff_x);
	der->computeDeriv(Derivative::Y, diffusion, diff_y);
	der->computeDeriv(Derivative::LAPL, s, lapl_deriv);

#if 0
	for (int i=0; i < nb_rbf; i++) {
		printf("x,y deriv: %f, %f\n", x_deriv[i], y_deriv[i]);
	}
	double nxd = maxNorm(x_deriv);
	printf(" max norm of x_deriv = %f\n", nxd);
	double nyd = maxNorm(y_deriv);
	printf(" max norm of y_deriv = %f\n", nyd);
	exit(0);
#endif

	//der.computeDeriv(Derivative::X, x_deriv, xx_deriv);
	//der.computeDeriv(Derivative::Y, y_deriv, yy_deriv);
	//der.computeDeriv(Derivative::LAPL, s, lapl_deriv);

	//for (int i=0; i < s.size(); i++) {
	//lapl_deriv[i] = xx_deriv[i] + yy_deriv[i];
	//}

	// compute u^* = u^n + dt*lapl(u^n)

	printf("heat, dt= %f, time= %f\n", dt, time);

	// second order time advancement if SECOND is defined
#define SECOND

	// explicit scheme
	for (int i = 0; i < nb_stencils; i++) {
		// first order
		Vec3& v = (*rbf_centers)[i];
		//printf("dt= %f, time= %f\n", dt, time);
		double f = force(v[0], v[1], time);
		//printf("f= %f\n", f);
		double grad = diff_x[i] * x_deriv[i] + diff_y[i] * y_deriv[i];
#ifdef SECOND
		s1[i] = s[i] + 0.5 * dt * (grad + diffusion[i] * lapl_deriv[i] + f);
#else
		s[i] = s[i] + dt * (grad + diffusion[i]*lapl_deriv[i] + f);
#endif
	}

	time = time + dt;

	// reset boundary solution

	// assert bnd_sol.size() == bnd_index.size()
	vector<int>& bnd_index = *boundary_set; //grid.getBoundary();
	int sz = bnd_sol.size();

	//printf("nb bnd pts: %d\n", sz);

	for (int i = 0; i < sz; i++) {
		// first order
		Vec3& v = (*rbf_centers)[bnd_index[i]];
		//printf("bnd[%d] = %f, %f\n", i, v.x(), v.y());
#ifdef SECOND
		s1[bnd_index[i]] = bnd_sol[i];
#else
		s[bnd_index[i]] = bnd_sol[i];
#endif
	}

	double ns1 = maxNorm(s1);
	//printf(" max norm of s1 = %f\n", ns1);
	//exit(0);

#ifdef SECOND
	// compute laplace u^* 
	//der.computeDeriv(Derivative::LAPL, s1, lapl_deriv);
	// compute laplace u^n 
	computeDiffusion(s1);
	der->computeDeriv(Derivative::X, s1, x_deriv);
	der->computeDeriv(Derivative::Y, s1, y_deriv);
	der->computeDeriv(Derivative::X, diffusion, diff_x);
	der->computeDeriv(Derivative::Y, diffusion, diff_y);
	der->computeDeriv(Derivative::LAPL, s, lapl_deriv);

	// compute u^{n+1} = u^n + dt*lapl(u^*)
	for (int i = 0; i < nb_stencils; i++) {
		Vec3& v = (*rbf_centers)[i];
		//printf("i= %d, nb_rbf= %d\n", i, nb_rbf);
		//v.print("v");
		double f = force(v[0], v[1], time + 0.5 * dt);
		double grad = diff_x[i] * x_deriv[i] + diff_y[i] * y_deriv[i];
		s[i] = s[i] + dt * (grad + diffusion[i] * lapl_deriv[i] + f);
	}

	// reset boundary solution

	for (int i = 0; i < sz; i++) {
		s[bnd_index[i]] = bnd_sol[i];
	}
#endif

#if 0
	for (int i=0; i < nb_rbf; i++) {
		Vec3* v = rbf_centers[i];
		printf("(%f,%f): T(%d)=%f\n", v.x(), v.y(), i, s[i]);
	}
#endif

	// solution analysis

	vector<double> sol_ex;
	vector<double> sol_error;

	sol_ex.resize(nb_stencils);
	sol_error.resize(nb_stencils);

	// exact solution
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		sol_ex[i] = exactSolution(v[0], v[1], time);
		sol_error[i] = sol_ex[i] - s[i];
	}

	// print error to a file
	//printf("nb_rbf= %d\n", nb_rbf);
	//exit(0);

	FILE* fderr = fopen("error.out", "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fderr, "%f %f %f\n", v[0], v[1], sol_error[i]);
	}
	fclose(fderr);

	// print solution to a file
	FILE* fdsol = fopen("solution.out", "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fdsol, "%f %f %f\n", v.x(), v.y(), s[i]);
	}
	fclose(fdsol);

	double nrm_ex = maxNorm(sol_ex);
	printf("exact max norm: %f\n", nrm_ex);
	double nrm_sol0 = maxNorm(s);
	printf("max norm(s[0]): %f\n", nrm_sol0);
	double nrm_error = maxNorm(sol_error);
	printf("nrm_error= %f\n", nrm_error);

	return;

#undef SECOND
}
//----------------------------------------------------------------------
void Heat::initialConditions(std::vector<double> *solution) {
	vector<double>& s = sol[0];
	//double alpha = 1.00;
	//	vector<Vec3>* rbf_centers = getRbfCenters();

	//printf("%d, %d\n", s.size(), rbf_centers.size()); exit(0);

	for (int i = 0; i < s.size(); i++) {
		Vec3& v = (*rbf_centers)[i];
		//s[i] = exp(-alpha*v.square());
		//s[i] = 1. - v.square();
		s[i] = exactSolution(v[0], v[1], 0.);
		//s[i] = 1.0;
		//printf("s[%d]= %f\n", i, s[i]);
	}
	//exit(0);

	vector<int>& bnd_index = *boundary_set; //grid.getBoundary();
	int sz = bnd_index.size();
	bnd_sol.resize(sz);

	for (int i = 0; i < sz; i++) {
		bnd_sol[i] = s[bnd_index[i]];
		Vec3& v = (*rbf_centers)[bnd_index[i]];
		printf("bnd(x,y) = %f, %f\n", v.x(), v.y());
		printf("   sol(bnd)= %f\n", bnd_sol[i]);
	}

	if (solution != NULL) {
		cout << "USING solution (setting initial condition)" << endl;
		for (int i = 0; i < s.size(); i++) {
			(*solution)[i] = s[i];
		}
	}
	//exit(0);
}
//----------------------------------------------------------------------
void Heat::computeDiffusion(vector<double>& sol) {
	//	vector<Vec3>* rbf_centers = getRbfCenters();

	for (int i = 0; i < sol.size(); i++) {
		Vec3& v = (*rbf_centers)[i];
		diffusion[i] = sol[i];
	}
}
//----------------------------------------------------------------------
double Heat::maxNorm() {
	double nrm = 0.;
	for (int i = 0; i < sol[0].size(); i++) {
		double s = abs(sol[0][i]);
		if (s > nrm)
			nrm = s;
	}

	//printf("max norm: %f\n", nrm);

	return nrm;
}
//----------------------------------------------------------------------
double Heat::maxNorm(vector<double> sol) {
	double nrm = 0.;
	for (int i = 0; i < sol.size(); i++) {
		double s = abs(sol[i]);
		if (s > nrm)
			nrm = s;
	}

	//printf("max norm: %f\n", nrm);

	return nrm;
}
//----------------------------------------------------------------------
// solution at point v and time t
double Heat::exactSolution(double x, double y, double t) {
	double r = sqrt(x * x * maji2 + y * y * mini2);

	// if temporal decay is too large, time step will have to decrease

	double T = cos(freq * r) * exp(-decay * t);
	return T;
}
//----------------------------------------------------------------------
double Heat::force(double x, double y, double t) {
	double r2 = x * x * maji2 + y * y * mini2;
	double r4 = x * x * maji4 + y * y * mini4;
	double r = sqrt(r2);
	double f1;
	double f2;

	// if temporal decay is too large, time step will have to decrease

	double nn = freq * r;
	double freq2 = freq * freq;

	f1 = cos(nn) * ((freq2 / r2) * r4 - decay);
	f2 = freq2 * (maji2 + mini2 - r4 / r2);

	if (nn < 1.e-5) {
		f2 *= (1. - nn * nn / 6.);
	} else {
		f2 *= sin(nn) / nn;
	}

	f1 = (f1 + f2) * exp(-decay * t);
	//printf("t= %f, alpha= %f\n", t, alpha);
	//printf("exp= %f, nn= %f, f1= %f, f2= %f\n", exp(-alpha*t), nn, f1, f2);

	return f1;
}
//----------------------------------------------------------------------
