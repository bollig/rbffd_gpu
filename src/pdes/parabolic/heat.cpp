#include <math.h>
#include "grids/grid.h"
#include "rbffd/derivative.h"
#include "heat.h"
#include "exact_solutions/exact_solution.h"

using namespace std;

Heat::Heat(ExactSolution* _solution, std::vector<Vec3>* rbf_centers_, int stencil_size, std::vector<int>* global_boundary_nodes_, Derivative* der_, int rank) :	rbf_centers(rbf_centers_), boundary_set(global_boundary_nodes_), der(der_),	id(rank), subdomain(NULL), exactSolution(_solution) {
	nb_stencils = stencil_size;
	nb_rbf = rbf_centers->size();

	PI = acos(-1.);
	freq = PI / 2.;
	decay = 1.;

	time = 0.0; // physical time

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

Heat::Heat(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank) : exactSolution(_solution), rbf_centers(&subdomain_->G_centers), boundary_set(&subdomain_->global_boundary_nodes), der(der_), id(rank),subdomain(subdomain_) {
	nb_stencils = subdomain->Q_stencils.size();
	nb_rbf = subdomain->G_centers.size();

	PI = acos(-1.);
	freq = PI / 2.;
	decay = 1.;

	time = 0.0; // physical time

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

Heat::Heat(ExactSolution* _solution, Grid& grid_, Derivative& der_) : exactSolution(_solution), rbf_centers(&(grid_.getRbfCenters())),boundary_set(&(grid_.getBoundary())), der(&der_), id(0), subdomain(NULL) {

    //printf("WARNING!!!!  (BAD CODE DESIGN) DEFAULTING TO EXACT_REGULARGRID.H SOLUTION. YOU SHOULD USE A DIFFERENT CONSTRUCTOR FOR HEAT\n");
    //exactSolution = new ExactRegularGrid(acos(-1.) / 2., 1.);

	nb_stencils = grid_.getStencil().size();
	nb_rbf = grid_.getRbfCenters().size();
	PI = acos(-1.);
	freq = PI / 2.;
	decay = 1.;

	time = 0.0; // physical time

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
			printf("(local: %d), lapl(%f,%f,%f)= %f\t%f\n", i, v.x(), v.y(),v.z(),
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
			double f = exactSolution->tderiv(v, time) - exactSolution->laplacian(v, time);
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
			printf("bnd[%d] = {%d} %f, %f, %f\n", i, bnd_index[i], v.x(), v.y(), v.z());
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
			double f = exactSolution->tderiv(v, time + 0.5 * dt) - exactSolution->laplacian(v, time + 0.5 * dt);
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
			sol_ex[i] = exactSolution->at(v, time);
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
			fprintf(fderr, "%f %f %f %f\n", v[0], v[1], v[2], sol_error[i]);
		}
		fclose(fderr);

		sprintf(filename, "solution.out.%d", id);

		// print solution to a file
		FILE* fdsol = fopen(filename, "w");
		for (int i = 0; i < nb_stencils; i++) {
			Vec3& v = (*rbf_centers)[i];
			fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), s[i]);
		}
		fclose(fdsol);

		double nrm_ex = maxNorm(sol_ex);
                printf("(time: %f) exact max norm: %f\n", time, nrm_ex);
		double nrm_sol0 = maxNorm(s);
                printf("(time: %f) max norm(s[0]): %f\n", time, nrm_sol0);
		double nrm_error = maxNorm(sol_error);
                printf("(time: %f) nrm_error= %f\n", time, nrm_error);

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
                printf("(local: %d), lapl(%f,%f,%f)= %f\t UpdatedSol = %f\n", i, v.x(), v.y(), v.z(),
				lapl_deriv[i], s[i]);
	}
	//exit(0);
#endif

	// compute u^* = u^n + dt*lapl(u^n)

	printf("heat, dt= %f, time= %f\n", dt, time);

	// second order time advancement if SECOND is defined
//#define SECOND

	// explicit scheme
	for (int i = 0; i < nb_stencils; i++) {
		// first order
		Vec3& v = (*rbf_centers)[i];
		//printf("dt= %f, time= %f\n", dt, time);
		double f = force(v, time);
                //printf("force (local: %d) = %f\n", i, f);
                //printf("Before %f,", s[i]);
#ifdef SECOND
		s1[i] = s[i] + 0.5 * dt * (lapl_deriv[i] + f);
#else
		s[i] = s[i] + dt* ( lapl_deriv[i] + f);
#endif
                //printf("After %f\n", s[i]);
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
		printf("bnd[%d] = {%d} %f, %f, %f\n", i, bnd_index[i], v.x(), v.y(), v.z());
#ifdef SECOND
		s1[bnd_index[i]] = bnd_sol[i];
#else
		s[bnd_index[i]] = bnd_sol[i];
#endif
	}

#ifdef SECOND
	// compute laplace u^* 
	der->computeDeriv(Derivative::LAPL, s1, lapl_deriv);
        //cerr << "SECOND ORDER TIME" << endl;
	// compute u^{n+1} = u^n + dt*lapl(u^*)
	for (int i = 0; i < nb_stencils; i++) {
		Vec3& v = (*rbf_centers)[i];
		//printf("i= %d, nb_rbf= %d\n", i, nb_rbf);
		//v.print("v");
		double f = force(v, time+0.5*dt);
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
		printf("(%f,%f,%f): T(%d)=%f\n", v.x(), v.y(), v.z(), i, s[i]);
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
		sol_ex[i] = exactSolution->at(v, time);
		sol_error[i] = sol_ex[i] - s[i];
                printf("%d Force: %f\tLapl: %f\t Solution: %f\t Exact: %f \t Error: %f\n",i, force(v, time), lapl_deriv[i], s[i], sol_ex[i], sol_error[i]);
	}

	// print error to a file
	//	printf("nb_rbf= %d\n", nb_rbf);
	//	exit(0);

	char filename[256];
	sprintf(filename, "error.out.%d", id);

	FILE* fderr = fopen(filename, "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fderr, "%f %f %f %f\n", v[0], v[1], v[2], sol_error[i]);
	}
	fclose(fderr);

	sprintf(filename, "solution.out.%d", id);

	// print solution to a file
	FILE* fdsol = fopen(filename, "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), s[i]);
	}
	fclose(fdsol);

	double nrm_ex = maxNorm(sol_ex);
        printf("(final simulation time: %f) max norm of exact solution: %f\n", time, nrm_ex);
	double nrm_sol0 = maxNorm(s);
        printf("(final simulation time: %f) max norm of computed solution: %f\n", time, nrm_sol0);
	double nrm_error = maxNorm(sol_error);
        printf("(final simulationtime: %f) max nrm of error= %f\n", time, nrm_error);

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
		double f = exactSolution->tderiv(v, time) - exactSolution->laplacian(v, time);
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
		double f = exactSolution->tderiv(v, time + 0.5 * dt) - exactSolution->laplacian(v, time + 0.5 * dt);
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
		sol_ex[i] = exactSolution->at(v, time);
		sol_error[i] = sol_ex[i] - s[i];
	}

	// print error to a file
	//printf("nb_rbf= %d\n", nb_rbf);
	//exit(0);

	FILE* fderr = fopen("error.out", "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fderr, "%f %f %f %f\n", v[0], v[1], v[2], sol_error[i]);
	}
	fclose(fderr);

	// print solution to a file
	FILE* fdsol = fopen("solution.out", "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), s[i]);
	}
	fclose(fdsol);

	double nrm_ex = maxNorm(sol_ex);
        printf("(time: %f) exact max norm: %f\n", time, nrm_ex);
	double nrm_sol0 = maxNorm(s);
        printf("(time: %f) max norm(s[0]): %f\n", time, nrm_sol0);
	double nrm_error = maxNorm(sol_error);
        printf("(time: %f) nrm_error= %f\n", time, nrm_error);

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
		double f = exactSolution->tderiv(v, time) - exactSolution->laplacian(v, time);
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
		double f = exactSolution->tderiv(v, time + 0.5 * dt) - exactSolution->laplacian(v, time + 0.5 * dt);
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
		sol_ex[i] = exactSolution->at(v, time);
		sol_error[i] = sol_ex[i] - s[i];
	}

	// print error to a file
	//printf("nb_rbf= %d\n", nb_rbf);
	//exit(0);

	FILE* fderr = fopen("error.out", "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fderr, "%f %f %f %f\n", v[0], v[1], v[2], sol_error[i]);
	}
	fclose(fderr);

	// print solution to a file
	FILE* fdsol = fopen("solution.out", "w");
	for (int i = 0; i < nb_rbf; i++) {
		Vec3& v = (*rbf_centers)[i];
		fprintf(fdsol, "%f %f %f %f\n", v.x(), v.y(), v.z(), s[i]);
	}
	fclose(fdsol);

	double nrm_ex = maxNorm(sol_ex);
        printf("(time: %f) exact max norm: %f\n", time, nrm_ex);
	double nrm_sol0 = maxNorm(s);
        printf("(time: %f) max norm(s[0]): %f\n", time, nrm_sol0);
	double nrm_error = maxNorm(sol_error);
        printf("(time: %f) nrm_error= %f\n", time, nrm_error);

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
		s[i] = exactSolution->at(v, 0.);
                printf("filling: %f %f %f ==> %f\n", v.x(), v.y(), v.z(),s[i]);
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
		printf("bnd(x,y) = %f, %f, %f\n", v.x(), v.y(), v.z());
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
// The forcing term.
// Here we get the term using the method of manufactured solutions.
// (plugin lapl(u) to get rhs of the equation). 
double Heat::force(Vec3& v, double t) {

    //cout << "TDERIV: " << exactSolution->tderiv(v,t) << "\tLapl: " << exactSolution->laplacian(v,t) << endl;
    //return exactSolution->laplacian(v,t);
    return exactSolution->tderiv(v, t) - exactSolution->laplacian(v, t);
}
//----------------------------------------------------------------------
