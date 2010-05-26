#include "diffuse2d_2ndorder.h"

Diffuse2D_2NDOrder::Diffuse2D_2NDOrder(ExactSolution* _solution, std::vector<Vec3>* rbf_centers, std::vector<int> boundary_indices, std::vector<int>)
{

}

Diffuse2D_2NDOrder::~Diffuse2D_2NDOrder()
{
	
	
}


void Diffuse2D_2NDOrder::Initialize()
{
	vector<double>& s = solution[0];
	//double alpha = 1.00;
	//	vector<Vec3>* rbf_centers = getRbfCenters();

	//printf("%d, %d\n", s.size(), rbf_centers.size()); exit(0);

	for (int i = 0; i < s.size(); i++) {
		Vec3& v = (*rbf_centers)[i];
		//s[i] = exp(-alpha*v.square());
		//s[i] = 1. - v.square();
		s[i] = exactSolution->at(v, 0.);
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
	//	printf("bnd(x,y) = %f, %f, %f\n", v.x(), v.y(), v.z());
	//	printf("   sol(bnd)= %f\n", bnd_sol[i]);
	}
/*
	// copies solution into outbound buffer for us to print outside of class
	if (solution != NULL) {
		cout << "USING solution (setting initial condition)" << endl;
		for (int i = 0; i < s.size(); i++) {
			(*solution)[i] = s[i];
		}
	}
*/
	//exit(0);
}


void Diffuse2D_2NDOrder::Advance() 
{
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
				fprintf(fderr, "%f %f %f\n", v[0], v[1], sol_error[i]);
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

double Diffuse2D_2NDOrder::CheckNorm()
{
	
}