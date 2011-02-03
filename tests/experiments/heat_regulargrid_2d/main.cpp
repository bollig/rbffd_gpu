#include <stdlib.h>

#include "pdes/parabolic/heat.h"

#include "grids/regulargrid.h"
#include "grids/stencil_generator.h"

#include "grids/domain.h"
#include "rbffd/derivative_cl.h"
#include "rbffd/derivative_tests.h"

#include "exact_solutions/exact_regulargrid.h"

#include "timer_eb.h"
#include "utils/comm/communicator.h"

using namespace std;
using namespace EB;
//----------------------------------------------------------------------

int main(int argc, char** argv) {
	Timer tm("Total runtime for this processor");
    Timer tm2("Load project settings"); 
    Timer tm3("Setup domain decomposition"); 
    Timer tm4("Advance One Timestep"); 
	// grid should only be valid instance for MASTER
	Grid* grid; 
	Domain* subdomain; 

	tm.start(); 

	Communicator* comm_unit = new Communicator(argc, argv);

	cout << " Got Rank: " << comm_unit->getRank() << endl;
	cout << " Got Size: " << comm_unit->getSize() << endl;

    tm2.start(); 

	ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit);

	int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required); 

	if (comm_unit->getRank() == Communicator::MASTER) {


		int nx = settings->GetSettingAs<int>("NB_X", ProjectSettings::required); 
		int ny = 1; 
		int nz = 1; 
		if (dim > 1) {
			ny = settings->GetSettingAs<int>("NB_Y", ProjectSettings::required); 
		}
		if (dim > 2) {
			nz = settings->GetSettingAs<int> ("NB_Z", ProjectSettings::required); 
		} 
		if (dim > 3) {
			cout << "ERROR! Dim > 3 Not supported!" << endl;
			exit(EXIT_FAILURE); 
		}

		double minX = settings->GetSettingAs<double>("MIN_X", ProjectSettings::optional, "-1."); 	
		double maxX = settings->GetSettingAs<double>("MAX_X", ProjectSettings::optional, "1."); 	
		double minY = settings->GetSettingAs<double>("MIN_Y", ProjectSettings::optional, "-1."); 	
		double maxY = settings->GetSettingAs<double>("MAX_Y", ProjectSettings::optional, "1."); 	
		double minZ = settings->GetSettingAs<double>("MIN_Z", ProjectSettings::optional, "-1."); 	
		double maxZ = settings->GetSettingAs<double>("MAX_Z", ProjectSettings::optional, "1."); 

		double stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

		double dt = settings->GetSettingAs<double>("DT", ProjectSettings::optional, "0.000001"); 

        tm2.stop(); 

		if (dim == 1) {
			grid = new RegularGrid(nx, 1, minX, maxX, 0., 0.); 
		} else if (dim == 2) {
			grid = new RegularGrid(nx, ny, minX, maxX, minY, maxY); 
		} else if (dim == 3) {
			grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
		} else {
			cout << "ERROR! Dim > 3 Not Supported!" << endl;
		}

        // TODO: if (grid->loadFromFile()) 
        {
    		grid->setSortBoundaryNodes(true); 
	    	grid->generate();
		    std::cout << "Generating stencils\n";
		    grid->generateStencils(new StencilGenerator(stencil_size));   // nearest nb_points
		    grid->writeToFile(); 
        }

		int x_subdivisions = comm_unit->getSize();		// reduce this to impact y dimension as well 
		int y_subdivisions = (comm_unit->getSize() - x_subdivisions) + 1; 

		Domain* original_domain = new Domain(grid, dt, comm_unit->getSize()); 
		// pre allocate pointers to all of the subdivisions
		std::vector<Domain*> subdomain_list(x_subdivisions*y_subdivisions);
		// allocate and fill in details on subdivisions

		std::cout << "Generating subdomains\n";
		original_domain->generateDecomposition(subdomain_list, x_subdivisions, y_subdivisions); 

		subdomain = subdomain_list[0]; 
		for (int i = 1; i < comm_unit->getSize(); i++) {
			std::cout << "Sending subdomain[" << i << "]\n";
			comm_unit->sendObject(subdomain_list[i], i); 
		}

	} else {
        tm2.stop(); 
		cout << "MPI RANK " << comm_unit->getRank() << ": waiting to receive subdomain"
			<< endl;
        tm3.start(); 
		subdomain = new Domain(); // EMPTY object that will be filled by MPI

		int status = comm_unit->receiveObject(subdomain, 0); // Receive from CPU (0)
	}

	comm_unit->barrier();
    tm3.stop(); 
    
	subdomain->printVerboseDependencyGraph();

	subdomain->printCenters(subdomain->G_centers, "All Centers Needed by this CPU");

	printf("CHECKING STENCILS: \n");
	for (int irbf = 0; irbf < subdomain->Q_stencils.size(); irbf++) {
		printf("Stencil[%d] = ", irbf);
		if (irbf == subdomain->Q_stencils[irbf][0]) {
			printf("PASS\n");
			subdomain->printStencil(subdomain->Q_stencils[irbf], "S");
		} else {
			printf("FAIL\n");
		}
	}


#if 0
	comm_unit->broadcastObjectUpdates(subdomain);
	comm_unit->barrier();
	Grid* sub_grid = subdomain->getGrid(); 
	// TODO: Clean this up.


	Derivative* der = new DerivativeCL(sub_grid->getNodeList(), sub_grid->getStencils(), sub_grid->getBoundaryIndices().size(), dim, comm_unit->getRank()); 
#else 
	Derivative* der = new DerivativeCL(subdomain->G_centers, subdomain->Q_stencils, subdomain->global_boundary_nodes.size(), dim, comm_unit->getRank());
#endif 

	double epsilon = settings->GetSettingAs<double>("EPSILON");
	der->setEpsilon(epsilon);

#if 0
	printf("start computing weights\n");
	for (int irbf=0; irbf < sub_grid->getStencils().size(); irbf++) {
		der->computeWeights(sub_grid->getNodeList(), sub_grid->getStencil(irbf), irbf);
	}
	cout << "end computing weights" << endl;
#else 
	printf("start computing weights\n");
	for (int irbf=0; irbf < subdomain->Q_stencils.size(); irbf++) {
#if 1
        der->computeWeights(subdomain->G_centers, subdomain->Q_stencils[irbf], irbf);
#else 
        der->computeWeightsSVD(subdomain->G_centers, subdomain->Q_stencils[irbf], irbf, "x"); 
        der->computeWeightsSVD(subdomain->G_centers, subdomain->Q_stencils[irbf], irbf, "y"); 
        der->computeWeightsSVD(subdomain->G_centers, subdomain->Q_stencils[irbf], irbf, "z"); 
        der->computeWeightsSVD(subdomain->G_centers, subdomain->Q_stencils[irbf], irbf, "lapl"); 
#endif 
	}
	cout << "end computing weights" << endl;
#endif 


	if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS", ProjectSettings::optional, "1")) {
		DerivativeTests* der_test = new DerivativeTests();
		der_test->testAllFunctions(*der, *grid);
	}



	// SOLVE HEAT EQUATION

	//EB 4
#if 1
	// Exact Solution ( freq, decay )
	//ExactSolution* exact = new ExactRegularGrid(1.0, 1.0);
	ExactSolution* exact = new ExactRegularGrid(acos(-1.) / 2., 1.);

	Heat heat(exact, subdomain, der, comm_unit->getRank());
	heat.initialConditions(&subdomain->U_G);

	// Send updates according to MPISendable object.

	comm_unit->broadcastObjectUpdates(subdomain);
	comm_unit->barrier();

	// This is HARDCODED because we dont have the ability currently to call
	// maxEig = der.computeEig() and therefore we have a different timestep than
	// the original code. I will address this next.
	//heat.setDt(0.011122);
	heat.setDt(subdomain->dt);
	subdomain->printVector(subdomain->global_boundary_nodes, "INDICES OF GLOBAL BOUNDARY NODES: ");
	// Even with Cartesian, the max norm stays at one. Strange
	int iter;
	for (iter = 0; iter < 1000; iter++) {
		cout << "*********** COMPUTE DERIVATIVES (Iteration: " << iter
			<< ") *************" << endl;
		subdomain->printVector(subdomain->U_G, "INPUT_TO_HEAT_ADVANCE");

        tm4.start(); 
		heat.advanceOneStepWithComm(comm_unit);
        tm4.stop(); 
        char label[256]; 
        sprintf(label, "SOLUTION AFTER %d ITERATIONS", iter+1); 
		subdomain->printVector(subdomain->U_G, label);

		double nrm = heat.maxNorm();
		// TODO : Need to add a "comm_unit->sendTerminate()" to
		// break all processes when problem is encountered
		if (nrm > 1.)
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
		subdomain->writeFinal(grid->getNodeList(), (char*) "FINAL_SOLUTION.txt");
		// TODO print solution to file
		cout << "FINAL ITER: " << iter << endl;
		delete(grid);
	}
printf("REACHED THE END OF MAIN\n");

delete(subdomain);

delete(settings);
delete(comm_unit); 

cout.flush();

tm.end();
tm.printAll();
exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
