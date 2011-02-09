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
	double max_global_rel_error = settings->GetSettingAs<double>("MAX_GLOBAL_REL_ERROR", ProjectSettings::optional, "1e-2"); 
	int use_gpu = settings->GetSettingAs<double>("USE_GPU", ProjectSettings::optional, "1"); 

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

		double dt = settings->GetSettingAs<double>("DT", ProjectSettings::optional, "0.0001"); 

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

        std::cout << "Generating nodes\n"; 
        if (grid->loadFromFile()) 
        {
            printf("************** Generating new Grid **************\n"); 
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

    	original_domain->printVerboseDependencyGraph();
		std::cout << "Generating subdomains\n";
		original_domain->generateDecomposition(subdomain_list, x_subdivisions, y_subdivisions); 

    	original_domain->printVerboseDependencyGraph();
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

    subdomain->printNodeList("All Centers Needed by This Process"); 

	printf("CHECKING STENCILS: \n");
	for (int irbf = 0; irbf < subdomain->getStencilsSize(); irbf++) {
		printf("Stencil[%d] = ", irbf);
        StencilType& s = subdomain->getStencil(irbf); 
        if (irbf == s[0]) {
			printf("PASS\n");
            subdomain->printStencil(s, "S"); 
		} else {
			printf("FAIL\n");
		}
	}


    // TODO: Derivative constructor for Grid& instead of passing subcomps of subdomain
    Derivative* der; 
    if (use_gpu) {
        der = new DerivativeCL(subdomain->getNodeList(), subdomain->getStencils(), subdomain->getBoundaryIndices().size(), dim, comm_unit->getRank()); 
    } else {
        der = new Derivative(subdomain->getNodeList(), subdomain->getStencils(), subdomain->getBoundaryIndices().size(), dim); 
    }

	double epsilon = settings->GetSettingAs<double>("EPSILON");
	der->setEpsilon(epsilon);


	printf("start computing weights\n");
	for (int irbf=0; irbf < subdomain->getStencilsSize(); irbf++) {
		der->computeWeights(subdomain->getNodeList(), subdomain->getStencil(irbf), irbf);
	}
	cout << "end computing weights" << endl;


	if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS", ProjectSettings::optional, "1")) {
		DerivativeTests* der_test = new DerivativeTests();
		der_test->testAllFunctions(*der, *(subdomain));
	}



	// SOLVE HEAT EQUATION

	//EB 4
#if 1
	// Exact Solution ( freq, decay )
	//ExactSolution* exact = new ExactRegularGrid(1.0, 1.0);
	ExactSolution* exact = new ExactRegularGrid(acos(-1.) / 2., 1.);

    // TODO: udpate heat to construct on grid
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
    heat.setRelErrTol(max_global_rel_error); 


    subdomain->printBoundaryIndices("INDICES OF GLOBAL BOUNDARY NODES: ");
	int iter;
	for (iter = 0; iter < 1000; iter++) {
		cout << "*********** COMPUTE DERIVATIVES (Iteration: " << iter
			<< ") *************" << endl;
        char label[256]; 
        sprintf(label, "LOCAL INPUT SOLUTION [local_indx (global_indx)] FOR ITERATION %d", iter); 
        subdomain->printSolution(label); 
        tm4.start(); 
		heat.advanceOneStepWithComm(comm_unit);
        tm4.stop(); 
        sprintf(label, "LOCAL SOLUTION [local_indx (global_indx)] AFTER %d ITERATIONS", iter+1); 
        subdomain->printSolution(label); 

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


    // NOTE: all local subdomains have a U_G solution which is consolidated
    // into the MASTER process "global_U_G" solution. 
	comm_unit->consolidateObjects(subdomain);

	if (comm_unit->getRank() == 0) {
        // NOTE: the final solution is assembled, but we have to use the 
        // GLOBAL node list instead of a local subdomain node list
		subdomain->writeFinal(grid->getNodeList(), (char*) "FINAL_SOLUTION.txt");
		cout << "FINAL ITER: " << iter << endl;
        std::vector<double> final_sol(grid->getNodeListSize()); 
        ifstream fin; 
        fin.open("FINAL_SOLUTION.txt"); 
#if 1
        int count = 0; 
        for (int count = 0; count < final_sol.size(); count++) {
            Vec3 node; 
            double val;
            fin >> node[0] >> node[1] >> node[2] >> val;
            if (fin.good()) {
                final_sol[count] = val;
               // std::cout << "Read: " << node << ", " << final_sol[count] << std::endl; 
            }
        }
        fin.close();
        std::cout << "============== Verifying Accuracy of Final Solution =============\n"; 
        heat.checkError(final_sol, grid->getNodeList(), max_global_rel_error); 
        std::cout << "============== Solution Valid =============\n"; 
#endif 
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
