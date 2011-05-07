#include <stdlib.h>
#include <map> 

#include "pdes/parabolic/heat.h"

#include "grids/regulargrid.h"

#include "grids/domain.h"
#include "rbffd/derivative_cl.h"
#include "rbffd/derivative_tests.h"

#include "exact_solutions/exact_regulargrid.h"

#include "timer_eb.h"
#include "utils/comm/communicator.h"
#include "utils/io/pde_writer.h"

#if USE_VTK
#include "utils/io/vtu_pde_writer.h"
#endif 

using namespace std;
using namespace EB;
//----------------------------------------------------------------------

int main(int argc, char** argv) {
    TimerList tm;

    tm["total"] = new Timer("[Main] Total runtime for this proc");
    tm["grid"] = new Timer("[Main] Grid generation");
    tm["stencils"] = new Timer("[Main] Stencil generation");
    tm["settings"] = new Timer("[Main] Load settings"); 
    tm["decompose"] = new Timer("[Main] Decompose domain"); 
    tm["consolidate"] = new Timer("[Main] Consolidate subdomain solutions"); 
    tm["updates"] = new Timer("[Main] Broadcast subdomain update"); 
    tm["send"] = new Timer("[Main] Send subdomains to other processors (master only)"); 
    tm["receive"] = new Timer("[Main] Receive subdomain from master (clients only)"); 
    tm["timestep"] = new Timer("[Main] Advance One Timestep"); 
    tm["tests"] = new Timer("[Main] Test stencil weights"); 
    tm["weights"] = new Timer("[Main] Compute all stencils weights"); 
    tm["oneWeight"] = new Timer("[Main] Compute single stencil weights"); 
    tm["heat_init"] = new Timer("[Main] Initialize heat"); 
    // grid should only be valid instance for MASTER
    Grid* grid; 
    Domain* subdomain; 

    tm["total"]->start(); 

    Communicator* comm_unit = new Communicator(argc, argv);

    cout << " Got Rank: " << comm_unit->getRank() << endl;
    cout << " Got Size: " << comm_unit->getSize() << endl;

    tm["settings"]->start(); 

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit->getRank());

    int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required); 
    int max_num_iters = settings->GetSettingAs<int>("MAX_NUM_ITERS", ProjectSettings::required); 
    double max_global_rel_error = settings->GetSettingAs<double>("MAX_GLOBAL_REL_ERROR", ProjectSettings::optional, "1e-2"); 
    int use_gpu = settings->GetSettingAs<int>("USE_GPU", ProjectSettings::optional, "1"); 
    int local_sol_dump_frequency = settings->GetSettingAs<int>("LOCAL_SOL_DUMP_FREQUENCY", ProjectSettings::optional, "100"); 
    int global_sol_dump_frequency = settings->GetSettingAs<int>("GLOBAL_SOL_DUMP_FREQUENCY", ProjectSettings::optional, "200"); 

    int prompt_to_continue = settings->GetSettingAs<int>("PROMPT_TO_CONTINUE", ProjectSettings::optional, "0"); 
    int debug = settings->GetSettingAs<int>("DEBUG", ProjectSettings::optional, "0"); 

    double start_time = settings->GetSettingAs<double>("START_TIME", ProjectSettings::optional, "0.0"); 
    double end_time = settings->GetSettingAs<double>("END_TIME", ProjectSettings::optional, "1.0"); 
    double dt = settings->GetSettingAs<double>("DT", ProjectSettings::optional, "1e-5"); 

    if (comm_unit->isMaster()) {


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

        int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

        tm["settings"]->stop(); 

        if (dim == 1) {
            grid = new RegularGrid(nx, 1, minX, maxX, 0., 0.); 
        } else if (dim == 2) {
            grid = new RegularGrid(nx, ny, minX, maxX, minY, maxY); 
        } else if (dim == 3) {
            grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
        } else {
            cout << "ERROR! Dim > 3 Not Supported!" << endl;
        }

        grid->setMaxStencilSize(stencil_size); 

        std::cout << "Attempting to load Grid from files\n"; 
        if (grid->loadFromFile()) 
        {
            printf("************** Generating new Grid **************\n"); 
            grid->setSortBoundaryNodes(true); 
            tm["grid"]->start(); 
            grid->generate();
            tm["grid"]->stop(); 
            grid->writeToFile(); 
            std::cout << "Generating stencils using Grid::ST_HASH\n";
            tm["stencils"]->start(); 
            grid->generateStencils(Grid::ST_HASH);   
            tm["stencils"]->stop();
            grid->writeToFile(); 
            tm.writeToFile("gridgen_timer_log"); 
        }

        int x_subdivisions = comm_unit->getSize();		// reduce this to impact y dimension as well 
        int y_subdivisions = (comm_unit->getSize() - x_subdivisions) + 1; 

        // TODO: load subdomain from disk

        // Construct a new domain given a grid. 
        // TODO: avoid filling sets Q, B, etc; just think of it as a copy constructor for a grid
        Domain* original_domain = new Domain(grid, dt, comm_unit->getSize()); 
        // pre allocate pointers to all of the subdivisions
        std::vector<Domain*> subdomain_list(x_subdivisions*y_subdivisions);
        // allocate and fill in details on subdivisions

        std::cout << "Generating subdomains\n";
        tm["decompose"]->start();
        //original_domain->printVerboseDependencyGraph();
        original_domain->generateDecomposition(subdomain_list, x_subdivisions, y_subdivisions); 
        tm["decompose"]->stop();

        tm["send"]->start(); 
        subdomain = subdomain_list[0]; 
        for (int i = 1; i < comm_unit->getSize(); i++) {
            std::cout << "Sending subdomain[" << i << "]\n";
            comm_unit->sendObject(subdomain_list[i], i); 
        }
        tm["send"]->stop(); 

        printf("----------------------\nEND MASTER ONLY\n----------------------\n\n\n");

    } else {
        tm["settings"]->stop(); 
        cout << "MPI RANK " << comm_unit->getRank() << ": waiting to receive subdomain"
            << endl;

        tm["receive"]->start(); 
        subdomain = new Domain(); // EMPTY object that will be filled by MPI
        int status = comm_unit->receiveObject(subdomain, 0); // Receive from CPU (0)
        tm["receive"]->stop(); 
    }

    comm_unit->barrier();

    if (debug) {
        subdomain->printVerboseDependencyGraph();
        subdomain->printNodeList("All Centers Needed by This Process"); 

        printf("CHECKING STENCILS: ");
        for (int irbf = 0; irbf < subdomain->getStencilsSize(); irbf++) {
            //  printf("Stencil[%d] = ", irbf);
            StencilType& s = subdomain->getStencil(irbf); 
            if (irbf == s[0]) {
                //	printf("PASS\n");
                //    subdomain->printStencil(s, "S"); 
            } else {
                printf("FAIL on stencil %d\n", irbf);
                exit(EXIT_FAILURE);
            }
        }
        printf("OK\n");

    }

    // TODO: Derivative constructor for Grid& instead of passing subcomps of subdomain
    Derivative* der; 
    if (use_gpu) {
        der = new DerivativeCL(subdomain->getNodeList(), subdomain->getStencils(), subdomain->getBoundaryIndices().size(), dim, comm_unit->getRank()); 
    } else {
        der = new Derivative(subdomain->getNodeList(), subdomain->getStencils(), subdomain->getBoundaryIndices().size(), dim); 
    }

    int use_var_eps = settings->GetSettingAs<int>("USE_VAR_EPSILON", ProjectSettings::optional, "0");
    if (use_var_eps) {
        double alpha = settings->GetSettingAs<double>("VAR_EPSILON_ALPHA", ProjectSettings::optional, "1.0"); 
        double beta = settings->GetSettingAs<double>("VAR_EPSILON_BETA", ProjectSettings::optional, "1.0"); 
        der->setVariableEpsilon(subdomain->getStencilRadii(), subdomain->getStencils(), alpha, beta); 
    } else {
        double epsilon = settings->GetSettingAs<double>("EPSILON", ProjectSettings::required);
        der->setEpsilon(epsilon);
    }



    printf("start computing weights\n");
    tm["weights"]->start(); 
    for (int irbf=0; irbf < subdomain->getStencilsSize(); irbf++) {
        tm["oneWeight"]->start(); 
        der->computeWeights(subdomain->getNodeList(), subdomain->getStencil(irbf), irbf);
        tm["oneWeight"]->stop();
    }
    der->writeToFile(Derivative::X, "x_weights.mtx"); 
    der->writeToFile(Derivative::Y, "y_weights.mtx"); 
    der->writeToFile(Derivative::Z, "z_weights.mtx"); 
    der->writeToFile(Derivative::LAPL, "lapl_weights.mtx"); 
    tm["weights"]->stop(); 
    cout << "end computing weights" << endl;

    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS", ProjectSettings::optional, "1")) {
        tm["tests"]->start(); 
        DerivativeTests* der_test = new DerivativeTests();
        der_test->testAllFunctions(*der, *(subdomain));
        if (comm_unit->getSize()) {
            if (settings->GetSettingAs<int>("DERIVATIVE_EIGENVALUE_TEST", ProjectSettings::optional, "0")) {
                der_test->testEigen(*der, *(subdomain));
            }
        }
        tm["tests"]->stop();
    }



    // SOLVE HEAT EQUATION

    // Exact Solution ( freq, decay )
    //ExactSolution* exact = new ExactRegularGrid(1.0, 1.0);
    ExactSolution* exact = new ExactRegularGrid(acos(-1.) / 2., 1.);

    // TODO: udpate heat to construct on grid
    tm["heat_init"]->start(); 
    Heat* heat = new Heat(exact, subdomain, der, comm_unit->getRank());
    heat->initialConditions(&subdomain->U_G);
    tm["heat_init"]->stop(); 

    // This is HARDCODED because we dont have the ability currently to call
    // maxEig = der.computeEig() and therefore we have a different timestep than
    // the original code. I will address this next.
    //heat->setDt(0.011122);
    heat->setDt(subdomain->dt);
    heat->setRelErrTol(max_global_rel_error); 

    // Send updates according to MPISendable object.
    tm["updates"]->start(); 
    comm_unit->broadcastObjectUpdates(subdomain);
    comm_unit->barrier();
    tm["updates"]->stop();
    // Setup a logging class that will monitor our iteration and dump intermediate files
#if USE_VTK
    PDEWriter* writer = new VtuPDEWriter(subdomain, heat, comm_unit, local_sol_dump_frequency, global_sol_dump_frequency);
#else 
    PDEWriter* writer = new PDEWriter(subdomain, heat, comm_unit, local_sol_dump_frequency, global_sol_dump_frequency);
#endif 

    //subdomain->printBoundaryIndices("INDICES OF GLOBAL BOUNDARY NODES: ");
    int iter;

    int num_iters = (int) ((end_time - start_time) / dt);
    std::cout << "NUM_ITERS = " << num_iters << std::endl;

    for (iter = 0; iter < num_iters && iter < max_num_iters; iter++) {
        writer->update(iter);

        std::cout << "*********** Solve Heat (Iteration: " << iter << ") *************" << endl;

        char label[256]; 
        if (debug) {
            sprintf(label, "LOCAL INPUT SOLUTION [local_indx (global_indx)] FOR ITERATION %d", iter); 
            subdomain->printSolution(label); 
        }
        tm["timestep"]->start(); 
        heat->advanceOneStepWithComm(comm_unit);
        tm["timestep"]->stop(); 

        if (debug) {
            sprintf(label, "LOCAL UPDATED SOLUTION [local_indx (global_indx)] AFTER %d ITERATIONS", iter+1); 
            subdomain->printSolution(label); 
        }

        //        double nrm = heat->maxNorm();
        // TODO : Need to add a "comm_unit->sendTerminate()" to
        // break all processes when problem is encountered
        //        if (nrm > 1.)
        //            break;

        if (prompt_to_continue && comm_unit->isMaster()) {
            std::string buf; 
            cout << "Press [Enter] to continue" << std::endl;
            cin.get(); 
        }

    }

    printf("after heat\n");

    // NOTE: all local subdomains have a U_G solution which is consolidated
    // into the MASTER process "global_U_G" solution. 
    tm["consolidate"]->start(); 
    comm_unit->consolidateObjects(subdomain);
    comm_unit->barrier();
    tm["consolidate"]->stop(); 
    //    subdomain->writeGlobalSolutionToFile(-1); 
    std::cout << "Checking Solution on Master\n";
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
        heat->checkError(final_sol, grid->getNodeList(), max_global_rel_error); 
        std::cout << "============== Solution Valid =============\n"; 
#endif 
        delete(grid);
    }

cout.flush();
printf("Cleaning up objects\n");

// Writer first so we can dump final solution
delete(writer);
delete(heat);
delete(subdomain);
delete(settings);
delete(comm_unit); 


printf("REACHED THE END OF MAIN\n");

tm["total"]->stop();
tm["total"]->printAll();
tm["total"]->writeAllToFile();

exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
