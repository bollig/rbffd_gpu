#include <stdlib.h>
#include <map> 

#include "pdes/parabolic/heat_pde.h"
#include "pdes/parabolic/heat_pde_cl.h"

#include "grids/regulargrid.h"

#include "grids/domain.h"
#include "rbffd/derivative_tests.h"
#include "rbffd/rbffd.h"
#include "rbffd/rbffd_cl.h"

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
    tm["updates"] = new Timer("[Main] Broadcast solution updates"); 
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
    int timescheme = settings->GetSettingAs<int>("TIME_SCHEME", ProjectSettings::optional, "1"); 

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

        int ns_nx = settings->GetSettingAs<int>("NS_NB_X", ProjectSettings::optional, "10"); 
        int ns_ny = settings->GetSettingAs<int>("NS_NB_Y", ProjectSettings::optional, "10");
        int ns_nz = settings->GetSettingAs<int>("NS_NB_Z", ProjectSettings::optional, "10");

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
#if 0
        std::cout << "Attempting to load Grid from files\n"; 
        if (grid->loadFromFile()) 
        {
            printf("************** Generating new Grid **************\n"); 
            //grid->setSortBoundaryNodes(true); 
            grid->setSortBoundaryNodes(false); 
            tm["grid"]->start(); 
            grid->generate();
            tm["grid"]->stop(); 
            grid->writeToFile(); 
            std::cout << "Generating stencils using Grid::ST_HASH\n";
            tm["stencils"]->start(); 
            grid->setNSHashDims(ns_nx, ns_ny, ns_nz);
            grid->generateStencils(Grid::ST_HASH);   
            tm["stencils"]->stop();
            grid->writeToFile(); 
            tm.writeToFile("gridgen_timer_log"); 
        }
#endif 
        Grid::GridLoadErrType err = grid->loadFromFile(); 
        if (err == Grid::NO_GRID_FILES) 
        {
            printf("************** Generating new Grid **************\n"); 
            //grid->setSortBoundaryNodes(true); 
            grid->setSortBoundaryNodes(false); 
            tm["grid"]->start(); 
            grid->generate();
            tm["grid"]->stop(); 
            grid->writeToFile(); 
        } 
        if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
            std::cout << "Generating stencils using Grid::ST_HASH\n";
            tm["stencils"]->start(); 
            grid->setNSHashDims(ns_nx, ns_ny, ns_nz);
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
        Domain* original_domain = new Domain(grid, comm_unit->getSize()); 
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

    RBFFD* der;
    if (use_gpu) {
        der = new RBFFD_CL(subdomain, dim, comm_unit->getRank()); 
    } else {
        der = new RBFFD(subdomain, dim, comm_unit->getRank()); 
    }

    int use_var_eps = settings->GetSettingAs<int>("USE_VAR_EPSILON", ProjectSettings::optional, "0");
    if (use_var_eps) {
        double alpha = settings->GetSettingAs<double>("VAR_EPSILON_ALPHA", ProjectSettings::optional, "1.0"); 
        double beta = settings->GetSettingAs<double>("VAR_EPSILON_BETA", ProjectSettings::optional, "1.0"); 
        //der->setVariableEpsilon(subdomain->getStencilRadii(), subdomain->getStencils(), alpha, beta); 
        der->setVariableEpsilon(alpha, beta); 
    } else {
        double epsilon = settings->GetSettingAs<double>("EPSILON", ProjectSettings::required);
        der->setEpsilon(epsilon);
    }

    // We specify a rank on the filename because we compute these weights independently within subdomains
    char weight_name[256]; 
    sprintf(weight_name, "x_weights_rank%d.mtx", comm_unit->getRank()); 
    int err = der->loadFromFile(RBFFD::X, weight_name);
    sprintf(weight_name, "y_weights_rank%d.mtx", comm_unit->getRank()); 
    err += der->loadFromFile(RBFFD::Y, weight_name); 
    sprintf(weight_name, "z_weights_rank%d.mtx", comm_unit->getRank()); 
    err += der->loadFromFile(RBFFD::Z, weight_name); 
    sprintf(weight_name, "lapl_weights_rank%d.mtx", comm_unit->getRank()); 
    err += der->loadFromFile(RBFFD::LAPL, weight_name); 

    if (err) { 
        printf("start computing weights\n");
        tm["weights"]->start(); 

        // NOTE: good test for Direct vs Contour
        // Grid 11x11, vareps=0.05; Look at stencil 12. SHould have -100, 25,
        // 25, 25, 25 (i.e., -4,1,1,1,1) not sure why scaling is off.
        der->setWeightType(RBFFD::ContourSVD);
        der->computeAllWeightsForAllStencils();
        tm["weights"]->stop(); 

        cout << "end computing weights" << endl;

        sprintf(weight_name, "x_weights_rank%d.mtx", comm_unit->getRank()); 
        der->writeToFile(RBFFD::X, weight_name); 
        sprintf(weight_name, "y_weights_rank%d.mtx", comm_unit->getRank()); 
        der->writeToFile(RBFFD::Y, weight_name); 
        sprintf(weight_name, "z_weights_rank%d.mtx", comm_unit->getRank()); 
        der->writeToFile(RBFFD::Z, weight_name); 
        sprintf(weight_name, "lapl_weights_rank%d.mtx", comm_unit->getRank()); 
        der->writeToFile(RBFFD::LAPL, weight_name); 
        
        cout << "end write weights to file" << endl;
    }

    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS", ProjectSettings::optional, "1")) {
        bool weightsPreComputed = true; 
        tm["tests"]->start(); 
        // The test class only computes weights if they havent been done already
        DerivativeTests* der_test = new DerivativeTests(der, subdomain, weightsPreComputed);
        if (use_gpu) {
            // Applies weights on both GPU and CPU and compares results for the first 10 stencils
            der_test->compareGPUandCPUDerivs(10);
        }
        // Test approximations to derivatives of functions f(x,y,z) = 0, x, y, xy, etc. etc.
        der_test->testAllFunctions();
        // For now we can only test eigenvalues on an MPI size of 1 (we could distribute with Par-Eiegen solver)
        if (comm_unit->getSize() == 1) {
            if (settings->GetSettingAs<int>("DERIVATIVE_EIGENVALUE_TEST", ProjectSettings::optional, "0")) 
            {
                // FIXME: why does this happen? Perhaps because X Y and Z are unidirectional? 
                // Test X and 4 eigenvalues are > 0
                // Test Y and 30 are > 0
                // Test Z and 36 are > 0
                // NOTE: the 0 here implies we compute the eigenvalues but do not run the iterations of the random perturbation test
                der_test->testEigen(RBFFD::LAPL, 0);
            }
        }
        tm["tests"]->stop();
    }

    // SOLVE HEAT EQUATION

    // Exact Solution ( freq, decay )
    //ExactSolution* exact = new ExactRegularGrid(1.0, 1.0);
    ExactSolution* exact = new ExactRegularGrid(acos(-1.) / 2., 1.);

    TimeDependentPDE* pde; 
    tm["heat_init"]->start(); 
    // We need to provide comm_unit to pass ghost node info
    if (use_gpu) {
        pde = new HeatPDE(subdomain, der, comm_unit, true); 
    } else { 
        // Implies initial conditions are generated
        pde = new HeatPDE(subdomain, der, comm_unit, true);
    }

    pde->fillInitialConditions(exact);

    // Broadcast updates for timestep, initial conditions for ghost nodes, etc. 
    tm["updates"]->start(); 
    comm_unit->broadcastObjectUpdates(pde);
    comm_unit->barrier();
    tm["updates"]->stop();

    tm["heat_init"]->stop(); 

    //TODO:    pde->setRelErrTol(max_global_rel_error); 

    // Setup a logging class that will monitor our iteration and dump intermediate files
#if USE_VTK
    // TODO: update VtuPDEWriter for the new PDE classes
    PDEWriter* writer = new VtuPDEWriter(subdomain, pde, comm_unit, local_sol_dump_frequency, global_sol_dump_frequency);
#else 
    PDEWriter* writer = new PDEWriter(subdomain, pde, comm_unit, local_sol_dump_frequency, global_sol_dump_frequency);
#endif 

//    subdomain->printCenterMemberships(subdomain->G, "G = " );
    //subdomain->printBoundaryIndices("INDICES OF GLOBAL BOUNDARY NODES: ");
    int iter;

    int num_iters = (int) ((end_time - start_time) / dt);
    std::cout << "NUM_ITERS = " << num_iters << std::endl;

    for (iter = 0; iter < num_iters && iter < max_num_iters; iter++) {
        writer->update(iter);

        std::cout << "*********** Solve Heat (Iteration: " << iter << ") *************" << endl;

        char label[256]; 
#if 0
        sprintf(label, "LOCAL INPUT SOLUTION [local_indx (global_indx)] FOR ITERATION %d", iter); 
        pde->printSolution(label); 
#endif 

        tm["timestep"]->start(); 
        pde->advance((TimeDependentPDE::TimeScheme)timescheme, dt);
        tm["timestep"]->stop(); 

        // This just double checks that all procs have ghost node info.
        // pde->advance(..) should broadcast intermediate updates as needed,
        // but updated solution. 
        tm["updates"]->start(); 
        comm_unit->broadcastObjectUpdates(pde);
        comm_unit->barrier();
        tm["updates"]->stop();

        pde->checkLocalError(exact, max_global_rel_error); 
#if 0
        sprintf(label, "LOCAL UPDATED SOLUTION [local_indx (global_indx)] AFTER %d ITERATIONS", iter+1); 
        pde->printSolution(label); 
#endif 

        //        double nrm = pde->maxNorm();
        if (prompt_to_continue && comm_unit->isMaster()) {
            std::string buf; 
            cout << "Press [Enter] to continue" << std::endl;
            cin.get(); 
        }
    }
#if 1
    printf("after heat\n");

    // NOTE: all local subdomains have a U_G solution which is consolidated
    // into the MASTER process "global_U_G" solution. 
    tm["consolidate"]->start(); 
    comm_unit->consolidateObjects(pde);
    comm_unit->barrier();
    tm["consolidate"]->stop(); 
    //    subdomain->writeGlobalSolutionToFile(-1); 
    std::cout << "Checking Solution on Master\n";
    if (comm_unit->getRank() == 0) {
        pde->writeGlobalGridAndSolutionToFile(grid->getNodeList(), (char*) "FINAL_SOLUTION.txt");
#if 0
        // NOTE: the final solution is assembled, but we have to use the 
        // GLOBAL node list instead of a local subdomain node list
        cout << "FINAL ITER: " << iter << endl;
        std::vector<double> final_sol(grid->getNodeListSize()); 
        ifstream fin; 
        fin.open("FINAL_SOLUTION.txt"); 

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
#endif 
        std::cout << "============== Verifying Accuracy of Final Solution =============\n"; 
        pde->checkGlobalError(exact, grid, max_global_rel_error); 
        std::cout << "============== Solution Valid =============\n"; 

        delete(grid);
    }


    cout.flush();
    printf("Cleaning up objects\n");

    // Writer first so we can dump final solution
    delete(writer);
    delete(pde);
#endif 
    delete(subdomain);
    delete(settings);
    delete(comm_unit); 



    tm["total"]->stop();
    tm["total"]->printAll();
    tm["total"]->writeAllToFile();

    printf("REACHED THE END OF MAIN\n");
    
    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
