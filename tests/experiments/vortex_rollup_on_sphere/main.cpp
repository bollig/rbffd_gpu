#include <stdlib.h>
#include <map> 

#include "pdes/hyperbolic/vortex_rollup_cl.h"
#include "pdes/hyperbolic/vortex_rollup.h"


#include "grids/grid_reader.h"

#include "grids/domain.h"
#include "rbffd/derivative_tests.h"
#include "rbffd/rbffd.h"
#include "rbffd/rbffd_cl.h"

#include "./exact_rollup.h"

#include "timer_eb.h"
#include "utils/comm/communicator.h"
#include "utils/io/pde_writer.h"

#if USE_VTK
#include "utils/io/vtu_pde_writer.h"
#endif 

std::string md_grid_filename;


// Get specific settings for this test case
void fillGlobalProjectSettings(int dim_num, ProjectSettings* settings) {
    md_grid_filename = settings->GetSettingAs<string>("GRID_FILENAME", ProjectSettings::required); 
}


// Choose a specific Solution to this test case
ExactSolution* getExactSolution(int dim_num) {
    //double Re = 2.;
    ExactSolution* exact = new ExactRollup(); 
    return exact;
}

// Choose a specific type of Grid for the test case
Grid* getGrid(int dim_num) {
    Grid* grid = new GridReader(md_grid_filename);
    return grid; 
}


using namespace std;
using namespace EB;

//----------------------------------------------------------------------
//NOTE: EVERYTHING BELOW IN MAIN WAS COPIED FROM heat_regulargrid_2d/main.cpp
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

    //-----------------
    fillGlobalProjectSettings(dim, settings);
    //-----------------

    int max_num_iters = settings->GetSettingAs<int>("MAX_NUM_ITERS", ProjectSettings::required); 
    double max_global_rel_error = settings->GetSettingAs<double>("MAX_GLOBAL_REL_ERROR", ProjectSettings::optional, "1e-1"); 
    double max_local_rel_error = settings->GetSettingAs<double>("MAX_LOCAL_REL_ERROR", ProjectSettings::optional, "1e-1"); 

    // USE_GPU can be 0, 1, 2. If 0 done use GPU. if 1 use a block approach. if 2 use a thread per stencil approach
    int use_gpu = settings->GetSettingAs<int>("USE_GPU", ProjectSettings::optional, "1"); 
    int benchmark_only = settings->GetSettingAs<int>("BENCHMARK_ONLY", ProjectSettings::optional, "0"); 

    int local_sol_dump_frequency = settings->GetSettingAs<int>("LOCAL_SOL_DUMP_FREQUENCY", ProjectSettings::optional, "100"); 
    int global_sol_dump_frequency = settings->GetSettingAs<int>("GLOBAL_SOL_DUMP_FREQUENCY", ProjectSettings::optional, "200"); 
    int writeIntermediate = settings->GetSettingAs<int>("WRITE_INTERMEDIATE_FILES", ProjectSettings::optional, "0"); 

    int prompt_to_continue = settings->GetSettingAs<int>("PROMPT_TO_CONTINUE", ProjectSettings::optional, "0"); 
    int debug = settings->GetSettingAs<int>("DEBUG", ProjectSettings::optional, "0"); 

    double start_time = settings->GetSettingAs<double>("START_TIME", ProjectSettings::optional, "0.0"); 
    double end_time = settings->GetSettingAs<double>("END_TIME", ProjectSettings::optional, "1.0"); 
    double dt = settings->GetSettingAs<double>("DT", ProjectSettings::optional, "1e-5"); 
    int timescheme = settings->GetSettingAs<int>("TIME_SCHEME", ProjectSettings::optional, "1"); 
    int weight_method = settings->GetSettingAs<int>("WEIGHT_METHOD", ProjectSettings::optional, "1"); 
    int compute_eigenvalues = settings->GetSettingAs<int>("DERIVATIVE_EIGENVALUE_TEST", ProjectSettings::optional, "0");
    int useHyperviscosity = settings->GetSettingAs<int>("USE_HYPERVISCOSITY", ProjectSettings::optional, "0");
    int use_eigen_dt = settings->GetSettingAs<int>("USE_EIG_DT", ProjectSettings::optional, "1");
    int use_cfl_dt = settings->GetSettingAs<int>("USE_CFL_DT", ProjectSettings::optional, "1");

    int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

    if (comm_unit->isMaster()) {

        int ns_nx = settings->GetSettingAs<int>("NS_NB_X", ProjectSettings::optional, "10"); 
        int ns_ny = settings->GetSettingAs<int>("NS_NB_Y", ProjectSettings::optional, "10");
        int ns_nz = settings->GetSettingAs<int>("NS_NB_Z", ProjectSettings::optional, "10");

        tm["settings"]->stop(); 

        grid = getGrid(dim);

        grid->setMaxStencilSize(stencil_size); 

        Grid::GridLoadErrType err = grid->loadFromFile(); 
        if (err == Grid::NO_GRID_FILES) 
        {
            printf("************** Generating new Grid **************\n"); 
            //grid->setSortBoundaryNodes(true); 
            //            grid->setSortBoundaryNodes(true); 
            tm["grid"]->start(); 
            grid->generate();
            tm["grid"]->stop(); 
            if (writeIntermediate) {
                grid->writeToFile(); 
            }
        } 
        if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
            std::cout << "Generating stencils files\n";
            tm["stencils"]->start(); 
            grid->setNSHashDims(ns_nx, ns_ny, ns_nz);
            //            grid->generateStencils(Grid::ST_BRUTE_FORCE);   
            //            grid->generateStencils(Grid::ST_KDTREE);   
            grid->generateStencils(Grid::ST_HASH);   
            tm["stencils"]->stop();
            if (writeIntermediate) {
                grid->writeToFile(); 
            }
            //tm.writeToFile("gridgen_timer_log"); 
        }

        int x_subdivisions = comm_unit->getSize();		// reduce this to impact y dimension as well 
        int y_subdivisions = (comm_unit->getSize() - x_subdivisions) + 1; 

        // TODO: load subdomain from disk

        // Construct a new domain given a grid. 
        // TODO: avoid filling sets Q, B, etc; just think of it as a copy constructor for a grid
        Domain* original_domain = new Domain(dim, grid, comm_unit->getSize()); 
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
        //subdomain->writeToFile();
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
        der = new RBFFD_CL(RBFFD::THETA | RBFFD::LAMBDA | RBFFD::HV, subdomain, dim, comm_unit->getRank()); 
    } else {
        der = new RBFFD(RBFFD::THETA | RBFFD::LAMBDA | RBFFD::HV, subdomain, dim, comm_unit->getRank()); 
    }

    der->setUseHyperviscosity(useHyperviscosity);

    // NOTE: must come before calling setHVScalars 
    der->setEpsilonByStencilSize(stencil_size);

    int hv_k = settings->GetSettingAs<int>("HV_K", ProjectSettings::optional, "4");
    double hv_gamma = settings->GetSettingAs<double>("HV_GAMMA", ProjectSettings::optional, "800");
    der->setHVScalars(hv_k, hv_gamma);
    der->setWeightType((RBFFD::WeightType)weight_method);
    der->setComputeConditionNumber(true);

    // Try loading all the weight files
    int err = der->loadAllWeightsFromFile();

    if (err) { 
        printf("start computing weights\n");
        tm["weights"]->start(); 

        // NOTE: good test for Direct vs Contour
        // Grid 11x11, vareps=0.05; Look at stencil 12. SHould have -100, 25,
        // 25, 25, 25 (i.e., -4,1,1,1,1) not sure why scaling is off.
        der->computeAllWeightsForAllStencils();
        tm["weights"]->stop(); 

        cout << "end computing weights" << endl;

        if (writeIntermediate) {
            der->writeAllWeightsToFile(); 
            cout << "end write weights to file" << endl;
        }
    }

    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS", ProjectSettings::optional, "1")) {
        bool weightsPreComputed = true; 
        bool exitIfTestFailed = settings->GetSettingAs<int>("BREAK_ON_DERIVATIVE_TESTS", ProjectSettings::optional, "1");
        bool exitIfEigTestFailed = settings->GetSettingAs<int>("BREAK_ON_EIG_TESTS", ProjectSettings::optional, "1");
        tm["tests"]->start(); 
        // The test class only computes weights if they havent been done already
        DerivativeTests* der_test = new DerivativeTests(dim, der, subdomain, weightsPreComputed);
        if (use_gpu) {
            // Applies weights on both GPU and CPU and compares results for the first 10 stencils
            der_test->compareGPUandCPUDerivs(10);
        }
        // Test approximations to derivatives of functions f(x,y,z) = 0, x, y, xy, etc. etc.
        der_test->testAllFunctions(exitIfTestFailed);
        // For now we can only test eigenvalues on an MPI size of 1 (we could distribute with Par-Eiegen solver)
        if (comm_unit->getSize() == 1) {
            if (compute_eigenvalues) 
            {
                // FIXME: why does this happen? Perhaps because X Y and Z are unidirectional? 
                // Test X and 4 eigenvalues are > 0
                // Test Y and 30 are > 0
                // Test Z and 36 are > 0
                // NOTE: the 0 here implies we compute the eigenvalues but do not run the iterations of the random perturbation test
                //                der_test->testEigen(RBFFD::LAPL, exitIfEigTestFailed, 0);                
                // NOTE: testHyperviscosity boolean allow us to write the
                // effect of hyperviscosity on the eigenvalues
                der_test->testEigen(RBFFD::LAMBDA, exitIfEigTestFailed, 0);
            }
        }
        tm["tests"]->stop();
    }

    // SOLVE HEAT EQUATION

    ExactSolution* exact = getExactSolution(dim); 

    TimeDependentPDE* pde; 
    tm["heat_init"]->start(); 

    // We need to provide comm_unit to pass ghost node info
    if (use_gpu) {
        pde = new VortexRollup_CL(subdomain, (RBFFD_CL*)der, comm_unit, use_gpu, useHyperviscosity, true);
    } else {
        pde = new VortexRollup(subdomain, der, comm_unit, useHyperviscosity, true);
    }

    // This should not influence anything. 
    pde->setStartEndTime(start_time, end_time);

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

    // Test DT: 
    // 1) get the minimum avg stencil radius (for stencil area--i.e., dx^2)
    double avgdx = 1000.;
    std::vector<StencilType>& sten = subdomain->getStencils();
    for (size_t i=0; i < sten.size(); i++) {
        // In FD stencils we divide by h^2 for the laplacian. That is the 
        double dx = subdomain->getMinStencilRadius(i);
        if (dx < avgdx) {
            avgdx = dx; 
        }
    }
    // Laplacian = d^2/dx^2
    double sten_area = avgdx*avgdx;

    double max_dt = 0;//(0.5*sten_area)/decay;

    // Not sure where Gordon came up with this parameter.
    // for second centered difference and euler time we have nu = 0.5
    //          dt <= nu/dx^2 
    // is valid for stability in some FD schemes. 
    // double max_dt = 0.2*(sten_area);
    printf("dt = %f, min h=%f\n", dt, avgdx); 
    printf("(FD suggested max_dt(0.5*dx^2/K)= %f; 0.5dx^2 = %f)\n", max_dt, 0.5*sten_area);

    // Only use the CFL dt if our current choice is greater and we insist it be used
    if (use_cfl_dt && dt > max_dt) {
        dt = 0.9999*max_dt;
    }

    // This appears to be consistent with Chinchipatnam2006 (Thesis)
    // TODO: get more details on CFL for RBFFD
    // note: checking stability only works if we have all weights for all
    // nodes, so we dont do it in parallel
    if (compute_eigenvalues && (comm_unit->getSize() == 1)) {
        RBFFD::EigenvalueOutput eigs = der->getEigenvalues();
        // Not sure why this is 2 (doesnt seem to be correct)
        max_dt = 2. / eigs.max_neg_eig;
        printf("Suggested max_dt based on eigenvalues (1/lambda_max)= %f\n", max_dt);

        // CFL condition:
        if (dt > max_dt) {
            std::cout << "WARNING! your choice of timestep (" << dt << ") is TOO LARGE for to maintain stability of system. According to eigenvalues, it must be less than " << max_dt << std::endl;
            if (use_eigen_dt) {
                dt = max_dt;
            } else {
                //exit(EXIT_FAILURE);
            }
        }
    }

    std::cout << "[MAIN] ********* USING TIMESTEP dt=" << dt << " ********** " << std::endl;

    //    subdomain->printCenterMemberships(subdomain->G, "G = " );
    //subdomain->printBoundaryIndices("INDICES OF GLOBAL BOUNDARY NODES: ");
    int iter;

    int num_iters = (int) ((end_time - start_time) / dt);
    std::cout << "NUM_ITERS = " << num_iters << std::endl;

    for (iter = 0; iter < num_iters && iter < max_num_iters; iter++) {
        if (writeIntermediate) {
            writer->update(iter);
        }
#if 1
        if (!(iter % local_sol_dump_frequency) && !benchmark_only) {

            std::cout << "\n*********** Rank " << comm_unit->getRank() << " Local Solution [ Iteration: " << iter << " (t = " << pde->getTime() << ", dt = " << dt << ") ] *************" << endl;
            pde->checkLocalError(exact, max_local_rel_error); 
            pde->checkNorms();
        }
        if (!(iter % global_sol_dump_frequency) && !benchmark_only) {
            tm["consolidate"]->start(); 
            comm_unit->consolidateObjects(pde);
            comm_unit->barrier();
            tm["consolidate"]->stop(); 
            if (comm_unit->isMaster()) {
                std::cout << "\n*********** Global Solution [ Iteration: " << iter << " (t = " << pde->getTime() << ", dt = " << dt << ") ] *************" << endl;
                pde->checkGlobalError(exact, grid, max_global_rel_error); 
            }
        }
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

        char label[256]; 
#if 0
        sprintf(label, "LOCAL INPUT SOLUTION [local_indx (global_indx)] FOR ITERATION %d", iter); 
        pde->printSolution(label); 
#endif 

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


    }
#if 1
    printf("DONE SOLVING\n");

    if(writeIntermediate) {
        writer->writeLocal(iter);
    }

    // NOTE: all local subdomains have a U_G solution which is consolidated
    // into the MASTER process "global_U_G" solution. 
    tm["consolidate"]->start(); 
    comm_unit->consolidateObjects(pde);
    comm_unit->barrier();
    tm["consolidate"]->stop(); 
    //    subdomain->writeGlobalSolutionToFile(-1); 
    std::cout << "Checking Solution on Master\n";
    if (comm_unit->getRank() == 0) {
        if(writeIntermediate) {
            pde->writeGlobalGridAndSolutionToFile(grid->getNodeList(), (char*) "FINAL_SOLUTION.txt");
        }
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
        if (!benchmark_only) {
            std::cout << "============== Verifying Accuracy of Final Solution =============\n"; 
            std::cout << "\n*********** Global Solution [ Iteration: " << iter << " (t = " << pde->getTime() << ") ] *************" << endl;
            pde->checkGlobalError(exact, grid, max_global_rel_error); 
            std::cout << "============== Solution Valid =============\n"; 
        }

        delete(grid);
    }


    cout.flush();
    printf("Cleaning up objects\n");

    // Writer first so we can dump final solution
    std::cout << "Writing final solution\n";
    delete(writer);
    delete(pde);
#endif 
    delete(subdomain);
    delete(settings);
    delete(comm_unit); 

    tm["total"]->stop();
    tm["total"]->printAll();
    tm["total"]->writeAllToFile();

    printf("\n\nREACHED THE END OF MAIN\n\n");

    return 0;
}
//----------------------------------------------------------------------
