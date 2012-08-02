#include <stdlib.h>
#include <map> 

#include "pdes/hyperbolic/cosine_bell.h"
#include "pdes/hyperbolic/cosine_bell_cl.h"
#include "pdes/hyperbolic/cosine_bell_vcl.h"

#include "grids/grid_reader.h"

#include "grids/domain.h"
#include "rbffd/derivative_tests.h"
#include "rbffd/rbffd.h"
#include "rbffd/rbffd_cl.h"
#include "rbffd/rbffd_vcl.h"

#include "./exact_advection.h"

#include "timer_eb.h"
#include "utils/comm/communicator.h"
#include "utils/io/pde_writer.h"

#if USE_VTK
#include "utils/io/vtu_pde_writer.h"
#endif 

std::string md_grid_filename;
int         md_grid_size;
int         md_grid_num_cols;

double sphere_radius; 
double velocity_angle; 
double time_for_one_revolution; 
double start_time; 
double end_time; 
double dt;

int num_timesteps; 
int num_revolutions; 

// Get specific settings for this test case
void fillGlobalProjectSettings(int dim_num, ProjectSettings* settings) {
    md_grid_filename = settings->GetSettingAs<string>("GRID_FILENAME", ProjectSettings::required); 
    md_grid_size = settings->GetSettingAs<int>("GRID_SIZE", ProjectSettings::required); 
    md_grid_num_cols = settings->GetSettingAs<int>("GRID_FILE_NUM_COLS", ProjectSettings::optional, "4"); 
    sphere_radius = settings->GetSettingAs<double>("SPHERE_RADIUS", ProjectSettings::optional, "1.0"); 
    double velocity_angle_denom = settings->GetSettingAs<double>("VELOCITY_ANGLE_DENOM", ProjectSettings::optional, "2"); 
    if (fabs(velocity_angle_denom) > 0.) {  
        velocity_angle = M_PI/velocity_angle_denom;
    } else {
        velocity_angle = 0.;
    }
    start_time = settings->GetSettingAs<double>("START_TIME", ProjectSettings::optional, "0"); 
    num_revolutions = settings->GetSettingAs<double>("NUM_REVOLUTIONS", ProjectSettings::optional, "1"); 
    time_for_one_revolution = settings->GetSettingAs<double>("TIME_FOR_ONE_REVOLUTION", ProjectSettings::optional, "1036800"); 

    num_timesteps = settings->GetSettingAs<double>("NUM_TIMESTEPS", ProjectSettings::optional, "300"); 

    end_time = time_for_one_revolution * num_revolutions;  
    dt = (time_for_one_revolution) / num_timesteps;  
}


// Choose a specific Solution to this test case
ExactSolution* getExactSolution(int dim_num) {
    //double Re = 2.;
    ExactSolution* exact = new ExactAdvection(sphere_radius); 
    return exact;
}

// Choose a specific type of Grid for the test case
Grid* getGrid(int dim_num) {
    Grid* grid = new GridReader(md_grid_filename, md_grid_num_cols, md_grid_size);
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
    tm["gridReader"] = new Timer("[Main] Grid Reader Load File From Disk");
    tm["loadGrid"] = new Timer("[Main] Load Grid (and Stencils) from Disk");
    tm["writeGrid"] = new Timer("[Main] Write Grid (and Stencils) to Disk");
    tm["stencils"] = new Timer("[Main] Stencil generation");
    tm["settings"] = new Timer("[Main] Load settings"); 
    tm["decompose"] = new Timer("[Main] Decompose domain"); 
    tm["consolidate"] = new Timer("[Main] Consolidate subdomain solutions"); 
    tm["updates"] = new Timer("[Main] Broadcast solution updates"); 
    tm["send"] = new Timer("[Main] Send subdomains to other processors (master only)"); 
    tm["receive"] = new Timer("[Main] Receive subdomain from master (clients only)"); 
    tm["timestep"] = new Timer("[Main] Advance One Timestep"); 
    tm["derSetup"] = new Timer("[Main] Setup RBFFD Derivative Settings");
    tm["tests"] = new Timer("[Main] Test stencil weights"); 
    tm["weights"] = new Timer("[Main] Compute all stencils weights"); 
    tm["writeWeights"] = new Timer("[Main] Write Weights to Disk");
    tm["loadWeights"] = new Timer("[Main] Load Weights from Disk");
    tm["oneWeight"] = new Timer("[Main] Compute single stencil weights"); 
    tm["heat_init"] = new Timer("[Main] Initialize heat"); 
    tm["cleanup"] = new Timer("[Main] Destruct objects");
    tm["misc"] = new Timer("[Main] Misc.");
    tm["CFL"] = new Timer("[Main] Compute CFL and max dt.");
    tm["pdewriter"] = new Timer("[Main] Construct PDEWriter");
    // grid should only be valid instance for MASTER
    
    Grid* grid = NULL; 
    Domain* subdomain; 

    tm["total"]->start(); 

    tm["misc"]->start();
    Communicator* comm_unit = new Communicator(argc, argv);

    cout << " Got Rank: " << comm_unit->getRank() << endl;
    cout << " Got Size: " << comm_unit->getSize() << endl;
    
    tm["misc"]->stop();

    tm["settings"]->start(); 

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit->getRank());

    int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required); 

    //-----------------
    fillGlobalProjectSettings(dim, settings);
    //-----------------

    //int max_num_iters = settings->GetSettingAs<int>("MAX_NUM_ITERS", ProjectSettings::required); 
    double max_global_rel_error = settings->GetSettingAs<double>("MAX_GLOBAL_REL_ERROR", ProjectSettings::optional, "1e-1"); 
    double max_local_rel_error = settings->GetSettingAs<double>("MAX_LOCAL_REL_ERROR", ProjectSettings::optional, "1e-1"); 

    int use_gpu = settings->GetSettingAs<int>("USE_GPU", ProjectSettings::optional, "1"); 

    // 0: NO files; <1: Write grid and stencils; <2: Write stencil weights; 3+: Write solution 
    int writeIntermediate = settings->GetSettingAs<int>("WRITE_INTERMEDIATE_FILES", ProjectSettings::optional, "0"); 
    
    int local_err_dump_frequency = settings->GetSettingAs<int>("LOCAL_ERR_DUMP_FREQ", ProjectSettings::optional, "1"); 
    int global_err_dump_frequency = settings->GetSettingAs<int>("GLOBAL_ERR_DUMP_FREQ", ProjectSettings::optional, "5"); 
    int sol_dump_frequency = settings->GetSettingAs<int>("SOL_DUMP_FREQ", ProjectSettings::optional, "-1"); 
    if (sol_dump_frequency == -1) {
        sol_dump_frequency = num_timesteps;
    }

    int prompt_to_continue = settings->GetSettingAs<int>("PROMPT_TO_CONTINUE", ProjectSettings::optional, "0"); 
    int debug = settings->GetSettingAs<int>("DEBUG", ProjectSettings::optional, "0"); 

    int timescheme = settings->GetSettingAs<int>("TIME_SCHEME", ProjectSettings::optional, "2"); 
    int weight_method = settings->GetSettingAs<int>("WEIGHT_METHOD", ProjectSettings::optional, "0"); 
    int compute_eigenvalues = settings->GetSettingAs<int>("DERIVATIVE_EIGENVALUE_TEST", ProjectSettings::optional, "0");
    int useHyperviscosity = settings->GetSettingAs<int>("USE_HYPERVISCOSITY", ProjectSettings::optional, "0");
    //int use_eigen_dt = settings->GetSettingAs<int>("USE_EIG_DT", ProjectSettings::optional, "1");
    //int use_cfl_dt = settings->GetSettingAs<int>("USE_CFL_DT", ProjectSettings::optional, "1");

    int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

    if (comm_unit->isMaster()) {

        int ns_nx = settings->GetSettingAs<int>("NS_NB_X", ProjectSettings::optional, "10"); 
        int ns_ny = settings->GetSettingAs<int>("NS_NB_Y", ProjectSettings::optional, "10");
        int ns_nz = settings->GetSettingAs<int>("NS_NB_Z", ProjectSettings::optional, "10");

        tm["settings"]->stop(); 

        tm["gridReader"]->start();
        grid = getGrid(dim);

        grid->setMaxStencilSize(stencil_size); 
        tm["gridReader"]->stop();

        tm["loadGrid"]->start(); 
        Grid::GridLoadErrType err = grid->loadFromFile(); 
        tm["loadGrid"]->stop(); 
        if (err == Grid::NO_GRID_FILES) 
        {
            printf("************** Generating new Grid **************\n"); 
            //grid->setSortBoundaryNodes(true); 
//            grid->setSortBoundaryNodes(true); 
            tm["grid"]->start(); 
            grid->generate();
            tm["grid"]->stop(); 
            if(writeIntermediate > 0) {
                tm["writeGrid"]->start();
                grid->writeToFile(); 
                tm["writeGrid"]->stop();
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
            if(writeIntermediate > 0) {
                tm["writeGrid"]->start();
                grid->writeToFile(); 
                tm["writeGrid"]->stop();
            }
            tm.writeToFile("gridgen_timer_log"); 
        }

    tm["misc"]->start();
        int x_subdivisions = comm_unit->getSize();		// reduce this to impact y dimension as well 
        int y_subdivisions = (comm_unit->getSize() - x_subdivisions) + 1; 

        // TODO: load subdomain from disk

        // Construct a new domain given a grid. 
        // TODO: avoid filling sets Q, B, etc; just think of it as a copy constructor for a grid
        Domain* original_domain = new Domain(dim, grid, comm_unit->getSize()); 
        // pre allocate pointers to all of the subdivisions
        std::vector<Domain*> subdomain_list(x_subdivisions*y_subdivisions);
        // allocate and fill in details on subdivisions
    tm["misc"]->stop();

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
        comm_unit->receiveObject(subdomain, 0); // Receive from CPU (0)
        tm["receive"]->stop(); 
        //subdomain->writeToFile();
    }

    tm["misc"]->start();
    comm_unit->barrier();

    if (debug) {
        subdomain->printVerboseDependencyGraph();
        subdomain->printNodeList("All Centers Needed by This Process"); 

        printf("CHECKING STENCILS: ");
        for (int irbf = 0; irbf < (int)subdomain->getStencilsSize(); irbf++) {
            //  printf("Stencil[%d] = ", irbf);
            StencilType& s = subdomain->getStencil(irbf); 
            if (irbf == s[0]) {
                //	printf("PASS\n");
                //    subdomain->printStencil(s, "S"); 
            } else {
                printf("FAIL on stencil %d\n", irbf);
                
                tm["total"]->stop();
                tm.printAll();
                tm.writeAllToFile();

                exit(EXIT_FAILURE);
            }
        }
        printf("OK\n");
    }
    tm["misc"]->stop();

    tm["derSetup"]->start();
    RBFFD* der;
    if (use_gpu > 2) {
        der = new RBFFD_VCL(RBFFD::LAMBDA | RBFFD::THETA | RBFFD::HV, subdomain, dim, comm_unit->getRank()); 
    } else if (use_gpu) {
        der = new RBFFD_CL(RBFFD::LAMBDA | RBFFD::THETA | RBFFD::HV, subdomain, dim, comm_unit->getRank()); 
    } else {
        der = new RBFFD(RBFFD::LAMBDA | RBFFD::THETA | RBFFD::HV, subdomain, dim, comm_unit->getRank()); 
    }

    der->setUseHyperviscosity(useHyperviscosity);
    double eps_c1 = settings->GetSettingAs<double>("EPSILON_C1", ProjectSettings::optional, "0.");
    double eps_c2 = settings->GetSettingAs<double>("EPSILON_C2", ProjectSettings::optional, "0.");
    // If both are zero assume we havent set anything
    if (eps_c1 || eps_c2) {
        der->setEpsilonByParameters(eps_c1, eps_c2); 
    } else {
        der->setEpsilonByStencilSize();
    }
    int hv_k = settings->GetSettingAs<int>("HV_K", ProjectSettings::optional, "-1");
    double hv_gamma = settings->GetSettingAs<double>("HV_GAMMA", ProjectSettings::optional, "0");
    if (hv_k != -1) {
        der->setHVScalars(hv_k, hv_gamma);
    }
    der->setWeightType((RBFFD::WeightType)weight_method);
    der->setComputeConditionNumber(true);
    tm["derSetup"]->stop();

    printf("Attempting to load Stencil Weights\n");
 
    // Try loading all the weight files
    tm["loadWeights"]->start();
    int err = der->loadAllWeightsFromFile();
    tm["loadWeights"]->stop();

    if (err) { 
        printf("start computing weights\n");
        tm["weights"]->start(); 

        // NOTE: good test for Direct vs Contour
        // Grid 11x11, vareps=0.05; Look at stencil 12. SHould have -100, 25,
        // 25, 25, 25 (i.e., -4,1,1,1,1) not sure why scaling is off.
        der->computeAllWeightsForAllStencils();
        tm["weights"]->stop(); 

        cout << "end computing weights" << endl;

        if (writeIntermediate > 1) {
            tm["writeWeights"]->start();
            der->writeAllWeightsToFile(); 
            cout << "end write weights to file" << endl;
            tm["writeWeights"]->stop();
        }
    }

    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS", ProjectSettings::optional, "1")) {
        bool weightsPreComputed = false; // X,Y,Z and LAPL are required for tests and were not selected
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

    tm["heat_init"]->start(); 
    ExactSolution* exact = getExactSolution(dim); 

    TimeDependentPDE* pde; 

    if (use_gpu) {
        pde = new CosineBell_VCL(subdomain, (RBFFD_VCL*)der, comm_unit, sphere_radius, velocity_angle, time_for_one_revolution, use_gpu, useHyperviscosity, true);
    } else if (use_gpu) {
        pde = new CosineBell_CL(subdomain, (RBFFD_CL*)der, comm_unit, sphere_radius, velocity_angle, time_for_one_revolution, use_gpu, useHyperviscosity, true);
    } else {
        pde = new CosineBell(subdomain, der, comm_unit, sphere_radius, velocity_angle, time_for_one_revolution, useHyperviscosity, true);
    }

    // This should not influence anything. 
    pde->setStartEndTime(start_time, end_time);

    pde->fillInitialConditions(exact);

#if 0
    // Broadcast updates for timestep, initial conditions for ghost nodes, etc. 
    tm["updates"]->start(); 
    comm_unit->broadcastObjectUpdates(pde);
    comm_unit->barrier();
    tm["updates"]->stop();
#endif 
    tm["heat_init"]->stop(); 

    tm["pdewriter"]->start();
    //TODO:    pde->setRelErrTol(max_global_rel_error); 

    // Setup a logging class that will monitor our iteration and dump intermediate files
#if USE_VTK
    // TODO: update VtuPDEWriter for the new PDE classes
    printf("Using VtuPDEWriter\n");
    PDEWriter* writer = new VtuPDEWriter(subdomain, pde, comm_unit, sol_dump_frequency,0);
#else 
    printf("Using PDEWriter\n");
    PDEWriter* writer = new PDEWriter(subdomain, pde, comm_unit, sol_dump_frequency, 0);
#endif 
    tm["pdewriter"]->stop();

    tm["CFL"]->start();
    // Test DT: 
    // 1) get the minimum avg stencil radius (for stencil area--i.e., dx^2)
    double min_dx = 1000.;
    std::vector<StencilType>& sten = subdomain->getStencils();
    for (size_t i=0; i < sten.size(); i++) {
        // In FD stencils we divide by h^2 for the laplacian. That is the 
        double dx = subdomain->getMinStencilRadius(i);
        if (dx < min_dx) {
            min_dx = dx; 
        }
    }
    // Laplacian = d^2/dx^2
    //double min_sten_area = min_dx*min_dx;
    
    // Get the max velocity at t=0.
    double max_vel = 0.;
    if (use_gpu) {
        max_vel  = ((CosineBell_CL*)pde)->getMaxVelocity(start_time);
    } else {
        max_vel  = ((CosineBell*)pde)->getMaxVelocity(start_time);
    }

	printf("dt = %f, min dx=%f, abs(vel)= %f\n", dt, min_dx, max_vel); 
    printf("CFL Number (for specified dt) = %f\n", max_vel * (dt / min_dx)); 

    // Got this via trial and error for my code. Roughly 0.5 for RBF-FD + RK4.
    double CFL_NUM = 0.50;
    // Assume that we have uniform velocity in each dimension
    double cfl_dt = (CFL_NUM*min_dx) / max_vel;

    printf("Max dt (for RK4-4) = %f\n", cfl_dt); 
    tm["CFL"]->stop();

    // Only use the CFL dt if our current choice is greater and we insist it be used
    if (dt > cfl_dt) {
        printf("ERROR: dt too high. Adjust and re-execute.\n");

        tm["cleanup"]->start();
        delete(der);
        delete(subdomain);
        delete(settings);
        delete(comm_unit); 
        tm["cleanup"]->stop();

        tm["total"]->stop();
        tm.printAll();
        exit(EXIT_FAILURE);
        
//        dt = cfl_dt;
    } 

#if 0
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
#endif 
        std::cout << "[MAIN] ********* USING TIMESTEP dt=" << dt << " ********** " << std::endl;

        //    subdomain->printCenterMemberships(subdomain->G, "G = " );
        //subdomain->printBoundaryIndices("INDICES OF GLOBAL BOUNDARY NODES: ");
        int iter = 0;

        int num_iters = (int) ((end_time - start_time) / dt);
        std::cout << "NUM_ITERS = " << num_iters << std::endl;
                
        if (writeIntermediate > 2) {
            writer->update(iter);
        }

    //    for (iter = 0; iter < num_iters && iter < max_num_iters; iter++) {
        for (int revs = 1; revs <= num_revolutions; revs++) {
            for (int rev_iter =0; rev_iter < num_timesteps; rev_iter++) {

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

                iter++;
                if (writeIntermediate > 2) {
                    writer->update(iter);
                }
            }
#if 1
            if (!(revs % local_err_dump_frequency)) {
                std::cout << "\n*********** Rank " << comm_unit->getRank() << " Local Solution [ Iteration: " << iter << " (t = " << pde->getTime() << ", dt = " << dt << ") ] *************" << endl;
                pde->checkLocalError(exact, max_local_rel_error); 
                pde->checkNorms();
            }

            if (!(revs % global_err_dump_frequency)) {
                tm["consolidate"]->start(); 
                comm_unit->consolidateObjects(pde);
                comm_unit->barrier();
                tm["consolidate"]->stop(); 
                if (comm_unit->isMaster()) {
                    std::cout << "\n*********** Global Solution [ Iteration: " << iter << " (t = " << pde->getTime() << ", dt = " << dt << ") ] *************" << endl;
                    pde->checkGlobalError(exact, grid, max_global_rel_error); 
                }
            }
#endif 
            //        double nrm = pde->maxNorm();
            if (prompt_to_continue && comm_unit->isMaster()) {
                std::string buf; 
                cout << "Press [Enter] to continue" << std::endl;
                cin.get(); 
            }
        }
#if 1
        printf("after %d revolutions, %d iters\n", num_revolutions, iter);

        // NOTE: all local subdomains have a U_G solution which is consolidated
        // into the MASTER process "global_U_G" solution. 
        tm["consolidate"]->start(); 
        comm_unit->consolidateObjects(pde);
        comm_unit->barrier();
        tm["consolidate"]->stop(); 
        //    subdomain->writeGlobalSolutionToFile(-1); 
        std::cout << "Checking Solution on Master\n";
        if (comm_unit->getRank() == 0) {
            if(writeIntermediate > 2) {
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
            std::cout << "============== Verifying Accuracy of Final Solution =============\n"; 
            std::cout << "\n*********** Global Solution [ Iteration: " << iter << " (t = " << pde->getTime() << ") ] *************" << endl;
            pde->checkGlobalError(exact, grid, max_global_rel_error); 
            std::cout << "============== Solution Valid =============\n"; 

            delete(grid);
        }

    cout.flush();
    printf("Cleaning up objects\n");

    tm["cleanup"]->start(); 
    // Writer first so we can dump final solution
    delete(writer);
    delete(pde);
    delete(der);
#endif 
    delete(subdomain);
    delete(settings);
    delete(comm_unit); 
    tm["cleanup"]->stop();

    tm["total"]->stop();
    tm.printAll();
    tm.writeAllToFile();

    printf("\n\nREACHED THE END OF MAIN\n\n");

    return 0;
}
//----------------------------------------------------------------------
