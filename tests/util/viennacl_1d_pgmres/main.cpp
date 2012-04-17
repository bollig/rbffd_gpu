#include <stdlib.h>
#include <map> 

#include "grids/regulargrid.h"

#include "grids/domain.h"
#include "rbffd/derivative_tests.h"
#include "rbffd/rbffd.h"
#include "rbffd/rbffd_cl.h"

#include "timer_eb.h"
#include "utils/comm/communicator.h"
#include "utils/io/pde_writer.h"

#if USE_VTK
#include "utils/io/vtu_pde_writer.h"
#endif 

#include "poisson_1d.h"

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
    tm["weights"] = new Timer("[Main] Compute all stencils weights"); 
    tm["oneWeight"] = new Timer("[Main] Compute single stencil weights"); 

    // grid should only be valid instance for MASTER
    Grid* grid=NULL; 
    Domain* subdomain; 

    int dim = 1; 

    tm["total"]->start(); 

    Communicator* comm_unit = new Communicator(argc, argv);

    cout << " Got Rank: " << comm_unit->getRank() << endl;
    cout << " Got Size: " << comm_unit->getSize() << endl;

    tm["settings"]->start(); 

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit->getRank());

    if (comm_unit->isMaster()) {

        int ns_nx = settings->GetSettingAs<int>("NS_NB_X", ProjectSettings::optional, "10"); 
        int ns_ny = settings->GetSettingAs<int>("NS_NB_Y", ProjectSettings::optional, "10");
        int ns_nz = settings->GetSettingAs<int>("NS_NB_Z", ProjectSettings::optional, "10");

        int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

        int N = settings->GetSettingAs<int>("NB_X", ProjectSettings::optional, "10"); 
        double minX = settings->GetSettingAs<double>("MIN_X", ProjectSettings::optional, "0"); 
        double maxX = settings->GetSettingAs<double>("MAX_X", ProjectSettings::optional, "1"); 

        tm["settings"]->stop(); 

        grid = new RegularGrid(N, minX, maxX); 

        grid->setMaxStencilSize(stencil_size); 

        Grid::GridLoadErrType err = grid->loadFromFile(); 
        if (err == Grid::NO_GRID_FILES) 
        {
            printf("\n************** Generating new Grid **************\n"); 
            grid->setSortBoundaryNodes(true); 
            tm["grid"]->start(); 
            grid->generate();
            tm["grid"]->stop(); 
            grid->writeToFile(); 
        } 
        if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
            std::cout << "Generating stencils files\n";
            tm["stencils"]->start(); 
            //            grid->setNSHashDims(ns_nx, ns_ny, ns_nz);
            //            grid->generateStencils(Grid::ST_HASH);   
            //            grid->generateStencils(Grid::ST_BRUTE_FORCE);  
            grid->generateStencils(Grid::ST_KDTREE);   
            tm["stencils"]->stop();
            grid->writeToFile(); 
            tm.writeToFile("gridgen_timer_log"); 
        }

        int x_subdivisions = comm_unit->getSize();		// reduce this to impact y dimension as well 
        int y_subdivisions = (comm_unit->getSize() - x_subdivisions) + 1; 

        // TODO: load subdomain from disk

        // Construct a new domain given a grid. 
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

            // Now that its sent, we can free the memory for domains on other processors.
            delete(subdomain_list[i]);
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

    comm_unit->barrier();

#if 0
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
            exit(EXIT_FAILURE);
        }
    }
    printf("OK\n");
#endif 

    RBFFD* der = new RBFFD(RBFFD::LAPL | RBFFD::X | RBFFD::Y | RBFFD::Z, subdomain, dim, comm_unit->getRank()); 

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

    // Try loading all the weight files
    int err = der->loadFromFile(RBFFD::X); 
    err += der->loadFromFile(RBFFD::Y); 
    err += der->loadFromFile(RBFFD::Z); 
    err += der->loadFromFile(RBFFD::LAPL); 

    if (err) { 
        printf("start computing weights\n");
        tm["weights"]->start(); 

        // NOTE: good test for Direct vs Contour
        // Grid 11x11, vareps=0.05; Look at stencil 12. SHould have -100, 25,
        // 25, 25, 25 (i.e., -4,1,1,1,1) not sure why scaling is off.
        der->computeAllWeightsForAllStencils();
        tm["weights"]->stop(); 

        cout << "end computing weights" << endl;

        der->writeToFile(RBFFD::X);
        der->writeToFile(RBFFD::Y);
        der->writeToFile(RBFFD::Z);
        der->writeToFile(RBFFD::LAPL);

        cout << "end write weights to file" << endl;
    }

    Poisson1D_PDE_VCL* pde = new Poisson1D_PDE_VCL(subdomain, der, comm_unit); 

    pde->assemble();
    pde->write_System(); 
    pde->solve();

#if 0
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
        double dx = subdomain->getStencilRadius(i);
        if (dx < avgdx) {
            avgdx = dx; 
        }
    }
    // Laplacian = d^2/dx^2
    double sten_area = avgdx*avgdx;

    double max_dt = (0.5*sten_area)/ddecay;

    // Not sure where Gordon came up with this parameter.
    // for second centered difference and euler time we have nu = 0.5
    //          dt <= nu/dx^2 
    // is valid for stability in some FD schemes. 
    // double max_dt = 0.2*(sten_area);
    printf("dt = %f (FD suggested max_dt(0.5*dx^2/K)= %f; 0.5dx^2 = %f)\n", dt, max_dt, 0.5*sten_area);
    // This appears to be consistent with Chinchipatnam2006 (Thesis)
    // TODO: get more details on CFL for RBFFD
    // note: checking stability only works if we have all weights for all
    // nodes, so we dont do it in parallel
    if (compute_eigenvalues && (comm_unit->getSize() == 1)) {
        RBFFD::EigenvalueOutput eigs = der->getEigenvalues();
        // Not sure why this is 2
        max_dt = 2. / eigs.max_neg_eig;
        printf("Suggested max_dt based on eigenvalues (2/lambda_max)= %f\n", max_dt);

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
        writer->update(iter);

#if 0
        char label[256]; 
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

        if (!(iter % local_sol_dump_frequency)) {

            std::cout << "\n*********** Rank " << comm_unit->getRank() << " Local Solution [ Iteration: " << iter << " (t = " << pde->getTime() << ") ] *************" << endl;
            pde->checkLocalError(exact, max_local_rel_error); 
        }
        if (!(iter % global_sol_dump_frequency)) {
            tm["consolidate"]->start(); 
            comm_unit->consolidateObjects(pde);
            comm_unit->barrier();
            tm["consolidate"]->stop(); 
            if (comm_unit->isMaster()) {
                std::cout << "\n*********** Global Solution [ Iteration: " << iter << " (t = " << pde->getTime() << ") ] *************" << endl;
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
    }
#if 1
    printf("after heat\n");
    std::cout << "\n*********** Rank " << comm_unit->getRank() << " Final Local Solution [ Iteration: " << iter << " (t = " << pde->getTime() << ") ] *************" << endl;
    pde->checkLocalError(exact, max_local_rel_error); 

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
#endif 

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

    printf("\n\nREACHED THE END OF MAIN\n\n");

    return 0;
}
//----------------------------------------------------------------------
