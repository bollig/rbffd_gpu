#include <stdlib.h>
#include <map> 

#include "grids/grid_reader.h"

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

#include "stokes_steady_vcl.h"

using namespace std;
using namespace EB;

//----------------------------------------------------------------------
//NOTE: EVERYTHING BELOW IN MAIN WAS COPIED FROM heat_regulargrid_2d/main.cpp
//----------------------------------------------------------------------

int main(int argc, char** argv) {

    std::vector<std::string> grids; 
    grids.push_back("~/GRIDS/md/md031.01024"); 

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


    for (size_t i = 0; i < grids.size(); i++) {
        std::string& grid_name = grids[i]; 
        if (comm_unit->isMaster()) {

            int ns_nx = settings->GetSettingAs<int>("NS_NB_X", ProjectSettings::optional, "10"); 
            int ns_ny = settings->GetSettingAs<int>("NS_NB_Y", ProjectSettings::optional, "10");
            int ns_nz = settings->GetSettingAs<int>("NS_NB_Z", ProjectSettings::optional, "10");

            int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

            int N = settings->GetSettingAs<int>("NB_X", ProjectSettings::optional, "10"); 
            double minX = settings->GetSettingAs<double>("MIN_X", ProjectSettings::optional, "0"); 
            double maxX = settings->GetSettingAs<double>("MAX_X", ProjectSettings::optional, "1"); 

            tm["settings"]->stop(); 

            grid = new GridReader(grid_name, 4); 

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
            subdomain->setCommSize(comm_unit->getSize());
            tm["receive"]->stop(); 
            //subdomain->writeToFile();
        }

        comm_unit->barrier();

        RBFFD* der = new RBFFD(RBFFD::LSFC | RBFFD::XSFC | RBFFD::YSFC | RBFFD::ZSFC, subdomain, dim, comm_unit->getRank()); 

        double alpha = settings->GetSettingAs<double>("EPSILON_C1", ProjectSettings::required); 
        double beta = settings->GetSettingAs<double>("EPSILON_C2", ProjectSettings::required); 
        //der->setVariableEpsilon(subdomain->getStencilRadii(), subdomain->getStencils(), alpha, beta); 
        der->setVariableEpsilon(alpha, beta); 

        // Try loading all the weight files
        int err = der->loadFromFile(RBFFD::XSFC); 
        err += der->loadFromFile(RBFFD::YSFC); 
        err += der->loadFromFile(RBFFD::ZSFC); 
        err += der->loadFromFile(RBFFD::LSFC); 

        if (err) { 
            printf("start computing weights\n");
            tm["weights"]->start(); 

            // NOTE: good test for Direct vs Contour
            // Grid 11x11, vareps=0.05; Look at stencil 12. SHould have -100, 25,
            // 25, 25, 25 (i.e., -4,1,1,1,1) not sure why scaling is off.
            der->computeAllWeightsForAllStencils();
            tm["weights"]->stop(); 

            cout << "end computing weights" << endl;

            der->writeToFile(RBFFD::XSFC);
            der->writeToFile(RBFFD::YSFC);
            der->writeToFile(RBFFD::ZSFC);
            der->writeToFile(RBFFD::LSFC);

            cout << "end write weights to file" << endl;
        }

        Poisson1D_PDE_VCL* pde = new Poisson1D_PDE_VCL(subdomain, der, comm_unit); 

        pde->assemble();
        pde->write_System(); 
        pde->solve();
        pde->write_Solution(); 
    } 

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
