#include <stdlib.h>
#include <iostream>

#include "utils/conf/projectsettings.h"

#include "grids/regulargrid.h"
#include "grids/stencil_generator.h"

//#include "grids/domain_decomposition.h" 

#include "utils/comm/communicator.h"

using namespace std;

int main(int argc, char** argv) {

    Communicator* comm_unit = new Communicator(argc, argv);

    if (comm_unit->getRank() == Communicator::MASTER) { // MASTER THREAD 0
        std::cout << "MPI RANK " << comm_unit->getRank() << ": loading project configuration and generating grid." << endl;
        ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit);


        int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required);
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

        int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::optional, "15");

        int sort_nodes = settings->GetSettingAs<int>("SORT_NODES", ProjectSettings::optional, "0");
        double debug = settings->GetSettingAs<int>("DEBUG", ProjectSettings::optional, "0");


        // Regular Stencil generator uses brute force nearest neighbor search.
        // KDTreeStencilGenerator uses a kdtree to find the N nearest neighbors for the stencil
        // Other types of StencilGenerators can exist
        StencilGenerator* stencil_generator = new StencilGenerator(stencil_size);

        Grid* grid = new RegularGrid(nx, ny,  nz, minX, maxX, minY, maxY, minZ, maxZ);
        grid->setSortBoundaryNodes(sort_nodes);
        grid->setDebug(debug);
        grid->generate();
        grid->generateStencils(stencil_generator);	// populates the stencil map stored inside the grid class
        grid->writeToFile();

#if 0
        std::vector<GPU*> subdomain_list;
        for (int i = 0; i < comm_unit->getSize(); i++) {
            // TODO: get a grid decomposition
            // subdomain_list[i] = new GPU(grid, local_xmin, local_xmax, local_ymin, local_ymax, local_zmin, local_zmax);
        }

        drivePDE(subdomain); 
#endif 
        // TODO: solve equation
        //Heat heat(subdomain_list[0]->getGrid());
        delete(grid);
        delete(settings);

    } else {
        cout << "MPI RANK " << comm_unit->getRank() << ": waiting to receive subdomain" << endl;
#if 0 
        subdomain = new GPU(); // EMPTY object that will be filled by MPI

        int status = comm_unit->receiveObject(subdomain, Communicator::MASTER); // Receive from CPU (0)
        drivePDE(subdomain); 
#endif 

    }

    // NOTE: if we run with MPI, any EXIT_FAILURE on any of the processors prior to this point
    // WILL cause mpirun to EXIT_FAILURE and our test will fail
    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
