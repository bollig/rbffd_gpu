#include "utils/comm/communicator.h"
#include <stdlib.h>
#include <iostream>

#include "utils/conf/projectsettings.h"

#include "grids/regulargrid.h"

#include "grids/domain.h"


using namespace std;

int main(int argc, char** argv) {

    Communicator* comm_unit = new Communicator(argc, argv);
    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit->getRank());

    // All processes should full this subdomain pointer
    Domain* subdomain; 

    if (comm_unit->getRank() == Communicator::MASTER) { // MASTER THREAD 0
        std::cout << "MPI RANK " << comm_unit->getRank() << ": loading project configuration and generating grid." << endl;

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

        //int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::optional, "15");

        int sort_nodes = settings->GetSettingAs<int>("SORT_NODES", ProjectSettings::optional, "0");
        double debug = settings->GetSettingAs<int>("DEBUG", ProjectSettings::optional, "0");


        Grid* grid = new RegularGrid(nx, ny,  nz, minX, maxX, minY, maxY, minZ, maxZ);
        grid->setSortBoundaryNodes(sort_nodes);
        grid->setDebug(debug);
        grid->generate();
        grid->generateStencils(Grid::ST_BRUTE_FORCE);	// populates the stencil map stored inside the grid class
        grid->writeToFile();

        int x_subdivisions = comm_unit->getSize();		// reduce this to impact y dimension as well 
        int y_subdivisions = (comm_unit->getSize() - x_subdivisions) + 1; 

        Domain* original_domain = new Domain(dim, grid, comm_unit->getSize()); 
        //domain_decomp->printVerboseDependencyGraph(); 

        // pre allocate pointers to all of the subdivisions
        std::vector<Domain*> subdomain_list(x_subdivisions*y_subdivisions);
        // allocate and fill in details on subdivisions
        original_domain->generateDecomposition(subdomain_list, x_subdivisions, y_subdivisions); 

        subdomain = subdomain_list[0]; 
        for (int i = 1; i < comm_unit->getSize(); i++) {
            std::cout << "Sending subdomain[" << i << "]\n";
            comm_unit->sendObject(subdomain_list[i], i); 
        }

        delete(grid);

    } else {
        cout << "MPI RANK " << comm_unit->getRank() << ": waiting to receive subdomain" << endl;

        subdomain = new Domain(); // EMPTY object that will be filled by MPI
        comm_unit->receiveObject(subdomain, Communicator::MASTER); // Receive from CPU (0)
    }


    subdomain->printVerboseDependencyGraph(); 

    delete(subdomain); 
    delete(settings);
    delete(comm_unit);

    // NOTE: if we run with MPI, any EXIT_FAILURE on any of the processors prior to this point
    // WILL cause mpirun to EXIT_FAILURE and our test will fail
    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
