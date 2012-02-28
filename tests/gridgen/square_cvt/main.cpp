#include <stdlib.h>

#include "utils/conf/projectsettings.h"

#include "grids/cvt/cvt.h"

#include "utils/comm/communicator.h"

using namespace std;

int main(int argc, char** argv) {

    Communicator* comm_unit = new Communicator(argc, argv);

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit->getRank());

	
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

    double debug = settings->GetSettingAs<int>("DEBUG", ProjectSettings::optional, "0"); 

	// Generate a CVT with nx*ny*nz nodes, in 1, 2 or 3D with 0 locked boundary nodes, 
	// 20000 samples per iteration for 30 iterations
    Grid* grid = new CVT(nx * ny * nz, dim, 0, NULL, 20000, 60); 
    grid->setDebug(debug);
    grid->setExtents(minX, maxX, minY, maxY, minZ, maxZ); 
    grid->generate(); 
    grid->writeToFile(); 
    
    delete(grid);
    delete(settings);
#if 0
    Grid* grid2 = new Grid(); 
    grid2->loadFromFile("initial_grid.ascii"); 
    grid2->writeToFile("final_grid.ascii");
   
    cout.flush();
#endif 
    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
