#include <stdlib.h>

#include "utils/conf/project_settings.h"

#include "grids/regulargrid.h"

#include "utils/comm/communicator.h"

using namespace std;

int main(int argc, char** argv) {

    Communicator* comm_unit = new Communicator(argc, argv);

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit);

	
    int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required); 
    int nx = settings->GetSettingAs<int>("NB_X", ProjectSettings::required); 
    int ny = 1; 
    int nz = 1; 
    if (dim > 1) {
    	int ny = settings->GetSettingAs<int>("NB_Y", ProjectSettings::required); 
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

    Grid* grid; 

    if (dim == 1) {
	    grid = RegularGrid(nx, minX, maxX); 
    } else if (dim == 2) {
	    grid = RegularGrid(nx, ny, minX, maxX, minY, maxY); 
    } else if (dim == 3) {
	    grid = RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
    } else {
	    cout << "ERROR! Dim > 3 Not Supported!" << endl;
    }

    grid->generateGrid(); 

    grid->writeToFile("initial_grid"); 
    
    delete(grid);
    delete(settings);

    Grid* grid2 = Grid(); 
    grid2->readFromFile("initial_grid"); 

    cout.flush();

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
