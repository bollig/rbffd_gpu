#include <stdlib.h>
#include <iostream>

#include "utils/conf/projectsettings.h"

#include "grids/regulargrid.h"
#include "grids/domain_decomposition/stencil.h"

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

    Grid* grid = new RegularGrid(nx, ny,  nz, minX, maxX, minY, maxY, minZ, maxZ); 
    grid->setDebug(debug);
    grid->generate(); 
    grid->writeToFile(); 
   
    Stencil* stencil = new Stencil(grid, 15, 1.0);
    stencil->generate();  

    std::cout << "All stencils:" << std::endl;
    std::cout << *stencil << std::endl;

    std::vector<int>& sten = stencil->getStencil(0); 

    if (dim == 1) {
	// Extra test for ctest: 
	// When we have a 1D grid, two nodes are boundary. 
	// Recall that these nodes are sorted to the front of the
	// list so we expect this pattern: 	
	// B1	[0]  [9]  [2]  [3]  [4] 
	// B2	[1]  [8]  [7]  [6]  [5] 
	// I1 	[2]  [9]  [3]  [0]  [4] 
	if ((sten[0] != 0)
  	||  (sten[1] != 9)
	||  (sten[2] != 2)
	||  (sten[3] != 3)
	||  (sten[4] != 4))
	{
	 	exit(EXIT_FAILURE); 
	}
    }

    delete(grid);
    delete(settings);
    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
