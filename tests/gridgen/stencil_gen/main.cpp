#include <stdlib.h>
#include <iostream>

#include "utils/conf/projectsettings.h"

#include "grids/regulargrid.h"
#include "grids/stencil_generator.h"

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

    std::vector<StencilType>& stencil = grid->getStencils();    
	StencilType& sten = grid->getStencil(0); 

	std::cout << "ALL STENCILS: " << std::endl;	
	for (int i = 0; i < stencil.size(); i++) {
		for (int j = 0; j < stencil[i].size(); j++) {
			std::cout << " [" << stencil[i][j] << "] "; 
		}
		std::cout << std::endl;
	}

    if (dim == 1) {
       // Extra test for ctest: 
       // When we have a 1D grid, two nodes are boundary. 
       // Recall that these nodes are sorted to the front of the
       // list so we expect this pattern:      
       // B1   [0]  [9]  [2]  [3]  [4] 
       // B2   [1]  [8]  [7]  [6]  [5] 
       // I1   [2]  [9]  [3]  [0]  [4] 
       if ((sten[0] != 0)
       ||  (sten[1] != 1)
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
