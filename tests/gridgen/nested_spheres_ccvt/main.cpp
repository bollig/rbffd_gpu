#include <stdlib.h>

#include "utils/conf/projectsettings.h"

#include "grids/cvt/nested_sphere_cvt.h"

#include "utils/comm/communicator.h"

using namespace std;

int main(int argc, char** argv) {

    Communicator* comm_unit = new Communicator(argc, argv);

    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit);

	
    int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required); 
    int nb_interior = settings->GetSettingAs<int>("NB_INTERIOR", ProjectSettings::required); 
    int nb_inner_boundary = settings->GetSettingAs<int>("NB_INNER_BOUNDARY", ProjectSettings::required); 
    int nb_outer_boundary = settings->GetSettingAs<int>("NB_OUTER_BOUNDARY", ProjectSettings::required); 
    int nb_boundary = nb_inner_boundary + nb_outer_boundary; 

    if (dim > 3) {
	cout << "ERROR! Dim > 3 Not supported!" << endl;
	exit(EXIT_FAILURE); 
    }

    double inner_r = settings->GetSettingAs<double>("INNER_RADIUS", ProjectSettings::optional, "0.5"); 
    double outer_r = settings->GetSettingAs<double>("OUTER_RADIUS", ProjectSettings::optional, "1.0"); 

    double minX = settings->GetSettingAs<double>("MIN_X", ProjectSettings::optional, "-1."); 	
    double maxX = settings->GetSettingAs<double>("MAX_X", ProjectSettings::optional, "1."); 	
    double minY = settings->GetSettingAs<double>("MIN_Y", ProjectSettings::optional, "-1."); 	
    double maxY = settings->GetSettingAs<double>("MAX_Y", ProjectSettings::optional, "1."); 	
    double minZ = settings->GetSettingAs<double>("MIN_Z", ProjectSettings::optional, "-1."); 	
    double maxZ = settings->GetSettingAs<double>("MAX_Z", ProjectSettings::optional, "1."); 

    double debug = settings->GetSettingAs<int>("DEBUG", ProjectSettings::optional, "0"); 

	// Generate a CVT with nx*ny*nz nodes, in 1, 2 or 3D with 0 locked boundary nodes, 
	// 20000 samples per iteration for 30 iterations
    NestedSphereCVT* cgrid = new NestedSphereCVT(nb_interior, nb_inner_boundary, nb_outer_boundary, dim, 0, 20000, 60); 
    cgrid->setExtents(minX, maxX, minY, maxY, minZ, maxZ);
    cgrid->setInnerRadius(inner_r); 
    cgrid->setOuterRadius(outer_r); 

    Grid* grid = cgrid; 
    grid->setDebug(debug);
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
