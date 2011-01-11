#include <stdlib.h>

#include "pdes/parabolic/new_heat.h"

#include "grids/regulargrid2d.h"
#include "grids/regulargrid3d.h"

#include "rbffd/derivative_cl.h"
//#include "rbffd/new_derivative_tests.h"

#include "exact_solutions/exact_regulargrid.h"

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

    double stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

    OriginalGrid* grid; 

    if (dim == 1) {
	    grid = new RegularGrid2D(nx, 1, minX, maxX, 0., 0., stencil_size); 
    } else if (dim == 2) {
	    grid = new RegularGrid2D(nx, ny, minX, maxX, minY, maxY, stencil_size); 
    } else if (dim == 3) {
	    grid = new RegularGrid3D(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ, stencil_size); 
    } else {
	    cout << "ERROR! Dim > 3 Not Supported!" << endl;
    }

    grid->generateGrid();
    grid->computeStencils();   // nearest nb_points
    grid->avgStencilRadius(); 

//    grid->writeToFile(((RegularGrid*)grid)->getFullName("regular_grid",0)); 

    // Compute stencils given a set of generators
    GPU* subdomain = new GPU(-1.,1.,-1.,1.,-1.,1.,0.,comm_unit->getRank(),comm_unit->getSize());      // TODO: get these extents from the cvt class (add constructor to GPU)

    // Clean this up. Have GPU class fill data on constructor. Pass Grid class to constructor.
    // Remove need for extents in constructor.
    subdomain->fillLocalData(grid->getRbfCenters(), grid->getStencil(), grid->getBoundary(), grid->getAvgDist()); // Forms sets (Q,O,R) and l2g/g2l maps
    subdomain->fillVarData(grid->getRbfCenters()); // Sets function values in U

    // Verbosely print the memberships of all nodes within the subdomain
    //subdomain->printCenterMemberships(subdomain->G, "G");

    // 0: 2D problem; 1: 3D problem
    ExactSolution* exact_heat_regulargrid = new ExactRegularGrid(1.0, 1.0);

    // Clean this up. Have the Poisson class construct Derivative internally.
    Derivative* der = new DerivativeCL(subdomain->G_centers, subdomain->Q_stencils, subdomain->global_boundary_nodes.size(), dim);
    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);

    printf("start computing weights\n");
    vector<vector<int> >& stencil = grid->getStencil();
    vector<Vec3>& rbf_centers = grid->getRbfCenters();
    for (int irbf=0; irbf < rbf_centers.size(); irbf++) {
		der->computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "x");
		der->computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "y");
		der->computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "lapl");
	}
    cout << "end computing weights" << endl;

    vector<double> u;
    cout << "start computing derivative (on GPU)" << endl;
	    
    vector<double> xderiv(rbf_centers.size());	
    vector<double> yderiv(rbf_centers.size());    
    vector<double> lapl_deriv(rbf_centers.size());

    der->computeDeriv(Derivative::X, u, xderiv);
    der->computeDeriv(Derivative::Y, u, yderiv);
    der->computeDeriv(Derivative::LAPL, u, lapl_deriv);


#if 0
    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS")) {
        DerivativeTests* der_test = new DerivativeTests();
        der_test->testAllFunctions(*der, *grid);
    }
#endif 


    delete(subdomain);
    delete(grid);
    delete(settings);

    cout.flush();

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
