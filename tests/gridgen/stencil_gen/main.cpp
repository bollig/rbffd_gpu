#include "utils/comm/communicator.h"
#include "timer_eb.h" 
#include <stdlib.h>
#include <iostream>
#include "utils/conf/projectsettings.h"
#include "grids/regulargrid.h"

using namespace std;

int main(int argc, char** argv) {
    EB::TimerList tm;
    tm["grid"] = new EB::Timer("Generate grid"); 
    tm["kdtree"] = new EB::Timer("Generate stencils using KDTree"); 
    tm["hash"] = new EB::Timer("Generate stencils using Hash"); 
    tm["brute"] = new EB::Timer("Generate stencils using Brute Force"); 

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
        nz = settings->GetSettingAs<int>("NB_Z", ProjectSettings::required); 
    } 
    if (dim > 3) {
        cout << "ERROR! Dim > 3 Not supported!" << endl;
        exit(EXIT_FAILURE); 
    }

    int ns_nx = settings->GetSettingAs<int>("NS_NB_X", ProjectSettings::optional, "10"); 
    int ns_ny = 1; 
    int ns_nz = 1; 
    if (dim > 1) {
        ns_ny = settings->GetSettingAs<int>("NS_NB_Y", ProjectSettings::optional, "10"); 
    }
    if (dim > 2) {
        ns_nz = settings->GetSettingAs<int>("NS_NB_Z", ProjectSettings::optional, "10"); 
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

    tm["grid"]->start(); 
    Grid* grid = new RegularGrid(nx, ny,  nz, minX, maxX, minY, maxY, minZ, maxZ); 
    grid->setSortBoundaryNodes(sort_nodes); 
    grid->setDebug(debug);
    grid->generate();  
    tm["grid"]->stop();

#if 1
    tm["hash"]->start(); 
    grid->setNSHashDims(ns_nx, ns_ny, ns_nz); 
    grid->generateStencils(stencil_size, Grid::ST_HASH);	// populates the stencil map stored inside the grid class 
    tm["hash"]->stop(); 

    if (debug) {
        std::vector<StencilType>& stencil2 = grid->getStencils();    
        std::cout << "ALL STENCILS: " << std::endl;	
        for (size_t i = 0; i < stencil2.size(); i++) {
            for (size_t j = 0; j < stencil2[i].size(); j++) {
                std::cout << " [" << stencil2[i][j] << "] "; 
            }
            std::cout << std::endl;
        }
    }
#endif 


#if 1
    tm["kdtree"]->start(); 
    grid->generateStencils(stencil_size, Grid::ST_KDTREE);	// populates the stencil map stored inside the grid class 
    tm["kdtree"]->stop(); 

    if (debug) {
        std::vector<StencilType>& stencil2 = grid->getStencils();    
        std::cout << "ALL STENCILS: " << std::endl;	
        for (size_t i = 0; i < stencil2.size(); i++) {
            for (size_t j = 0; j < stencil2[i].size(); j++) {
                std::cout << " [" << stencil2[i][j] << "] "; 
            }
            std::cout << std::endl;
        }
    }
#endif 


#if 0
    tm["brute"]->start(); 
    grid->generateStencils(stencil_size, Grid::ST_BRUTE_FORCE);	// populates the stencil map stored inside the grid class 
    tm["brute"]->stop(); 

    if(debug) {
        std::vector<StencilType>& stencil2 = grid->getStencils();    
        std::cout << "ALL STENCILS: " << std::endl;	
        for (int i = 0; i < stencil2.size(); i++) {
            for (int j = 0; j < stencil2[i].size(); j++) {
                std::cout << " [" << stencil2[i][j] << "] "; 
            }
            std::cout << std::endl;
        }
    }
#endif 

    grid->writeToFile(); 

    //std::vector<StencilType>& stencil = grid->getStencils();    
    StencilType& sten = grid->getStencil(0); 

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
        //    exit(EXIT_FAILURE); 
        }
    }

    delete(grid);
    delete(settings);

    tm.printAll();
    tm.writeToFile();

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
