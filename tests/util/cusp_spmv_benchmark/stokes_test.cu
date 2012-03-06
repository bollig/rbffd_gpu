// THIS IS adapted from verbose_monitor.cu
// PROVIDED BY THE CUSP v0.1 EXAMPLES

#include "grids/grid_reader.h"
#include "rbffd/rbffd.h"

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/gmres.h>
#include <cusp/gallery/poisson.h>
#include <cusp/print.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>

#include <iostream>
using namespace std;

int main(void)
{
    bool writeIntermediate = true; 

    std::vector<std::string> grids; 
    grids.push_back("~/GRIDS/md/md031.01024"); 
    grids.push_back("~/GRIDS/md/md050.02601"); 
    grids.push_back("~/GRIDS/md/md063.04096"); 
    grids.push_back("~/GRIDS/md/md089.08100"); 
    grids.push_back("~/GRIDS/md/md127.16384"); 
    grids.push_back("~/GRIDS/md/md165.27556"); 


    for (size_t i = 0; i < grids.size(); i++) {
        std::string grid_name = grids[i]; 
        // Get contours from rbfzone.blogspot.com to choose eps_c1 and eps_c2 based on stencil_size (n)
        unsigned int stencil_size = 40;
        float eps_c1 = 0.027;
        float eps_c2 = 0.274;


        GridReader grid(grid_name, 4); 
        grid.setMaxStencilSize(stencil_size); 
        // We do not read until generate is called: 

        Grid::GridLoadErrType err = grid.loadFromFile(); 
        if (err == Grid::NO_GRID_FILES) 
        {
            grid.generate();
            if (writeIntermediate) {
                grid.writeToFile(); 
            }
        } 
        if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
            //            grid.generateStencils(Grid::ST_BRUTE_FORCE);   
            grid.generateStencils(Grid::ST_KDTREE);   
            //grid.generateStencils(Grid::ST_HASH);   
            if (writeIntermediate) {
                grid.writeToFile(); 
            }
        }
        
        
        RBFFD der(RBFFD::LSFC | RBFFD::XSFC | RBFFD::YSFC | RBFFD::ZSFC, &grid, 3, 0); 
        int der_err = der.loadAllWeightsFromFile(); 

        if (der_err) {
            der.setEpsilonByParameters(eps_c1, eps_c2);
            der.computeAllWeightsForAllStencils(); 

            if (writeIntermediate) {
                der.writeAllWeightsToFile(); 
            }
        }

        cout << "Allocating device memory\n" << std::endl;
#if 0
        cusp::array1d<double, cusp::host_memory> x(A.num_rows, 1); 
        cusp::array1d<double, cusp::host_memory> b = x; 

        std::cout << "Multiplying matrices\n";
        cusp::multiply(A,x,b); 
#endif 
    }
#if 0
    exit(-1);
    // Convert the 2D memory block to a sparse representation
    cusp::coo_matrix<int, double, cusp::device_memory> A_dev(A);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<double, cusp::device_memory> x_dev(A.num_rows, 0);
    cusp::array1d<double, cusp::device_memory> b_dev(A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-8
    cusp::verbose_monitor<double> monitor(b, 100, 1e-8);

    cout << "Starting GMRES\n" << std::endl;

    // solve the linear system A x = b
    cusp::krylov::gmres(A_dev, x_dev, b_dev, 30, monitor);

    cout << "GMRES complete\n" << std::endl;

    // monitor will report solver progress and results
#endif 
    return EXIT_SUCCESS;
}

