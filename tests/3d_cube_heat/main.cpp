#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "Vec3.h"
#include "communicator.h"
#include "regular_grid_3D.h"
#include "domain.h"
#include "heat_pde.h"
#include "parallelpde.h"

using namespace std;

//----------------------------------------------------------------------

int main(int argc, char** argv) {
    double dt = 0.01;
    
    // Valid for all processors
    Communicator* comm_unit;
    ParallelPDE* heat;

    // These will be valid for MASTER only
    Domain* globaldomain;
    AbstractGrid* grid;
 
    comm_unit = new Communicator(argc, argv);
    
    if (comm_unit->isMaster()) { // Master
        // Domain will be sampled regularly in [0,1]x[0,1]x[0,1] with 10 samples per dimension
        grid = new RegularGrid3D(0, 1, 0, 1, 0, 1, 10, 10, 10);

        // The PDE covers the full Domain (of course there could be multiple grids..)
        globaldomain = new Domain(grid);
    }

    // The default ParallelPDE constructor computes a domain decomposition
    // And distributes the subdomains before a comm_unit->barrier()
    heat = new HeatPDE(globaldomain, comm_unit);

    // Continue until our iteration or norm limits break (short circuited)
    int iter = 0;
    while (iter++ < 100 && heat->CheckNorm() < 0.01) {
        // All processors advance by one step
        heat->AdvanceTimestep(dt);

        // Write results to disk every 10 iterations
        if (iter % 10 == 0) {
            // consolidate results back into the global domain object
            heat->Consolidate(globaldomain);

            if (comm_unit->isMaster()) {
                // Dump to "heat_solutionXXXX.dat"
                globaldomain->DumpToFile("heat_solution", iter);
            }
        }
    }

    printf("after heat\n");

    heat->Consolidate(globaldomain);

    if (comm_unit->isMaster()) {
        globaldomain->DumpToFile("final_solution", iter);
        delete(globaldomain);
        delete(grid);
    }
    delete(heat);
    delete(comm_unit);
    
    printf("REACHED THE END OF MAIN\n");

    return 0;
}
//----------------------------------------------------------------------
