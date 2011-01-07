#include <stdlib.h>
#include <stdio.h>
#include "regulargrid.h"

using namespace std;


/*----------------------------------------------------------------------*/
RegularGrid::RegularGrid(int n_x, double minX, double maxX)
	: Grid(n_x), nx(n_x), ny(1), nz(1), 
	  xmin(minX), xmax(maxX), ymin(0.), ymax(0.), zmin(0.), zmax(0.)
{
    this->node_list.clear();
    this->boundary_indices.clear();
    this->boundary_normals.clear();
}

/*----------------------------------------------------------------------*/
RegularGrid::RegularGrid(int n_x, int n_y, double minX, double maxX, double minY, double maxY)
	: Grid(n_x * n_y ), nx(n_x), ny(n_y), nz(1), 
	  xmin(minX), xmax(maxX), ymin(minY), ymax(maxY), zmin(0.), zmax(0.)
{
    this->node_list.clear();
    this->boundary_indices.clear();
    this->boundary_normals.clear();
}

/*----------------------------------------------------------------------*/
RegularGrid::RegularGrid(int n_x, int n_y, int n_z, double minX, double maxX, double minY, double maxY, double minZ, double maxZ)
	: Grid(n_x * n_y * n_z), nx(n_x), ny(n_y), nz(n_z), 
	  xmin(minX), xmax(maxX), ymin(minY), ymax(maxY), zmin(minZ), zmax(maxZ)
{
    this->node_list.clear();
    this->boundary_indices.clear();
    this->boundary_normals.clear();
}

/*----------------------------------------------------------------------*/
RegularGrid::~RegularGrid() {

}

/*----------------------------------------------------------------------*/
void RegularGrid::generateGrid() {
    dx = (nx > 1) ? (xmax - xmin) / (nx - 1.) : 0.;
    dy = (ny > 1) ? (ymax - ymin) / (ny - 1.) : 0.;
    dz = (nz > 1) ? (zmax - zmin) / (nz - 1.) : 0.;

    printf("dx = %f\t", dx);
    printf("dy = %f\t", dy);
    printf("dz = %f\n", dz);

	// Should be set inside Grid:: already, but just in case something has changed...
    nb_nodes = nx * ny * nz;

    if (nb_nodes != node_list.size()) {
	    node_list.resize(nb_nodes); 
	    boundary_indices.resize(nb_nodes); 
	    boundary_normals.resize(nb_nodes); 
    }

    unsigned int count = 0;

    for (unsigned int i = 0; i < nx; i++) {
        for (unsigned int j = 0; j < ny; j++) {
   	    for (unsigned int k = 0; k < nz; k++) {

		// Pert defined in Grid::
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                double z = zmin + k * dz;

                node_list.push_back(Vec3(x, y, z));

		if (i == 0 || i == (nx - 1) || j == 0 || j == (ny - 1) || k == 0 || k == (nz - 1)) {
                    boundary_indices.push_back(count); // boundary point

		    double normal_x; 
		    double normal_y; 
		    double normal_z; 
    		    if ( i == 0 ) {
			normal_x = -1.; 
		    } else if (i == nx-1) {
		    	normal_x = 1.;
		    } else {
			normal_x = 0.;
		    }

    		    if ( j == 0 ) {
			normal_y = -1.; 
		    } else if (j == ny-1) {
		    	normal_y = 1.;
		    } else {
			normal_y = 0.;
		    }

    		    if ( k == 0 ) {
			normal_z = -1.; 
		    } else if (k == nz-1) {
		    	normal_z = 1.;
		    } else {
			normal_z = 0.;
		    }
		    boundary_normals.push_back(Vec3(normal_x, normal_y, normal_z)); 

		}
                count++;
            }
        }
    }

    //TODO: Sorting nodes could be disabled when we know our setup is correct
    this->sortNodes();
}


