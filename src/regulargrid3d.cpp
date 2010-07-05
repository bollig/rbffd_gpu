#include <stdlib.h>
#include <set>
#include "regulargrid3d.h"

using namespace std;

/*----------------------------------------------------------------------*/
RegularGrid3D::RegularGrid3D(int n_x, int n_y, int n_z, double minX, double maxX, double minY, double maxY, double minZ, double maxZ, int stencil_size) : Grid(n_x, n_y, stencil_size) {

    // grid size
    nb_x = nx = n_x;
    nb_y = ny = n_y;
    nb_z = nz = n_z;

    xmin = minX;
    xmax = maxX;

    ymin = minY;
    ymax = maxY;

    zmin = minZ;
    zmax = maxZ;

    dx = (nb_x > 1) ? (xmax - xmin) / (nb_x - 1.) : 0.;
    dy = (nb_y > 1) ? (ymax - ymin) / (nb_y - 1.) : 0.;
    dz = (nb_z > 1) ? (zmax - zmin) / (nb_z - 1.) : 0.;

    // We dont want random pert of the points yet...
    pert = 0.;

    laplacian.resize(nb_x * nb_y * nb_z);

    subgrid_index_list = new ArrayT<vector<int> >(nx, ny, nz);

    this->stencil_size = stencil_size;
    
    this->rbf_centers.clear();
    this->boundary.clear();
}

/*----------------------------------------------------------------------*/
RegularGrid3D::~RegularGrid3D() {

}

/*----------------------------------------------------------------------*/
void RegularGrid3D::generateGrid() {
    // Create a regular perturbed grid
    printf("dx = %f\n", dx);
    printf("dy = %f\n", dy);
    printf("dz = %f\n", dz);

    nb_rbf = nb_x * nb_y * nb_z;
    maxint = (double) ((1 << 31) - 1);

    //  NOTE: for ArrayT, use 3 as first param because index i varies fastest!
    coord = new ArrayT<double>(3, nb_rbf); //  1000 rbfs (3 dims each)
    ArrayT<double>& coordr = *coord;

    int count = 0;

    for (int k = 0; k < nb_z; k++) {
        for (int j = 0; j < nb_y; j++) {
            for (int i = 0; i < nb_x; i++) {
                double x = xmin + i * dx + randf(-pert, pert);
                double y = ymin + j * dy + randf(-pert, pert);
                double z = zmin + k * dz + randf(-pert, pert);
                // TODO: Constrain x,y,z to the limits even when pert is enabled
                
                coordr(0, count) = x;
                coordr(1, count) = y;
                coordr(2, count) = z;
                rbf_centers.push_back(Vec3(x, y, z));
                // given i,j, rbf index i: i+nb_x*j
                if (i == 0 || i == (nb_x - 1) || j == 0 || j == (nb_y - 1) || k == 0 || k == (nb_z - 1)) {
                    boundary.push_back(count); // boundary point
                }
                count++;
            }
        }
    }

    nb_rbf = rbf_centers.size();
    nb_bnd = boundary.size();
    
    this->sortNodes();

    printf("count = %d\n", count);
    printf("nb_x, nb_y, nb_z= %d, %d, %d\n", nb_x, nb_y, nb_z);
    printf("rbf_centers.size= %d\n", (int) rbf_centers.size());
    printf("boundary.size= %d\n", (int) boundary.size());
}

/*----------------------------------------------------------------------*/
// Read the specified "file" and interpret it as a regulargrid3d format
// Each line: {X, Y, Z}
// First nb_bnd lines are boundary points
// Read a total of npts
// file: input file with grid points 1 per row
// nb_bnd: number of boundary points
// npts: total number of points

void RegularGrid3D::generateGrid(const char* file, int nb_bnd, int npts) {
    float x, y, z;

    FILE* fd = fopen(file, "r");
    printf("fd= %ld\n", (long) fd);
    if (fd == 0) {
        printf("could not open file %s\n", file);
        exit(0);
    }

    coord = new ArrayT<double>(3, npts);
    ArrayT<double>& coordr = *coord;

    for (int i = 0; i < npts; i++) {
        fscanf(fd, "%g%g%g\n", &x, &y, &z);
        //printf("x,y= %f, %f\n", x,y);
        coordr(0, i) = x;
        coordr(1, i) = y;
        coordr(2, i) = z;
        rbf_centers.push_back(Vec3(x, y, z));
    }

    for (int i = 0; i < nb_bnd; i++) {
        boundary.push_back(i);
    }

    nb_rbf = rbf_centers.size();
    nb_bnd = boundary.size();
}

