#include <stdlib.h>
#include "set"
#include "grid3d.h"

using namespace std;

//----------------------------------------------------------------------

void Grid3D::generateGrid(const char* file, int nb_bnd, int npts)
// npts: total number of points
{
    FILE* fd = fopen(file, "r");
    printf("fd= %ld\n", (long) fd);
    if (fd == 0) {
        printf("could not open file %s\n", file);
        exit(0);
    }

    float x, y, z;

    int nb_rbf = npts;
    coord = new ArrayT<double>(3, nb_rbf);
    ArrayT<double>& coordr = *coord;

    for (int i = 0; i < npts; i++) {
        fscanf(fd, "%g%g\n", &x, &y, &z);
        //printf("x,y= %f, %f\n", x,y);
        coordr(0, i) = x;
        coordr(1, i) = y;
        coordr(2, i) = z;
        rbf_centers.push_back(Vec3(x, y, z));
    }

    for (int i = 0; i < nb_bnd; i++) {
        boundary.push_back(i);
    }
}

void Grid3D::generateGrid() {
    // Create a regular perturbed grid
    rmax = 1.;
    xmin = -rmax;
    xmax = rmax;
    ymin = -rmax;
    ymax = rmax;
    zmin = -rmax;
    zmax = rmax;
    dx = (xmax - xmin) / (nb_x - 1.); // nb_x = nb points
    dy = (ymax - ymin) / (nb_y - 1.);
    dz = (zmax - ymin) / (nb_z - 1.);
    pert = 0.0 * dx;

    printf("dx = %f\n", dx);
    printf("dy = %f\n", dy);
    printf("dz = %f\n", dz);


    nb_rbf = nb_x * nb_x * nb_z;
    maxint = (double) ((1 << 31) - 1);

    coord = new ArrayT<double>(3, nb_rbf); //  1000 rbfs.
    ArrayT<double>& coordr = *coord;

    int count = 0;

    for (int k = 0; k < nb_z; k++) {
        for (int j = 0; j < nb_y; j++) {
            for (int i = 0; i < nb_x; i++) {
                double x = xmin + i * dx + randf(-pert, pert) + 0.12 * dx * 0.5;
                double y = ymin + j * dy + randf(-pert, pert) + 0.12 * dy * 0.5;
                double z = zmin + j * dz + randf(-pert, pert) + 0.12 * dz * 0.5;
                coordr(0, count) = x;
                coordr(1, count) = y;
                coordr(2, count) = z;
                rbf_centers.push_back(Vec3(x, y, z));
                // given i,j, rbf index i: i+nb_x*j
                if (i == 0 || i == (nb_x - 1) || j == 0 || j == (nb_y - 1) || k == 0 || k == (nb_z - 1)) {
                    boundary.push_back(i + nb_x * j + (nb_y * nb_x * k)); // boundary point
                }
                count++;
            }
        }
    }

    printf("count= %d\n", count);
    printf("nb_x, nb_y, nb_z= %d, %d, %d\n", nb_x, nb_y, nb_z);
    printf("rbf_centers.size= %d\n", (int) rbf_centers.size());
    printf("boundary.size= %d\n", (int) boundary.size());
}

//----------------------------------------------------------------------

void Grid3D::generateSubGrid() {
    // subgrid size
    count = 0;

    ArrayT<double>& coordr = *coord;

    // make the grid overlay extend beyond original grid
    double sdx = (xmax - xmin) / (nx); // number of cells: nx*ny
    double sdy = (ymax - ymin) / (ny);
    double sdz = (zmax - zmin) / (nz);
    sdx += sdx / nx;
    sdy += sdy / ny;
    sdz += sdz / nz;

    double sdxmin = xmin - 0.5 * sdx;
    double sdxmax = xmax + 0.5 * sdx;
    double sdymin = ymin - 0.5 * sdy;
    double sdymax = ymax + 0.5 * sdy;
    double sdzmin = zmin - 0.5 * sdz;
    double sdzmax = zmax + 0.5 * sdz;

    sdx = (sdxmax - sdxmin) / nx; // number of cells: nx*ny
    sdy = (sdymax - sdymin) / ny;
    sdz = (sdzmax - sdzmin) / nz;

    /*	double sdz;
            if (nz == 1) {
                    sdz = 0.; // 2D
            } else {
                    sdz = (zmax - zmin) / (nz-1.);
            }
     */

    printf("sdx,sdy,sdz= %f, %f, %f\n", sdx, sdy, sdz);
    printf("xmin,ymin,zmin= %f, %f, %f\n", xmin, ymin, zmin);

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                (*subgrid_index_list)(i, j, k).resize(0);
            }
        }
    }

    //printf("nb_rbf= %d\n", nb_rbf);
    //printf("====\n");
    for (int i = 0; i < nb_rbf; i++) {
        int ix = (int) ((coordr(0, i) - sdxmin) / sdx);
        int iy = (int) ((coordr(1, i) - sdymin) / sdy);
        int iz = (int) ((coordr(2, i) - sdzmin) / sdz);

        //printf("ix,iy= %d, %d\n", ix, iy);
        //printf("coord: %f, %f\n", (coord(0,i)), (coord(1,i)));

        if (ix == nx) ix--;
        if (iy == ny) iy--;
        if (iz == nz) iz--;
        (*subgrid_index_list)(ix, iy, iz).push_back(i); // global index

        // replace by 2D array
        sub_cell.push_back(ix + nx * iy + nx * ny * iz); // which sub_cell is rbf[i] within?
    }

    int sum = 0;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                //printf("size subgrid[%d,%d] = %d\n", i, j, (*subgrid_index_list)(i,j).size());
                sum += (*subgrid_index_list)(i, j, k).size();
            }
        }
    }
    printf("sum = %d\n", sum);
}