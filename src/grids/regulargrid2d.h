#ifndef _REGULAR_GRID_2D_H_
#define _REGULAR_GRID_2D_H_

#include <vector>
#include <Vec3.h>
#include <ArrayT.h>
#include "grid.h"

class RegularGrid2D : public Grid {

public:
        RegularGrid2D(int n_x, int n_y, double minX=0., double maxX=1., double minY=0., double maxY=1., int stencil_size=9);
	~RegularGrid2D();

        virtual void generateGrid();

        // Read the specified "file" and interpret it as a regulargrid2D format
        // Each line: {X, Y, Z}
        // First nb_bnd lines are boundary points
        // Read a total of npts
	// file: input file with grid points 1 per row
	// nb_bnd: number of boundary points
	// npts: total number of points
	virtual void generateGrid(const char* file, int nb_bnd, int npts);
//----------------------------------------------------------------------
};


#endif //_REGULAR_GRID_2D_H_
