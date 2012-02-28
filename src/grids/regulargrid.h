#ifndef _REGULAR_GRID_H_
#define _REGULAR_GRID_H_

#include "grid_interface.h"

class RegularGrid : public Grid {
	protected: 
		unsigned int nx, ny, nz; 
		double dx, dy, dz;
		
public:

	// 1D
        RegularGrid(int n_x, double minX=0., double maxX=1.);
	// 2D
        RegularGrid(int n_x, int n_y, double minX=0., double maxX=1., double minY=0., double maxY=1.);
	// 3D
        RegularGrid(int n_x, int n_y, int n_z, double minX=0., double maxX=1., double minY=0., double maxY=1., double minZ=0., double maxZ=1.);
	virtual ~RegularGrid();

	// Overrides Grid::generate()
        virtual void generate();

	// Overrides Grid::	
	virtual std::string getFileDetailString(); 

	virtual std::string className() {return "regulargrid";}
};


#endif //_REGULAR_GRID_H_
