#ifndef _GRID3D_H_
#define _GRID3D_H_

#include <vector>
#include <Vec3.h>
#include <ArrayT.h>
#include "grid.h"

class Grid3D : public Grid {
public: 
	virtual void generateGrid(); 
	virtual void generateGrid(const char* file, int nb_bnd, int npts);
	virtual void generateSubGrid();
	
};

#endif //_GRID3D_H_
