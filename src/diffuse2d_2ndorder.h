#ifndef __DIFFUSE2D_2NDORDER_H__
#define __DIFFUSE2D_2NDORDER_H__

#include "parallel_pde.h"

class Diffuse2D_2NDOrder : public ParallelPDE
{

// Implemented methods from abstract class
public: 
	virtual void Initialize(); 
	virtual void Advance();
	virtual double CheckNorm();
	
};

#endif // __DIFFUSE2D_2NDORDER_H__
