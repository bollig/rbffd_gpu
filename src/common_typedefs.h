#ifndef __COMMON_TYPEDEFS_H__
#define __COMMON_TYPEDEFS_H__

// Common types in here: 

// Set single or double precision here.
#if 1
typedef double FLOAT;
#else
typedef float FLOAT;
#endif

#include "Vec3.h"

typedef Vec3 NodeType; 
typedef std::vector<int> StencilType; 

// Start with a scalar solution to our equations
typedef double SolutionType;



#endif

