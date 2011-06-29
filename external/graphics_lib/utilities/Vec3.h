
#include "Vec3f.h"
#include "Vec3d.h"

#ifdef USE_DOUBLE_VEC3
#define Vec3 Vec3d
#else
#define Vec3 Vec3f
#endif 

//#ifndef _VEC3_H_
//#define _VEC3_H_
//#endif 
