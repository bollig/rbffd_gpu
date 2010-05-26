#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

/**
 * Geometry is an abstract class that represents a collection of patches
 * that form a closed object. Therefore, any geometry has both a surface
 * and volume integral, as well as the ability to project interior or
 * exterior points onto the surface.
 */

#include "parametric_patch.h"

#include <vector>

class Geometry
{
private:
	std::vector<ParametricPatch> patches;

public:

        // Project a point to the boundary defined by the geometry
        virtual Vec3 project(Vec3 pt) = 0;

        // Compute the surface integral of the geometry
	virtual double surfaceIntegral() = 0;
};

#endif //__GEOMETRY_H__
