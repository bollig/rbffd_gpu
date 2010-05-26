/* 
 * File:   closed_geometry.h
 * Author: bollig
 *
 * Created on April 26, 2010, 6:46 AM
 */

#ifndef _CLOSED_GEOMETRY_H
#define	_CLOSED_GEOMETRY_H

#include "geometry.h"

class ClosedGeometry : public Geometry {
    
public:
        // Compute the interior volume integral
	virtual double volumeIntegral() = 0;

        // Project a point to the boundary defined by the geometry
        virtual Vec3 project(Vec3 pt) = 0;

        // Compute the surface integral of the geometry
	virtual double surfaceIntegral() = 0;
        
private:
    

};

#endif	/* _CLOSED_GEOMETRY_H */

