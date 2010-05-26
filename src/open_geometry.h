/* 
 * File:   open_geometry.h
 * Author: bollig
 *
 * Created on April 26, 2010, 6:47 AM
 */

#ifndef _OPEN_GEOMETRY_H
#define	_OPEN_GEOMETRY_H

class OpenGeometry : public Geometry {

	OpenGeometry();
	~OpenGeometry();

        // Project a point to the boundary defined by the geometry
        virtual Vec3 project(Vec3 pt) = 0;

        // Compute the surface integral of the geometry
	virtual double surfaceIntegral() = 0;
};

#endif	/* _OPEN_GEOMETRY_H */

