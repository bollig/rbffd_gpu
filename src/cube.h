/* 
 * File:   cube.h
 * Author: bollig
 *
 * Created on April 25, 2010, 4:03 PM
 */

#ifndef _CUBE_H
#define	_CUBE_H

#include <stdio.h>
#include "closed_geometry.h"
#include "Vec3.h"

// TODO:
//  - Add routines to randomly sample the cube rather than regular samples

class CubeGeometry : public ClosedGeometry {
    // To start we assume this cube will be sampled regularly and
    // not passed through the CCVT. When we get ready for the CCVT we
    // will need to add the projection feature from interior to surface(see below)

    // For projection example: Real Time Collision Detecction by Christer Ericson
private:
    // The min and max values of the bounding box for the cube (corresp. to (x, y, z))
    Vec3 bound_min;
    Vec3 bound_max;

    // Number of divisions in cube
    Vec3 num_divs;
    // delta* (e.g., deltaX) for each dimension
    Vec3 delta;

    Vec3** samples;

public:

    CubeGeometry(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, int nX, int nY, int nZ) :
    bound_min(xmin, ymin, zmin), bound_max(xmax, ymax, zmax), num_divs(nX, nY, nZ) {

        delta[0] = (xmax - xmin) / ((double) (nX - 1));
        delta[1] = (ymax - ymin) / ((double) (nY - 1));
        delta[2] = (zmax - zmin) / ((double) (nZ - 1));
        fprintf(stderr, "dx, dy, dz = %f, %f, %f\n", delta.x(), delta.y(), delta.z());

        samples = new Vec3*[nX * nY * nZ];

        for (int i = 0; i < nX; i++) {
            for (int j = 0; j < nY; j++) {
                for (int k = 0; k < nZ; k++) {
                    samples[i*nY*nZ + nZ*j + k] = new Vec3(i*delta.x(), j*delta.y(), k * delta.z());
                }
            }
        }


    }

    ~CubeGeometry() {
        delete [] samples;
    }

    Vec3** getSamples() {
        return samples;
    }

    // Compute the interior volume integral

    virtual double volumeIntegral() {
        double vol = 1.;
        //fprintf(stderr, "ERROR! VOLUME NOT IMPLEMENTED %s\n", __FILE__);
        for (int i = 0; i < 3; i++) {
            vol *= bound_max[i] - bound_min[i];
        }
        return vol;
    }

    // Project a point to the boundary defined by the geometry

    virtual Vec3 project(Vec3 pt) {
        Vec3 projection;
        bool exterior_pt = false; // Test if the point was outside box to
        // avoid work in interior projection

        // For each dimension we check to see if its outside the faces of
        // cube. In those cases, constrain back to face.
        for (int i = 0; i < 3; i++) {
            double val = pt[i];
            if (val < bound_min[i]) {
                val = bound_min[i];
                exterior_pt = true;
            }
            if (val > bound_max[i]) {
                val = bound_max[i];
                exterior_pt = true;
            }
            projection[i] = val;
        }
        // TODO: this only constrains outside points onto the surface.
        // We also need to project interior points to the surface by
        //  - get the distance from point to planes
        //  - choose shortest distance
        //  - project to that plane by adding dist to

        if (!exterior_pt) {
            // Need to find the closest point from the interior now.
            // NOTE: worst case Im at center and the point is projected to arbitrary face
            //      second worst case is that Im midway between 3 faces

            // Point projection onto plane:
            //      R = Q - (n dot (Q - P) / ( n dot n)) n
            // where
            //      R : projection point
            //      Q : test point
            //      n : normal of plane
            //      P : center of plane
            // Then the closest projection will be R, s.t. it has minimum (R-Q).magnitude()

            // To speed it up, if we subtract center of volume from P, then P becomes
            // signed according to the planes it it closest to. For example,
            // (-x, -y, z) implies it is closer to xmin, ymin, zmax (top half, 4th quadrant)
            // Then we only have 3 planes left to test against.
            // Also, if we look at the magnitude of each component max(-x, -y, z)
            // we know which of the 3 planes it is closest to. If 2 or 3 maximums we just pick
            // the first. This reduces us to one projection. Also, regarding the magnitude: it
            // should be the ratio of the distance traversed to the total distance to the boundary
            //Vec3 cubeDims((bound_max[0] - bound_min[0])/2.,(bound_max[1] - bound_min[1])/2.,(bound_max[2] - bound_min[2])/2.);
            Vec3 cubeDims = bound_max - bound_min;
            cubeDims *= 0.5;

            // Center of Cube (not face)
            Vec3 c = cubeDims + bound_min;

            // Adjusted point:
            Vec3 Q = pt - c;

            int maxDim = 0;
            if (fabs(Q[maxDim] / c[maxDim]) < fabs(Q[1] / c[1])) {
                maxDim = 1;
            }
            if (fabs(Q[maxDim] / c[maxDim]) < fabs(Q[2] / c[2])) {
                maxDim = 2;
            }

            // fprintf(stderr, "Interior point is closest to %saxis[%d] (%f)\n", (Q[maxDim] > 0) ? "+" : "-", maxDim, Q[maxDim]);

            // Normal of plane
            Vec3 N(0., 0., 0.);
            double sign = (Q[maxDim] > 0) ? 1. : -1.;
            N[maxDim] = sign;

            //Adjust sign of cubeDims
            cubeDims[0] *= N[0];
            cubeDims[1] *= N[1];
            cubeDims[2] *= N[2];

            Vec3 P = pt + cubeDims;

#if 0
            // Center of plane
            Vec3 P = c + cubeDims; // Translate P away from center according along Normal

            Vec3 R = Q - ((N * (Q - P)) / (N * N)) * N;
#endif 
            // For each dimension we check to see if its outside the faces of
            // cube. In those cases, constrain back to face.
            for (int i = 0; i < 3; i++) {
                double val = P[i];
                if (val < bound_min[i]) {
                    val = bound_min[i];
                }
                if (val > bound_max[i]) {
                    val = bound_max[i];
                }
                projection[i] = val;
            }

            return projection;
        }

        return projection;
    }

    // Compute the surface integral of the geometry
    // NOTE: this does not use a numerical method like the ellipse

    virtual double surfaceIntegral() {
        //fprintf(stderr, "ERROR! SURFACE INTEGRAL NOT IMPLEMENTED %s\n", __FILE__);

        // Surface area of cube: 2ab + 2bc + 2bc
        double a = (bound_max[0] - bound_min[0]);
        double b = (bound_max[1] - bound_min[1]);
        double c = (bound_max[2] - bound_min[2]);

        // For unit cube should return 6
        return 2. * a * b + 2. * b * c + 2. * a*c;
    }

};


#endif	/* _CUBE_H */
