#ifndef _SQUARE_PARAMETRIC_PATCH_H
#define _SQUARE_PARAMETRIC_PATCH_H

/**
 * TODO: remove advanced features and keep this class simple.
 * I want to have the square lying in the xy plane (z = 0). It should use
 * conditionals to test if point is outside of boundaries and projection should
 * be simple z projection. For points outside the bounds of the square that are
 * projected onto the boundaries, we do a simple orthogonal projection of point
 * to line segment.
 */

#include <math.h>

#include <vector>
#include <Vec3.h>
#include "parametric_patch.h"

class SquarePatch : public ParametricPatch {
private:
    Vec3 scale;
    Vec3 center;
    Vec3 normal;
    Vec3 princ_axis_1; // principal axis 1 (orthogonal to normal)
    Vec3 princ_axis_2;

public:
    // notice that there is discrete information in this definition

    SquarePatch(double minU, double maxU, double minV, double maxV,
            int numU, int numV, double scaleX, double scaleY) :
    ParametricPatch(minU, maxU, minV, maxV, numU, numV), scale(scaleX, scaleY, 0.),
    center(0.f, 0.f, 0.f)
    {
        princ_axis_1 = Vec3(1.0, 0.0, 0.0);
        princ_axis_2 = Vec3(0.0, 1.0, 0.0);
    }

    ~SquarePatch() {
        ;
    }

    // Given a point, P0 = (x0, y0, z0), and a normal vector, N = <A, B, C>, the plane can be expressed as
    //	x = u
    // 	y = v
    // 	z = (1 - u - v) * P0 + u*P1 + v*P2
    // Where P1 = <x0 - B, y0 + A, z0>
    // and   P2 = <x0 - C, y0, z0 + A)
    //		( NOTE: N = (P1 x P2) / A )

    double x(double u, double v) {
        return u * scale.x();
    }

    double y(double u, double v) {
        return v * scale.y();
    }

    double z(double u, double v) {
        return 0.;
    }

    Vec3 getXYZ(double u, double v) {
        Vec3 xyz(u, v, 0.);
        return xyz;
    }

    // Tangent space
    // derivative of x(u,v) wrt u
    // NOTE: This class is normalized here. Some will not be. 

    double xu(double u, double v) {
        return 1.;
    }

    double yu(double u, double v) {
        return 0;
    }

    double zu(double u, double v) {
        return 0.;
    }

    double xv(double u, double v) {
        return 0.;
    }

    double yv(double u, double v) {
        return 1.;
    }

    double zv(double u, double v) {
        return 0.;
    }
#if 0

    virtual Vec3& gradient(double x, double y, double z) {
        // Plane is a*(x-x0) + b*(y-y0) + c*(z-z0) + d = 0
        // <a, b, c> is normal
        grad.setValue(, 0., 0.);
        return grad;
    }
#endif 
    //----------------------------------------------------------------------

    Vec3 singleProjectStep(Vec3& pt, Vec3& dir) {
        // For projection and signed distance we can make a few assumptions:
        // Given a point in the square shadow we can project it to the square using
        // point to plane projection (N dot P / |N|)
        // Given point outside of a square and its shadow, we can project it to the
        // square edges by point to line segment projection


        /*	dir.normalize();
                double F = how_far(pt);

                Vec3 gradF;
                double gx = 2.*pt.x()/(a*a);
                double gy = 2.*pt.y()/(b*b);
                double gz = 2.*pt.z()/(c*c);
                gradF.setValue(gx,gy,gz);

                double lam;
                lam = -F / (gradF*dir);
                Vec3 pp = pt+lam*dir;
        return pt + lam*dir;	*/

        // TODO
        return pt;
    }

    //----------------------------------------------------------------------

    // negative if inside, positive if outside

    virtual double how_far(Vec3& pt) {
        //TODO

        return 0;
    }
    //----------------------------------------------------------------------

    // Return the first principal axis

    Vec3 getP1() {
        return princ_axis_1;
    }

    Vec3 getP2() {
        return princ_axis_2;
    }


    // Test to see if a point is below the square (i.e. on or below the plane that
    // the square lies in). Since the square is not infinite, call isInside()
    // to find out if a point lies within square.

    bool isBelow(Vec3 pt) {
        return (pt.z() < 0.) ? true : false;
    }

    // Test to see if the point is within the shadow of the square
    // Same as isBelow, but constrained to within square's boundaries.
    // Requires lots more computation.

    bool isInShadow(Vec3 pt) {

        // If inside all bounds
        return true;
        
        return false;
    }


    //----------------------------------------------------------------------
private:

    // Adjust range to [-1,1].

    double adjustRangeU(double u) {
        return ((u / maxU) - 0.5)*2.;
    }
    // Adjust range to [-1,1].

    double adjustRangeV(double v) {
        return ((v / maxV) - 0.5)*2.;
    }

};

#endif 
