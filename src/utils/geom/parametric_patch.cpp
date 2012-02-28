#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <Vec3.h>
#include "parametric_patch.h"

using namespace std;

vector<Vec3> ParametricPatch::getBoundaryPoints()
{
	Vec3 pt;

	for (int j=0; j < numV; j++) {
		double v = minV + j*dv;
	for (int i=0; i < numU; i++) {
		double u = minU + i*du;
		pt.setValue(x(u,v), y(u,v), z(u,v));
		boundary_pts.push_back(pt);
	}}
	return boundary_pts;
}
//----------------------------------------------------------------------
Vec3& ParametricPatch::nearest(Vec3& pt)
// bad idea to call functions on const argument. Generates error: 
//   " discards qualifiers" (What does this mean?)
// Problem is that the function could modify the argument, which would
// be inconsistent with argument statement of constancy
{
	double min_dist2 = 1.e10;
	double dist2; // distance squared
	int imin = 0;

	int npts = boundary_pts.size();
	for (int i=0; i < npts; i++) {
		dist2 = pt.distance2(boundary_pts[i]);
		//boundary_pts[i].print("pt");
		if (dist2 < min_dist2) {
			min_dist2 = dist2;
			//umin = minU+i*du;
			//vmin = minV+j*dv;
			imin = i;
		//printf("---------\nmin_dist2= %f\n", min_dist2);
        //boundary_pts[i].print("bnd");
		//pt.print("pt");
		}
	} //}

	return boundary_pts[imin];
}
//----------------------------------------------------------------------
// Newton-Raphson 
Vec3 ParametricPatch::find(Vec3& pt, int& umin, int& umax, 
	int& vmin, int& vmax)
{
    std::cout << "[ParametricPatch] Error: find not implemented\n";
    return Vec3(0.,0.,0.); 
}
//----------------------------------------------------------------------
Vec3 ParametricPatch::F(Vec3& pt_surf, Vec3& seed)
// rsd: seed point
// rsf: surface point
//(rsd-rsf)xgrad(rsf) = 0 = F(rsf) 
//(seed-pt_surf)xgrad(pt_surf) = 0 = F(rsf) 
{
	// Check out how to write substraction operators that return references
	Vec3& grad = gradient(pt_surf.x(), pt_surf.y(), pt_surf.z());
	Vec3 diff = seed - pt_surf; // why is temporary type Vec3 ?
	Vec3 cr = diff.cross(grad);
    return cr; 
}
//----------------------------------------------------------------------
// Line: from vector rsd with direction "dir"
// Compute intersection of this line with a plane passing through pt0, 
// with normal "normal"
// intersection of a plane with a inearnormal grad, pt0 is on the plane
Vec3 ParametricPatch::planeIntersect(Vec3& rsd, Vec3& dir, Vec3& pt0, Vec3& normal)
{
	// r = rsd + lam*dir
	// plane: (r-pt0).normal = 0
  	//r.normal = pt0.normal
  	//(rsd+lam*dir).normal = pt0.normal 
  	//lam = (pt0.normal - rsd.normal) / dir.normal
  	//r = rsd + lam.dir ("." is scalar project)

	double lam = (pt0 - rsd)*normal / (normal*dir);
	Vec3 newpt = rsd+lam*dir;
	return newpt;
}
//----------------------------------------------------------------------
Vec3 ParametricPatch::project(Vec3 pt_off_surface)
{
		double tolerance = 1.e-4;

		// Find nearest point to ellipsoid
		//pt_off_surface.print("original pt_off_surface"); // not on the surface
		getBoundaryPoints();
		Vec3 pt_on_surface = nearest(pt_off_surface);
		Vec3 pt_on_surface1;

#if 0
		pt_off_surface.print("original seed");
		pt_on_surface.print("nearest point from original");
#endif 
		// Outer iteration
		double err;
		int nb_outer = 4;

#if 0
			printf("****** before outer iterations ****\n");
#endif 
			Vec3 diff = pt_off_surface - pt_on_surface;
			Vec3& p = pt_on_surface;
			Vec3 grad = gradient(p.x(), p.y(), p.z());
			Vec3 cross = diff.cross(grad); // should be zero
			//grad.print("initial grad");
			//printf("initial cross mag: %g\n", cross.magnitude());
			//printf("|grad x (pt_off - pt_on)| = %f\n", cross.magnitude());
	
		for (int i=0; i < nb_outer; i++) {
			// pt is new point on the surface
			pt_on_surface1 = singleIteration(pt_off_surface, pt_on_surface);
			//pt_on_surface1.print("pt on surface");
			Vec3 diff = pt_on_surface - pt_on_surface1;
			err = sqrt(diff.square());
			//printf("err= %f\n", err);
			if (err < tolerance) break;
			if (err > 1000.) {
				printf("[ParametricPatch] project point: max tolerance exceeded. Cannot continumUe.\n");
				exit(0);
			}
			//printf("===========================\n");
			//pt_off_surface.print("off surface");
			//pt_on_surface1.print("on surface");
			diff = pt_off_surface - pt_on_surface1;
			Vec3& p = pt_on_surface;
			Vec3 grad = gradient(p.x(), p.y(), p.z());
			//grad.print("grad");
			Vec3 cross = diff.cross(grad); // should be zero
			//printf("initial cross mag: %g\n", cross.magnitude());
			//printf("|grad x (pt_off - pt_on)| = %f\n", cross.magnitude());
			pt_on_surface = pt_on_surface1;
		}

		if (err > tolerance) {
		//	printf("[ParametricPatch] Warning: *** project point on surface: NOT CONVERGED ***\n");
		}

		// pt is new point on the surface
	
		//pt_on_surface.print("new pt on surface");
	
		return pt_on_surface;
}
//----------------------------------------------------------------------
Vec3 ParametricPatch::singleIteration(Vec3& pt_off_surface, Vec3& pt_on_surface)
{
	
		// project along the gradient direction from the off surface point
		Vec3& rsd = pt_off_surface;
		Vec3& dir = grad;
		Vec3& pt = pt_on_surface;
		Vec3 normal = gradient(pt.x(), pt.y(), pt.z());
#if 0
		normal.print("normal");
#endif 
		Vec3 pti = planeIntersect(rsd, dir, pt, normal);

		//double dist = pti.distance2(rsd);

		// new point on the surface
		return projectToBoundary(pti, normal);
}
//----------------------------------------------------------------------
// Exact projection (overrides inexact one in superclass
// MUST STILL DEBUG
// Starting from pt, project along a specified direction
Vec3 ParametricPatch::projectToBoundary(Vec3& pt2, Vec3& dir)
{
		double tolerance = 1.e-5;
		Vec3 pt1, pt, verror;

		pt = pt2;

		// Inner iteration
		int nb_inner = 5;

		for (int i=0; i < nb_inner; i++) {
			//pt.print("pt");
			pt1 = singleProjectStep(pt, dir);
			verror = (pt1-pt);
			double err_norm = sqrt(verror.square());
			//printf("** singleProj: err = %g\n", err_norm);
			pt = pt1;
			if (err_norm < tolerance) break;
		}
		// NEED ERROR CRITERIA

		return pt;
}
//----------------------------------------------------------------------
Vec3 ParametricPatch::intersectWithLine(Vec3& pt0, Vec3& pt1)
{
	// Only makes sense for closed surfaces
	double pt0_inside = how_far(pt0);
	double pt1_inside = how_far(pt1);
	printf("p0_inside*p1_inside(initial)= %f\n", pt0_inside*pt1_inside);
	Vec3 pt;

	if (pt0_inside * pt1_inside < 0) {
		printf("Line intersects the surface");
	} else {
		printf("Line does not intersect the surface");
		if (pt0_inside > 0) {
			if (pt0_inside < pt1_inside) {
				pt = pt1 - (1.5*pt1_inside / (pt1_inside - pt0_inside))*(pt1 - pt0);
			} else {
				pt = pt0 - (1.5*pt0_inside / (pt0_inside - pt1_inside))*(pt0 - pt1);
			}
		} else if (pt0_inside < 0) {
			if (pt0_inside < pt1_inside) {
				pt = pt0 + (1.5*pt0_inside / (pt0_inside - pt1_inside))*(pt0 - pt1);
			} else {
				pt = pt1 + (1.5*pt1_inside / (pt1_inside - pt0_inside))*(pt1 - pt0);
			}
		} else { 		// pt0_inside == 0 (on the surface)
			
		}
	}

	pt0_inside = how_far(pt0);
	pt1_inside = how_far(pt1);
	printf("p0_inside*p1_inside(final)= %f\n", pt0_inside*pt1_inside);

	return pt;
}
//----------------------------------------------------------------------
