#ifndef _ADVANCED_SQUARE_PARAMETRIC_PATCH_H
#define _ADVANCED_SQUARE_PARAMETRIC_PATCH_H

#include <math.h>

#include <vector>
#include <Vec3.h>
#include "parametric_patch.h"

class SquarePatch : public ParametricPatch
{
private:
	Vec3 scale; 
	Vec3 center; 
	Vec3 normal; 
	Vec3 princ_axis_1; 		// principal axis 1 (orthogonal to normal)
	Vec3 princ_axis_2; 
	
public:
// notice that there is discrete information in this definition
	SquarePatch(double minU, double maxU, double minV, double maxV,
		int numU, int numV, double normalX, double normalY, double normalZ) :
//	  double normalX, double normalY, double normalZ) : 
	     ParametricPatch(minU,maxU,minV,maxV,numU,numV), normal(normalX,normalY,normalZ), 
			//center(1.,2.,3.)
			center(0.f,0.f,0.f) { 
			princ_axis_1 = guessP1(normal);
			princ_axis_1.normalize();	 
			princ_axis_2 = normal.cross(princ_axis_1);
			princ_axis_2.normalize();
		 }
	
	SquarePatch(double minU, double maxU, double minV, double maxV,
		int numU, int numV, double normalX, double normalY, double normalZ, 
		double center_x, double center_y, double center_z) :
//	  double normalX, double normalY, double normalZ) : 
	     ParametricPatch(minU,maxU,minV,maxV,numU,numV), normal(normalX,normalY,normalZ),
			center(center_x, center_y, center_z)
	{ 
		princ_axis_1 = guessP1(normal);
		princ_axis_1.normalize();	 
		princ_axis_2 = normal.cross(princ_axis_1);
		princ_axis_2.normalize();
	}
	
	~SquarePatch() {;}

	// Given a point, P0 = (x0, y0, z0), and a normal vector, N = <A, B, C>, the plane can be expressed as
	//	x = u
	// 	y = v
	// 	z = (1 - u - v) * P0 + u*P1 + v*P2
	// Where P1 = <x0 - B, y0 + A, z0> 
	// and   P2 = <x0 - C, y0, z0 + A) 
	//		( NOTE: N = (P1 x P2) / A )

	double x(double u, double v) 
	{
		// for u in [-1,1]
		// x = x0 + a*u where a is the scale
		Vec3 uv = getXYZ(u, v);
		return uv.x();
	}
	double y(double u, double v)
	{		
		Vec3 uv = getXYZ(u, v);
		return uv.y();
	}
	double z(double u, double v)
	{ 
		Vec3 uv = getXYZ(u, v);
		return uv.z();
	}
	
	Vec3 getXYZ(double u, double v) {
		Vec3 p1 = getP1();
		//p1 = p1+ center; 
		Vec3 p2 = getP2();
		p1 = p1 + center; 
		p2 = p2 + center; 
		//		Vec3 p2(center.x()-normal.z(), center.y(), center.z()+normal.x());
		if ((fabs(u) > 1.0f) || (fabs(v) > 1.0f)) {fprintf(stderr, "ERROR! u, v must be in [-1,1]\n"); }
		Vec3 uv = (1. - u - v)*center + u*(p1) + v*(p2);
		return uv;
	}


	// Tangent space 
	// derivative of x(u,v) wrt u
	double xu(double u, double v) {
		return 2./maxU;
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
		return 2./maxV;
	}
	double zv(double u, double v) {
		return 0.;
	}
#if 0
	virtual Vec3& gradient(double x, double y, double z) {
		// Plane is a*(x-x0) + b*(y-y0) + c*(z-z0) + d = 0
		// <a, b, c> is normal
		grad.setValue(,0.,0.);
		return grad;
	}
#endif 
	//----------------------------------------------------------------------
	Vec3 singleProjectStep(Vec3& pt, Vec3& dir)
	{
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
		// If the angle from the vector PC (i.e., center to pt) to the normal is
		// > 90 degrees then the point is behind the square (i.e. below)
		Vec3 pc = pt - center; 
	//	printf("pc: %f, %f, %f\n", pc.x(), pc.y(), pc.z());
		if (!(pc.magnitude() > 0)) {	// The point is the center
			return true;
		}
		Vec3 n = normal;
		double angle = acos( n*pc / (n.magnitude() * pc.magnitude()));
		//fprintf(stderr, "ANGLE: %f \t%f \t%f\n", angle, acos(-1.), acos(-1.)/2.); 
		
		if (angle >= acos(-1.)/2.) {
			return true;
		}
		return false;
	}
	
	// Test to see if the point is within the shadow of the square
	// Same as isBelow, but constrained to within square's boundaries.
	// Requires lots more computation.
	bool isInShadow(Vec3 pt) {
		// Assume that square is for u,v in [-1, 1] 
		// Then we check to make sure the point is within the square:
		// D 	 C		if (acos(AB * PB / |AB||PB|) <= PI/2 ) it is ABOVE
		//	  p			if (acos(BC * PC / |BC||PC|) <= PI/2 ) it is LEFT
		// A 	 B		if (acos(CD * PD / |CD||PD|) <= PI/2 ) it is BELOW
		//				if (acos(DA * PA / |DA||PA|) <= PI/2 ) it is RIGHT
		// if any of those fail, P is outside. 
		
		double pi_2 = acos(-1.) / 2.;
		
		Vec3 A = getXYZ(minU, minV);
		Vec3 B = getXYZ(maxU, minV);
		Vec3 C = getXYZ(maxU, maxV);
		Vec3 D = getXYZ(minU, maxV);
		
		Vec3 AB = A-B;
		Vec3 PB = pt-B;
		
		Vec3 BC = B-C;
		Vec3 PC = pt-C;
		
		Vec3 CD = C-D;
		Vec3 PD = pt-D;
		
		Vec3 DA = D-A;
		Vec3 PA = pt-A;
		
		if (!(PB.magnitude() > 0) || !(PC.magnitude() > 0) || !(PD.magnitude() > 0) || !(PA.magnitude() > 0)) {	// The point is a vertex
			return true;
		}
		
		if ((acos(AB * PB / (AB.magnitude()*PB.magnitude())) <= pi_2 ) 
			&& (acos(BC * PC / (BC.magnitude()*PC.magnitude())) <= pi_2 ) 
			&& (acos(CD * PD / (CD.magnitude()*PD.magnitude())) <= pi_2 ) 
			&& (acos(DA * PA / (DA.magnitude()*PA.magnitude())) <= pi_2 ))
			return true; 
		return false; 
	}
	
	
//----------------------------------------------------------------------
private: 
	
	// Adjust range to [-1,1]. 
	double adjustRangeU(double u) 
	{
		return ((u/maxU)-0.5)*2.;
	}
	// Adjust range to [-1,1]. 
	double adjustRangeV(double v) 
	{
		return ((v/maxV)-0.5)*2.;
	}


	// This is almost a hack. Well, its valid, but it may not behave the way 
	// you expect at first. 
	// 
	// If a normal vector (a, b, c) is given it is true that the orthogonal vector
	// pair (-b, a, 0) and (-c, 0, a) are orthogonal to the normal (and therefore
	// valid principal axes). We start with the assumption that (-b, a, 0) will 
	// be p1 (i.e. corresponding to parameter u). Then p2 is normal.cross(p1).
	// However, if b and a are both 0, then our assumption results in 
	// a zero vector, so we tentatively choose the other vector (-c, 0, a) as p1.
	// This however should correspond to v, so
	// we actually want the inverse of this: (c, 0, -a). This maintains the 
	// right hand rule when calculating p2 as norm.cross(p1);
	// NOTE: there is a limitation here. Specifying a normal does not specify
	// rotation of the square in the plane orthogonal to the normal. This is a 
	// problem, for example, when forming a cube which is rotated 45 degrees around
	// the Z-axis. The end caps parallel the X-Y plane are not rotated 45 degrees
	// like the rest of the faces. This is a problem because the squares will not
	// properly align with the edges of the other faces. 
	// It is better to use a constructor which 
	// clearly specifies the principal axes if forming general geometries where
	// this can cause a problem. 
	Vec3 guessP1(Vec3 normal) {
		bool ab = false,ca = false; 
		
		if (fabs(normal.x()) > 0.) {
			ab = true; ca = true;
		} else if (fabs(normal.y()) > 0.) {
			ab = true; 
		} else if (fabs(normal.z()) > 0.) {
			ca = true; 
		} else {
			fprintf(stderr, "ERROR, NORMAL IS ZERO!\n");
			exit(-1); 
		}
		if (ab) {
			Vec3 p1(-normal.y(), normal.x(), 0.0f);
			return p1; 
		} else if (ca) {
			//fprintf(stderr, "CHOOSING ca\n");
			Vec3 p1(normal.z(), 0.0f, -normal.x());
			return p1; 
		} else {
			fprintf(stderr, "ERROR, FAILED ALL CASES\n");
		}
	}

};

#endif 
