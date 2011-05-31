#ifndef _ELLIPSOID_PARAMETRIC_PATCH_H
#define _ELLIPSOID_PARAMETRIC_PATCH_H


// CHECK sphere
// x = r*cos(u)*cos(v), y=r*cos(u)*sin(v), z=r*sin(u)
// Surface = 4*pi*r^2
// CHECK ellipse
// x = a*cos(u)*cos(v), y=b*cos(u)*sin(v), z=c*sin(u)
// Surface = Use Mathemtica or Maple to get approximate formulas

#include <math.h>

#include <vector>
#include <Vec3.h>
#include "parametric_patch.h"

class EllipsoidPatch : public ParametricPatch
{
private:
	double a, b, c; // ellipsoid axes

public:
// notice that there is discrete information in this definition
	EllipsoidPatch(double u1, double u2, double v1, double v2, 
	  int nu, int nv, double a_, double b_, double c_) : 
	     ParametricPatch(u1,u2,v1,v2,nu,nv), a(a_), b(b_), c(c_)
	{ }
	~EllipsoidPatch() {;}

	// Surface definition
	// u in [0,2 pi]
	// v in [0,pi]
	double x(double u, double v) 
	{
		return a*cos(u)*cos(v);
	}
	double y(double u, double v)
	{
		return b*cos(u)*sin(v);
	}
	double z(double u, double v)
	{
		return c*sin(u);
	}

	// Tangent space 
	// derivative of x(u,v) wrt u
	double xu(double u, double v) {
		return (-a*sin(u)*cos(v));
	}
	double yu(double u, double v) {
		return (-b*sin(u)*sin(v));
	}
	double zu(double u, double v) {
		return c*cos(u);
	}
	double xv(double u, double v) {
		return (-a*cos(u)*sin(v));
	}
	double yv(double u, double v) {
		return (b*cos(u)*cos(v));
	}
	double zv(double u, double v) {
		return 0.;
	}

	virtual Vec3& gradient(double x, double y, double z) {
		//printf(" ellip grad\n");
		grad.setValue(x/(a*a),y/(b*b),z/(c*c));
		return grad;
	}

	#if 1
	//----------------------------------------------------------------------
	// Exact projection (overrides inexact one in superclass
	// MUST STILL DEBUG
	// Starting from pt, compute the gradient of the volume function at pt, 
	// and project along the gradient direction
	virtual Vec3 projectToBoundary(Vec3& pt) {
		//pt.print("pt: project_to_boundary");
		grad = gradient(pt.x(), pt.y(), pt.z());
		grad.normalize();
		grad.print("grad: project_to_boundary");
		double ma = (grad.x()/a)*(grad.x()/a) + 
		            (grad.y()/b)*(grad.y()/b) + 
		            (grad.z()/c)*(grad.z()/c);
		double mb = pt.x()*grad.x()/(a*a) + 
		            pt.y()*grad.y()/(b*b) + 
		            pt.z()*grad.z()/(c*c);
		double root = sqrt(ma*ma + mb);
		double lam1 = (-mb + root) / ma;
		double lam2 = (-mb - root) / ma;
		double lam = (fabs(lam1) < fabs(lam2)) ? lam1 : lam2;
		//printf("lam= %f, lam1, lam2= %f, %f\n", lam, lam1, lam2);
		Vec3 pt_new = pt + lam*grad;
		return pt_new;
	}
	#endif

	//----------------------------------------------------------------------
	Vec3 singleProjectStep(Vec3& pt, Vec3& dir)
	{
		dir.normalize();
		double F = how_far(pt);

		Vec3 gradF;
		double gx = 2.*pt.x()/(a*a);
		double gy = 2.*pt.y()/(b*b);
		double gz = 2.*pt.z()/(c*c);
		gradF.setValue(gx,gy,gz);

		double lam;
		lam = -F / (gradF*dir);
		//printf("lam= %f\n", lam);
		Vec3 pp = pt+lam*dir;
	//printf("... how_far: %f\n", how_far(pp));
		return pt + lam*dir;
	}

	//----------------------------------------------------------------------

	// negative if inside, positive if outside
	virtual double how_far(Vec3& pt) {
		//pt.print("how_far");
		//double xx = pt.x() / a;
		double val = pow(pt.x()/a, 2.) + pow(pt.y()/b, 2.) + 
		   pow(pt.z()/c, 2.) - 1.;
		//printf("val= %f\n", val);
		return val;
	}
	//----------------------------------------------------------------------
//----------------------------------------------------------------------

};

#endif 
