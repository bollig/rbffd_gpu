#ifndef _PARAMETRIC_PATCH_
#define _PARAMETRIC_PATCH_

// CHECK sphere
// x = r*cos(u)*cos(v), y=r*cos(u)*sin(v), z=r*sin(u)
// Surface = 4*pi*r^2
// CHECK ellipse
// x = a*cos(u)*cos(v), y=b*cos(u)*sin(v), z=c*sin(u)
// Surface = Use Mathemtica or Maple to get approximate formulas

#include <math.h>
#include <stdio.h>
#include <vector>
#include <Vec3.h>

class ParametricPatch
{
protected:
	double minU, maxU;
	double minV, maxV;
	int numU;
	int numV;
	double du, dv;
	int umin, umax; // index corresponding to nearest point
	int vmin, vmax;

	Vec3 ru; // tangent vector
	Vec3 rv; // tangent vector
	Vec3 pt;
	Vec3 vzero;
	Vec3 tmp; // for temporary usage

	std::vector<Vec3> boundary_pts;

	Vec3 grad;

public:
    // Parametric patch defined over domain [minU, maxU]x[minV, maxV] with 
    // numU_ : the number of subdivisions in parameter u for patch (patch defined as f(u,v)
    // numV_ : number of subdivisions in parameter v
	ParametricPatch(double minU_, double maxU_, double minV_, double maxV_, 
	   int numU_, int numV_) : minU(minU_), maxU(maxU_), minV(minV_), maxV(maxV_), 
	   numU(numU_), numV(numV_)  {
		du = (maxU - minU) / (numU-1); // non-periodic
		dv = (maxV - minV) / (numV-1); // non-periodic
		vzero.setValue(0.,0.,0.);
	}
	~ParametricPatch() {;}

	// Surface definition
	virtual double x(double u, double v) = 0;
	virtual double y(double u, double v) = 0;
	virtual double z(double u, double v) = 0;

	// Tangent space
	virtual double xu(double u, double v) = 0;
	virtual double yu(double u, double v) = 0;
	virtual double zu(double u, double v) = 0;
	virtual double xv(double u, double v) = 0;
	virtual double yv(double u, double v) = 0;
	virtual double zv(double u, double v) = 0;

	// In case the patch has a non-parametric representation
	// if not, return vzero so abstract does not create problems
	virtual Vec3& gradient(double x, double y, double z) {
	    //printf("grad(double) in parametric_patch\n");
		return vzero;
	}

	virtual Vec3& gradient(Vec3& r)  {
	    //printf("grad(vec) in parametric_patch\n");
		return gradient(r.x(), r.y(), r.z());
	}

    double random(double a, double b) {
        double r = ::random() / (double) RAND_MAX;
        return a + r * (b - a);
    }

// Generate a random U in [minU_, maxU_]
    double randomU() {
        return random(minU, maxU); 
    }

    double randomV() {
        return random(minV, maxV); 
    }

	virtual double surfaceElement(double u, double v) {
		double xx = x(u,v);
		double yy = y(u,v);
		double zz = z(u,v);
		ru.setValue(xu(u,v), yu(u,v), zu(u,v));
		rv.setValue(xv(u,v), yv(u,v), zv(u,v));
		//ru.print("ru");
		//rv.print("rv");
		//Vec3 vv = ru.cross(rv).magnitude(); // VERIFY ROUTINE
		Vec3 vv = ru.cross(rv); // VERIFY ROUTINE
		//vv.print("ru x rv");
		return vv.magnitude(); //ru.cross(rv).magnitude(); // VERIFY ROUTINE
	}

	virtual double surfaceIntegral() {
		double intg = 0.0;
		double u, v;

		for (int j=0; j < (numV-1); j++) {
		for (int i=0; i < (numU-1); i++) {
			u = minU + i*du;
			v = minV + j*dv;
			//printf("u,v= %f, %f\n", u,v);
			intg += surfaceElement(u+0.5*du,v+0.5*dv);
		}}

		// WARNING: this assumes uniform sampling in u and v.
		// Single surface element (dS) = || Xu(t) x Xv(t) ||*du(t)*dv(t) 
		// so the du and dv may vary.
		return intg*du*dv;
	}

	// project a point onto the surface
	virtual Vec3 ProjectToBoundary(double x, double y, double z) 
	{
		// assumes one is close to the boundary
		pt.setValue(x,y,z);
		return projectToBoundary(pt);
	}

	//----------------------------------------------------------------------
	// project the point ptc onto the boundary in the direction given by "dir"
	// Assumes that the point is already close to the boundary
	//virtual Vec3 projectToBoundary(Vec3& ptc, Vec3& dir) {
	//}
	//----------------------------------------------------------------------

	virtual Vec3 projectToBoundary(Vec3& ptc) {
		//ptc.print("ptc");
		Vec3 grad = gradient(ptc.x(), ptc.y(), ptc.z());
		///grad.print("grad(ptc), before iter 0");

		int nb_outer = 1;
		int nb_inner = 3;

		for (int outer=0; outer < nb_outer; outer++) {
			pt = ptc;
			//printf("\n *** outer iter %d\n", outer);
	
			for (int i=0; i < nb_inner; i++) {
				pt = send_to_boundary(pt, grad);
			}

			// update gradient
			//pt.print("pt: end of inner iteration\n");
			grad = gradient(pt.x(), pt.y(), pt.z());
			//grad.print("grad: end of inner iteration\n");
		}
		return pt;
	}

	//----------------------------------------------------------------------
	// send in a direction parallel to grad(F)
	// Assume original point is close to the proposed solution
	virtual Vec3& send_to_boundary(Vec3& pt, Vec3& grad)
	{
		double f = how_far(pt);
		printf("before: how_far= %f\n", f);
		// assumes that grad() is never zero (no singularity)
		double lam = -f / (grad.x() + grad.y() + grad.z());
		pt = pt + lam * grad;
		printf("after: how_far= %f\n", how_far(pt));
		return pt;
	}

	//----------------------------------------------------------------------
	// non-parametric evaluation of F(x,y,z) (=0 on the surface)
	double how_far(double x, double y, double z) {
		tmp.setValue(x,y,z);
		return how_far(tmp);
	}
	virtual double how_far(Vec3& pt) = 0;

	// collection of boundary points
	std::vector<Vec3> getBoundaryPoints();

	inline int getMinIndices(int& min_u, int&  max_u, 
      int& min_v, int& max_v) {
	  	min_u = umin;
	  	max_u = umax;
	  	min_v = vmin;
	  	max_v = vmax;
	}

	// near point to the boundary
	Vec3& nearest(Vec3& pt);

	Vec3 find(Vec3& pt, int& umin, int& umax, int& vmin, int& vmax);
	Vec3 planeIntersect(Vec3& rsd, Vec3& dir, Vec3& pt0, Vec3& normal);

	// project pt_off_surface to the surface. 
	virtual Vec3 project(Vec3 pt_off_surface);
	virtual Vec3 projectToBoundary(Vec3& pt, Vec3& dir);

	// intersection of patch with line pt = pt0 + t*(pt1-pt0)
	// Intersection occurs when t in [0,1] and pt is on the patch
	virtual Vec3 intersectWithLine(Vec3& pt0, Vec3& pt1);

private:
	virtual Vec3 singleIteration(Vec3& pt_off_surface, Vec3& pt_on_surface);
	virtual Vec3 singleProjectStep(Vec3& pt, Vec3& dir) = 0;

private:
//  (rsd-rsf)xgrad(rsf) = 0 = F(rsf) 
	Vec3& F(Vec3& pt_surf, Vec3& pt_vol);
//----------------------------------------------------------------------
};

#endif 
