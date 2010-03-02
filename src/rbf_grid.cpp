// Generate rbfs in a spherical shell

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <Vec3i.h>
#include <Vec3.h>
#include <ArrayT.h>
#include <vector> 
#include <algorithm> 
#include <functional> 
#include <write_lineset.h>
#include <write_psi.h>

#include "gpu.h"
#include "rbf_gaussian.h"
#include "derivative.h"

using namespace std;
using namespace arma;

typedef ArrayT<double> AF;
double maxint;

// I need a datastructure  (i,x,y)

#define OTHER 0
#define CENTER 1
#define STENCIL 2

struct Dist {
	int i;  // rbf index
	double d; // distance
	double x,y,z;
	// 0: other, 1: center, 2: stencil
	int id; // center, stencil or other
};

// GLOBAL VARIABLES
vector<Vec3> rbf_centers;
		
//----------------------------------------------------------------------
struct mySort 
{
  bool operator()(int i, int j) { return i < j; }
  bool operator()(const Dist& d1, const Dist& d2) { return d1.d < d2.d; }
} myObject;
//----------------------------------------------------------------------
int randi(int i1, int i2)
{
    double r  = (double) (random() / maxint);
    return (i1 + r*(i2-i1));
}
//----------------------------------------------------------------------
double randf(double f1, double f2)
{
    return  f1 + (f2-f1)*(random() / maxint);
}
//----------------------------------------------------------------------
void createSubgrid(int nx, int ny, int nz, ArrayT<vector<int> >** subgrid_index_list)
// overlay a 2D grid of size (nx,ny). RBFs are thus in one and 
// only one subgrid. Neighbors of a points will be located in 
// its subcell and those surrounding it. 
{
	// each element represents one subgrid. List of nodes in subgrid. 

	*subgrid_index_list = new ArrayT<vector<int> >(nx, ny, nz);
}
//----------------------------------------------------------------------
int main()
{
	int nb_x = 80;
	int nb_y = 80;
	int nb_z = 1;
	int nb_rbf = nb_x * nb_x * nb_x;
    maxint = (double) ((1 << 31) - 1);

	AF coord(3, nb_rbf); //  1000 rbfs.

	// Create a regular perturbed grid
	double pi = acos(-1.);
	double rmin = 11.;
	double rmax = 15.;
	double xmin = -rmax;
	double xmax =  rmax;
	double ymin = -rmax;
	double ymax =  rmax;
	double zmin = 0.; // 2D
	double zmax =  0.;
	double dx = (xmax-xmin) / (nb_x-1.);
	double dy = (ymax-ymin) / (nb_y-1.);
	double dz = 1.0;
	double pert = 0.0*dx;

	// subgrid size
	int nx = 40;
	int ny = 40;
	int nz = 1;
	int count = 0;

	printf("dx,dy= %f, %f\n", dx, dy);

	// Generate random points in a 3D grid, and only keep the points in the spherical shell

	for (int k=0; k < 1; k++) { // 2D
	for (int j=0; j < (nb_y-1); j++) {
	for (int i=0; i < (nb_x-1); i++) {
		double x = xmin + i*dx + randf(-pert, pert) + dx*0.5;
		double y = ymin + j*dy + randf(-pert, pert) + dy*0.5;
		double z = 0.0; // 2D
		coord(0, count) = x;
		coord(1, count) = y;
		coord(2, count) = z;
		rbf_centers.push_back(Vec3(x,y,z));
		count++;
		//printf("x,y,z=  %f, %f, %f\n", x, y, z);
	}}}
	//exit(0);

	nb_rbf = count;

	ArrayT<vector<int> >* subgrid_index_list;
	printf("create: nx,ny,nz= %d, %d, %d\n", nx, ny, nz);
	createSubgrid(nx, ny, nz, &subgrid_index_list);
	double sdx = (xmax - xmin) / (nx);  // number of cells: nx*ny
	double sdy = (ymax - ymin) / (ny);
	double sdz;
	if (nz == 1) {
		sdz = 0.; // 2D
	} else {
		sdz = (zmax - zmin) / (nz-1.);
	}

	printf("sdx,sdy= %f, %f\n", sdx, sdy);
	printf("xmin,ymin= %f, %f\n", xmin, ymin);

	vector<int> sub_cell; // which subcell contains a particular node

	for (int i=0; i < nb_rbf; i++) {
		int ix = (int) ((coord(0, i)-xmin) / sdx);
		int iy = (int) ((coord(1, i)-ymin) / sdy);
		//printf("coord: %f, %f\n", (coord(0,i)), (coord(1,i)));
		int iz = 1; // 2D
		if (ix == nx) ix--;
		if (iy == ny) iy--;
		//printf("ix,iy= %d, %d\n", ix, iy);
		(*subgrid_index_list)(ix,iy).push_back(i); // global index
		sub_cell.push_back(ix+nx*iy);  // which sub_cell is rbf[i] within? 
	}

	#if 0
	for (int k=0; k < nz; k++) {
	for (int j=0; j < ny; j++) {
	for (int i=0; i < nx; i++) {
		printf("(%d,%d)= %d\n", i, j, (*subgrid_index_list)(i,j).size());
	}}}
	#endif

	// subgrid_index_list now contains in each slot the list of global indices

	// create stencils

	vector<vector<int> > stencil; // for each node, a vector of stencil nodes (global indexing)
	stencil.resize(nb_rbf);
	//printf("stencil size: %d, nb_rbf= %d\n", stencil.size(), nb_rbf);
	//exit(0);

	#if 1
	for (int i=0; i < nb_rbf; i++) {
		int which_cell = sub_cell[i];
		int wy = which_cell / nx;
		int wx = which_cell - nx*wy;
		//printf("wx,wy= %d, %d\n", wx, wy);

	    // Create stencils for each node
		int w; 
		for (int wwx = wx-1; wwx <= wx+1; wwx++) {
			if (wwx < 0 || wwx == nx) continue;
		for (int wwy = wy-1; wwy <= wy+1; wwy++) {
			if (wwy < 0 || wwy == ny) continue;
			
			vector<int> lst = (*subgrid_index_list)(wwx,wwy);
			for (int l=0; l < lst.size(); l++) {
				stencil[i].push_back(lst[l]);
			}
		}}
		//printf("rbf %d, stencil_size= %d\n", i, stencil[i].size());
	}
	#endif

	// Assume a grid of (gx, gy) GPUs. 
	int gx = 2;
	int gy = 2;
	vector<GPU*> gpus;
	gpus.resize(gx*gy);

	double deltax  = (xmax-xmin) / gx;
	double deltay  = (ymax-ymin) / gy;

	printf("delta gpu x, y= %f\, %f\n", deltax, deltay);

	// Initialize GPU datastructures
	for (int id=0; id < gx*gy; id++) {
		int igy = id / gx;
		int igx = id - igy * gx;
		double xm = xmin + igx*deltax;
		double ym = ymin + igy*deltay;
		printf("xm,ym= %f\n", xm, ym);
		gpus[id] = new GPU(xm, xm+deltax, ym, ym+deltay, id);
	}

	// Figure out the sets Bi, Oi Qi

	printf("nb gpus: %d\n", gpus.size());
	for (int i=0; i < gpus.size(); i++) {
		gpus[i]->fillLocalData(rbf_centers, stencil);
		gpus[i]->fillVarData(rbf_centers);
	}

	printf("gpu structures (B,O,Q) are initialized\n");
	printf("initialized on scalar variable to liaenr function\n");

	// Compute derivative on a single GPU. Check against analytical result
	// du/dx=1, du/dx=2, du/dx=3

	RBF_Gaussian rbf(1.);
	const Vec3 xi(.5,0.,0.);
	for (int i=0; i < 10; i++) {
		const Vec3 xvec(i*.1, 0.,0.);
		printf("%d, phi=%f, phi'=%f\n", i, rbf.eval(xvec, xi), rbf.xderiv(xvec,xi));
	}

	Derivative der(rbf_centers.size());

	printf("start computing weights\n");
	for (int irbf=0; irbf < rbf_centers.size(); irbf++) {
		//printf("irbf= %d\n", irbf);
		der.computeWeights(rbf_centers, stencil[irbf], irbf);
		//weights[irbf].print("weights");
	}
	printf("after all computeWeights\n");

	vector<mat>& x_weights = der.getXWeights();
	vector<mat>& y_weights = der.getYWeights();
	vector<mat>& lapl_weights = der.getLaplWeights();

	// change the classes the variables are located in
	vector<double> xderiv(rbf_centers.size());
	vector<double> yderiv(rbf_centers.size());
	vector<double> lapl_deriv(rbf_centers.size());

	//printf("deriv size: %d\n",rbf_centers.size());
	for (int n=0; n < 1; n++) {
		der.computeDeriv(x_weights, stencil, rbf_centers, xderiv);
		der.computeDeriv(y_weights, stencil, rbf_centers, yderiv);
		der.computeDeriv(lapl_weights, stencil, rbf_centers, lapl_deriv);
		printf("computed all derivatives\n");
	}

	for (int i=0; i < xderiv.size(); i++) {
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("(%d), (%f,%f), xder[%d]= %f, yderiv= %f\n", st.size(), v.x(), v.y(), i, xderiv[i], yderiv[i]);
	}

	for (int i=0; i < lapl_deriv.size(); i++) {
		Vec3& v = rbf_centers[i];
		vector<int>& st = stencil[i];
		printf("(%d), (%f,%f), lapl_der[%d]= %f\n", st.size(), v.x(), v.y(), i, lapl_deriv[i]);
	}

	exit(0);

	// step 1: create code on a single CPU
	// step 2: create code on a single GPU
	// step 3: create partitioning of RBFs
	// step 3a: calculate domains Q, O, B for each GPU

	exit(0);

	printf("nb_rbf= %d\n", nb_rbf);
	//exit(0);

	vector<Dist> dist; //  1000 rbfs.
	double distance = 1.e10;
	double min_dist = 1.e10;
	int ix=-1;

	// compute point closest to R=fmax-2.;
	double rfind = (rmax-2.);
	printf("rfind= %f\n", rfind);

	for (int i=0; i < nb_rbf; i++) {
		distance = coord(0,i)*coord(0,i)+coord(1,i)*coord(1,i)+coord(2,i)*coord(2,i) - rfind*rfind;
		if (distance < 0.) distance *= -1.;
		if (distance < min_dist) {
			min_dist = distance;
			ix = i;
		}
	}
	//printf("min_dist= %f, ix= %d\n", min_dist, ix); exit(0);
	printf("ix= %f, %f, %f\n", coord(0,ix), coord(1,ix), coord(2,ix));
	printf("dist= %f\n", coord(0,ix)*coord(0,ix)+coord(1,ix)*coord(1,ix)+coord(2,ix)*coord(2,ix));

	Dist tmp;

	// Compute all distances to point ix;
	for (int i=0; i < nb_rbf; i++) {
		double cx = coord(0,i) - coord(0,ix);
		double cy = coord(1,i) - coord(1,ix);
		double cz = coord(2,i) - coord(2,ix);
		tmp.i = i;
		tmp.d = cx*cx + cy*cy + cz*cz;
		tmp.x = coord(0,i);
		tmp.y = coord(1,i);
		tmp.z = coord(2,i);
		tmp.id = OTHER;
		dist.push_back(tmp);
	}
	printf("ix= %d\n", ix);

	// does not work
	//sort(dist.begin(), dist.end(), greater<int>());

	//works
	sort(dist.begin(), dist.end(), myObject);

	for (int i=0; i < 100; i++) {
		printf("dist[%d] = %f, %f, %f, %f\n", dist[i].i, dist[i].d, dist[i].x, dist[i].y, dist[i].z);
	}
	
	// Pick 30 nearest rbfs
	int nb_stencil = 30;

	exit(0);
}
//----------------------------------------------------------------------
