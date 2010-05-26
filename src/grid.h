#ifndef _GRID_H_
#define _GRID_H_

#include <vector>
#include <Vec3.h>
#include <ArrayT.h>

class Grid {

public:
        double xmin; // = -rmax;
	double xmax; // =  rmax;
	double ymin; // = -rmax;
	double ymax; // =  rmax;
	double zmin; // = 0.; // 2D
	double zmax; //  =  0.;

protected:
// grid
	double rmax; // = 15.;
	
	double dx; // = (xmax-xmin) / (nb_x-1.);
	double dy; // = (ymax-ymin) / (nb_y-1.);
	double dz; // = 1.0;
	double pert; // = 0.0*dx;
	int nb_rbf; // total number of points
	int nb_x;
	int nb_y;
	int nb_z;
	double maxint;
	int stencil_size;
	std::vector<double> avg_distance;;		// Computed in avgStencilRadius(..); computeStencils has a locally scoped copy.

	int nb_bnd; // number o fpoints on the domain boundary
	//double major, minor;
	double princ_axis[3]; 

	// Cartesion 5-point Laplacian
	std::vector<double> laplacian;

	std::vector<std::vector<int> > stencil; 		// List of stencils
	std::vector<Vec3> rbf_centers;					// Nodal positions
	
	ArrayT<std::vector<int> >* subgrid_index_list;
	ArrayT<double>* coord; //(3, nb_rbf); //  1000 rbfs.

	std::vector<int> sub_cell; // which subcell contains a particular node

	// specialized to 2D Cartesian grid
	std::vector<std::vector<int> > cart_stencil; 
	std::vector<std::vector<double> > cart_weights; 

	//std::vector<std::vector<double> > avg_dist; 

	// List of stencil lists (stencil is vec<vec<int>>)
	//std::vector< std::vector< std::vector<int> > > decomposed_domain; 
	
	// list of boundary points
	std::vector<int> boundary; 

// subgrid
	int nx; // = 10;
	int ny; // = 10;
	int nz; // = 1;
	int count; // = 0;

public:
	Grid(int n_x, int n_y, int stencil_size=9); // maximum stencil size
	~Grid();
	virtual void generateGrid();

	// file: input file with grid points 1 per row
	// nb_bnd: number of boundary points
	// npts: total number of points
	virtual void generateGrid(const char* file, int nb_bnd, int npts);

	virtual void generateSubGrid();

	// compute stencil that contains the "n" nearest nodes 
	void computeStencils();
	void avgStencilRadius();

	// use subgrid to compute stencils: a stencil is all points in all 
	// cells that contain the point and the surrounding ones
	void computeStencilsRegular();

	std::vector<Vec3>& getRbfCenters()
		{return rbf_centers;}
	int randi(int i1, int i2);
	double randf(double f1, double f2);
	std::vector<std::vector<int> >& getStencil() 
		{
			printf("Regular::getStencil::stencil[0]= %ld\n", (long int) &stencil[0]);
			return stencil;
		} 
	std::vector<int>& getBoundary() 
		{return boundary;}

	// compute stencil and weight for 5-point Laplacian on Carteian grid
	// (not corresponding to RBFs)
	void laplace();
	std::vector<double>& computeCartLaplacian(std::vector<double>& scalar);

	int getNbBnd() { return nb_bnd; }
	void setNbBnd(int nb_bnd_) { this->nb_bnd = nb_bnd_; }
	/*
	double getMajor() { return major; }
	void setMajor(double major_) { this->major = major_; } 
	double getMinor() { return minor; }
	void setMinor(double minor_) { this->minor = minor_; } 
	*/
	double getPrincipalAxis(int i) { return princ_axis[i+1]; }
	void setPrincipalAxes(double axis1, double axis2, double axis3) { 
		this->princ_axis[0] = axis1; 
		this->princ_axis[1] = axis2; 
		this->princ_axis[2] = axis3; 
	}
	
	std::vector<double>& getAvgDist() { return avg_distance; };
	double minimum(std::vector<double>& vec);
	
	// Decompose the domain into num_cpus subdomains and return a list
	// of stencil lists (one for each cpu)
	std::vector< std::vector< Vec3* > > decomposeDomain(int num_cpus);

//----------------------------------------------------------------------
};


#endif
