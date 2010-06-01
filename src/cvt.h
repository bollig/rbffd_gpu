#ifndef _BURKARDT_CVT_H_
#define _BURKARDT_CVT_H_

#include <vector>
#include "Vec3.h"
#include "parametric_patch.h"
#include "density.h"

class CVT
{
private:
	std::vector<double> rd;
	int print_sample_count;
	int nb_bnd; // number of seeds on the boundary
	int nb_pts; // total number of seeds
	Density* rho;

	// better to pass a grid structure and different types of grids will be represented by 
	// different subclasses, which are accessed by individual samplers (that populate the domain 
	// with random points
	double major;
	double minor;
	double midax; // good for 3D case
	std::vector<Vec3> bndry_pts;

	ParametricPatch *outer_geom; // not quite the correct class. In reality, the 
	ParametricPatch *geom; // not quite the correct class. In reality, the 
	   // correct class is Geometry = vector<ParametricPatch>

public:
	CVT();

	char ch_cap ( char c );
	bool ch_eqi ( char c1, char c2 );
	int ch_to_digit ( char c );
	void cvt ( int dim_num, int n, int batch, int init, int sample, int sample_num, int it_max, int it_fixed, int *seed, double r[], int *it_num, double *it_diff, double *energy );
	void cvt3d ( int dim_num, int n, int batch, int init, int sample, int sample_num, int it_max, int it_fixed, int *seed, double r[],int *it_num, double *it_diff, double *energy );
	double cvt_energy ( int dim_num, int n, int batch, int sample, bool initialize,
  	int sample_num, int *seed, double r[] );
	void cvt_iterate ( int dim_num, int n, int batch, int sample, bool initialize, 
  	int sample_num, int *seed, double r[], double *it_diff, double *energy );
	void cvt_sample ( int dim_num, int n, int n_now, int sample, bool initialize, 
  	int *seed, double r[] );
	void cvt_write ( int dim_num, int n, int batch, int seed_init, int seed, 
  	    const char *init_string, int it_max, int it_fixed, int it_num, 
  	    double it_diff, double energy, const char *sample_string, int sample_num, double r[], 
  	    const char *file_out_name, bool comment );
	void data_read ( const char *file_in_name, int dim_num, int n, double r[] );
	char digit_to_ch ( int i );
	void find_closest ( int dim_num, int n, int sample_num, double s[], double r[],
  	int nearest[] );
	int get_seed ( void );
	bool halham_leap_check ( int dim_num, int leap[] );
	bool halham_n_check ( int n );
	bool halham_dim_num_check ( int dim_num );
	bool halham_seed_check ( int dim_num, int seed[] );
	bool halham_step_check ( int step );
	bool halton_base_check ( int dim_num, int base[] );
	int i4_log_10 ( int i );
	int i4_max ( int i1, int i2 );
	int i4_min ( int i1, int i2 );
	void i4_to_halton_sequence ( int dim_num, int n, int step, int seed[], int leap[],
  	int base[], double r[] );
	char *i4_to_s ( int i );
	int prime ( int n );
	double r8_epsilon ( void );
	double r8_huge ( void );
	void r8mat_transpose_print ( int m, int n, double a[], const char *title );
	void r8mat_transpose_print_some ( int m, int n, double a[], int ilo, int jlo, 
  	int ihi, int jhi, const char *title );
	void r8mat_uniform_01 ( int m, int n, int *seed, double r[] );
	unsigned long random_initialize ( int seed );
	void s_blank_delete ( const char *s );
	void s_cap ( const char *s );
	bool s_eqi ( const char *s1, const char *s2 );
	int s_len_trim ( const char* s );
	double s_to_r8 ( const char *s, int *lchar, bool *error );
	bool s_to_r8vec ( const char *s, int n, double rvec[] );
	void timestamp ( void );
	char *timestring ( void );
	void tuple_next_fast ( int m, int n, int rank, int x[] );
	void user ( int dim_num, int n, int *seed, double r[] );
	
	// Gordon Erlebacher, 9/1/2009
	void ellipse ( int dim_num, int& n, int& nb_bnd, int *seed, std::vector<double>& r );
	void ellipse ( int dim_num, int& n, int& nb_bnd, int *seed, double r[] );
	void ellipse_init ( int dim_num, int& n, int& nb_bnd, int *seed, double r[] );
	void setNbBnd(int nb_bnd_) { this->nb_bnd = nb_bnd_; }
	int getNbBnd() { return nb_bnd; }
	void setNbPts(int nb_pts_) { this->nb_pts = nb_pts_; }
	int getNbPts() { return nb_pts; }
	void setDensity(Density* rho_) { this->rho = rho_; }
	void setEllipseAxes(double major_, double minor_) {
		this->major = major_;
		this->minor = minor_;
	}
	void setEllipsoidAxes(double major_, double minor_, double midax_=1.) {
		this->major = major_;
		this->minor = minor_;
		this->midax = midax_;
	}
	void setBoundaryPts(std::vector<Vec3>& bndry_pts_) {
		this->bndry_pts = bndry_pts_;
	}
	//void rejection2d(double area, double weighted_area, Density& density);
	void rejection2d(int nb_samples, double area, double weighted_area, Density& density, std::vector<Vec3>& samples);
	Vec3 singleRejection2d(double area, double weighted_area, Density& density);
	double random(double a, double b=1.);

	// Gordon Erlebacher: March 14, 2010
	void ellipsoid ( int dim_num, int& n, int *seed, double r[] );
	void ellipsoid_init ( int dim_num, int& n, int *seed, double r[] );
	void rejection3d(int nb_samples, Density& density, std::vector<Vec3>& samples);
	Vec3 singleRejection3d(Density& density);
	void cvt_iterate_3d ( int dim_num, int n, int batch, int sample, bool initialize, int sample_num, int *seed, double r[], double *it_diff, double *energy );

	void setGeometry(ParametricPatch* geom_) {
		geom = geom_;
	}
	void setOuterGeometry(ParametricPatch* geom_) {
		outer_geom = geom_;
	}
	void cvt_sample_3d ( int dim_num, int n, int n_now, int sample, bool initialize, int *seed, double r[] );

};

#endif
