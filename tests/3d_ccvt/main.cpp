#include <stdlib.h>
#include "ellipsoid_cvt.h"
#include <vector>
#include <string.h>
#include <ellipsoid_patch.h>
#include <parametric_patch.h>

using namespace std;

// GLOBAL VARIABLES
EllipsoidCVT* cvt;
double major;
double minor;
double midax;
ParametricPatch* geometry;
ParametricPatch* outer_geometry;

//----------------------------------------------------------------------
void create_3d_CVT(int N, Density& rho)
{
//# define N 200
# define DIM_NUM 3
  int batch;
  bool comment;
  double energy;
  char file_out_name[80] = "cvt_circle.txt";
  int init;
  char init_string[80];
  double it_diff;
  int it_fixed;
  int it_max;
  int it_num;
  double r[DIM_NUM*N];
  int sample;
  int sample_num;
  char sample_string[80];
  int seed;
  int seed_init;

  batch = 1000;
  init = 4; // user initialize boundary


  strcpy ( init_string, "user" );
  it_max =  1000;
  it_fixed = 1;
  sample = 3;
  sample_num = 3000; // 10k for 300 pts, 30k for 1000 pts)
  strcpy ( sample_string, "user" );
  seed = 123456789;

  seed_init = seed;
  cvt->setNbPts(N);
  cvt->setDensity(&rho);
  cvt->setEllipsoidAxes(major, minor, midax);

  int n1 = 30; // not really need here
  int n2 = 30;
  double pi = acos(-1.);
  geometry = new EllipsoidPatch(0., pi, 0., 2.*pi, n1, n2, major, minor, midax);
  outer_geometry = new EllipsoidPatch(0., pi, 0., 2.*pi, n1, n2, major+1.0, minor+1.0, midax+1.);
  cvt->setGeometry(geometry);
  cvt->setOuterGeometry(outer_geometry);

  //cvt->ellipsoid_init(DIM_NUM, N, &seed, r );
  //cvt->cvt3d( DIM_NUM, N, batch, init, sample, sample_num, it_max, it_fixed,
  cvt->cvt( DIM_NUM, N, batch, init, sample, sample_num, it_max, it_fixed,
    &seed, r, &it_num, &it_diff, &energy );
  exit(0);

  comment = false; // comment lines at the top of the output file

  cvt->cvt_write ( DIM_NUM, N, batch, seed_init, seed, init_string, 
    it_max, it_fixed, it_num, it_diff, energy, sample_string, sample_num, r, 
    file_out_name, comment );

	return;

# undef DIM_NUM
# undef N
}
//----------------------------------------------------------------------
int main (int argc, char** argv)
{
	// domain dimensions
	double pi = acos(-1.);

	// sphere
	major = 5.;
	minor = 3.;
	midax = 4.;

	Density rho;

	vector<Vec3> bndry_pts;
	vector<double> intg;

	// total nb points used to compute Voronoi mesh. 
	// nb seeds
	int tot_nb_pts = 1000;

	bool create_cvt;
	create_cvt = false;
	create_cvt = true;

	if (create_cvt) {
		cvt = new EllipsoidCVT();
		create_3d_CVT(tot_nb_pts, rho);
		exit(0);
	}

	if (argc > 1) {
		return EXIT_FAILURE; 	// FAIL TEST
	} 

	return EXIT_SUCCESS; 		// PASS TEST
}
