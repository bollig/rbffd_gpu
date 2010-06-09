# include <cstdlib>
# include <cmath>
# include <ctime>
# include <iostream>
# include <iomanip>
# include <fstream>
#include <string.h>

#include <math.h>
#include <vector>

#include "ellipsoid_cvt.h"

using namespace std;

//****************************************************************************80
EllipsoidCVT::EllipsoidCVT(double major_, double minor_, double midax_, int DEBUG_)
: CVT(DEBUG_) 
{
	nb_bnd = 0;
}
//****************************************************************************80

//----------------------------------------------------------------------
void EllipsoidCVT::ellipsoid_init ( int dim_num, int& n, int *seed, double r[] )
{
# define PI 3.141592653589793

// r[] is a vector of with 3 components at each point. Make sure memory is properly allocated!

	vector<Vec3> samples;
	samples.resize(n); // total number of points

	// random seeds with rejection2d.
	printf("ellipsoid_init: rejection3d\n");
	rejection3d(n,  *rho, samples);
	printf("n= %d\n", n);
	//exit(0);

  	for ( int j = 0; j < n; j++ ) {
    	r[0+j*3] = samples[j].x();
    	r[1+j*3] = samples[j].y();
    	r[2+j*3] = samples[j].z();
  	}

  // print initial seeds
  printf("Initial seed positions, %d seeds\n", n);
  for (int i=0; i < n; i++) {
	printf("(%d): \n", i);
  	for (int j=0; j < dim_num; j++) {
  		printf("%f ", r[j+i*dim_num]);
  	}
	printf("\n");
  }
  printf(" -----  end initial seeds --------------------\n");

  return;
# undef PI
}
//----------------------------------------------------------------------
void EllipsoidCVT::ellipsoid ( int dim_num, int& n, int *seed, double r[] )

// return number boundary points
// n = total number of points

//****************************************************************************80
//
//  Purpose:
//
//    USER samples points within an ellipse, given the boundary points
//    Grid points are uniform. For each point, compute r and theta. Remove the points outside the ellipse
//
//  Modified:
//    14 March 2010
//
//  Author:
//    Gordon Erlebacher
//
//  Parameters:
//    Input, integer DIM_NUM, the spatial dimension.
//    Input, integer N, the number of sample points desired.
//    Input/output, int *SEED, the "seed" value.  On output, SEED has
//       been updated.
//    Output, double R[DIM_NUM*N], the sample values.
//
// Ellipsoid: (phi in [0,2*pi], theta in [0,pi]
//   x= a*cos(theta)*cos(phi)
//   y= b*sin(theta)*cos(phi)
//   z= c*sin(phi)
{
# define PI 3.141592653589793

// na : is the number of points along the major axis
// nb : is the number of points along the minor axis
// nc : is the number of points along the middle axis

	printf("enter ellipsoid\n");

	vector<Vec3> samples;
	samples.resize(n);
	rejection3d(n, *rho, samples);

	for (int i=0; i < n; i++) {
		r[0+i*3] = samples[i].x();
		r[1+i*3] = samples[i].y();
		r[2+i*3] = samples[i].z();
  	}

  return;
# undef PI
}
//----------------------------------------------------------------------
void EllipsoidCVT::rejection3d(int nb_samples, Density& density, vector<Vec3>& samples)
// Given a pdf p(x,y) (zero outside the ellipse, with unit integral over the ellipsoid domain)
// and given a constant bound M, such that p(x,y,z) < M q(x,y,z), where q(x,y,z) is uniform distribution over the ellipse.
// Generate a sample (x*,y*,z*), and a sample u* in [0,1]
// x* \in [-a,a], y* \in [-b,b], z* \in [-c,c]
// Two cases:
// 1.   u < p(x,y,z) / M q(x,y,z) : accept (x*,y*,z*)
// 2.   u > p(x,y,z) / M q(x,y,z)   reject (x*,y*,z*)
//
// What is q(x,y): simply   1. / volume(ellipsoid)
// Choose M = max_{x,y}   p(x,y)*volume(ellipsoid)
// Given x*,y*,z* compute r*^2 = (x*/a)^2 + (y*/b*)^2 + (z*/c*)^2
// if r*^2 > 1., reject the point (outside the ellipsoid)

// Used in ellipse_init (constructor helper) and ellipsoid
// 3D version should be written though, but the ellipse is a special case
// (no hurry)
{
	samples.resize(nb_samples);
	printf("nb_samples= %d\n", nb_samples);

	for (int i=0; i < nb_samples; i++) {
		samples[i] = singleRejection3d(density);
		samples[i].print("samples3d");
		Vec3& s = samples[i];
		//printf("dist=%f\n", outer_geom->how_far(s.x(), s.y(), s.z()));
	}
}
//----------------------------------------------------------------------
Vec3 EllipsoidCVT::singleRejection3d(Density& density)
{
// Apparently not used in this file
// 3D version should be written though, but the ellipse is a special case
// (no hurry)

	double xs, ys, zs;
	double u;
	double r2;
	double maxrhoi = 1. / rho->getMax();

	double dist;

	while (1) {
		// should use other geometry
		xs = random(-major, major);
		ys = random(-major, major);
		zs = random(-major, major);

		// use outer geometry for seed and random point distribution
		dist = outer_geom->how_far(xs,ys,zs);
		if (dist > 0.) continue;  // outside the ellipse

		// rejection part if non-uniform distribution
		u = random(0.,1.);
		if (u < (density(xs,ys,zs))*maxrhoi) break;
	}
	return Vec3(xs,ys,zs);
}
//----------------------------------------------------------------------
void EllipsoidCVT::cvt3d ( int dim_num, int n, int batch, int init, int sample, int sample_num, int it_max, int it_fixed, int *seed, double r[],int *it_num, double *it_diff, double *energy )
{
  bool DEBUG = false;
  int i;
  bool initialize;
  int seed_base;
  int seed_init;

  if ( batch < 1 )
  {
    cout << "\n";
    cout << "EllipsoidCVT - Fatal error!\n";
    cout << "  The input value BATCH < 1.\n";
    exit ( 1 );
  }

  if ( seed <= 0 )
  {
    cout << "\n";
    cout << "EllipsoidCVT - Fatal error!\n";
    cout << "  The input value SEED <= 0.\n";
    exit ( 1 );
  }

  if ( DEBUG )
  {
    cout << "\n";
    cout << "  Step       SEED          L2-Change        Energy\n";
    cout << "\n";
  }

  *it_num = 0;
  *it_diff = 0.0;
  *energy = 0.0;
  seed_init = *seed;
//
//  Initialize the data, unless the user has already done that.
//
  if ( init != 4 )
  {
    initialize = true;
	// Initial seed
    //NO NEED FOR ELLIPSE cvt_sample ( dim_num, n, n, init, initialize, seed, r );
  }
  if ( DEBUG )
  {
    cout                          << "  "
         << setw(4)  << *it_num   << "  "
         << setw(12) << seed_init << "\n";
  }
//
//  If the initialization and sampling steps use the same random number
//  scheme, then the sampling scheme does not have to be initialized.
//
  if ( init == sample )
  {
    initialize = false;
  }
  else
  {
    initialize = true;
  }
//
//  Carry out the iteration.
//
  while ( *it_num < it_max )
  {
//
//  If it's time to update the seed, save its current value
//  as the starting value for all iterations in this cycle.
//  If it's not time to update the seed, restore it to its initial
//  value for this cycle.
//
    if ( ( (*it_num) % it_fixed ) == 0 )
    {
      seed_base = *seed;
    }
    else
    {
      *seed = seed_base;
    }

    *it_num = *it_num + 1;
    seed_init = *seed;

	//exit(0);
	// I should modify so that the number of samples changes from one iteration to the next (for non-rectangular
	// or non-regular domains
    cvt_iterate_3d ( dim_num, n, batch, sample, initialize, sample_num, seed,
      r, it_diff, energy );

    initialize = false;

    if ( DEBUG )
    {
      cout                          << "  "
           << setw(4)  << *it_num   << "  "
           << setw(12) << seed_init << "  "
           << setw(14) << *it_diff  << "  "
           << setw(14) << *energy    << "\n";
    }

	if ((*it_num) % 20 == 0) {
      cout                          << "  "
           << setw(4)  << *it_num   << "  "
           << setw(12) << seed_init << "  "
           << setw(14) << *it_diff  << "  "
           << setw(14) << *energy    << "\n";

  		cvt_write ( dim_num, n, batch, seed_init, *seed, "none",
    		it_max, it_fixed, *it_num, *it_diff, *energy, "none", sample_num, r,
    		"voronoi_tmp.txt", false );
			//exit(0);
	}
  }
  return;
}
//****************************************************************************

void EllipsoidCVT::cvt_iterate_3d ( int dim_num, int n, int batch, int sample, bool initialize,
  int sample_num, int *seed, double r[], double *it_diff, double *energy )

//****************************************************************************80
//
//  Purpose:
//
//    EllipsoidCVT_ITERATE takes one step of the EllipsoidCVT iteration.
//
//  Discussion:
//
//    The routine is given a set of points, called "generators", which
//    define a tessellation of the region into Voronoi cells.  Each point
//    defines a cell.  Each cell, in turn, has a centroid, but it is
//    unlikely that the centroid and the generator coincide.
//
//    Each time this EllipsoidCVT iteration is carried out, an attempt is made
//    to modify the generators in such a way that they are closer and
//    closer to being the centroids of the Voronoi cells they generate.
//
//    A large number of sample points are generated, and the nearest generator
//    is determined.  A count is kept of how many points were nearest to each
//    generator.  Once the sampling is completed, the location of all the
//    generators is adjusted.  This step should decrease the discrepancy
//    between the generators and the centroids.
//
//    The centroidal Voronoi tessellation minimizes the "energy",
//    defined to be the integral, over the region, of the square of
//    the distance between each point in the region and its nearest generator.
//    The sampling technique supplies a discrete estimate of this
//    energy.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    20 September 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Qiang Du, Vance Faber, and Max Gunzburger,
//    Centroidal Voronoi Tessellations: Applications and Algorithms,
//    SIAM Review, Volume 41, 1999, pages 637-676.
//
//  Parameters:
//
//    Input, int DIM_NUM, the spatial dimension.
//
//    Input, int N, the number of Voronoi cells.
//
//    Input, int BATCH, sets the maximum number of sample points
//    generated at one time.  It is inefficient to generate the sample
//    points 1 at a time, but memory intensive to generate them all
//    at once.  You might set BATCH to min ( SAMPLE_NUM, 10000 ), for instance.
//    BATCH must be at least 1.
//
//    Input, int SAMPLE, specifies how the sampling is done.
//    -1, 'RANDOM', using C++ RANDOM function;
//     0, 'UNIFORM', using a simple uniform RNG;
//     1, 'HALTON', from a Halton sequence;
//     2, 'GRID', points from a grid;
//     3, 'USER', call "user" routine.
//
//    Input, bool INITIALIZE, is TRUE if the SEED must be reset to SEED_INIT
//    before computation.  Also, the pseudorandom process may need to be
//    reinitialized.
//
//    Input, int SAMPLE_NUM, the number of sample points.
//
//    Input/output, int *SEED, the random number seed.
//
//    Input/output, double R[DIM_NUM*N], the Voronoi
//    cell generators.  On output, these have been modified
//
//    Output, double *IT_DIFF, the L2 norm of the difference
//    between the iterates.
//
//    Output, double *ENERGY,  the discrete "energy", divided
//    by the number of sample points.
//
{
  int *count;
  int get;
  int have;
  int i;
  int j;
  int j2;
  int *nearest;
  double *r2;
  double *s;
  bool success;
  double term;
//
//  Take each generator as the first sample point for its region.
//  This can slightly slow the convergence, but it simplifies the
//  algorithm by guaranteeing that no region is completely missed
//  by the sampling.
//

	printf("enter cvt_iterate_3d\n");

  *energy = 0.0;
  r2 = new double[dim_num*n];
  count = new int[n];
  nearest = new int[sample_num];
  s = new double[dim_num*sample_num];

  for ( j = 0; j < n; j++ ) {
    for ( i = 0; i < dim_num; i++ ) {
      r2[i+j*dim_num] = r[i+j*dim_num];
  }}
  for ( j = 0; j < n; j++ ) {
    count[j] = 0;   // 3/12/10
    //count[j] = 1; // original code (wrong in GE opinion)
  }
//
//  Generate the sampling points S.
//
  have = 0;

	// GE: track whether Voronoi cell "intersects the boundary
	int* intersect_bnd = new int [n]; // n seeds
	//printf("n= %d\n", n); exit(0);

	for (int i=0; i < n; i++) {
		intersect_bnd[i] = 0;
	}

  while ( have < sample_num ) {   // ***********************
    get = i4_min ( sample_num - have, batch );
    cvt_sample_3d ( dim_num, sample_num, get, sample, initialize, seed, s );

	//printf("cvt_sample3d\n");exit(0);

    initialize = false;
    have = have + get;
//
//  Find the index N of the nearest cell generator to each sample point S.
//
	// brute force (GE: MUST IMPROVE!!)
	// find nearest r point to each s point
    find_closest ( dim_num, n, get, s, r, nearest );
//
//  Add S to the centroid associated with generator N.
//  Gordon Erlebacher: Build in check for including within the geometry
//    at this point. Each time a sample point is inside the geometry, set
//    a flag stating that the seed "intersects" the boundary
//
	Vec3 inside;

    for ( j = 0; j < get; j++ ) {
      j2 = nearest[j];
      for ( i = 0; i < dim_num; i++ ) {
		// r2: centroid, s: sample point
        r2[i+j2*dim_num] = r2[i+j2*dim_num] + s[i+j*dim_num];
	  }
	  inside.setValue(s[0+j*dim_num],s[1+j*dim_num],s[2+j*dim_num]);
	  double dist = geom->how_far(inside); // use actual geometry
	  if (dist > 0.) { // outside
		//printf("** dist= %f, j2= %d\n", dist, j2);
		//inside.print("inside vec");
		intersect_bnd[j2] += 1; // j2 is the seed
	  } else {
		intersect_bnd[j2] -= 1;
	  }
	  //printf("dist (< 0 if inside) = %f\n", dist);

      for ( i = 0; i < dim_num; i++ ) {
        *energy = *energy + pow ( r[i+j2*dim_num] - s[i+j*dim_num], 2 );
      }
      count[j2] = count[j2] + 1;
    }
  } //******************

//
//  Estimate the centroids.
//
  for ( j = 0; j < n; j++ ) {
    for ( i = 0; i < dim_num; i++ ) {
	  if (count[j] > 0) {
      	r2[i+j*dim_num] = r2[i+j*dim_num] / ( double ) ( count[j] );
	  } else {
      	printf("count[j] should never be zero\n");
		//exit(1);
	  }
  }}
//
//  Determine the sum of the distances between generators and centroids.
//
  *it_diff = 0.0;

  for ( j = 0; j < n; j++ ) {
    term = 0.0;
    for ( i = 0; i < dim_num; i++ ) {
      term = term + ( r2[i+j*dim_num] - r[i+j*dim_num] )
                  * ( r2[i+j*dim_num] - r[i+j*dim_num] );
    }
    *it_diff = *it_diff + sqrt ( term );
  }
//
//  Replace the generators by the centroids.
//

// Identify which cells intersect the boundary.
// When a cell intersects a boundary (i.e., a sample point is outside the
// ellipse), project the centroid onto the boundary.

// Displace all the points
// after displacement, project  "boundary centroids" to the surface

  for ( j = 0; j < n; j++ ) {
    for ( i = 0; i < dim_num; i++ ) {
      r[i+j*dim_num] = r2[i+j*dim_num];
  }}

  int cnt = 0;
  Vec3 pt, ptc;
  for (j=0; j < n; j++) {
    // if count == abs(intersect_bnd), then there is no intersection
	if (count[j] != abs(intersect_bnd[j])) {
		cnt++;
		printf("===========================\n");
		//printf("intersect with bndry\n");
		ptc.setValue(r[0+j*dim_num],r[1+j*dim_num],r[2+j*dim_num]);
		//pt = geom->projectToBoundary(ptc);
		pt = geom->project(ptc);
		//ptc.print("pt off bnd (ptc)");
		//pt.print("pt on bnd");

		double howfar = geom->how_far(pt);
		Vec3 slope = (pt-ptc);
		slope.normalize();
		Vec3 grad = geom->gradient(pt.x(), pt.y(), pt.z());
		Vec3 gradptc = geom->gradient(ptc.x(), ptc.y(), ptc.z());
		grad.normalize();
		gradptc.normalize();
		ptc.print("ptc");
		pt.print("pt");
		grad.print("grad(@pt)");
		gradptc.print("grad(@ptc)");
		slope.print("slope (pt-ptc)");
		grad.print("grad(pt)");
	}
  	//printf("count: %d, intersect: %d, diff= %d\n", count[j], intersect_bnd[j],
	     //count[j]-abs(intersect_bnd[j]));
	// project points with intersect == 1
  }
  printf("nb intersections: %d\n", cnt);

  printf("(%s, Line: %d) EXIT_FAILURE\n", __FILE__, __LINE__);
  exit(EXIT_FAILURE);


  //printf("r2: %f, %f\n", r2[0+nb_bnd*dim_num], r2[1+nb_bnd*dim_num]);
  //exit(0);
//
//  Normalize the discrete energy estimate.
//
  *energy = *energy / sample_num;

  printf("exit iterate_3d\n");

  delete [] count;
  delete [] nearest;
  delete [] r2;
  delete [] s;

  return;
}
//****************************************************************************80
void EllipsoidCVT::cvt_sample_3d ( int dim_num, int n, int n_now, int sample, bool initialize, int *seed, double r[] )

//****************************************************************************80
//
//  Purpose:
//
//    EllipsoidCVT_SAMPLE returns sample points.
//
//  Discussion:
//
//    N sample points are to be taken from the unit box of dimension DIM_NUM.
//
//    These sample points are usually created by a pseudorandom process
//    for which the points are essentially indexed by a quantity called
//    SEED.  To get N sample points, we generate values with indices
//    SEED through SEED+N-1.
//
//    It may not be practical to generate all the sample points in a
//    single call.  For that reason, the routine allows the user to
//    request a total of N points, but to require that only N_NOW be
//    generated now (on this call).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    23 June 2005
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int DIM_NUM, the spatial dimension.
//
//    Input, int N, the number of Voronoi cells.
//
//    Input, int N_NOW, the number of sample points to be generated
//    on this call.  N_NOW must be at least 1.
//
//    Input, int SAMPLE, specifies how the sampling is done.
//    -1, 'RANDOM', using C++ RANDOM function;
//     0, 'UNIFORM', using a simple uniform RNG;
//     1, 'HALTON', from a Halton sequence;
//     2, 'GRID', points from a grid;
//     3, 'USER', call "user" routine.
//
//    Input, bool INITIALIZE, is TRUE if the pseudorandom process should be
//    reinitialized.
//
//    Input/output, int *SEED, the random number seed.
//
//    Output, double R[DIM_NUM*N_NOW], the sample points.
//
{
  double exponent;
  static int *halton_base = NULL;
  static int *halton_leap = NULL;
  static int *halton_seed = NULL;
  int halton_step;
  int i;
  int j;
  int k;
  static int ngrid;
  static int rank;
  int rank_max;
  static int *tuple = NULL;

  if ( n_now < 1 )
  {
    cout << "\n";
    cout << "EllipsoidCVT_SAMPLE - Fatal error!\n";
    cout << "  N_NOW < 1.\n";
    exit ( 1 );
  }

  if ( sample == -1 )
  {
    if ( initialize )
    {
      random_initialize ( *seed );
    }

    for ( j = 0; j < n_now; j++ )
    {
      for ( i = 0; i < dim_num; i++ )
      {
        r[i+j*dim_num] = random(0,1);
      }
    }
    *seed = ( *seed ) + n_now * dim_num;
  }
  else if ( sample == 0 )
  {
    r8mat_uniform_01 ( dim_num, n_now, seed, r );
  }
  else if ( sample == 1 )
  {
    halton_seed = new int[dim_num];
    halton_leap = new int[dim_num];
    halton_base = new int[dim_num];

    halton_step = *seed;

    for ( i = 0; i < dim_num; i++ )
    {
      halton_seed[i] = 0;
    }

    for ( i = 0; i < dim_num; i++ )
    {
      halton_leap[i] = 1;
    }

    for ( i = 0; i < dim_num; i++ )
    {
      halton_base[i] = prime ( i + 1 );
    }

    i4_to_halton_sequence ( dim_num, n_now, halton_step, halton_seed,
      halton_leap, halton_base, r );

    delete [] halton_seed;
    delete [] halton_leap;
    delete [] halton_base;

    *seed = *seed + n_now;
  }
  else if ( sample == 2 )
  {
    exponent = 1.0 / ( double ) ( dim_num );
    ngrid = ( int ) pow ( ( double ) n, exponent );
    rank_max = ( int ) pow ( ( double ) ngrid, ( double ) dim_num );
    tuple = new int[dim_num];

    if ( rank_max < n )
    {
      ngrid = ngrid + 1;
      rank_max = ( int ) pow ( ( double ) ngrid, ( double ) dim_num );
    }

    if ( initialize )
    {
      rank = -1;
      tuple_next_fast ( ngrid, dim_num, rank, tuple );
    }

    rank = ( *seed ) % rank_max;

    for ( j = 0; j < n_now; j++ )
    {
      tuple_next_fast ( ngrid, dim_num, rank, tuple );
      rank = rank + 1;
      rank = rank % rank_max;
      for ( i = 0; i < dim_num; i++ )
      {
        r[i+j*dim_num] = double ( 2 * tuple[i] - 1 ) / double ( 2 * ngrid );
      }
    }
    delete [] tuple;
    *seed = *seed + n_now;
  }
  else if ( sample == 3 )
  {
	#if 0
    user ( dim_num, n_now, seed, r );
	#else
	int nb_bnd = 0;
	if (initialize) {
		ellipsoid(dim_num, nb_pts, seed, r);
	} else {
		ellipsoid(dim_num, n_now, seed, r);
	}
	#endif
	#if 0
	for (int i=n_now-5; i < n_now; i++) {
		printf("r= %f\n", r[i]);
	}
	printf("\n");
	#endif
  }
  else
  {
    cout << "\n";
    cout << "EllipsoidCVT_SAMPLE - Fatal error!\n";
    cout << "  The value of SAMPLE = " << sample << " is illegal.\n";
    exit ( 1 );
  }

  // print seeds
  if (DEBUG) {
    printf("Initial seed positions\n");
     for (int i=0; i < n_now; i++) {
	printf("(%d): \n", i);
  	for (int j=0; j < dim_num; j++) {
  		printf("%f ", r[j+i*dim_num]);
  	}
	printf("\n");
    }
    printf("  -  end initial seeds --------------------\n");
  }
  
}
//****************************************************************************80
