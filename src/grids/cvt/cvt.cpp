#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <sstream>

#include <math.h>
#include <vector>

#include "cvt.h"

#include "grids/regulargrid.h"
#include "utils/random.h" 

using namespace std;

CVT::CVT(unsigned int nb_generators, unsigned int dimension, unsigned int nb_locked, unsigned int num_samples, unsigned int max_num_iters, unsigned int write_frequency, unsigned int sample_batch_size) 
	: Grid(nb_generators), 
	kdtree(NULL), cvt_iter(0),
	generatorsInitialized(false), 
	dim_num(dimension), nb_locked_nodes(nb_locked), nb_samples(num_samples),
	it_max(max_num_iters),sample_batch_size(sample_batch_size), write_freq(write_frequency)
{
	initTimers(); 
}

CVT::CVT (std::vector<NodeType>& nodes, unsigned int dimension, unsigned int nb_locked, unsigned int num_samples, unsigned int max_num_iters, unsigned int write_frequency, unsigned int sample_batch_size)
	: Grid(nodes), 
	kdtree(NULL), cvt_iter(0),
	generatorsInitialized(true), 
	dim_num(dimension), nb_locked_nodes(nb_locked), nb_samples(num_samples),
	it_max(max_num_iters),sample_batch_size(sample_batch_size), write_freq(write_frequency)
{
	initTimers();
}

void CVT::initTimers() {
	timers["total"] = new Timer("Total time in CVT::generate()");
	timers["iter"] = new Timer("perform one iteraton of Lloyd's");
	timers["initial"] = new Timer("Initialize CVT generators"); 
	timers["sample"] = new Timer("generate set of probabilistic samples");
	timers["energy"] = new Timer("compute system energy");
	timers["kbuild"] = new Timer("construct new KDTree");
	timers["kupdate"] = new Timer("update KDTree");
	timers["neighbor"] = new Timer("nearest neighbor search");
}

CVT::~CVT() {
	// dump timers (in case they havent been dumped prior to this
	timers.begin()->second->printAll(); 
	// erase all timers
	std::map<std::string, Timer*>::iterator iter = timers.begin(); 
	while(iter != timers.end()) {
		delete(iter->second);
		iter++;
	}
}

void CVT::generate() {
	if (!generatorsInitialized || (nb_nodes != node_list.size())) {
		this->node_list.resize(nb_nodes); 
		// Random in unit square
	//	this->cvt_sample(this->node_list, 0, nb_nodes, CVT::RANDOM, true); 
		// Regular sampling in (0,1) (NOTE: excludes 0 and 1) 
	//	this->cvt_sample(this->node_list, 0, nb_nodes, CVT::GRID, true); 
		// Random sampling in unit CIRCLE 
		this->cvt_sample(this->node_list, 0, nb_nodes, CVT::USER_INIT, true); 

		generatorsInitialized = true;
	}
		
	// Should we generate a KDTree to accelerate sampling? Cost of tree generation is 
	// high and should be amortized by VERY MANY samples. However, too many samples 
	// argues in favor of discrete Voronoi transform
	
	while (cvt_iter < it_max) {
		if (cvt_iter % write_freq == 0) {
			this->writeToFile();
		}
//		cvt_iterate(sample_batch_size, nb_samples, CVT::RANDOM);
		cvt_iter++;
	}
	
	cvt_iter = -1;
	this->writeToFile();

	std::cout << "CVT GENERATE NOT IMPLEMENTED" << std::endl;
	exit(EXIT_FAILURE); 
}

std::string CVT::getFileDetailString() {
	std::stringstream ss(std::stringstream::out); 
	ss << nb_nodes << "generators_" << dim_num << "d"; 
	return ss.str();	
}



//****************************************************************************80

void CVT::user_init(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand)

// Originally "user(...)" 
//****************************************************************************80
//
//  Purpose:
//
//    USER_INIT samples points in a user-specified region with given density.
//
//  Discussion:
//
//    This routine can be used to 
//
//    * specify an interesting initial configuration for the data,
//      by specifing that USER be used for initialization (INIT = 3);
//
//    * specify the shape of the computational region, by specifying
//      that sample points are to be generated by this routine, 
//      (SAMPLE = 3) and then returning sample points uniformly at random.
//
//    * specify the distribution or density function, by specifying
//      that sample points are to be generated by this routine, 
//      (SAMPLE = 3 ) and then returning sample points according to a 
//      given probability density function.
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
//    Input, integer N_NOW, the number of sample points desired.
//
//    Input, bool INIT_RAND, if TRUE, initialize the seed for any RNG used within
//    the routine
//
//    Output, a list of nodes
{
    double angle;
    int j;
    double radius;

    for (j = indx_start; j < indx_start+n_now; j++) {
        angle = 2.0 * M_PI * randf(0., 1.);
        radius = sqrt(randf(0., 1.));
        user_node_list[j][0] = radius * cos(angle);
        user_node_list[j][1] = radius * sin(angle);
    }

    return;
}


//****************************************************************************80

void CVT::user_sample(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand)

// Originally "user(...)" 
//****************************************************************************80
//
//  Purpose:
//
//    USER_INIT samples points in a user-specified region with given density.
//
//  Discussion:
//
//    This routine can be used to 
//
//    * specify an interesting initial configuration for the data,
//      by specifing that USER be used for initialization (INIT = 3);
//
//    * specify the shape of the computational region, by specifying
//      that sample points are to be generated by this routine, 
//      (SAMPLE = 3) and then returning sample points uniformly at random.
//
//    * specify the distribution or density function, by specifying
//      that sample points are to be generated by this routine, 
//      (SAMPLE = 3 ) and then returning sample points according to a 
//      given probability density function.
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
//    Input, integer N_NOW, the number of sample points desired.
//
//    Input, bool INIT_RAND, if TRUE, initialize the seed for any RNG used within
//    the routine
//
//    Output, a list of nodes
{
    double angle;
    int j;
    double radius;

    for (j = indx_start; j < indx_start+n_now; j++) {
        angle = 2.0 * M_PI * randf(0., 1.);
        radius = sqrt(randf(0., 1.));
        user_node_list[j][0] = radius * cos(angle);
        user_node_list[j][1] = radius * sin(angle);
    }

    return;
}






//****************************************************************************80
void CVT::cvt_sample(std::vector<NodeType>& sample_node_list, int indx_start, int n_now, int sample, bool init_rand) 
//****************************************************************************80
//
//  Purpose:
//
//    CVT_SAMPLE returns sample points.
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
//    request that only N_NOW be generated now (on this call) starting from
//    index N_START.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    6/24/10
//
//  Author:
//
//    Evan Bollig
//
//  Parameters:
//    Input, int INDX_START, the starting index in node_list at which we will
//    generate N_NOW nodes
//
//    Input, int N_NOW, the number of sample points to be generated
//    on this call.  N_NOW must be at least 1.
//
//    Input, sample_type SAMPLE, specifies how the sampling is done.
//    	(this is an enum and part of this class) 
//    -1, 'RANDOM', using C++ RANDOM function;
//    REMOVED:  0, 'UNIFORM', using a simple uniform RNG;
//    REMOVED:  1, 'HALTON', from a Halton sequence;
//     2, 'GRID', points from a grid;
//     3, 'USER', call "user" routine.
//
//    Input, bool INIT_RAND, is TRUE if the pseudorandom process should be
//    reinitialized.
//
//    Output: NONE. Modifies sample_node_list directly. 
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

    if (n_now < 1) {
        cout << "\n";
        cout << "CVT_SAMPLE - Called, but nothing to do!\n";
	return;
    }

    if (sample == CVT::RANDOM) {
        if (init_rand) {
            random_initialize(this->rand_seed);
        }

        for (j = indx_start; j < indx_start+n_now; j++) {
            for (i = 0; i < dim_num; i++) {
                // This is Gordon Erlebacher's random(,) routine (defined below)
                sample_node_list[j][i] = randf(0, 1);
            }
        }
        this->rand_seed = (this->rand_seed) + n_now * dim_num;
    } else if (sample == CVT::GRID) {

	// NOTE: this is uniform sampling between (0,1)^{dim_num}
	// in 1D it samples [0.5,0.95] for 10 nodes 

        exponent = 1.0 / (double) (dim_num);
//        ngrid = (int) pow((double) n, exponent);
	ngrid = (int) pow((double) nb_nodes, exponent);        
        rank_max = (int) pow((double) ngrid, (double) dim_num);
        tuple = new int[dim_num];

//        if (rank_max < n) {
	if (rank_max < nb_nodes) {        
            ngrid = ngrid + 1;
            rank_max = (int) pow((double) ngrid, (double) dim_num);
        }

        if (init_rand) {
            rank = -1;
            tuple_next_fast(ngrid, dim_num, rank, tuple);
        }

        rank = (this->rand_seed) % rank_max;

        for (j = indx_start; j < indx_start+n_now; j++) {
            tuple_next_fast(ngrid, dim_num, rank, tuple);
            rank = rank + 1;
            rank = rank % rank_max;
            for (i = 0; i < dim_num; i++) {
                sample_node_list[j][i] = double ( 2 * tuple[i] - 1) / double ( 2 * ngrid);
            }
        }
        delete [] tuple;
        this->rand_seed = this->rand_seed + n_now;
    } else if (sample == CVT::USER_INIT) {
        user_init(sample_node_list, indx_start, n_now, init_rand);
    } else {
	cout << "\n"; 
	cout << "CVT_INIT - Unsupported sample type. Only RANDOM and USER sampling are available at this time.\n"; 
	exit(EXIT_FAILURE);
    } 

    // print seeds
    if (DEBUG) {
        printf("Initial seed positions\n");
        for (int i = indx_start; i < indx_start+n_now; i++) {
            printf("(%d): \n", i);
            for (int j = 0; j < dim_num; j++) {
                printf("%f ", sample_node_list[i][j]);
            }
            printf("\n");
        }
        printf("  -  end initial seeds --------------------\n");
    }
    
    return;
}


//****************************************************************************80

unsigned long CVT::random_initialize(int seed)

//****************************************************************************80
//
//  Purpose:
//
//    RANDOM_INITIALIZE initializes the RANDOM random number generator.
//
//  Discussion:
//
//    If you don't initialize RANDOM, the random number generator, 
//    it will behave as though it were seeded with value 1.  
//    This routine will either take a user-specified seed, or
//    (if the user passes a 0) make up a "random" one.  In either
//    case, the seed is passed to SRAND (the appropriate routine 
//    to call when setting the seed for RANDOM).  The seed is also
//    returned to the user as the value of the function.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    07 December 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int SEED, is either 0, which means that the user
//    wants this routine to come up with a seed, or nonzero, in which
//    case the user has supplied the seed.
//
//    Output, unsigned long RANDOM_INITIALIZE, is the value of the seed
//    passed to SRAND, which is either the user's input value, or if
//    that was zero, the value selected by this routine.
//
{
    unsigned long ul_seed;

    if (seed != 0) {
        if (DEBUG) {
            cout << "\n";
            cout << "RANDOM_INITIALIZE\n";
            cout << "  Initialize RANDOM with user SEED = " << seed << "\n";
        }
    } else {
        seed = get_seed();
        if (DEBUG) {
            cout << "\n";
            cout << "RANDOM_INITIALIZE\n";
            cout << "  Initialize RAND with arbitrary SEED = " << seed << "\n";
        }
    }
    //
    //  Now set the seed.
    //
    ul_seed = (unsigned long) seed;
    srand(ul_seed);

    return ul_seed;
}
//****************************************************************************80

//****************************************************************************80

void CVT::tuple_next_fast(int m, int n, int rank, int x[])

//****************************************************************************80
//
//  Purpose:
//
//    TUPLE_NEXT_FAST computes the next element of a tuple space, "fast".
//
//  Discussion:
//
//    The elements are N vectors.  Each entry is constrained to lie
//    between 1 and M.  The elements are produced one at a time.
//    The first element is
//      (1,1,...,1)
//    and the last element is
//      (M,M,...,M)
//    Intermediate elements are produced in lexicographic order.
//
//  Example:
//
//    N = 2,
//    M = 3
//
//    INPUT        OUTPUT
//    -------      -------
//    Rank          X
//    ----          ----
//   -1            -1 -1
//
//    0             1  1
//    1             1  2
//    2             1  3
//    3             2  1
//    4             2  2
//    5             2  3
//    6             3  1
//    7             3  2
//    8             3  3
//    9             1  1
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    11 August 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int M, the maximum entry in each component.
//    M must be greater than 0.
//
//    Input, int N, the number of components.
//    N must be greater than 0.
//
//    Input, integer RANK, indicates the rank of the tuples.
//    Typically, 0 <= RANK < N**M; values larger than this are legal
//    and meaningful, and are equivalent to the corresponding value
//    MOD N**M.  If RANK < 0, this indicates that this is the first
//    call for the given values of (M,N).  Initialization is done,
//    and X is set to a dummy value.
//
//    Output, int X[N], the next tuple, or a dummy value if initialization
//    is being done.
//
{
    static int *base = NULL;
    int i;
    //
    if (rank < 0) {
        if (m <= 0) {
            cout << "\n";
            cout << "TUPLE_NEXT_FAST - Fatal error!\n";
            cout << "  The value M <= 0 is not legal.\n";
            cout << "  M = " << m << "\n";
            exit(1);
        }
        if (n <= 0) {
            cout << "\n";
            cout << "TUPLE_NEXT_FAST - Fatal error!\n";
            cout << "  The value N <= 0 is not legal.\n";
            cout << "  N = " << n << "\n";
            exit(1);
        }

        if (base) {
            delete [] base;
        }
        base = new int[n];

        base[n - 1] = 1;
        for (i = n - 2; 0 <= i; i--) {
            base[i] = base[i + 1] * m;
        }
        for (i = 0; i < n; i++) {
            x[i] = -1;
        }
    } else {
        for (i = 0; i < n; i++) {
            x[i] = ((rank / base[i]) % m) + 1;
        }
    }
    return;
}
//****************************************************************************80

int CVT::get_seed(void)

//****************************************************************************80
//
//  Purpose:
//
//    GET_SEED returns a random seed for the random number generator.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    15 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, int GET_SEED, a random seed value.
//
{
#define I_MAX 2147483647

    time_t clock;
    int i;
    int ihour;
    int imin;
    int isec;
    int seed;
    struct tm *lt;
    time_t tloc;
    //
    //  If the internal seed is 0, generate a value based on the time.
    //
    clock = time(&tloc);
    lt = localtime(&clock);
    //
    //  Hours is 1, 2, ..., 12.
    //
    ihour = lt->tm_hour;

    if (12 < ihour) {
        ihour = ihour - 12;
    }
    //
    //  Move Hours to 0, 1, ..., 11
    //
    ihour = ihour - 1;

    imin = lt->tm_min;

    isec = lt->tm_sec;

    seed = isec + 60 * (imin + 60 * ihour);
    //
    //  We want values in [1,43200], not [0,43199].
    //
    seed = seed + 1;
    //
    //  Remap ISEED from [1,43200] to [1,IMAX].
    //
    seed = (int)
            (((double) seed)
            * ((double) I_MAX) / (60.0 * 60.0 * 12.0));
    //
    //  Never use a seed of 0.
    //
    if (seed == 0) {
        seed = 1;
    }

    return seed;
#undef I_MAX
}
//****************************************************************************80
