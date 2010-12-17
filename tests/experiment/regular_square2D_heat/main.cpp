// Generate rbfs in a spherical shell

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <getopt.h>		// for getopt
#include <Vec3i.h>
#include <Vec3.h>
#include <ArrayT.h>
#include <vector> 
#include <algorithm> 
#include <functional> 
//#include <write_lineset.h>
//#include <write_psi.h>

#include <armadillo>
#include "grids/domain_decomposition/gpu.h"
#include "rbffd/rbfs/rbf_mq.h"
//#include "rbf_gaussian.h"
#include "rbffd/derivative.h"
#include "grids/regulargrid2d.h"
#include "exact_solutions/exact_regulargrid.h"
#include "pdes/parabolic/heat.h"
#include "density.h"
#include "grids/cvt/cvt.h"
#include "utils/comm/communicator.h"

// used go generate random seed that changes between runs
#include <time.h> 

using namespace std;
using namespace arma;

typedef ArrayT<double> AF;

// I need a datastructure  (i,x,y)

#define OTHER 0
#define CENTER 1
#define STENCIL 2

struct Dist {
    int i; // rbf index
    double d; // distance
    double x, y, z;
    // 0: other, 1: center, 2: stencil
    int id; // center, stencil or other
};

//----------------------------------------------------------------------
// not sure what this does any longer
//----------------------------------------------------------------------

// GLOBAL VARIABLES
CVT* cvt;
vector<Vec3> rbf_centers;
void checkDerivatives(Derivative& der, Grid& grid);
void checkXDerivatives(Derivative& der, Grid& grid);
double computeBoundaryIntegral(Density& rho, int npts, vector<double>& intg);
void computeBoundaryPointDistribution(double tot_length, int npts, int nb_bnd,
        vector<double> intg, vector<Vec3>& bnd);
double minimum(vector<double>& vec);
double major;
double minor;

vector<double> avgDist;

//----------------------------------------------------------------------
double minimum(vector<double>& vec) {
        double min = 1.e10;

        for (int i = 0; i < vec.size(); i++) {
                if (vec[i] < min) {
                        min = vec[i];

                }
        }
        return min;
}


//----------------------------------------------------------------------

GPU* distributeNodesAcrossGPUs(Grid *grid, Communicator* comm_unit, double dt) {
#if 1
    // Assume a grid of (gx, gy) GPUs.
    // MPI can create a Cartesian grid for us (TODO)
    // MPI_Cart_create()
    //
    int gx = comm_unit->getSize();
    int gy = 1; // Update this for Cart MPI
    int gz = 1;
    vector<GPU*> gpus;
    gpus.resize(gx * gy * gz);

    // The GPU class partitions the points as if they are in [-1,1]x[-1,1]x[-1,1]
    // We should generalize this.
    int xmin = grid->xmin;
    int xmax = grid->xmax;
    int ymin = grid->ymin;
    int ymax = grid->ymax;
    int zmin = grid->zmin;
    int zmax = grid->zmax;

    double deltax = (double) (xmax - xmin) / (double) gx;
    double deltay = (double) (ymax - ymin) / (double) gy;
    double deltaz = (double) (zmax - zmin) / (double) gz;

    printf("delta gpu x, y, z= %f, %f, %f\n", deltax, deltay, deltaz);

    // Initialize GPU datastructures
    for (int id = 0; id < gx * gy * gz; id++) {
        // Derived these on paper. They work, but it takes a while to verify

        // 1) Find the slice in which we lie (NOTE: "i" or "x" is varying fastest;
        //      for "k" switch gx to gz, and swap igz and igx equations)
        int igz = id / gx*gy;
        // 2) Find the row within the slice
        int igy = (id - igz * (gx*gy)) / gx;
        // 3) Find the column within the row
        int igx = (id - igz * (gx*gy)) - igy * gx;
       
        printf("igx = %d, igy = %d, igz = %d\n", igx, igy, igz);
        double xm = xmin + igx * deltax;
        double ym = ymin + igy * deltay;
        double zm = zmin + igz * deltaz;
        printf("xm= %f, ym= %f, zm=%f, dx = %f, dy = %f, dz = %f\n", xm, ym, zm, deltax, deltay, deltaz);
        gpus[id] = new GPU(xm, xm + deltax, ym, ym + deltay,  zm, zm + deltaz, dt, id, comm_unit->getSize());
    }

    // Figure out the sets Bi, Oi Qi

    printf("nb gpus: %d\n", (int) gpus.size());
    for (int i = 0; i < gpus.size(); i++) {
        printf("\n ***************** CPU %d ***************** \n", i);
        gpus[i]->fillLocalData(grid->getRbfCenters(), grid->getStencil(),
                grid->getBoundary(), grid->getAvgDist()); // Forms sets (Q,O,R) and l2g/g2l maps
        gpus[i]->fillVarData(grid->getRbfCenters()); // Sets function values in U
    }

    for (int i = 0; i < gpus.size(); i++) {
        printf(
                "\n ***************** FILLING O_by_rank for CPU%d ***************** \n",
                i);
        for (int j = 0; j < gpus.size(); j++) {
            gpus[i]->fillDependencyList(gpus[j]->R, j); // appends to O_by_rank	any nodes required by gpu[j]
        }

    }

    printf("gpu structures (Q\\O,O,R) are initialized\n");
    printf("initialized on scalar variable to linear function\n");

    // Distribute nodes to each GPU.
    for (int i = 1; i < gpus.size(); i++) {
        printf("Distributing to GPU[%d]\n", i);
        comm_unit->sendObject(gpus[i], i);
    }

    return gpus[0];

    // Compute derivative on a single GPU. Check against analytical result
    // du/dx=1, du/dx=2, du/dx=3
#if 0
    RBF_Gaussian rbf(1.);
    const Vec3 xi(.5, 0., 0.);
    for (int i = 0; i < 10; i++) {
        const Vec3 xvec(i * .1, 0., 0.);
        printf("%d, phi=%f, phi'=%f\n", i, rbf.eval(xvec, xi), rbf.xderiv(xvec, xi));
    }
#endif
#endif
}
//----------------------------------------------------------------------


void closeLogFile(void) {
    fprintf(stderr, "Closing STDOUT file\n");
    fclose(stdout);
}

void debugExit(void) {
    fprintf(stderr, "EXIT CALLED\n");
}

int parseCommandLineArgs(int argc, char** argv, int my_rank) {
    // Borrowed from Getopt-Long Example in GNU LibC manual
    int verbose_flag = 0; /* Flag set by '--verbose' and '--brief'. */
    int c;
    int hostname_flag = 0; // Non-zero if hostname is set;

    while (1) {
        // Struct order: { Long_name, Arg_required?, FLAG_TO_SET, Short_name}
        // NOTE: if an option sets a flag, it is set to the Short_name value
        static struct option long_options[] = {
            // NOTE: following are defined in getopt.h
            // 		no_argument			--> no arg to the option is expected
            //		required_argument 	--> arg is mandatory
            // 		optional_argument	--> arg is not necessary
            /* These options set a flag. */
            { "verbose", no_argument, &verbose_flag, 1},
            { "brief",
                no_argument, &verbose_flag, 0},
            /* These options don't set a flag.
             We distinguish them by their indices. */
            { "hostname", required_argument, &hostname_flag, 'h'},
            {
                "output-file", required_argument, 0, 'o'
            },
            { "file",
                required_argument, 0, 'o'},
            { "dir",
                required_argument, 0, 'd'},
            { "help", no_argument, 0,
                '?'},
            { 0, 0, 0, 0}
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        // the : here indicates a required argument
        c = getopt_long(argc, argv, "?o:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c) {
            case 0: // Any long option that sets a flag
                printf("option %s", long_options[option_index].name);
                if (optarg)
                    printf(" with arg %s", optarg);
                printf("\n");
                break;

            case 'o':
                char logname[256];
                sprintf(logname, "%s.%d", optarg, my_rank);
                printf("Redirecting STDOUT to file: `%s'\n", logname);
                freopen(logname, "w", stdout);
                atexit(closeLogFile);
                break;

            case 'h':
                printf("[DISABLED] option -h with value `%s'\n", optarg);
                break;

            case 'd':
                // 1) copy name into logdir (global) variable = argv/RANK
                // 2) mkdir logdir if not already made
                // 3) set logdir in all classes that write files (Heat, Derivative)
                break;

            case '?':
                printf("\nUsage: %s [options=arguments]\n", argv[0]);
                printf("Options:\n");
                printf(
                        "\t-o (--output-file, --file) \tSpecify the filename for process to redirect STDOUT to.\n");
                printf("\t-? (--help) \t\t\tPrint this message\n\n");
                exit(EXIT_FAILURE);
                break;

            default:
                printf("IN DEFAULT ARG OPTION (WHY?)\n");
                abort(); // abort loop when nothing is left
                break;
        }
    } // END WHILE

    /* Instead of reporting ‘--verbose’
     and ‘--brief’ as they are encountered,
     we report the final status resulting from them. */
    if (verbose_flag)
        printf("verbose flag is set\n");

    /* Print any remaining command line arguments (not options). */
    if (optind < argc) {
        printf("non-option ARGV-elements: ");
        while (optind < argc)
            printf("%s ", argv[optind++]);
        putchar('\n');
    }

    // Tell us whenever program terminates with an exit is called
    atexit(debugExit);

    return 0;
}

//----------------------------------------------------------------------

int main(int argc, char** argv) {
    Communicator* comm_unit = new Communicator(argc, argv);

    // An empty GPU object for each CPU. CPU0 (master) will
    // return this from distributeNodesAcrossGPUs while all
    // other CPUs get it from comm_unit->receiveObject()
    GPU* subdomain = NULL;

    Grid* grid = NULL;

    parseCommandLineArgs(argc, argv, comm_unit->getRank());

    int my_rank = comm_unit->getRank();
    
    cout << " Got Rank: " << comm_unit->getRank() << endl;
    cout << " Got Size: " << comm_unit->getSize() << endl;

    if (!my_rank) { // Master thread 0

        int stencil_size = 9;
        int nx = 10;
        int ny = 10;
        int nz = 10;

        grid = new RegularGrid2D(nx, ny, -1.,1., -1.,1., stencil_size);

        // 2nd argument: known number of boundary points (stored ahead of interior points)
        //grid->generateGrid("cvt_circle.txt", nb_bnd, tot_nb_pts);
        grid->generateGrid();
        grid->computeStencils(); // nearest nb_points
        //EB (NO NEED FOR THIS? CLEANUP: )
        // grid->avgStencilRadius(); // Fill variable grid.avg_distance (std::vector<double>)
        vector<double> avg_dist = grid->getAvgDist(); // get average stencil radius for each point

        //grid.computeStencilsRegular();   // regular 4 point stencil
        vector<vector<int> >& stencil = grid->getStencil();

        // global variable
        rbf_centers = grid->getRbfCenters();
        //vector<Vec3>& rbf_centers = grid.getRbfCenters();
        //		int nb_rbf = rbf_centers.size(); //grid.getRbfCenters().size();

        // Perform domain decomposition and distribute the subcollection of nodes to each CPU
        // parameter to distributeStencils should be a vector of vectors (list of node lists)
        // comm_unit handles message passing in blocking/non-blocking fashion but hides details
        // from this routine.
        //comm_unit->distributeStencils(grid.decomposeDomain(comm_unit->getSize()));
        subdomain = distributeNodesAcrossGPUs(grid, comm_unit, 0.001);

    }// endif (!my_rank)
    else {
        cout << "MPI RANK " << my_rank << ": waiting to receive subdomain"
                << endl;
        subdomain = new GPU(); // EMPTY object that will be filled by MPI

        int status = comm_unit->receiveObject(subdomain, 0); // Receive from CPU (0)

    }
     
#if 0
       comm_unit->consolidateObjects(subdomain);
        subdomain->writeFinal(grid->getRbfCenters(), (char*) "INITIAL_SOLUTION.txt");
        exit(0);

    char buf[32];
    sprintf(&buf[0], "Set AVG_DISTS", my_rank);
    subdomain->printVector(subdomain->Q_avg_dists, buf);
#endif 

    comm_unit->barrier(); // Called by all CPUs regardless of rank

    // EB NEED ALL STENCILS TO BE LOCAL INDEX BY THIS POINT
    Derivative der(subdomain->G_centers, subdomain->Q_stencils,
            subdomain->global_boundary_nodes.size());
    der.setAvgStencilRadius(subdomain->Q_avg_dists);

    // Set things up for variable epsilon
    int nb_rbfs = subdomain->Q_stencils.size();
    //EB 2
    //vector<double> avg_stencil_radius = grid.getAvgDist();
    vector<double> avg_stencil_radius = subdomain->Q_avg_dists;
    vector<double> epsv(nb_rbfs);

    for (int i = 0; i < nb_rbfs; i++) {
        epsv[i] = 1. / avg_stencil_radius[i];
        printf("avg rad(%d) = %f\n", i, avg_stencil_radius[i]);
    }
    double mm = minimum(avg_stencil_radius);
    printf("min avg_stencil_radius= %f\n", mm);

    der.setVariableEpsilon(epsv); // has no effect (vareps was disabled in derivative.cpp)
    //exit(0);

    subdomain->printCenters(subdomain->G_centers,
            "All Centers Needed by this CPU");

    printf("CHECKING STENCILS: \n");
    for (int irbf = 0; irbf < subdomain->Q_stencils.size(); irbf++) {
        printf("Stencil[%d] = ", irbf);
        if (irbf == subdomain->Q_stencils[irbf][0]) {
            printf("PASS\n");
            subdomain->printStencil(subdomain->Q_stencils[irbf], "S");
	} else {
            printf("FAIL\n");
        }
    }

    printf("start computing weights\n");
    for (int irbf = 0; irbf < subdomain->Q_stencils.size(); irbf++) {
        //		der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "x");
        //		der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "y");
        //		der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "lapl");
        // 0 here implies the first element of the stencil is the center
        char label[256];
        sprintf(label, "Stencil[%d]", irbf);
        subdomain->printStencil(subdomain->Q_stencils[irbf], label);
       
        // Centers are vec<Vec3> where Vec3=(double,double,double)
        // Stencils are vec<int> where int=index into centers vec<Vec3>
        der.computeWeightsSVD(subdomain->G_centers,
                subdomain->Q_stencils[irbf], irbf, "x");
        der.computeWeightsSVD(subdomain->G_centers,
                subdomain->Q_stencils[irbf], irbf, "y");
        der.computeWeightsSVD(subdomain->G_centers,
                subdomain->Q_stencils[irbf], irbf, "lapl");
    }
    printf("after all computeWeights\n");
    // MY BUG IS INSIDE HERE:
    //double maxEig = der.computeEig(subdomain->Q_stencils); // needs lapl_weights
    printf("after all computeEig\n");
    //exit(0);

    // Before solving the PDE, we will do several checks (each time)
    // Accuracy of (1st, 2nd) derivative of (constant, linear, quadratic, cubic functions)
    // Accuracy of lapl() or of (div.grad) on these three functions
    // Eigenvalues of lapl and grad^2

#if 0
    // I must compute the average error of the derivatives, with and without the boundary points
    //EB -1
    //checkDerivatives(der, grid);
    checkXDerivatives(der, grid);
    exit(0);
#endif

    comm_unit->barrier(); // Called by all CPUs regardless of rank

    // Loop this:
#if 0	
    for (int iter = 0; iter < 5; iter++) { // practice iterating the derivatives
        cout << "*********** COMPUTE DERIVATIVES (Iteration: " << iter << ") *************" << endl;
        subdomain->printVector(subdomain->U_G, "U_G");

        // Test computing derivatives
        vector<double> lapl_deriv(subdomain->Q_stencils.size());
        der.computeDeriv(Derivative::LAPL, subdomain->U_G, lapl_deriv);

        subdomain->printVector(lapl_deriv, "LAPL_U_G");

        // Imitation timestep:
        for (int j = 0; j < lapl_deriv.size(); j++) {
            subdomain->U_G[j] += lapl_deriv[j];
        }

        // Send updates according to MPISendable object.
        comm_unit->broadcastObjectUpdates(subdomain);
        comm_unit->barrier();
    }
#endif 

    //EB 3
#if 0
    // Need to add to GPU class: notion of global domain boundary. If the global domain
    // is represented by a vector of points where the first nb_bnd points in the
    // vector are boundary points, then we can specify a global number of boundary
    // elements and have routines like this work on "while(l2g(index) < nb_bnd)"

    printf("test Derivatives\n");
    //printf("rbf_center size: %d\n", nb_rbf); exit(0);
    testDeriv(C, der, grid);
    testDeriv(X, der, grid);
    testDeriv(Y, der, grid);
    testDeriv(X2, der, grid);
    testDeriv(XY, der, grid);
    testDeriv(Y2, der, grid);
    testDeriv(X3, der, grid);
    testDeriv(X2Y, der, grid);
    testDeriv(XY2, der, grid);
    testDeriv(Y3, der, grid);
#endif 

    //exit(0);

    //----------------------------------------------------------------------
#if 0
    vector<mat>& we = der.getLaplWeights();
    printf("Laplacian weights\n");
    for (int i = 0; i < we.size(); i++) {
        mat& w = we[i];
        //printf("weight %d, nb pts in stencil: %d\n", i, w.size());
        for (int j = 0; j < w.n_elem; j++) {
            printf("%f ", w(j));
        }
        printf("\n");
    }
    exit(0);
#endif
    //----------------------------------------------------------------------

    // box dimensions: hardcoded!!! [-1,1] x [-1,1]

    // 5 point laplacian (FD stencil)
    //dt = 0.24*dx*dx; // works for Cartesian mesh with 5 points stencil (non-rbf)
    // 9 point laplacian (rbf)
    // area of th ellipsoid = pi*a*b = 3.14*1*0.8 =
    // average area of voronoi cell: 3.14
    //exit(0);

    // SOLVE HEAT EQUATION

    //EB 4
#if 1
    // Exact Solution ( freq, decay )
    ExactSolution* exact = new ExactRegularGrid(acos(-1.) / 2., 1.);

    Heat heat(exact, subdomain, &der, comm_unit->getRank());
    heat.initialConditions(&subdomain->U_G);

    // Send updates according to MPISendable object.
    comm_unit->broadcastObjectUpdates(subdomain);
    comm_unit->barrier();

    // This is HARDCODED because we dont have the ability currently to call
    // maxEig = der.computeEig() and therefore we have a different timestep than
    // the original code. I will address this next.
    //heat.setDt(0.011122);
    heat.setDt(subdomain->dt);
    subdomain->printVector(subdomain->global_boundary_nodes,
            "GLOBAL BOUNDARY NODES: ");
    // Even with Cartesian, the max norm stays at one. Strange
    int iter;
    for (iter = 0; iter < 1000; iter++) {
        cout << "*********** COMPUTE DERIVATIVES (Iteration: " << iter
                << ") *************" << endl;
        subdomain->printVector(subdomain->U_G, "INPUT_TO_HEAT_ADVANCE");

        heat.advanceOneStepWithComm(comm_unit);
        subdomain->printVector(subdomain->U_G, "AFTER HEAT");

        double nrm = heat.maxNorm();

        // TODO : Need to add a "comm_unit->sendTerminate()" to
        // break all processes when problem is encountered
        if (nrm > 5.)
            break;
        //if (iter > 0) break;
    }

    printf("after heat\n");
    //	exit(0);
#endif 
    //}

    comm_unit->consolidateObjects(subdomain);

    if (comm_unit->getRank() == 0) {
        // TODO assemble final solution
        subdomain->writeFinal(grid->getRbfCenters(), (char*) "FINAL_SOLUTION.txt");
        // TODO print solution to file
        cout << "FINAL ITER: " << iter << endl;
    }
    printf("REACHED THE END OF MAIN\n");
}
//----------------------------------------------------------------------
