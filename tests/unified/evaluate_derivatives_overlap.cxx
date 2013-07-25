/*
 * 
 * Load metis domain
 * Load all weights
 * Load simple function (option to select 1's, sin(x), input file). 
 * Apply weights to function using GPU SpMV
 * Note: this SpMV overlaps
 * 
 */
#include <mpi.h> 

#include <stdlib.h>
#include <sstream>
#include <map>
#include <iostream> 
#include <cmath>

#include "utils/mpi_norms.h"

#include "grids/grid_reader.h"
#include "grids/domain.h"
#include "grids/metis_domain.h"
#include "rbffd/rbffd_vcl_overlap.h"
#include "rbffd/spmv_test_vcl.h"

#include "utils/opencl/viennacl_typedefs.h"

#include <viennacl/linalg/vector_operations.hpp> 
#include <viennacl/vector_proxy.hpp> 
#include <viennacl/linalg/prod.hpp> 
#include "viennacl/linalg/parallel_norm_1.hpp"                                                                                     
#include "viennacl/linalg/parallel_norm_2.hpp"
#include "viennacl/linalg/parallel_norm_inf.hpp"

#include <boost/program_options.hpp>

#include "timer_eb.h"


using namespace std;
using namespace EB;
using namespace boost; 

namespace po = boost::program_options;


int main(int argc, char** argv) {

    TimerList tm;

    tm["total"] = new Timer("[Main] Total runtime for this proc");
    tm["grid"] = new Timer("[Main] Grid generation");
    tm["gridReader"] = new Timer("[Main] Grid Reader Load File From Disk");
    tm["loadGrid"] = new Timer("[Main] Load Grid (and Stencils) from Disk");
    tm["loadDomain"] = new Timer("[Main] Load Domain from Disk");
    tm["writeGrid"] = new Timer("[Main] Write Grid (and Stencils) to Disk");
    tm["stencils"] = new Timer("[Main] Stencil generation");
    tm["writeStencils"] = new Timer("[Main] Write Stencils to Disk");
    tm["initialize"] = new Timer("[Main] Load settings and MPI_Init");
    tm["derSetup"] = new Timer("[Main] Setup RBFFD class");
    tm["assembleTest"] = new Timer("[Main] Assemble test vectors");
    tm["SpMV"] = new Timer("[Main] Compute Derivatives");
    tm["iteration"] = new Timer("[Main] Complete One Iteration");
    tm["finish"] = new Timer("[Main] ViennaCL Finish queue");
    tm["SpMV_w_comm"] = new Timer("[Main] SpMV + Comm");
    tm["computeNorms"] = new Timer("[Main] Compute Norms");
    tm["computeUpdate"] = new Timer("[Main] Compute mock timestep update");
    tm["synchronize"] = new Timer("[Main] Synchronize (perform MPI Comm)");
    tm["loadWeights"] = new Timer("[Main] Read weights from file");
    tm["tests"] = new Timer("[Main] Derivative tests");
    tm["cleanup"] = new Timer("[Main] Cleanup (delete) Domain and print final norms");

    tm["total"]->start();

    //-----------------
    tm["initialize"]->start();

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("debug,d", "enable verbose debug messages")
        ("grid_filename,g", po::value<string>(), "Grid filename (flat file, tab delimited columns). Required.")
        ("grid_num_cols,c", po::value<int>(), "Number of columns to expect in the grid file (X,Y,Z first). Note: columns in grid file, not problem dimensions")
        ("grid_size,N", po::value<int>(), "Number of nodes to expect in the grid file") 
        ("grid_dim,D", po::value<int>(), "Grid dimensions. Note: dimensions may be fewer than columns in grid file.") 
        ("stencil_size,n", po::value<int>(), "Number of nodes per stencil (assume all stencils are the same size)")
        ("partition_filename,p", po::value<string>(), "METIS Output Partition Filename (*.part.<P-processors>)")
        ("use_hyperviscosity", po::value<int>(), "Enable the computation of Hyperviscosity weights")
        ("hv_k", po::value<int>(), "Power of hyperviscosity")
        ("hv_gamma", po::value<double>(), "Scaling parameter on hyperviscosity")
        ("eps_c1", po::value<double>(), "Choose Epsilon as function of eps_c1 and eps_c2")
        ("eps_c2", po::value<double>(), "Choose Epsilon as function of eps_c1 and eps_c2")
        ("weight_method", po::value<int>(), "Set the method used to compute weights: 0 -> Direct Inversion of Ax=B; 1 -> ContourSVD") 
        ("ascii_weights,a", "Write weights in ASCII Matrix Market format (Default: off)") 
        ("weights,w", po::value<unsigned int>(), "Select the weights to compute. Argument should be an unsigned integer similar to the chmod comand. For example, the combination of weights X, Y, Z are 0x1 | 0x2 | 0x4 -> 0x7 == 7. Current choices are: X=0x1, Y=0x2, Z=0x4, LAPL=0x8, R=0x10, HV=0x20, LAMBDA=0x40, THETA=0x80, LSFC=0x100, XSFC=0x200, YSFC=0x400, ZSFC=0x800, ALT_XSFC=0x1000, ALT_YSFC=0x2000, ALT_ZSFC=0x4000, INTERP=0x8000")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    int debug = 0;
    if (vm.count("debug")) {
        debug = 1; 
    }

    int ascii_weights = 0;
    if (vm.count("ascii_weights")) {
        ascii_weights = 1;
    }

    string grid_filename; 
    if (vm.count("grid_filename")) {
        grid_filename = vm["grid_filename"].as<string>(); 
        cout << "Loading grid: " << grid_filename<< ".\n";
    } else {
        cout << "ERROR: grid_filename not specified\n";
        exit(-1); 
    }

    string partition_filename; 
    bool part_file_loaded = false;
    if (vm.count("partition_filename")) {
        partition_filename = vm["partition_filename"].as<string>(); 
        cout << "Loading partition file: " << partition_filename << ".\n";
        part_file_loaded = true; 
    } else {
        cout << "WARNING: partition_filename not specified, defaulting to all stencils for each processor\n";
    }

    int grid_dim = 3;
    if (vm.count("grid_dim")) {
        grid_dim = vm["grid_dim"].as<int>(); 
    }

    int grid_num_cols; 
    if (vm.count("grid_num_cols")) {
        grid_num_cols = vm["grid_num_cols"].as<int>(); 
        cout << "Number of expected columns: " << grid_num_cols << ".\n";
    } else {
        cout << "grid_num_cols was not set. Defaulting to grid_dim = " << grid_dim << ".\n";
        grid_num_cols = grid_dim;
    }

    int grid_size; 
    if (vm.count("grid_size")) {
        grid_size = vm["grid_size"].as<int>(); 
        cout << "Number of expected nodes: " << grid_size << ".\n";
    } else {
        cout << "ERROR: grid_size was not set.\n";
        exit(-2); 
    }

    // Select all derivative types (just in case)
    // Equivalent to: RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL | [...] | RBFFD::INTERP
    unsigned int weight_choices = (0x1 << RBFFD::NUM_DERIVATIVE_TYPES) - 1;
    if (vm.count("weights")) {
        weight_choices = vm["weights"].as<unsigned int>(); 
        cout << "Weight choices overridden to compute: " << weight_choices << ".\n";
    } else {
        cout << "Computing all weights: " << weight_choices << ", " << RBFFD::NUM_DERIVATIVE_TYPES <<  std::endl;
    }

    int stencil_size; 
    if (vm.count("stencil_size")) {
        stencil_size = vm["stencil_size"].as<int>(); 
        cout << "Number of nodes per stencil: " << stencil_size << ".\n";
    } else {
        cout << "ERROR: stencil_size was not set.\n";
        exit(-3); 
    }

    int use_hyperviscosity = 0; 
    int hv_k = -1;
    double hv_gamma = 0;
    if (vm.count("use_hyperviscosity")) {
        use_hyperviscosity = vm["use_hyperviscosity"].as<int>(); 
        cout << "Use Hyperviscosity: " << use_hyperviscosity<< ".\n";
        if (vm.count("hv_k")) {
            hv_k = vm["hv_k"].as<int>(); 
            cout << "HV_K : " << use_hyperviscosity<< ".\n";
        } else { 
            cout << "ERROR: hv_k required for use_hyperviscosity\n";
            exit(-3); 
        }	
        if (vm.count("hv_gamma")) {
            hv_gamma = vm["hv_gamma"].as<double>(); 
            cout << "hv_gamma: " << hv_gamma << ".\n";
        } else { 
            cout << "ERROR: hv_gamma required for use_hyperviscosity\n";
            exit(-3); 
        }
    }



    double eps_c1 = 1.;
    double eps_c2 = 0.;
    bool eps_c1_c2 = false; 
    if (vm.count("eps_c1")) {
        eps_c1 = vm["eps_c1"].as<double>(); 
        cout << "Epsilon c1: " << eps_c1 << ".\n";
        eps_c1_c2 = true;	
    } else {
        cout << "ERROR: eps_c1 was not set.\n";
        exit(-3); 
    }
    if (vm.count("eps_c2")) { 
        if (eps_c1_c2) {
            eps_c2 = vm["eps_c2"].as<double>(); 
            cout << "Epsilon c2: " << eps_c2 << ".\n";
        } else {
            cout << "ERROR: eps_c2 requires eps_c1\n"; 
            exit(-3); 
        }
    }

    int weight_method = 0; 
    if (vm.count("weight_method")) {
        weight_method = vm["weight_method"].as<int>(); 
        cout << "Weight Calculation Method: " << weight_method << ".\n";
    }


    MPI_Init(&argc, &argv);
    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    std::cout << "MPI Initialized: " << mpi_rank << ", " << mpi_size << std::endl;
    tm["initialize"]->stop();

    tm["loadDomain"]->start();
#if MASTER_LOAD_GRID
    Grid* grid;

    // Master process loads grid in case we we need collectives like MPI_Reduce
    if (!mpi_rank) { 
        tm["gridReader"]->start();
        grid = new GridReader(grid_filename, grid_num_cols, grid_size);
        grid->setMaxStencilSize(stencil_size);
        tm["gridReader"]->stop();

        tm["loadGrid"]->start();
        Grid::GridLoadErrType err = grid->loadFromFile(grid_filename);
        tm["loadGrid"]->stop();
        if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
            std::cout << "ERROR: Master process unable to read grid. Exiting..." << std::endl;
            exit(-1);
        }
    }
#endif 

    // Less memory efficient but will get the job done: 
    // Every proc can: 
    // 	read whole grid, 
    // 	read whole stencils, 
    // 	compute weights for subset of grid
    // 	write subset of weights to weights_*_...part<rank>_of_<size>
    // NOTE: no need to determine sets Q,O,R,B, etc. here. 


    // Similar to GridReader. Although it should not read in the stencils unless they end in a rank #. 

    Domain* subdomain; 
    subdomain = new METISDomain(mpi_rank, mpi_size, grid_dim, stencil_size); 
    Grid::GridLoadErrType d_load_err = subdomain->loadFromFile(); 
    if (d_load_err) { 
        std::cout << "ERROR: process " << mpi_rank << " could not load domain\n";
        exit(-1);
    }

    std::cout << "DECOMPOSED\n";

    tm["loadDomain"]->stop();

#if 0
        subdomain->printVerboseDependencyGraph();
        subdomain->printNodeList("All Centers Needed by This Process");

        printf("CHECKING STENCILS: ");
        for (int irbf = 0; irbf < (int)subdomain->getStencilsSize(); irbf++) {
            //  printf("Stencil[%d] = ", irbf);
            StencilType& s = subdomain->getStencil(irbf);
            if (irbf == s[0]) {
                //	printf("PASS\n");
                //    subdomain->printStencil(s, "S");
            } else {
                printf("FAIL on stencil %d\n", irbf);

                tm["total"]->stop();
                tm.printAll();
                tm.writeAllToFile();

                exit(EXIT_FAILURE);
            }
        }
        printf("OK\n");
#endif 

    tm["derSetup"]->start();
    RBFFD_VCL_OVERLAP* der = new RBFFD_VCL_OVERLAP(weight_choices, subdomain, grid_dim, mpi_rank);

    der->setUseHyperviscosity(use_hyperviscosity);
    // If both are zero assume we havent set anything
    if (eps_c1 || eps_c2) {
        der->setEpsilonByParameters(eps_c1, eps_c2);
    } else {
        der->setEpsilonByStencilSize();
    }
    if (hv_k != -1) {
        der->setHVScalars(hv_k, hv_gamma);
    }
    der->setWeightType((RBFFD::WeightType)weight_method);
    tm["derSetup"]->stop();

    tm["loadWeights"]->start();
    printf("Attempting to load Stencil Weights\n");

    // Try loading all the weight files
    der->overrideFileDetail(true);
    der->setAsciiWeights(ascii_weights);
    int load_err = der->loadAllWeightsFromFile();
    der->updateWeightsOnGPU(true); 
    tm["loadWeights"]->stop();



    tm["assembleTest"]->start(); 
    // By now our weights are loaded as Differentiation Matrices
    // 
    // Lets go ahead and use them to compute derivatives
    // Our local matrix is NxM with N < M (its under-determined)
    unsigned int N_part = subdomain->getStencilsSize();
    unsigned int M_part = subdomain->getNodeListSize();

    std::vector<double> u_cpu(M_part,1.);
    std::vector<double> u_x(M_part,1.);
    std::vector<double> u_y(M_part,1.);
    std::vector<double> u_z(M_part,1.);
    std::vector<double> u_l(M_part,1.);
    std::vector<double> u_new(M_part,1.);

    VCL_VEC_t u_gpu(M_part);
    VCL_VEC_t u_new_gpu(M_part);



    for (int i = 0; i < M_part; i++) {
        NodeType& node = subdomain->getNode(i); 
        //u[i] = sin((double)node[0]) + 2.*cos((double)node[1]) + exp(5 * (double)node[2]);
#if 1
        u_cpu[i] = sin((double)node[0]) + 2.*cos((double)node[1]) ;
#else 
        u_cpu[i] = 1;
#endif 
        u_x[i] = cos(node[0]); 
        u_y[i] = -2*sin(node[1]); 
        //u_z[i] = 5.*exp(5.*node[2]); 
        u_z[i] = 0.; 
        //u_l[i] = -sin(node[0]) - 2. * cos(node[1]) + 25. * exp(5.*node[2]); 
        u_l[i] = -sin(node[0]) - 2. * cos(node[1]) ;
    } 
    viennacl::copy(u_cpu, u_gpu);
    viennacl::copy(u_new, u_new_gpu);

    // If we are not synchronizing or if we are using bad mem locations -100000
    // should be enough to reflect in the norms
    std::vector<double> xderiv_cpu(M_part, -100000);	
    std::vector<double> yderiv_cpu(M_part, -100000);	
    std::vector<double> zderiv_cpu(M_part, -100000);	
    std::vector<double> lderiv_cpu(M_part, -100000);	

    VCL_VEC_t xderiv_gpu(M_part);	
    VCL_VEC_t yderiv_gpu(M_part);	
    VCL_VEC_t zderiv_gpu(M_part);	
    VCL_VEC_t lderiv_gpu(M_part);	



    //TODO: need to make apply work with synchronization
    double u_l2, u_l1, u_linf; 
    double x_l2, x_l1, x_linf;
    double y_l2, y_l1, y_linf; 
    double z_l2, z_l1, z_linf; 
    double l_l2, l_l1, l_linf; 
    double n_l2, n_l1, n_linf; 

    cout << "start computing derivative (on CPU)" << endl;
    tm["assembleTest"]->stop();

    // Need a class that will run: 
    //   a) SpMV on DM and solution vector
    //   b) alltoallv communication to synchronize output vector
    //   c) compute norms
    //   d) Need a class that inherits test and uses VCL, clSpMV formats
    std::cout << " Builing SpMVTest\n";

    // Run an SpMV test that uses MPI_Alltoallv for synchronization of the
    // solution vector
    SpMVTest *derTest = new SpMVTest(der, subdomain, mpi_rank, mpi_size ); 
    std::cout << " Built SpMVTest\n";

    derTest->disableTimers(); 
    // Prime the tubes: 
#if 0
    if (mpi_rank == 0) { 
        std::ofstream fout("U_gpu_before"); 
        fout << u_gpu; 
        fout.close();
        std::ofstream fout3("xderiv_gpu_before"); 
        fout3 << xderiv_gpu; 
        fout3.close();
        derTest->SpMV(RBFFD::X, u_gpu, xderiv_gpu);
        std::ofstream fout2("xderiv_gpu"); 
        fout2 << subdomain->QmB_size << "\n"; 
        fout2 << subdomain->B_size << "\n"; 
        fout2 << xderiv_gpu;
        fout2.close();
    exit(-1);
    }
#endif

    derTest->SpMV(RBFFD::X, u_gpu, xderiv_gpu);
    derTest->SpMV(RBFFD::Y, u_gpu, yderiv_gpu);
    derTest->SpMV(RBFFD::Z, u_gpu, zderiv_gpu);
    derTest->SpMV(RBFFD::LAPL, u_gpu, lderiv_gpu);
    derTest->enableTimers(); 

    // Prime AXPY. 
    u_new_gpu = 0.5*u_gpu + 1.*xderiv_gpu; 
 
    viennacl::copy(xderiv_cpu, xderiv_gpu);
    viennacl::copy(yderiv_cpu, yderiv_gpu);
    viennacl::copy(zderiv_cpu, zderiv_gpu);
    viennacl::copy(lderiv_cpu, lderiv_gpu);
    viennacl::copy(u_new, u_new_gpu);
    viennacl::copy(u_cpu, u_gpu);

    // Flush the queues before starting a benchmark
    viennacl::ocl::get_queue().finish();

    std::cout << " Entering loop: " << N_part << " rows \n";
    for (int i = 0; i < 100; i++) { 
        tm["iteration"]->start();

        // Verify that the CPU works
        // NOTE: we pass booleans at the end of the param list to indicate that
        // the function "u" is new (true) or same as previous calls (false). This
        // helps avoid overhead of passing "u" to the GPU.

        // NOTE: all SpMVs here assume internal comm to synchronize vector U in
        // order to compute the deriv. 

        tm["SpMV"]->start();
        derTest->SpMV(RBFFD::X, u_gpu, xderiv_gpu);
        tm["SpMV"]->stop();

        // We simulate an RK4 which has intermediate steps
        tm["SpMV"]->start();
        derTest->SpMV(RBFFD::Y, xderiv_gpu, yderiv_gpu);
        tm["SpMV"]->stop();

        tm["SpMV"]->start();
        derTest->SpMV(RBFFD::Z, yderiv_gpu, zderiv_gpu);
        tm["SpMV"]->stop();
 
        tm["SpMV"]->start();
        derTest->SpMV(RBFFD::LAPL, zderiv_gpu, lderiv_gpu);
        tm["SpMV"]->stop();

        tm["computeUpdate"]->start();
        // Ensure that the compiler is not trimming this loop by 
        // pretending to calc an updated solution
        // TODO: use views

        // Expanded these to ensure that they are asynchronous
        u_new_gpu = u_gpu; 
        u_new_gpu += (1./6.) * xderiv_gpu; 
        u_new_gpu += (2./6.) * yderiv_gpu; 
        u_new_gpu += (2./6.) * zderiv_gpu; 
        u_new_gpu += (1./6.) * lderiv_gpu; 

        tm["finish"]->start();
        viennacl::backend::finish();
        tm["finish"]->stop(); 
        tm["computeUpdate"]->stop();

        tm["iteration"]->stop();
    }

    // Compute the norms to make sure we have a complete picture. 
    viennacl::copy(u_gpu, u_cpu);
    viennacl::copy(xderiv_gpu, xderiv_cpu);
    viennacl::copy(yderiv_gpu, yderiv_cpu);
    viennacl::copy(zderiv_gpu, zderiv_cpu);
    viennacl::copy(lderiv_gpu, lderiv_cpu);
    viennacl::copy(u_new_gpu, u_new);

    viennacl::backend::finish();

    std::cout << "u: " << viennacl::linalg::norm_1(u_gpu) << std::endl;
    std::cout << "x: " << viennacl::linalg::norm_1(xderiv_gpu) << std::endl;
    std::cout << "y: " << viennacl::linalg::norm_1(yderiv_gpu) << std::endl;
    std::cout << "z: " << viennacl::linalg::norm_1(zderiv_gpu) << std::endl;
    std::cout << "l: " << viennacl::linalg::norm_1(lderiv_gpu) << std::endl;
    std::cout << "u_n: " << viennacl::linalg::norm_1(u_new_gpu) << std::endl;

    tm["computeNorms"]->start();
    u_l2 = l2norm( mpi_rank, u_cpu, 0, N_part);
    u_l1 = l1norm( mpi_rank, u_cpu, 0, N_part);
    u_linf = linfnorm( mpi_rank, u_cpu);
    tm["computeNorms"]->stop();

    tm["computeNorms"]->start();
    x_l2 = l2norm( mpi_rank, u_x, xderiv_cpu , 0 , N_part);
    x_l1 = l1norm( mpi_rank, u_x, xderiv_cpu, 0 , N_part );
    x_linf = linfnorm( mpi_rank, u_x, xderiv_cpu , 0 , N_part);
    tm["computeNorms"]->stop();


    tm["computeNorms"]->start();
    y_l2 = l2norm( mpi_rank, u_y, yderiv_cpu , 0 , N_part);
    y_l1 = l1norm( mpi_rank, u_y, yderiv_cpu , 0 , N_part);
    y_linf = linfnorm( mpi_rank, u_y, yderiv_cpu , 0 , N_part);
    tm["computeNorms"]->stop();


    tm["computeNorms"]->start();
    z_l2 = l2norm( mpi_rank, u_z, zderiv_cpu , 0 , N_part);
    z_l1 = l1norm( mpi_rank, u_z, zderiv_cpu , 0 , N_part);
    z_linf = linfnorm( mpi_rank, u_z, zderiv_cpu , 0 , N_part);
    tm["computeNorms"]->stop();

    tm["computeNorms"]->start();
    l_l2 = l2norm( mpi_rank, u_l, lderiv_cpu , 0 , N_part);
    l_l1 = l1norm( mpi_rank, u_l, lderiv_cpu , 0 , N_part);
    l_linf = linfnorm( mpi_rank, u_l, lderiv_cpu , 0 , N_part);
    tm["computeNorms"]->stop();

    tm["computeNorms"]->start();
    n_l2 = l2norm( mpi_rank, u_new, 0 , N_part);
    n_l1 = l1norm( mpi_rank, u_new, 0 , N_part);
    n_linf = linfnorm( mpi_rank, u_new, 0 , N_part);
    tm["computeNorms"]->stop();

    tm["cleanup"]->start();
    delete(derTest); 

    // We used MPI_reduce for norms, so only the master needs to print
    if (mpi_rank == 0) {
        std::cout << "U (L1, L2, Linf): " << u_l1 << ", " << u_l2 << ", " << u_linf << "\n"; 
        std::cout << "X (L1, L2, Linf): " << x_l1 << ", " << x_l2 << ", " << x_linf << "\n"; 
        std::cout << "Y (L1, L2, Linf): " << y_l1 << ", " << y_l2 << ", " << y_linf << "\n";
        std::cout << "Z (L1, L2, Linf): " << z_l1 << ", " << z_l2 << ", " << z_linf << "\n";
        std::cout << "Lapl (L1, L2, Linf): " << l_l1 << ", " << l_l2 << ", " << l_linf << "\n";
        std::cout << "U_new (L1, L2, Linf): " << n_l1 << ", " << n_l2 << ", " << n_linf << "\n";
    }
        

    std::cout << "Done checking apply on CPU and GPU\n";
#if MASTER_LOAD_GRID
    if (!mpi_rank) { 
        delete(grid);
        std::cout << "Deleted grid\n";
    }
#endif 
    delete(subdomain); 
    std::cout << "Deleted subdomain\n";
    tm["cleanup"]->stop();

    tm["total"]->stop();
    tm.printAll();

    std::cout << "----------------  END OF MAIN ------------------\n";
    char buf[256]; 
    sprintf(buf, "time_log.derivs.%d", mpi_rank); 
    tm.writeAllToFile(buf);
    tm.clear();
    MPI::Finalize();

    return 0;
}
//----------------------------------------------------------------------
