/*
 *  
 *  load full grid
 *  if metis part file not exists
 *  	load full stencils
 *  else 
 *  	load metis part file
 *  	for each metis line
 *  		if part equals mpi_rank
 *  			read stencil
 *  		else 
 *  			discard stencil
 *  for each stencil
 *  	compute weight
 *  
 *  for each stencil
 *  	write weights
 **/

#include <mpi.h> 
#include "grids/grid_reader.h"
#include "grids/domain.h"
#include "grids/metis_domain.h"

#ifdef USE_VTU_WRITER
#include "utils/io/vtu_domain_writer.h"
#endif 

#include "rbffd/rbffd.h"

#include <boost/program_options.hpp>

#include "timer_eb.h"

#include <stdlib.h>
#include <sstream>
#include <map>
#include <iostream> 


using namespace std;
using namespace EB;
using namespace boost; 

namespace po = boost::program_options;

//----------------------------------------------------------------------
//NOTE: EVERYTHING BELOW IN MAIN WAS COPIED FROM heat_regulargrid_2d/main.cpp
//----------------------------------------------------------------------

int main(int argc, char** argv) {

	TimerList tm;

	tm["total"] = new Timer("[Main] Total runtime for this proc");
	tm["grid"] = new Timer("[Main] Grid generation");
	tm["gridReader"] = new Timer("[Main] Grid Reader Load File From Disk");
	tm["loadGrid"] = new Timer("[Main] Load Grid (and Stencils) from Disk");
	tm["writeGrid"] = new Timer("[Main] Write Grid (and Stencils) to Disk");
	tm["stencils"] = new Timer("[Main] Stencil generation");
	tm["writeStencils"] = new Timer("[Main] Write Stencils to Disk");
	tm["settings"] = new Timer("[Main] Load settings");
	tm["derSetup"] = new Timer("[Main] Setup RBFFD class");
	tm["weights"] = new Timer("[Main] Compute Weights");
	tm["writeWeights"] = new Timer("[Main] Output weights to file");

	tm["total"]->start();

	//-----------------
	tm["settings"]->start();

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


#if 1
	MPI::Init(argc, argv);
	int mpi_rank = MPI::COMM_WORLD.Get_rank();
	int mpi_size = MPI::COMM_WORLD.Get_size();
#else 
	int mpi_rank = 0; 
	int mpi_size = 1;
#endif 

	tm["gridReader"]->start();
	Grid* grid = new GridReader(grid_filename, grid_num_cols, grid_size);
	grid->setMaxStencilSize(stencil_size);
	tm["gridReader"]->stop();

	tm["loadGrid"]->start();
	Grid::GridLoadErrType err = grid->loadFromFile(grid_filename);
	tm["loadGrid"]->stop();
	if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
		std::cout << "ERROR: unable to read grid. Exiting..." << std::endl;
		exit(-1);
	}

	// Less memory efficient but will get the job done: 
	// Every proc can: 
	// 	read whole grid, 
	// 	read whole stencils, 
	// 	compute weights for subset of grid
	// 	write subset of weights to weights_*_...part<rank>_of_<size>
	// NOTE: no need to determine sets Q,O,R,B, etc. here. 


	// Similar to GridReader. Although it should not read in the stencils unless they end in a rank #. 

	Domain* subdomain; 
	subdomain = new METISDomain(mpi_rank, mpi_size, grid, grid_dim, partition_filename, part_file_loaded); 
	subdomain->writeToFile(); 

#ifdef USE_VTU_WRITER
	VtuDomainWriter* vtu = new VtuDomainWriter(subdomain, mpi_rank, mpi_size);
	std::cout << "DECOMPOSED\n";
#endif 

#if 0
	if (debug) {
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
	}
#endif 

	tm["derSetup"]->start();
    std::cout << "Computing Weights\n";
	RBFFD* der = new RBFFD(weight_choices, subdomain, grid_dim, mpi_rank);

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
	der->setComputeConditionNumber(true);
	tm["derSetup"]->stop();

	printf("start computing weights\n");
	tm["weights"]->start();
	// NOTE: good test for Direct vs Contour
	// Grid 11x11, vareps=0.05; Look at stencil 12. SHould have -100, 25,
	// 25, 25, 25 (i.e., -4,1,1,1,1) not sure why scaling is off.
	der->computeAllWeightsForAllStencils();
	tm["weights"]->stop();

	cout << "end computing weights" << endl;

	tm["writeWeights"]->start();
	der->overrideFileDetail(true);
	der->setAsciiWeights(ascii_weights);
	der->writeAllWeightsToFile();
	cout << "end write weights to file" << endl;
	tm["writeWeights"]->stop();

#if 1
	delete(grid);
	std::cout << "Deleted grid\n";
	delete(subdomain); 
	std::cout << "Deleted subdomain\n";
#endif 

	tm["total"]->stop();
	tm.printAll();


	std::cout << "----------------  END OF MAIN ------------------\n";
    char buf[256]; 
    sprintf(buf, "time_log.weights.%d", mpi_rank); 
	tm.writeAllToFile(buf);

	tm.clear();
	MPI::Finalize();

	return 0;
}
//----------------------------------------------------------------------
