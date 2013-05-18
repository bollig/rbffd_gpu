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

#include <stdlib.h>
#include <sstream>
#include <map>
#include <iostream> 

#include "grids/grid_reader.h"
#include "grids/domain.h"
#include "grids/metis_domain.h"
#include "rbffd/rbffd.h"

#include <boost/program_options.hpp>

#include "timer_eb.h"

#include <mpi.h> 

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
		("grid_num_cols,c", po::value<int>(), "Number of columns to expect in the grid file (X,Y,Z first)")
		("grid_size,N", po::value<int>(), "Number of nodes to expect in the grid file") 
		("stencil_size,n", po::value<int>(), "Number of nodes per stencil (assume all stencils are the same size)")
		("partition_filename,p", po::value<string>(), "METIS Output Partition Filename (*.part.<P-processors>)")
		("use_hyperviscosity", po::value<int>(), "Enable the computation of Hyperviscosity weights")
		("hv_k", po::value<int>(), "Power of hyperviscosity")
		("hv_gamma", po::value<double>(), "Scaling parameter on hyperviscosity")
		("eps_c1", po::value<double>(), "Choose Epsilon as function of eps_c1 and eps_c2")
		("eps_c2", po::value<double>(), "Choose Epsilon as function of eps_c1 and eps_c2")
		("weight_method", po::value<int>(), "Set the method used to compute weights: 0 -> Direct Inversion of Ax=B; 1 -> ContourSVD") 
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

	string grid_filename; 
	if (vm.count("grid_filename")) {
		grid_filename = vm["grid_filename"].as<string>(); 
		cout << "Loading grid: " << grid_filename<< ".\n";
	} else {
		cout << "ERROR: grid_filename not specified\n";
		exit(-1); 
	}

	string partition_filename; 
	if (vm.count("partition_filename")) {
		partition_filename = vm["partition_filename"].as<string>(); 
		cout << "Loading partition file: " << partition_filename << ".\n";
	} else {
		cout << "WARNING: partition_filename not specified, defaulting to 'metis_stencils.graph'\n";
		partition_filename = "metis_stencils.graph"; 
	}

	int grid_num_cols; 
	if (vm.count("grid_num_cols")) {
		grid_num_cols = vm["grid_num_cols"].as<int>(); 
		cout << "Number of expected columns: " << grid_num_cols << ".\n";
	} else {
		cout << "grid_num_cols was not set. Defaulting to 3.\n";
		grid_num_cols = 3;
	}

	int grid_size; 
	if (vm.count("grid_size")) {
		grid_size = vm["grid_size"].as<int>(); 
		cout << "Number of expected nodes: " << grid_size << ".\n";
	} else {
		cout << "ERROR: grid_size was not set.\n";
		exit(-2); 
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
	subdomain = new METISDomain(mpi_rank, mpi_size, grid, partition_filename); 
	subdomain->writeToFile(); 
	std::cout << "DECOMPOSED\n";
	
#if 1
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
    RBFFD* der = new RBFFD(RBFFD::LAMBDA | RBFFD::THETA | RBFFD::HV, subdomain, 3, mpi_rank);

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
	tm.writeAllToFile("time_log.stencils");
	tm.clear();
	MPI::Finalize();

	return 0;
}
//----------------------------------------------------------------------
