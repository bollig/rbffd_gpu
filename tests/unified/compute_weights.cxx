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
		("neighbor_method,w", po::value<int>(), "Set neighbor query method (0:LSH, 1:KDTree, 2:BruteForce)")
		("lsh_resolution,l", po::value<int>(), "Set the coarse grid resolution for LSH overlay (same for all dimensions)") 
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

	int neighbor_method = 0; 
	if (vm.count("neighbor_method")) {
		neighbor_method = vm["neighbor_method"].as<int>(); 
		cout << "Weight method is set to: "
			<< neighbor_method << ".\n";
	} else {
		cout << "neighbor_method was not set. Defaulting to 0.\n";
	}

	int lsh_resolution = 100;
	// Why is neighbor_method == 0 required?
	if ((neighbor_method == 0) && (vm.count("lsh_resolution"))) {
		lsh_resolution = vm["lsh_resolution"].as<int>(); 
		cout << "Number of coarse grid cells per dimension: " << lsh_resolution << ".\n";
	} else {
		cout << "lsh_resolution was not set. Defaulting to 100 per dimension.\n";
	}

	int ns_nx, ns_ny, ns_nz; 
	ns_nx = ns_ny = ns_nz = lsh_resolution; 

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
