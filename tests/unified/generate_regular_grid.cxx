#include <stdlib.h>
#include <sstream>
#include <map>

#include "grids/regulargrid.h"

#include "timer_eb.h"

#include <boost/program_options.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/graphviz.hpp>


using namespace std;
using namespace EB;
using namespace boost; 

namespace po = boost::program_options;

int main(int argc, char** argv) {
	TimerList tm;

	tm["total"] = new Timer("[Main] Total runtime for this proc");
	tm["grid"] = new Timer("[Main] Grid generation");
	tm["settings"] = new Timer("[Main] Load settings");

	tm["total"]->start();

	tm["settings"]->start();
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("output_filename,o", po::value<string>(), "Grid filename (flat file, tab delimited columns). If none specified will default to \"regulargrid_{nx}x_{ny}y_{nz}z.ascii\".")
		("nx,x", po::value<int>(), "Grid resolution in the X direction")
		("ny,y", po::value<int>(), "Grid resolution in the Y direction")
		("nz,z", po::value<int>(), "Grid resolution in the Z direction")
		("hash_resolution,l", po::value<int>(), "Hashing grid overlay resolution in each direction")
		("stencil_size,s", po::value<int>(), "Stencil size")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
		cout << desc << "\n";
		return 1;
	}

	string output_filename; 
	bool output_filename_specified = false; 
	if (vm.count("output_filename")) {
		output_filename = vm["output_filename"].as<string>(); 
		output_filename_specified = true; 
	}

	int nx = 1; 
	int ny = 1; 
	int nz = 1; 
	int stencil_size = 5; 
	int hash_resolution = 100; 
	if (vm.count("nx")) {
		nx = vm["nx"].as<int>() ;
	} 
	if (vm.count("ny")) {
		ny = vm["ny"].as<int>() ;
	} 
	if (vm.count("nz")) {
		nz = vm["nz"].as<int>() ;
	} 

	if (vm.count("stencil_size")) {
		stencil_size = vm["stencil_size"].as<int>() ;
	}

	if (vm.count("hash_resolution")) {
		hash_resolution = vm["hash_resolution"].as<int>() ;
	}


	int dim = 3;

	double minX = -1.;
	double maxX = 1.;
	double minY = -1.;
	double maxY = 1.;
	double minZ = -1.;
	double maxZ = 1.;
	tm["settings"]->stop(); 

	tm["grid"]->start();
	Grid* grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
	grid->setSortBoundaryNodes(true); 
	grid->generate();
	int grid_size = grid->getNodeListSize();
	grid->setNSHashDims(hash_resolution, hash_resolution, hash_resolution);
	grid->generateStencils(stencil_size, Grid::ST_HASH);   // nearest nb_points
	if (output_filename_specified) { 
		grid->writeToFile(output_filename); 
	} else { 
		grid->writeToFile(); 
	}
	tm["grid"]->stop();

	delete(grid);

	tm["total"]->stop();
	tm.printAll();
	cout.flush();

	std::cout << "----------------  END OF MAIN ------------------\n";
    char buf[256]; 
    sprintf(buf, "time_log.generate"); 
	tm.writeAllToFile(buf);
	tm.clear();

	exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
