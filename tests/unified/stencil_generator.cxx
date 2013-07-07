#include <stdlib.h>
#include <sstream>
#include <map>

#include "grids/grid_reader.h"

#include <boost/program_options.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/graphviz.hpp>

#include "timer_eb.h"

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
    tm["METIS_graph"] = new Timer("[Main] Assemble and write METIS graph to file"); 
    tm["assemble_METIS_graph"] = new Timer("[Main] Assemble METIS graph"); 
    tm["output_METIS_graph"] = new Timer("[Main] write METIS graph to file"); 

	tm["total"]->start();

	//-----------------
	tm["settings"]->start();

	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("grid_filename,g", po::value<string>(), "Grid filename (flat file, tab delimited columns). Required.")
		("grid_num_cols,c", po::value<int>(), "Number of columns to expect in the grid file (X,Y,Z first)")
		("grid_size,N", po::value<int>(), "Number of nodes to expect in the grid file") 
		("stencil_size,n", po::value<int>(), "Number of nodes per stencil (assume all stencils are the same size)")
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

	string grid_filename; 
	if (vm.count("grid_filename")) {
		grid_filename = vm["grid_filename"].as<string>(); 
		cout << "Loading grid: " << grid_filename<< ".\n";
	} else {
		cout << "ERROR: grid_filename not specified\n";
		exit(-1); 
	}

	int grid_num_cols; 
	if (vm.count("grid_num_cols")) {
		grid_num_cols = vm["grid_num_cols"].as<int>(); 
		cout << "Number of expected columns: " << grid_num_cols << ".\n";
	} else {
		cout << "grid_num_cols was not set. Defaulting to 3.\n";
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


	tm["gridReader"]->start();
	Grid* grid = new GridReader(grid_filename, grid_num_cols, grid_size);
	grid->setMaxStencilSize(stencil_size);
	tm["gridReader"]->stop();

	tm["loadGrid"]->start();
	Grid::GridLoadErrType err = grid->loadFromFile();
	tm["loadGrid"]->stop();
	if (err == Grid::NO_GRID_FILES)
	{
		std::cout << "************** Generating new Grid **************\n";
		grid->setSortBoundaryNodes(true);
		tm["grid"]->start();
		grid->generate();
		tm["grid"]->stop();
		tm["writeGrid"]->start();
		grid->writeToFile("input_grid.ascii");
		tm["writeGrid"]->stop();
	}
	if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
		std::cout << "************** Generating Stencils **************\n"; 
		tm["stencils"]->start();
		switch (neighbor_method) {
			case 2: 
				grid->generateStencils(Grid::ST_BRUTE_FORCE);
				break; 
			case 1: 
				grid->generateStencils(Grid::ST_KDTREE);
				break; 
			case 0: 
			default: 
				grid->setNSHashDims(ns_nx, ns_ny, ns_nz);
				grid->generateStencils(Grid::ST_HASH);
				break; 
		}
		tm["stencils"]->stop();
		tm["writeStencils"]->start();
		grid->writeToFile("input_grid.ascii");
		tm["writeStencils"]->stop();
	}

    tm["METIS_graph"]->start(); 
    std::cout << "Generating Adjacency Graph for METIS\n";
	{
        tm["assemble_METIS_graph"]->start(); 
        // Assemble a DIRECTED graph that is the spadjacency_list our stencils
        //		typedef adjacency_list <boost::vecS, boost::setS, boost::bidirectionalS> Graph;

#if 0
        // TODO: we might need this for backwards compat.
#if BOOST_VERSION > 104900
        typedef boost::undirected_graph<> Graph;
#else
        // For Older versions of boost.
        typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS> Graph;
#endif
        Graph g;
        Graph::vertex_descriptor vds[grid_size]; 

        // Since the graph uses setS we cant use int indices for
        // vertices. These will be our ref indices
        for (int i = 0; i < grid_size; i++) {
            vds[i] = g.add_vertex();
        }

        for (int i = 0; i < grid_size; i++) { 
            StencilType& s = grid->getStencil(i); 
            // Start with index 1 to neglect the connection to
            // itself (the 1 on diag of matrix). Its assumed by metis. 
            for (int j = 1; j < stencil_size; j++) {
                if (s[j] < grid_size) 
                {
                    g.add_edge(vds[i], vds[s[j]]);
                }
            }
        }
#else
        typedef boost::adjacency_matrix<boost::undirectedS> UGraph;
        UGraph ug(grid->getNodeListSize());

        for (int i = 0; i < grid_size; i++) { 
            StencilType& s = grid->getStencil(i); 
            // Start with index 1 to neglect the connection to
            // itself (the 1 on diag of matrix). Its assumed by metis. 
            for (int j = 1; j < stencil_size; j++) {
                if (s[j] < grid_size) 
                {
                    boost::add_edge(i, s[j], ug);
                }
            }
        }
#endif  
        tm["assemble_METIS_graph"]->stop(); 
        std::cout << "Assembled.\n";

        tm["output_METIS_graph"]->start(); 
        // Dump the graph file for METIS
        
        // Assemble the graph inplace in temp file:
        char* outname = "metis_stencils.graph";
        std::ofstream grout(outname);

        // Assume we have an adjacency graph with all edges present as described
        // by our stencils. Now we need to get the edges not present. To do
        // this, we first iterate through all nodes and get any edges that
        // connect to those nodes. Then, 
        //  
        //
        // First the number of vertices, edges
        // Then all connections for each node (assumes at least one connection per node
        unsigned int num_edges = 0;

        grout << boost::num_vertices(ug) << " " << boost::num_edges(ug) << "\n"; 
        for (int i = 0; i < boost::num_vertices(ug); i++) {
            boost::graph_traits<UGraph>::adjacency_iterator e, e_end;
            boost::graph_traits<UGraph>::vertex_descriptor s = boost::vertex(i, ug);
            // With the adjacency_matrix we assume 
            // that edges are bidirectional and unique
            for (tie(e, e_end) = boost::adjacent_vertices(s, ug); e != e_end; ++e) {
                grout << *e + 1 << " ";
                num_edges++; 
            }
            grout << "\n";
        }

        if (num_edges/2 != boost::num_edges(ug)) {
            std::cout << "ERROR: num_edges does not match!\n";
            exit(-1); 
        }    

        // This will delete the file:
        grout.close();

        tm["output_METIS_graph"]->stop(); 
        std::cout << "Wrote the METIS graph file: metis_stencils.graph" << std::endl;
    }
    tm["METIS_graph"]->stop(); 


    delete(grid);
	std::cout << "Deleted grid\n";

	tm["total"]->stop();
	tm.printAll();


	std::cout << "----------------  END OF MAIN ------------------\n";
    char buf[256]; 
    sprintf(buf, "time_log.stencils"); 
	tm.writeAllToFile(buf);
	tm.clear();

	return 0;
}
//----------------------------------------------------------------------
