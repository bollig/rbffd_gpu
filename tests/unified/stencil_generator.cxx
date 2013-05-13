#include <stdlib.h>
#include <sstream>
#include <map>

#include "grids/grid_reader.h"

#include <boost/program_options.hpp>
#include <boost/graph/adjacency_list.hpp>
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
		("weight_method,w", po::value<int>(), "Set weight method (0:LSH, 1:KDTree, 2:BruteForce)")
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

	int weight_method = 0; 
	if (vm.count("weight_method")) {
		weight_method = vm["weight_method"].as<int>(); 
		cout << "Weight method is set to: "
			<< weight_method << ".\n";
	} else {
		cout << "weight_method was not set. Defaulting to 0.\n";
	}

	int lsh_resolution = 100;
	if ((weight_method == 0) && (vm.count("lsh_resolution"))) {
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
		grid->writeToFile();
		tm["writeGrid"]->stop();
	}
	if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
		std::cout << "************** Generating Stencils **************\n"; 
		tm["stencils"]->start();
		switch (weight_method) {
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
		grid->writeToFile();
		tm["writeStencils"]->stop();
	}

	{
		// Assemble a DIRECTED graph that is the spadjacency_list our stencils
//		typedef adjacency_list <boost::vecS, boost::setS, boost::bidirectionalS> Graph;

#if 0
		typedef boost::undirected_graph<> Graph;
#else
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

		std::ofstream gvout("undirected_graph.graphviz"); 
		write_graphviz(gvout, g);
		gvout.close();

		// Dump the graph file for METIS

		std::ostringstream grouts;
		// First the number of vertices, edges
		// Then all connections for each node (assumes at least one connection per node
		unsigned int num_edges = 0;
	      	for (int i = 0; i < boost::num_vertices(g); i++) {
			boost::graph_traits<Graph>::adjacency_iterator e, e_end;
			boost::graph_traits<Graph>::vertex_descriptor 
				s = boost::vertex(i, g);
			//cout << "the edges incident to v: " << i+1 << "\n";
			std::set<unsigned int> unique_verts; 
			for (tie(e, e_end) = boost::adjacent_vertices(s, g); e != e_end; ++e) {
				unique_verts.insert(get_vertex_index(*e,g)); 
			}
			for (std::set<unsigned int>::iterator it = unique_verts.begin(); it != unique_verts.end(); it++) {
				// Add 1 to the index to make sure we are indexing from 1 in metis
			//	std::cout << (*it) + 1 << "\n";
				grouts << (*it) + 1 << " ";
				num_edges++; 
			}
			grouts << "\n";
		}	

		std::ofstream grout("metis_stencils.graph"); 
		// We can divide num_edges by 2 to get correct count because edges are symmetric
		grout << boost::num_vertices(g) << " " << num_edges / 2 << "\n"; 
		grout << grouts.str();
		
		grout.close();

#if 0
		// Now make the DIRECTED adjacency graph UNDIRECTED. 
		UGraph udirMat = g; 

		std::ofstream fout("undirected_graph.graphviz"); 
		write_graphviz(fout, udirMat);
		fout.close();
#endif
	}


	delete(grid);
	std::cout << "Deleted grid\n";

	tm["total"]->stop();
	tm.printAll();


	std::cout << "----------------  END OF MAIN ------------------\n";
	tm.writeAllToFile("time_log.stencils");
	tm.clear();

	return 0;
}
//----------------------------------------------------------------------
