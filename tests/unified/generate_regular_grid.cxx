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
		("output_filename,o", po::value<string>(), "Grid filename (flat file, tab delimited columns). If none specified will default to \"input_grid.ascii\".")
		("nx,x", po::value<int>(), "Grid resolution in the X direction")
		("ny,y", po::value<int>(), "Grid resolution in the Y direction")
		("nz,z", po::value<int>(), "Grid resolution in the Z direction")
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
	if (vm.count("output_filename")) {
		output_filename = vm["output_filename"].as<string>(); 
	} else {
		output_filename = "input_grid.ascii";
	}
	cout << "Output grid: " << output_filename << ".\n";


	int nx = 1; 
	int ny = 1; 
	int nz = 1; 
	int stencil_size = 5; 
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

	int dim = 3;

	double minX = -1.;
	double maxX = 1.;
	double minY = -1.;
	double maxY = 1.;
	double minZ = -1.;
	double maxZ = 1.;

	Grid* grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
	grid->setSortBoundaryNodes(true); 
	grid->generate();
	int grid_size = grid->getNodeListSize();
	grid->generateStencils(stencil_size, Grid::ST_KDTREE);   // nearest nb_points
	grid->writeToFile(output_filename); 

	{
		// Assemble a DIRECTED graph that is the spadjacency_list our stencils
		//		typedef adjacency_list <boost::vecS, boost::setS, boost::bidirectionalS> Graph;

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

#if 0 
	// Perhaps we can visualize with graphviz? 
		std::ofstream gvout("undirected_graph.graphviz"); 
		write_graphviz(gvout, g);
		gvout.close();
		std::cout << "Wrote the graphviz file: undirected_graph.graphviz" << std::endl;
#endif 

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
		std::cout << "Wrote the METIS graph file: metis_stencils.graph" << std::endl;
	}



	delete(grid);

	cout.flush();

	exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
