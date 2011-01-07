#ifndef __GRID_H__
#define __GRID_H__

#include <vector> 
#include <string>
#include "Vec3.h"

typedef Vec3 Node; 

class Grid 
{
	protected: 
		// Number of nodes this class is configured for. If this does not 
		// match the node_list.size() then we need to regenerate the node
		// set.
		unsigned int nb_nodes; 

		// Perturbation offsets for generated points
		double pert; 
	
		// A list of nodes that are N-dimensional and have no connectivity
		// NOTE: the Node type can determine its own coordinate system but
		// all of the nodes must share the same coordinate system. 
		std::vector<Node> node_list; 

		// True/False for every node: are you on the boundary?  
		std::vector<unsigned int> boundary_indices; 

		// If a node is on interior the normal is assumed to be 0-vector. 
		// Nodes on boundary have a non-zero vector. 
		std::vector<Vec3> boundary_normals; 

	public:
		Grid() : pert(0.), nb_nodes(0) {}
		Grid(unsigned int num_nodes) : pert(0), nb_nodes(num_nodes) {}
		Grid(std::vector<Node>& nodes) : pert(0), nb_nodes(nodes.size()), node_list(nodes) {} 

		virtual void generateGrid() =0; 

		// I/O of grid to/from file labeling the file with the specified iter number 
		virtual void writeToFile(std::string filename); 
		virtual void loadFromFile(std::string filename); 

		// Sort the nodes so boundary nodes are first in the lists
		virtual void sortNodes(); 

		std::string getFullName(std::string base_filename, int iter=0);

		// Select a random number [randf(-pert, pert)] for each node dimension and add
		// to randomly perturb nodes in space; store perturb_percent in pert to maintain
		// limited history of node perturbation
		virtual void perturbNodes(double perturb_amount);

}; 

#endif //__GRID_H__
