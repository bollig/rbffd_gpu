#ifndef __GRID_H__
#define __GRID_H__

#include <vector> 
#include <string>
#include "Vec3.h"

typedef Vec3 NodeType; 

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
		std::vector<NodeType> node_list; 

		// True/False for every node: are you on the boundary?  
		std::vector<unsigned int> boundary_indices; 

		// If a node is on interior the normal is assumed to be 0-vector. 
		// Nodes on boundary have a non-zero vector. 
		std::vector<Vec3> boundary_normals; 

	public:
		Grid() : pert(0.), nb_nodes(0) {}
		Grid(unsigned int num_nodes) : pert(0), nb_nodes(num_nodes) {}
		Grid(std::vector<NodeType>& nodes) : pert(0), nb_nodes(nodes.size()), node_list(nodes) {} 

		virtual ~Grid(){ }

		virtual void generate(); 

		// Write data to disk generating the filename appropriately for the class (calls to writeToFile(std::string)) 
		virtual	void writeToFile(); 
		virtual void writeToFile(std::string filename); 

		// Load data from disk using the class generated filename and specified iter
			void loadFromFile(int iter); 
		virtual void loadFromFile(std::string filename); 

		// Sort the nodes so boundary nodes are first in the lists
		virtual void sortNodes(); 


		std::vector<NodeType>& 	   getNodeList() { return node_list; }
		std::vector<unsigned int>& getBoundaryIndices() { return boundary_indices; } 
		std::vector<Vec3>& 	   getBoundaryNormals() { return boundary_normals; }


		// Convert a basic filename like "output_file" to something more descriptive
		// and appropriate to the grid like "output_file_120nodes_final.ascii"
		// or "output_file_356_nodes_final.bin"
			std::string getFilename(std::string base_filename, int iter=0);

		// Get a filename appropriate for output from this class
		// same as getFilename(std::string, int) however it uses 
		// the class's internal name instead of a user specified string. 
			std::string getFilename(int iter=0); 

		// Get a string that gives some detail about the grid (used by expandFilename(...))
		// NOTE: replace spaces with '_'
		virtual std::string getFileDetailString(); 

		virtual std::string className() { return "grid"; }


		// Select a random number [randf(-pert, pert)] for each node dimension and add
		// to randomly perturb nodes in space; store perturb_percent in pert to maintain
		// limited history of node perturbation
		virtual void perturbNodes(double perturb_amount);


		// NOTE: this grid does not have details of node connectivity. We could add this
		// in a subclass if we wanted like a TriangularGrid or RectilinearGrid. 
}; 

#endif //__GRID_H__
