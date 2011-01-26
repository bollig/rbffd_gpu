#ifndef __GRID_H__
#define __GRID_H__

#include <vector> 
#include <string>
#include "Vec3.h"

#include "stencil_generator.h" 
#include "common_typedefs.h"

class Grid 
{
	public: 
		double xmin, xmax; 
		double ymin, ymax; 
		double zmin, zmax; 

    protected: 
		// 0 = Debug output off; 1 = Verbose output and write intermediate files
    		int DEBUG;

		// Number of nodes this class is configured for. If this does not 
		// match the node_list.size() then we need to regenerate the node
		// set.
		size_t nb_nodes; 

		// Perturbation offsets for generated points
		double pert; 

		// Should we sort our nodes when generating so boundary nodes appear first in the list?
		// 0 = NO, non-zero = YES 
		int boundary_nodes_first;
	
		// A list of nodes that are N-dimensional and have no connectivity
		// NOTE: the Node type can determine its own coordinate system but
		// all of the nodes must share the same coordinate system. 
		std::vector<NodeType> node_list; 

		// True/False for every node: are you on the boundary?  
		std::vector<size_t> boundary_indices; 

		// If a node is on interior the normal is assumed to be 0-vector. 
		// Nodes on boundary have a non-zero vector. 
		std::vector<Vec3> boundary_normals; 

		// These are the stencils connecting our nodes in node_list.
		// These are not guaranteed to be generated and are usually computed by calling
		// generateStencils(StencilGenerator*) where the StencilGenerator can use any method
		// to determine neighbors that will be member to a stencil. For example we could have
		// separate generators for nearest-neighbor ball queries (all neighbors within max radius)
		// and one for a maximum number of nearest neighbors, and one that generates only FD stencils
		// by taking nodes due north/east/south/west (assuming a regular grid). 
		std::vector<StencilType> stencil_map; 

		// These are the average radii for each stencil. That is, the average distance between the 
		// stencil center and its connected neighbors.
		std::vector<double> avg_stencil_radii; 
	
	public:
		Grid() : pert(0.), nb_nodes(0), boundary_nodes_first(false), DEBUG(0) {}
		Grid(size_t num_nodes) : pert(0), nb_nodes(num_nodes), boundary_nodes_first(false), DEBUG(0) {}
		Grid(std::vector<NodeType>& nodes) : pert(0), nb_nodes(nodes.size()), boundary_nodes_first(false), DEBUG(0), node_list(nodes) {} 

		virtual ~Grid(){ }

		// PROBABLY THE MOST IMPORTANT ROUTINE: generates the nodes in node_list
		virtual void generate(); 

		// SECOND MOST IMPORTANT ROUTINE: generates stencil connectivity of node_list stored in stencil_map
		virtual void generateStencils(StencilGenerator* stencil_generator);	



		// Write data to disk generating the filename appropriately for the class (calls to writeToFile(std::string)) 
		virtual	void writeToFile(); 
		virtual void writeToFile(std::string filename); 

		// Load data from disk using the class generated filename and specified iter
			int loadFromFile(int iter=0); 
		virtual int loadFromFile(std::string filename); 

		// Sort the nodes so boundary nodes are first in the lists
		virtual void sortNodes(); 

		// Enable or disable node sorting (if a grid generator obeys the order)
		// so boundary nodes appear first in the node_list
		void setSortBoundaryNodes(int sort_boundary_first) { this->boundary_nodes_first = sort_boundary_first; }

		std::vector<NodeType>& 	    getNodeList() 	{ return node_list; }
		std::vector<size_t>&  getBoundaryIndices() { return boundary_indices; } 
		std::vector<Vec3>&          getBoundaryNormals() { return boundary_normals; }
		std::vector<StencilType>&   getStencils() 	{ return stencil_map; }
		StencilType&                getStencil(int indx) { return stencil_map[indx]; }
		std::vector<double>& 	    getStencilRadii(){ return avg_stencil_radii; }
		double 		                getStencilRadius(int indx) { return avg_stencil_radii[indx]; }



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


		// Set DEBUG to 0 or 1
		void setDebug(int debug_) { DEBUG = debug_; }

		// NOTE: this grid does not have details of node connectivity. We could add this
		// in a subclass if we wanted like a TriangularGrid or RectilinearGrid. 
}; 

#endif //__GRID_H__
