#ifndef __GRID_H__
#define __GRID_H__

#include <vector> 
#include <string>
#include "Vec3.h"
#include "KDTree.h"

#include "common_typedefs.h"

class Grid 
{
    public: 
        double xmin, xmax; 
        double ymin, ymax; 
        double zmin, zmax; 

    public: 
        // We allow multiple types of stencil generators (for backwards compat)
        enum st_generator_t {ST_BRUTE_FORCE=0, ST_KDTREE, ST_HASH};

    protected: 
        // 0 = Debug output off; 1 = Verbose output and write intermediate
        // files
        int DEBUG;

        // Number of nodes this class is configured for. If this does not 
        // match the node_list.size() then we need to regenerate the node
        // set.
        size_t nb_nodes; 

        // Maximum number of nodes to allow in a stencil (can be equal to
        // nb_nodes if we want to define a global function across a stencil)
        size_t max_st_size; 
        
        // Maximum radius allowed when searching stencil nodes
        double max_st_radius; 

        // Perturbation offsets for generated points
        double pert; 

        // Should we sort our nodes when generating so boundary nodes appear
        // first in the list?  0 = NO, non-zero = YES 
        int boundary_nodes_first;

        // A list of nodes that are N-dimensional and have no connectivity
        // NOTE: the Node type can determine its own coordinate system but all
        // of the nodes must share the same coordinate system. 
        std::vector<NodeType> node_list; 

        // True/False for every node: are you on the boundary?  
        std::vector<size_t> boundary_indices; 

        // If a node is on interior the normal is assumed to be 0-vector. 
        // Nodes on boundary have a non-zero vector. 
        std::vector<Vec3> boundary_normals; 

        // These are the stencils connecting our nodes in node_list.  These are
        // not guaranteed to be generated and are usually computed by calling
        // generateStencils(st_generator_t type)
        std::vector<StencilType> stencil_map; 

        // These are the average radii for each stencil. That is, the average
        // distance between the stencil center and its connected neighbors.
        std::vector<double> avg_stencil_radii; 

        // The shortest distance for each stencil
        std::vector<double> min_stencil_radii; 

        // The longest distance for each stencil
        std::vector<double> max_stencil_radii; 



        // An internal KDTree representation of the nodes. Useful for CVT
        // sampling (if updated) and also for neighbor queries
        KDTree* node_list_kdtree; 

    public:
        Grid() : 
            xmin(0.), xmax(1.), 
            ymin(0.), ymax(1.), 
            zmin(0.), zmax(0.),
            max_st_size(0), max_st_radius(DBL_MAX),
            pert(0.), nb_nodes(0),
            node_list_kdtree(NULL),
            boundary_nodes_first(false), DEBUG(0)
            {}
        Grid(size_t num_nodes) : 
            xmin(0.), xmax(1.), 
            ymin(0.), ymax(1.), 
            zmin(0.), zmax(0.),
            max_st_size(0), max_st_radius(DBL_MAX), 
            pert(0), nb_nodes(num_nodes), 
            node_list_kdtree(NULL),
            boundary_nodes_first(false), DEBUG(0) 
            {}
        Grid(std::vector<NodeType>& nodes) : 
            xmin(0.), xmax(1.), 
            ymin(0.), ymax(1.), 
            zmin(0.), zmax(0.),
            max_st_size(0), max_st_radius(DBL_MAX), 
            pert(0), nb_nodes(nodes.size()), 
            node_list_kdtree(NULL), 
            boundary_nodes_first(false), DEBUG(0), node_list(nodes) 
            {} 

        virtual ~Grid(){ }

        // PROBABLY THE MOST IMPORTANT ROUTINE: generates the nodes in node_list
        virtual void generate(); 

        // SECOND MOST IMPORTANT ROUTINE: generates stencil connectivity of node_list stored in stencil_map
//        virtual void generateStencils(StencilGenerator* stencil_generator);	
        void generateStencils(st_generator_t generator_choice = Grid::ST_BRUTE_FORCE);
        void generateStencils(size_t st_max_size, st_generator_t generator_choice = Grid::ST_BRUTE_FORCE);
        void generateStencilsBruteForce(); 
        void generateStencilsKDTree(); 
        void generateStencilsHash();

        void computeStencilRadii();



        // Write data to disk generating the filename appropriately for the class (calls to writeToFile(std::string)) 
        virtual	void writeToFile(); 
        virtual void writeToFile(std::string filename); 
        void writeBoundaryToFile(std::string filename); 
        void writeNormalsToFile(std::string filename); 
        void writeAvgRadiiToFile(std::string filename); 
        void writeStencilsToFile(std::string filename); 
        virtual void writeExtraToFile(std::string filename); 

        virtual void printNodeList(std::string label); 
        virtual void printBoundaryIndices(std::string label);  

        // Load data from disk using the class generated filename and specified iter
        int loadFromFile(int iter=-1); 
        virtual int loadFromFile(std::string filename); 
        int loadBoundaryFromFile(std::string grid_filename); 
        int loadNormalsFromFile(std::string grid_filename); 
        int loadAvgRadiiFromFile(std::string grid_filename); 
        int loadStencilsFromFile(std::string grid_filename); 
        virtual int loadExtraFromFile(std::string grid_filename); 

        // Sort the nodes so boundary nodes are first in the lists
        virtual void sortNodes(); 

        // Enable or disable node sorting (if a grid generator obeys the order)
        // so boundary nodes appear first in the node_list
        void setSortBoundaryNodes(int sort_boundary_first) { this->boundary_nodes_first = sort_boundary_first; }
        void setMaxStencilSize(size_t st_max_size) { this->max_st_size = st_max_size; }
        size_t getMaxStencilSize() { return this->max_st_size; }

        size_t                      getNodeListSize() { return node_list.size(); }
        std::vector<NodeType>& 	    getNodeList() 	{ return node_list; }
        NodeType& 	                getNode(size_t indx) 	{ return node_list[indx]; }
        size_t                      getBoundaryIndicesSize() { return boundary_indices.size();} 
        std::vector<size_t>&        getBoundaryIndices() { return boundary_indices; } 
        size_t                      getBoundaryIndex(size_t indx) { return boundary_indices[indx]; } 
        std::vector<Vec3>&          getBoundaryNormals() { return boundary_normals; }

        size_t                      getStencilsSize() { return stencil_map.size(); }
        size_t                      getStencilSize(int indx) { return stencil_map[indx].size(); }
        std::vector<StencilType>&   getStencils() 	{ return stencil_map; }
        StencilType&                getStencil(int indx) { return stencil_map[indx]; }
        std::vector<double>& 	    getStencilRadii(){ return avg_stencil_radii; }
        double 		                getStencilRadius(int indx) { return avg_stencil_radii[indx]; }

        
        
        KDTree* getNodeListAsKDTree() {
            if (node_list_kdtree == NULL) {
                // TODO: hardcoded dimension should be changed
                node_list_kdtree = new KDTree(node_list); 
            }
            return node_list_kdtree; 
        }


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

        void setExtents( double minX, double maxX, double minY, double maxY, double minZ, double maxZ ) {
            xmin = minX;
            xmax = maxX;
            ymin = minY; 
            ymax = maxY; 
            zmin = minZ; 
            zmax = maxZ;
        }




        // Set DEBUG to 0 or 1
        void setDebug(int debug_) { DEBUG = debug_; }

        // NOTE: this grid does not have details of node connectivity. We could add this
        // in a subclass if we wanted like a TriangularGrid or RectilinearGrid. 
}; 

// A small class that allows us to sort stencil nodes by their distance to to
// the stencil center
class ltvec {
public:
    static NodeType xi;
    static vector<NodeType>* rbf_centers;

    static void setXi(NodeType& xi) {
        ltvec::xi = xi;
    }

    static void setRbfCenters(vector<NodeType>& rbf_centers_) {
        rbf_centers = &rbf_centers_;
    }

    bool operator()(const int i, const int j) {
        double d1 = ((*rbf_centers)[i] - xi).square();
        double d2 = ((*rbf_centers)[j] - xi).square();
        // allows duplicates
        return d1 <= d2;
    }
};

#endif //__GRID_H__
