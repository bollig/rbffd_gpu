#ifndef __GRID_H__
#define __GRID_H__

#include <vector> 
#include <set> 
#include <string>
#include <sstream>
#include "Vec3.h"
#include "KDTree.h"

#include "common_typedefs.h"

class Grid 
{
    public: 
        // Bounding box of domain (should contain ALL points regardless of
        // geom)
        float xmin, xmax; 
        float ymin, ymax; 
        float zmin, zmax; 

        // Number of subdivisions in domain bounding box to use in each
        // direction for the cell overlay in the hash neighbor qeury 
        unsigned int ns_nbx, ns_nby, ns_nbz; 

    public: 
        // We allow multiple types of stencil generators (for backwards compat)
        enum st_generator_t {ST_BRUTE_FORCE=0, ST_KDTREE, ST_HASH};
        enum GridLoadErrType {GRID_AND_STENCILS_LOADED=0, NO_GRID_FILES, NO_STENCIL_FILES, NO_EXTRA_FILES};

    protected: 
        // 0 = Debug output off; 1 = Verbose output and write intermediate
        // files
        int DEBUG;

        // Number of nodes this class is configured for. If this does not 
        // match the node_list.size() then we need to regenerate the node
        // set.
        unsigned int nb_nodes; 

        // Maximum number of nodes to allow in a stencil (can be equal to
        // nb_nodes if we want to define a global function across a stencil)
        unsigned int max_st_size; 

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
        std::vector<unsigned int> boundary_indices; 

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


        // Sorted list of indices on the boundary and interior. This helps when
        // we dont have nodes sorted boundary first and want to access
        // boundary/interior only nodes. By default these are EMPTY. fill them
        // by calling partitionIndices()
        std::set<unsigned int> b_indices;
        std::set<unsigned int> i_indices; 
        bool partitioned_indices;

        bool stencilsComputed;

    public:
        Grid() : 
            xmin(0.), xmax(1.), 
            ymin(0.), ymax(1.), 
            zmin(0.), zmax(0.),
            ns_nbx(10), ns_nby(10), ns_nbz(10),
            max_st_size(0), max_st_radius(DBL_MAX),
            pert(0.), nb_nodes(0),
            node_list_kdtree(NULL),
            stencilsComputed(false),
            partitioned_indices(false),
            boundary_nodes_first(false), DEBUG(0)
    {}
        Grid(unsigned int num_nodes) : 
            xmin(0.), xmax(1.), 
            ymin(0.), ymax(1.), 
            zmin(0.), zmax(0.),
            ns_nbx(10), ns_nby(10), ns_nbz(10),
            max_st_size(0), max_st_radius(DBL_MAX), 
            pert(0), nb_nodes(num_nodes), 
            node_list_kdtree(NULL),
            stencilsComputed(false),
            partitioned_indices(false),
            boundary_nodes_first(false), DEBUG(0) 
    {}
        Grid(std::vector<NodeType>& nodes) : 
            xmin(0.), xmax(1.), 
            ymin(0.), ymax(1.), 
            zmin(0.), zmax(0.),
            ns_nbx(10), ns_nby(10), ns_nbz(10),
            max_st_size(0), max_st_radius(DBL_MAX), 
            pert(0), nb_nodes(nodes.size()), 
            node_list_kdtree(NULL), 
            stencilsComputed(false),
            partitioned_indices(false),
            boundary_nodes_first(false), DEBUG(0), node_list(nodes) 
    {} 

        virtual ~Grid(){
            if (node_list_kdtree != NULL) {
                delete(node_list_kdtree);
            }
        }

        // PROBABLY THE MOST IMPORTANT ROUTINE: generates the nodes in node_list
        virtual void generate(); 

        // SECOND MOST IMPORTANT ROUTINE: generates stencil connectivity of node_list stored in stencil_map
        void generateStencils(st_generator_t generator_choice = Grid::ST_BRUTE_FORCE);
        void generateStencils(unsigned int st_max_size, st_generator_t generator_choice = Grid::ST_BRUTE_FORCE);
        
        void generateStencilsBruteForce(); 
        void generateStencilsKDTree(); 
        void generateStencilsHash();

        void computeStencilRadii();

        // Write data to disk generating the filename appropriately for the class (calls to writeToFile(std::string)) 
        virtual	void writeToFile(int iter=-1); 
        virtual void writeToFile(std::string filename); 
        void writeBoundaryToFile(std::string filename); 
        void writeNormalsToFile(std::string filename); 
        void writeAvgRadiiToFile(std::string filename); 
        void writeMaxRadiiToFile(std::string filename); 
        void writeMinRadiiToFile(std::string filename); 
        void writeStencilsToFile(std::string filename); 
        virtual void writeExtraToFile(std::string filename); 

        virtual void printNodeList(std::string label); 
        virtual void printBoundaryIndices(std::string label);  

        // Load data from disk using the class generated filename and specified iter
        Grid::GridLoadErrType loadFromFile(int iter=-1); 
        virtual GridLoadErrType loadFromFile(std::string filename); 

        int loadBoundaryFromFile(std::string grid_filename); 
        int loadNormalsFromFile(std::string grid_filename); 
        int loadAvgRadiiFromFile(std::string grid_filename); 
        int loadStencilsFromFile(std::string grid_filename); 
        virtual int loadExtraFromFile(std::string grid_filename); 

        // Sort the nodes so boundary nodes are first in the lists
        virtual void sortNodes(); 

        void partitionIndices(); 
        // Return the sorted set of interior and boundary nodes
        std::set<unsigned int>& getSortedBoundarySet() { 
            if(!partitioned_indices) { 
                this->partitionIndices();
            }
            return b_indices; 
        } 
        std::set<unsigned int>& getSortedInteriorSet() {  
            if(!partitioned_indices) { 
                this->partitionIndices();
            }
            return i_indices; 
        }


        // Enable or disable node sorting (if a grid generator obeys the order)
        // so boundary nodes appear first in the node_list
        void setSortBoundaryNodes(int sort_boundary_first) 
        { this->boundary_nodes_first = sort_boundary_first; }

        void setMaxStencilSize(unsigned int st_max_size) 
        { this->max_st_size = st_max_size; }

        unsigned int getMaxStencilSize() 
        { return this->max_st_size; }

        unsigned int getNodeListSize() 
        { return node_list.size(); }

        std::vector<NodeType>& getNodeList() 	
        { return node_list; }

        NodeType& getNode(unsigned int indx) 	
        { return node_list[indx]; }
        void setNode(unsigned int indx, NodeType node)
        { node_list[indx] = node; }

        unsigned int getBoundaryIndicesSize() 
        { return boundary_indices.size();} 
        std::vector<unsigned int>& getBoundaryIndices() 
        { return boundary_indices; } 

        unsigned int& getBoundaryIndex(unsigned int indx) 
        { return boundary_indices[indx]; } 
        void setBoundaryIndex(unsigned int indx, unsigned int boundary_indx)
        { boundary_indices[indx] = boundary_indx; }

        std::vector<Vec3>& getBoundaryNormals() 
        { return boundary_normals; }

        Vec3& getBoundaryNormal(unsigned int indx) 
        { return boundary_normals[indx]; }
        void setBoundaryNormal(unsigned int indx, Vec3& normal) 
        { boundary_normals[indx] = normal; }

        unsigned int getStencilsSize() 
        { return stencil_map.size(); }
        unsigned int getStencilSize(int indx) 
        { return stencil_map[indx].size(); }
        std::vector<StencilType>& getStencils()
        { return stencil_map; }
        StencilType& getStencil(int indx)
        { return stencil_map[indx]; }

        std::vector<double>& getStencilRadii()
        { return avg_stencil_radii; }
        double getStencilRadius(int indx) 
        { return avg_stencil_radii[indx]; }
        
        std::vector<double>& getMaxStencilRadii()
        { return max_stencil_radii; }
        double getMaxStencilRadius(int indx) 
        { return max_stencil_radii[indx]; }

        std::vector<double>& getMinStencilRadii()
        { return min_stencil_radii; }
        double getMinStencilRadius(int indx) 
        { return min_stencil_radii[indx]; }



        KDTree* getNodeListAsKDTree() {
            if (node_list_kdtree == NULL) {
                // TODO: hardcoded dimension should be changed
                node_list_kdtree = new KDTree(node_list); 
            }
            return node_list_kdtree; 
        }


        // Convert a basic filename like "output_file" to something more
        // descriptive and appropriate to the grid like
        // "output_file_120nodes_final.ascii" or
        // "output_file_356_nodes_final.bin"
        std::string getFilename(std::string base_filename, int iter=0);

        // Get a filename appropriate for output from this class
        // same as getFilename(std::string, int) however it uses 
        // the class's internal name instead of a user specified string. 
        std::string getFilename(int iter=0); 

        // Get a string that gives some detail about the grid (used by
        // expandFilename(...)) 
        // NOTE: replace spaces with '_'
        virtual std::string getFileDetailString(); 
        
        std::string getStencilDetailString() { 
            std::stringstream ss(std::stringstream::out);
            ss << "stsize_" << this->getMaxStencilSize(); 
            return ss.str();
        }

        virtual std::string className() { return "grid"; }


        // Select a random number [randf(-pert, pert)] for each node dimension
        // and add to randomly perturb nodes in space; store perturb_percent in
        // pert to maintain limited history of node perturbation
        virtual void perturbNodes(float perturb_amount);

        void setExtents( double minX, double maxX, double minY, double maxY, double minZ, double maxZ ) {
            xmin = minX;
            xmax = maxX;
            ymin = minY; 
            ymax = maxY; 
            zmin = minZ; 
            zmax = maxZ;
        }

        void refreshExtents() {
            std::cout << "Updating extents" << std::endl;
            for (unsigned int i = 0; i < this->getNodeListSize(); i++) {
                NodeType& n = this->getNode(i);
                if (n.x() < xmin) {
                    xmin = n.x(); 
                }
                if (n.x() > xmax) {
                    xmax = n.x(); 
                }  
                if (n.y() < ymin) {
                    ymin = n.y(); 
                }
                if (n.y() > ymax) {
                    ymax = n.y(); 
                }   
                if (n.z() < zmin) {
                    zmin = n.z(); 
                }
                if (n.z() > zmax) {
                    zmax = n.z(); 
                }  
            }
        }

        void setNSHashDims(unsigned int overlay_nbx, unsigned int overlay_nby, unsigned int overlay_nbz) {
            ns_nbx = overlay_nbx; 
            ns_nby = overlay_nby; 
            ns_nbz = overlay_nbz; 
        }           

        // Verify that our configuraton for max_st_size is valid and adjust it if its too large
        void checkStencilSize(); 

        // Set DEBUG to 0 or 1
        void setDebug(int debug_) { DEBUG = debug_; }

        // NOTE: this grid does not have details of node connectivity. We could
        // add this in a subclass if we wanted like a TriangularGrid or
        // RectilinearGrid. 
        
    protected: 
        void writeVecToFile(std::string prefix, std::string suffix, std::vector<double> vals);
        void resizeBoundary(unsigned int nb_boundary_nodes)
        {
            this->boundary_indices.resize(nb_boundary_nodes);
            this->boundary_normals.resize(nb_boundary_nodes);
        }
        void resizeNodeList(unsigned int nb_pts)
        {
            this->node_list.resize(nb_pts);
        }

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

// allow insertion sort of <distance, node_indx> pairs for std::set
class ltdist {
    public: 
        bool operator() (const std::pair<float,unsigned int> i, const std::pair<float,unsigned int> j)
        {
            return i.first <= j.first; 
        }
};

#endif //__GRID_H__
