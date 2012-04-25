#ifndef _Domain_H_
#define _Domain_H_

#include "utils/comm/mpisendable.h"

#include <vector>
#include <set>
#include <map>

#include "grids/grid_interface.h"
#include "common_typedefs.h"

class Domain : public Grid, public MPISendable
{
    public: 		// Member Properties

        int dim_num;

        int id; 		// which Domain
        int comm_size; 	// Total number of Domains

#if 0
        double xmin;
        double xmax;
        double ymin;
        double ymax;
        double zmin;
        double zmax;
#endif 


        // ***************************************************
        //              N O T E: 
        //----------------------------------------------------
        //
        // Sets in GLOBAL INDEXING: 
        //   G, Q, D, O, B, QmD, QmB, BmO, R 
        //
        // Sets in LOCAL INDEXING: 
        //   node_list, boundary_indices, boundary_normals
        //
        //
        // ***************************************************


        // 1) These are the sets of stencil centers
        std::set<int> G; 			// All nodes required for computation
        std::set<int> Q;			// All stencil centers in this CPU's QUEUE							
        std::set<int> D;			// Set of stencil centers DEPENDENT on nodes in R before evaluation
        std::set<int> O;			// Stencil Centers that are OUTPUT to other Domains but might NOT depend on R
        //TODO: get rid of B since we dont do anything special with it.
        std::set<int> B; 			// Stencil Centers on DOMAIN BOUNDARY (in O and D) We group O and D together even 
                                    // if O can contain nodes that do not depend on R because set O will require a 
                                    // memcpy back to the CPU from the GPU. 

        std::set<int> QmD; 			// Interior stencil centers excluding stencils dependent on RECEIVING
        std::set<int> QmB;
        std::set<int> BmO;
        std::set<int> R;			// Nodes REQUIRED from other Domains (not stencil centers) 

        // 5) These are the maps between local and global indexing space
        // l2g[i] = global_indx_matching_i
        std::vector<int> loc_to_glob;
        // g2l[i] = local_indx_matching_i
        std::map<int, int> glob_to_loc;

        // 6) Additional properties (should be possible to avoid these)

        // These decompose sets R and O by CPU rank so we can see which CPU requires 
        // which subset of R and must send what subset of O. 
        std::vector<std::vector<int> > O_by_rank;
        std::vector<std::vector<int> > R_by_rank;

    private: 
        bool inclMX, inclMY, inclMZ;

    public: 	// Member Functions: 

        // Empty constructor for CPUs that will construct this instance via MPI
        Domain() {}
        Domain(const Domain& subdomain); // Copy constructor

        // Requires communicator to pass messages. This must be preconstructed comm_unit 
        Domain(int dim_num, unsigned int global_nb_nodes, double _xmin, double _xmax, double _ymin, double _ymax, double _zmin, double _zmax, int _comm_rank, int _comm_size);


        Domain(int dim_num, Grid* _grid, int _comm_size);


        //--------------------------
        // Override Grid:: functions
        //--------------------------
        virtual std::string className() { return "domain"; }

        virtual std::string getFileDetailString() {
            char prefix[256]; 
            sprintf(prefix, "_rank%d_of_%da", this->id, this->comm_size); 
            std::string s = Grid::getFileDetailString(); 
            s.append(prefix); 
            return s;
        }
        virtual void printNodeList(std::string label) { 
            // More verbose print: 
            printCenters(this->node_list, label); 
        }
        virtual void printBoundaryIndices(std::string label) {
            // More verbose print:
            printVector(this->boundary_indices, label); 
        }
        
        virtual void writeExtraToFile(std::string filename) {
            this->writeG2LToFile(filename);
            this->writeL2GToFile(filename);
           // this->writeLocalSolutionToFile(filename);
        }

        // TODO: allow subdomains to load from disk rather than always
        // initializing on master and then distributing. This would provide
        // restart capabiility!
        //virtual int loadExtraFromFile(std::string filename) {
           // this->loadG2LFromFile(filename);
           // this->loadL2GFromFile(filename);
           // this->loadLocalSolutionFromFile(filename);
        //}

        //--------------------------------------------------
        // Domain specific routines: 
        //--------------------------------------------------
        void writeG2LToFile(std::string filename); 
        void writeL2GToFile(std::string filename); 

        // Decompose the current domain into x_divisions by y_divisions by z_divisions. 	
        void generateDecomposition(std::vector<Domain*>& subdomains, int x_divisions, int y_divisions = 1, int z_divisions = 1); 


        // Fill this Domains stencil and position sets based on the global set of RBF centers and stencils. 
        void fillLocalData(std::vector<NodeType>& rbf_centers, std::vector<StencilType>& stencil, std::vector<unsigned int>& boundary, std::vector<double>& avg_dist, std::vector<double>& max_dist, std::vector<double>& min_dist);

        // Append to O_by_rank (find what subset of O is needed (i.e., in the
        // set R) of another subdomain
        // Param: subdomain_R specifies R for neighboring domain;
        // subdomain_rank specifies neighbors rank
        void fill_O_by_rank(std::set<int>& subdomain_R, int subdomain_rank); 

        // Append to R_by_rank (find what subset of R is provided (i.e., in the
        // set O) of another subdomain
        // Param: subdomain_O specifies R for neighboring domain;
        // subdomain_rank specifies neighbors rank
        void fill_R_by_rank(std::set<int>& subdomain_O, int subdomain_rank); 

        // When we move to 3D this should be updated to reflect zmin, zmax
        // We could also make this polar coords, striped subdomains etcs. 
        bool isInsideSubdomain(NodeType& pt) 
        {
            // TODO : need to support xmin != xmax && zmin != zmax but ymin==ymax 
            // 		  and other combinations

#if 1
//            std::cout << pt << "in [" << xmin << ", " << xmax << "]";
            bool inside = isInsideRange(pt.x(), xmin, xmax, inclMX); 
            if (dim_num > 1) {
              //  std::cout << "x[" << ymin << ", " << ymax << "]"; 
                inside &= isInsideRange(pt.y(), ymin, ymax, inclMY); 
            }
            if (dim_num > 2) { 
                //std::cout << "x[" << zmin << ", " << zmax << "]";
                inside &= isInsideRange(pt.z(), zmin, zmax, inclMZ); 
            }
//            std::cout << "==> " << inside << std::endl;
            return inside; 
#else 
            if (ymin == ymax) {
                return isInsideRange(pt.x(), xmin, xmax, inclMX); 
            } else if (zmin == zmax) {
                return isInsideRange(pt.x(), xmin, xmax, inclMX) && isInsideRange(pt.y(), ymin, ymax, inclMY); 
            } else {
                return isInsideRange(pt.x(), xmin, xmax, inclMX) && isInsideRange(pt.y(), ymin, ymax, inclMY) && isInsideRange(pt.z(), zmin, zmax, inclMZ); 
            } 
            
#endif 
        }

        bool isInsideRange(double pt_, double rmin, double rmax, bool inclusiveMax) {

#if 0
            double d1 = pt_ - rmin; 
            double d2 = rmax - pt_; 
            if (fabs(d2) < 1e-5) {
                std::cout << "+"; 
            } else if (fabs(d1) < 1e-5) {
                std::cout << "-"; 
            } else { 
                std::cout << ".";
            }
#endif
#if 0
            if (inclusiveMax) {
                // Use subtraction here to guarantee all nodes are accounted for
                return (d1 >= 0. && d2 >= 0.);
            } else {
                return (d1 >= 0. && d2 > 0.));
            }
#else
            if (inclusiveMax) {
                return ((pt_ >= rmin) && (pt_ <= rmax));
            } else {
                return ((pt_ >= rmin) && (pt_ < rmax));
            }
#endif 
        }

        void setInclusiveMaxBoundary(bool inclMaxX, bool inclMaxY, bool inclMaxZ) { 
            inclMX = inclMaxX; 
            inclMY = inclMaxY; 
            inclMZ = inclMaxZ; 
        }


        // local numbering: {Q\O, O, B}
        // local to global
        int l2g(int ix) {
            return loc_to_glob[ix];
        }
        // global to local
        // build a hash table
        int g2l(int ix) {
            return glob_to_loc[ix];
        }


        // NOT USED:
        // Not memory safe.
        StencilType& convert_g2l(StencilType& stencil) {
            StencilType* local_stencil = new StencilType(stencil);
            for (unsigned int j = 0; j < stencil.size(); j++) {
                (*local_stencil)[j] = g2l(stencil[j]);
            }
            return *local_stencil;
        }

        // ******** BEGIN MPISENDABLE ************
        // The following seven routines are required by MPISendable inheritence.
        void setCommSize(int size) { comm_size = size; } 
        virtual int send(int my_rank, int receiver_rank); 
        virtual int receive(int my_rank, int sender_rank);

        virtual int sendUpdate(int my_rank, int receiver_rank) { std::cout << "[Domain] sending node updates not allowed yet.\n"; return 0; }
        virtual int receiveUpdate(int my_rank, int sender_rank) { std::cout << "[Domain] receiving node updates not allowed yet.\n"; return 0; }

        virtual int sendFinal(int my_rank, int receiver_rank) { std::cout << "[Domain] nothing final to send\n"; return 0; }
        virtual int receiveFinal(int my_rank, int sender_rank) { std::cout << "[Domain] nothing final to receive\n"; return 0; }

        virtual int initFinal() { return 0; }
        virtual int updateFinal() { return 0; }
        // ******** END MPISENDABLE ************
        
        //public: 	// Member Functions
        // Fill sets Q, D, O, B, QmD and R to distinguish center memberships
        void fillCenterSets(std::vector<NodeType>& rbf_centers, std::vector<StencilType>& stencils);

        // Generate a set of ALL nodes that are used by the stencils in set s.
        // Uses stencil to lookup nodes required by s.
        // memory for return set is allocoted within stencilSet
        void stencilSet(std::set<int>& s, std::vector<StencilType>& stencil, std::set<int>& Sset_out);

        // Print all nodes in stencils and show display_char if they are in center_set; '.' otherwise. 
        void printStencilNodesIn(const std::vector<StencilType>& stencils, const std::set<int>& center_set, std::string display_char); 
        // Dump a bit of ascii art to show the dependency graph for a Domain
        void printVerboseDependencyGraph(); 

        // Print contents of a set
        void printSetL2G(const std::set<int>& center_set, std::string set_label) ; 
        void printSetG2L(const std::set<int>& center_set, std::string set_label) ; 

        // Print contents of a set
        void printVector(const std::vector<double>& stencil_radii, std::string set_label) ; 
        void printVector(const std::vector<unsigned int>& center_set, std::string set_label) ; 
        void printVectorL2G(const std::vector<int>& center_set, std::string set_label) ; 
        void printVectorG2L(const std::vector<int>& center_set, std::string set_label) ; 

        void printStencil(const StencilType& stencil, std::string stencil_label) ;

        // Print the stencil plus use the indices in the stencil to gather and print function values
        void printStencilPlus(const StencilType& stencil, const std::vector<double>& function_values, std::string stencil_label) ;

        void printCenters(const std::vector<NodeType>& centers, std::string center_label) ;

        // Print a table of memberships for a set of centers
        void printCenterMemberships(const std::set<int>& center_set, std::string display_name) ;

        // Tell if a stencil depends on nodes in center_set (i.e. Q_stencil[i] depends on R?)
        bool dependsOnSet( const int global_stencil_id, const std::set<int>& center_set);

        // Determine if a center is member of a set (center is global index)
        bool isInSet(const int center, const std::set<int>& center_set) const;

        // Determine if a center is member of a vector (center is global index)
        bool isInVector(const unsigned int center, const std::vector<unsigned int>& center_set) const;
};

#endif
