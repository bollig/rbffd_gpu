#ifndef _Domain_H_
#define _Domain_H_
#include <vector>
#include <set>
#include <map>
#include <Vec3.h>

//#include <boost/hash.hpp> // full library
#include "utils/comm/mpisendable.h"
#include "grids/grid_interface.h"
#include "common_typedefs.h"

class Domain : public MPISendable {
public: 		// Member Properties
	int id; 		// which Domain
	int comm_size; 	// Total number of Domains
	typedef std::map<int, int> GlobMap;

	double xmin;
	double xmax;
	double ymin;
	double ymax;
	double zmin;
	double zmax;
	double dt; 
	
	// 1) These are the sets of stencil centers
	std::set<int> G; 			// All nodes required for computation
	std::set<int> Q;			// All stencil centers in this CPUs QUEUE							
	std::set<int> D;			// Set of stencil centers DEPEND on nodes in R before evaluation
	std::set<int> O;			// Centers that are OUTPUT to other Domains
	std::set<int> B; 			// Centers on BOUNDARY (in O and D or both)
	std::set<int> QmB; 			// Interior centers (computed without communication) 
	std::set<int> R;			// Nodes REQUIRED from other Domains. 
	// Possibly add: 
	// std::set<int> BmD;

#if 1	
	// 2) These get us the connectivity of the stencils
	// Not responsible for any other stencils than QmB and B. 
	// We index these vectors with the integers contained in sets Q, O, D, etc.
	// NOTE: (index Q_centers[*][0] is always RBF stencil center)
	std::vector<StencilType> Q_stencils; 
	//	std::vector<std::vector<int> > Q_stencils_local; 		// (local index equivalent of Q_stencils) CPUs generate this after receiving the global set. 
	
	// We might want to decompose Q_stencils into this: 
	//std::vector<std::vector<int> > QmB_stencils; 		// Set of stencils we can evaluate without communication
	//std::vector<std::vector<int> > BmD_stencils; 		// Set of stencils that can be evaluated while sending O and receiving R
	//std::vector<std::vector<int> > D_stencils; 		// Set of stencils that block until R is received
	
	// 3) These get us the physical position of nodes in stencils
	std::vector<NodeType> G_centers; 		// The full set of centers needed for computation (Q + R = G)
	
	// Precomputed average distances (possibly stencil radii...code is not clear)
	// stencils in subdomain.
	std::vector<double> Q_avg_dists;
	
	// The points in this subdomain which are also part of the global PDE boundary
	// NOTE: global indices
	std::vector<size_t> global_boundary_nodes;

        // Since we have global_boundary_nodes we can get the indices of the interior
        // nodes by doing a difference on the set Q and the global boundary nodes.
#else 
	// Grid contains: node_list (equiv to G_centers), boundary_indices (equiv to global_boundary_nodes), stencil_map (equiv to Q_stencils), avg_stencil_radii (equiv to Q_avg_dists)
	Grid* grid;
#endif 

	// 4) These get us the func values of stencil nodes
	std::vector<double> U_G;			// U_Q union U_R
	// Possibly decompose to U_Q, U_R instead since U_R will update each iteration
	

	// 5) These are the maps between local and global indexing space
	std::vector<int> loc_to_glob;
	GlobMap globmap;

	// 6) Additional properties (should be possible to avoid these)
	
	// These decompose sets R and O by CPU rank so we can see which CPU requires 
	// which subset of R and must send what subset of O. 
	std::vector<std::vector<int> > O_by_rank;
	// IF we use this then we can send atomic updates (i.e. we dont need to ask
	// how many node updates we will receive from each Domain and can instead send
	// and receive exactly what is required): 
	// std::vector<std::set<int> > R_by_rank;
	
private:
	// A map for global INDEX=VALUE storage of the final solution
	// recvFinal will populate this and then we can call (TODO) getFinal()
	// to get the values as a vector.
	std::map<int,double> global_U_G;

public: 	// Member Functions: 

	// Empty constructor for CPUs that will construct this instance via MPI
	Domain() {}
	Domain(const Domain& subdomain); // Copy constructor

	// Requires communicator to pass messages. This must be preconstructed comm_unit 
	Domain(double _xmin, double _xmax, double _ymin, double _ymax, double _zmin, double _zmax, double _dt, int _comm_rank, int _comm_size);
	
	
	Domain(Grid* _grid, double _dt, int _comm_size);
	
	// Decompose the current domain into x_divisions by y_divisions by z_divisions. 	
	void generateDecomposition(std::vector<Domain*>& subdomains, int x_divisions, int y_divisions = 1, int z_divisions = 1); 
	
	
	// Fill this Domains stencil and position sets based on the global set of RBF centers and stencils. 
	void fillLocalData(std::vector<Vec3>& rbf_centers, std::vector<StencilType>& stencil, std::vector<size_t>& boundary, std::vector<double>& avg_dist);

	// Append to O_by_rank (find what subset of O is needed by rank subdomain_rank)
	void fillDependencyList(std::set<int>& subdomain_R, int subdomain_rank); 

	// When we move to 3D this should be updated to reflect zmin, zmax
	// We could also make this polar coords, striped subdomains etcs. 
	bool isInsideSubdomain(Vec3& pt) 
	{
		// TODO : need to support xmin != xmax && zmin != zmax but ymin==ymax 
		// 		  and other combinations
		if (ymin == ymax) {
			return (pt.x() < xmin || pt.x() > xmax);
		} else if (zmin == zmax) {
			return (pt.x() < xmin || pt.x() > xmax || pt.y() < ymin || pt.y() > ymax);
		} else {
			return (pt.x() < xmin || pt.x() > xmax || pt.y() < ymin || pt.y() > ymax || pt.z() < zmin || pt.z() > zmax);
		} 
	}

	// local numbering: {Q\O, O, B}
	// local to global
	int l2g(int ix) {
		return loc_to_glob[ix];
	}
	// global to local
	// build a hash table
	int g2l(int ix) {
		return globmap[ix];
	}
	
	
	StencilType& convert_g2l(StencilType& stencil) {
		StencilType* local_stencil = new StencilType(stencil);
		for (int j = 0; j < stencil.size(); j++) {
			(*local_stencil)[j] = globmap[stencil[j]];
		}
		return *local_stencil;
	}
	

	// fill data variable U
	void fillVarData(std::vector<Vec3>& rbf_centers);

	std::vector<double>& getU() { return U_G; };
	
	// The following four routines are required by MPISendable inheritence.
	virtual int send(int my_rank, int receiver_rank); 
	virtual int receive(int my_rank, int sender_rank);
	virtual int sendUpdate(int my_rank, int receiver_rank); 
	virtual int receiveUpdate(int my_rank, int sender_rank);

	virtual int sendFinal(int my_rank, int receiver_rank);
	virtual int receiveFinal(int my_rank, int sender_rank);

	virtual int initFinal();
	virtual void getFinal(std::vector<double>* final);

	// Dump the final solution to a file along with the vector of nodes that
	// the values correspond to.
	virtual int writeFinal(std::vector<NodeType>& nodes, char* filename);

//public: 	// Member Functions
	// Fill sets Q, D, O, B, QmB and R to distinguish center memberships
	void fillCenterSets(std::vector<NodeType>& rbf_centers, std::vector<StencilType>& stencils);
	
	// Generate a set of ALL nodes that are used by the stencils in set s.
	// Uses stencil to lookup nodes required by s.
	// memory for return set is allocoted within stencilSet
	std::set<int>& stencilSet(std::set<int>& s, std::vector<StencilType>& stencil);
	
	// Print all nodes in stencils and show display_char if they are in center_set; '.' otherwise. 
	void printStencilNodesIn(const std::vector<StencilType>& stencils, const std::set<int>& center_set, const char* display_char); 
    	// Dump a bit of ascii art to show the dependency graph for a Domain
    	void printVerboseDependencyGraph(); 
	
	// Print contents of a set
	void printSet(const std::set<int>& center_set, const char* set_label) ; 
	
	// Print contents of a set
	void printVector(const std::vector<double>& stencil_radii, const char* set_label) ; 
	void printVector(const std::vector<int>& center_set, const char* set_label) ; 
	void printVector(const std::vector<size_t>& center_set, const char* set_label) ; 
	
	void printStencil(const StencilType& stencil, const char* stencil_label) ;
	
	// Print the stencil plus use the indices in the stencil to gather and print function values
	void printStencilPlus(const StencilType& stencil, const std::vector<double>& function_values, const char* stencil_label) ;
	
	void printCenters(const std::vector<Vec3>& centers, const char* center_label) ;
	
	// Print a table of memberships for a set of centers
	void printCenterMemberships(const std::set<int>& center_set, const char* display_name) ;
	
	// Tell if a stencil depends on nodes in center_set (i.e. Q_stencil[i] depends on R?)
	bool dependsOnSet( const int global_stencil_id, const std::set<int>& center_set);
	
	// Determine if a center is member of a set (center is global index)
	bool isInSet(const int center, const std::set<int>& center_set) const;

        // Determine if a center is member of a vector (center is global index)
	bool isInVector(const size_t center, const std::vector<size_t>& center_set) const;
};

#endif
