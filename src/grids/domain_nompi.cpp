#define REMOVE_SOLUTION_FROM_DOMAIN

//#include <set>

#include "domain_nompi.h"
#include <algorithm>
#include <iostream>
#include <fstream>


using namespace std;
//using namespace boost;


DomainNoMPI::DomainNoMPI(const DomainNoMPI& subdomain) {
    cout << "INSIDE COPY CONSTRUCTOR" << endl;
    exit(EXIT_FAILURE);
}

DomainNoMPI::DomainNoMPI(int dimension, Grid* grid, int _comm_size)
    : 
        dim_num(dimension),
        id(0), comm_size(_comm_size),
        inclMX(true),inclMY(true),inclMZ(true)
{
    // Use this constructor to create a wrapper. 
  
    xmin = grid->xmin;
    xmax = grid->xmax;
    ymin = grid->ymin; 
    ymax = grid->ymax;
    zmin = grid->zmin;
    zmax = grid->zmax;

    this->grid = grid; // Perhaps temporary, at least until I get this routine working

    // We might need to know how many nodes are in the domain globally for things like Hyperviscosity
    this->global_num_nodes = grid->getNodeListSize();

    // Forms sets (Q,O,R) and l2g/g2l maps
    printf("inside DomainNoMPI constructor\n");
    printf("nodelist size: %d\n", grid->getNodeList().size());
    printf("stencil size: %d\n", grid->getStencils().size());
    printf("... fillLocalData from constructor\n");
  
    // Why fill data on the entire grid. I DO NOT FOLLOW.
    fillLocalData(grid->getNodeList(), grid->getStencils(), grid->getBoundaryIndices(), grid->getStencilRadii(), grid->getMaxStencilRadii(), grid->getMinStencilRadii()); 
    //printf("after fillLocalData from constructor\n"); exit(0);
    this->max_st_size = grid->getMaxStencilSize();
}


// Construct a new DomainNoMPI object
DomainNoMPI::DomainNoMPI(Grid* grid, int dimension, unsigned int global_nb_nodes, 
        double _xmin, double _xmax, double _ymin, double _ymax, double _zmin, double _zmax, 
        int _comm_rank, int _comm_size) :
    dim_num(dimension),
    id(_comm_rank), comm_size(_comm_size),
    inclMX(false),inclMY(false),inclMZ(false)
{

    this->grid = grid;

    // These are props of Grid inherited by DomainNoMPI
    global_num_nodes = global_nb_nodes;
    xmin = _xmin;
    xmax = _xmax;
    ymin = _ymin; 
    ymax = _ymax;
    zmin = _zmin;
    zmax = _zmax;
}


void DomainNoMPI::generateDecomposition(std::vector<DomainNoMPI*>& subdomains, int x_divisions, int y_divisions, int z_divisions) 
{
#if 0
    int gx = x_divisions;
    int gy = y_divisions;
    int gz = z_divisions;
    //    std::vector<DomainNoMPI*> subdomains;
    subdomains.resize(gx * gy * gz);

    // We partition points by divying up the extents
    double deltax = (double) (xmax - xmin) / (double) gx;
    double deltay = (double) (ymax - ymin) / (double) gy;
    double deltaz = (double) (zmax - zmin) / (double) gz;

    printf("domain decomposition deltas (dx, dy, dz) = (%f, %f, %f)\n", deltax, deltay, deltaz);

    // Initialize subdomains datastructures
    for (int id = 0; id < gx * gy * gz; id++) {
        // Derived these on paper. They work, but it takes a while to verify

        // 1) Find the slice in which we lie (NOTE: "i" or "x" is varying fastest;
        //      for "k" switch gx to gz, and swap igz and igx equations)
        //int igz = id / gx*gy;  // BUG (Gordon Erlebacher discovered, Aug. 2, 2013)
        int igz = id / (gx*gy);
        // 2) Find the row within the slice
        int igy = (id - igz * (gx*gy)) / gx;
        // 3) Find the column within the row
        int igx = (id - igz * (gx*gy)) - igy * gx;

        //        printf("igx = %d, igy = %d, igz = %d\n", igx, igy, igz);
        double xm = xmin + igx * deltax;
        double ym = ymin + igy * deltay;
        double zm = zmin + igz * deltaz;
        
        // Far end of boundary. doctor the set to make sure we get ALL nodes included. 
        // Had an issue with 3 processors and the 3rd proc should have gotten
        // (xmin, 1] as its domain but it was really (xmin,1) and the boundary
        // node was lost. This was not a problem for 2 processors though.
        //
        double left_bound = xm;
        double right_bound = (igx==gx-1) ? xmax : xm+deltax;

        double bottom_bound = ym;
        double top_bound = (igy==gy-1) ? ymax : ym+deltay;

        double front_bound = zm;
        double back_bound = (igz==gz-1) ? zmax : zm+deltaz;

        printf("Subdomain[%d (%d of %d)] Extents = (%f, %f) x (%f, %f) x (%f, %f)\n",id, id+1, comm_size, left_bound, right_bound, bottom_bound, top_bound, front_bound, back_bound);
        printf("Tile (ix, iy, iz) = (%d, %d, %d) of (%d, %d, %d)\n", igx, igy, igz,igx == gx-1, igy==gy-1, igz==gz-1); 
        subdomains[id] = new DomainNoMPI(dim_num, global_num_nodes, left_bound, right_bound, bottom_bound, top_bound, front_bound, back_bound, id, comm_size);
        subdomains[id]->setMaxStencilSize(this->max_st_size);
        subdomains[id]->setInclusiveMaxBoundary(igx == gx-1, igy == gy-1, igz == gz-1); 
    }

    // Figure out the sets Bi, Oi Qi

    printf("nb subdomains: %d\n", (int) subdomains.size());
    for (unsigned int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** CPU %d ***************** \n", i);
        // Forms sets (Q,O,R) and l2g/g2l maps
        std::cout << "GLOBAL STENCIL MAP SIZE= " << this->stencil_map.size() << std::endl;
        subdomains[i]->fillLocalData( this->node_list, this->stencil_map, this->boundary_indices, this->avg_stencil_radii, this->max_stencil_radii, this->min_stencil_radii); 
    }

    for (unsigned int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** FILLING O_by_rank for CPU%d ***************** \n", i);
        for (unsigned int j = 0; j < subdomains.size(); j++) {
            subdomains[i]->fill_O_by_rank(subdomains[j]->R, j); 
        }
    }

    for (unsigned int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** FILLING R_by_rank for CPU%d ***************** \n", i);
        for (unsigned int j = 0; j < subdomains.size(); j++) {
            subdomains[i]->fill_R_by_rank(subdomains[j]->O, j); 
        }
    }

 
    //return subdomains;
#endif
}
//----------------------------------------------------------------------
void DomainNoMPI::GEgenerateDecomposition(std::vector<DomainNoMPI*>& subdomains, int x_divisions, int y_divisions, int z_divisions) 
//  Written by G. Erlebacher, Aug. 2, 2013
{
    int gx = x_divisions;
    int gy = y_divisions;
    int gz = z_divisions;
    //    std::vector<Domain*> subdomains;
    subdomains.reserve(gx*gy*gz);
    subdomains.resize(0);
    printf("subdomain capacity= %d\n", subdomains.capacity());
    if (subdomains.size() != 0) {
        printf("subdomains size must be zero initially\n");
        exit(1);
    }

    // We partition points by divying up the extents
    double deltax = (double) (xmax - xmin) / (double) gx;
    double deltay = (double) (ymax - ymin) / (double) gy;
    double deltaz = (double) (zmax - zmin) / (double) gz;

    printf("domain decomposition deltas (dx, dy, dz) = (%f, %f, %f)\n", deltax, deltay, deltaz);

    // Initialize subdomains datastructures
    for (int igz = 0; igz < z_divisions; igz++) {
    for (int igy = 0; igy < y_divisions; igy++) {
    for (int igx = 0; igx < x_divisions; igx++) {
        double xm = xmin + igx * deltax;
        double ym = ymin + igy * deltay;
        double zm = zmin + igz * deltaz;
        
        // Far end of boundary. doctor the set to make sure we get ALL nodes included. 
        // Had an issue with 3 processors and the 3rd proc should have gotten
        // (xmin, 1] as its domain but it was really (xmin,1) and the boundary
        // node was lost. This was not a problem for 2 processors though.
        //
        double left_bound = xm;
        double right_bound = (igx==gx-1) ? xmax : xm+deltax;

        double bottom_bound = ym;
        double top_bound = (igy==gy-1) ? ymax : ym+deltay;

        double front_bound = zm;
        double back_bound = (igz==gz-1) ? zmax : zm+deltaz;

        int id = igx + x_divisions*(igy + y_divisions*igz);

        printf("Subdomain[%d (%d of %d)] Extents = (%f, %f) x (%f, %f) x (%f, %f)\n",id, id+1, comm_size, left_bound, right_bound, bottom_bound, top_bound, front_bound, back_bound);
        printf("Tile (ix, iy, iz) = (%d, %d, %d) of (%d, %d, %d)\n", igx, igy, igz,igx == gx-1, igy==gy-1, igz==gz-1); 
        DomainNoMPI* dom = new DomainNoMPI(grid, dim_num, global_num_nodes, left_bound, right_bound, bottom_bound, top_bound, front_bound, back_bound, id, comm_size);
        //subdomains[id] = new DomainNoMPI(dim_num, global_num_nodes, left_bound, right_bound, bottom_bound, top_bound, front_bound, back_bound, id, comm_size);
        //subdomains[id]->setMaxStencilSize(this->max_st_size);
        //subdomains[id]->setInclusiveMaxBoundary(igx == gx-1, igy == gy-1, igz == gz-1); 
        dom->setMaxStencilSize(this->max_st_size);
        dom->setInclusiveMaxBoundary(igx == gx-1, igy == gy-1, igz == gz-1); 
        subdomains.push_back(dom); // Must delete arrays to prevent memory leak
    }}}

    // Figure out the sets Bi, Oi Qi

    printf("nb subdomains: %d\n", (int) subdomains.size());
    for (unsigned int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** CPU %d ***************** \n", i);
        // Forms sets (Q,O,R) and l2g/g2l maps
        std::cout << "GLOBAL STENCIL MAP SIZE= " << this->stencil_map.size() << std::endl;
        printf("about to enter fillLocalData\n");
        //Subdomain& s = *subdomains[i];
        Grid& s = *grid;
        subdomains[i]->fillLocalData( grid->getNodeList(), s.getStencils(), s.getBoundaryIndices(), s.getStencilRadii(), s.getMaxStencilRadii(), s.getMinStencilRadii());
        printf("subdomain %d filled\n", i); 
        printf("subdomain %d STENCIL MAP SIZE: %d\n", i, subdomains[i]->stencil_map.size());
        printf("subdomain %d NB NODES: %d\n", i, subdomains[i]->getNodeListSize());
        printf("subdomain %d Q size: %d\n", i, subdomains[i]->Q.size());
        printf("subdomain %d R size: %d\n", i, subdomains[i]->R.size());
        //printCenterMemberships(subdomains[i]->G, "G");

        char file[80];
    #if 0
        sprintf(file, "g2l_%0d", i);
        subdomains[i]->writeG2LToFile(file); // works
        sprintf(file, "l2g_%0d", i);
        subdomains[i]->writeL2GToFile(file); // works
    #endif
        sprintf(file, "ellsubdomain_%0d.mtxb", i);
        //printf("file name for Ellpack: %s\n", file);
        //subdomains[i]->writeToEllpackBinaryFile(file);
    }

#if 0
    // The next method never return, but code is identical to Evan's original code. 
    // Perhaps does not work if no MPI? 
    for (unsigned int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** FILLING O_by_rank for CPU%d ***************** \n", i);
        for (unsigned int j = 0; j < subdomains.size(); j++) {
            subdomains[i]->fill_O_by_rank(subdomains[j]->R, j); 
        }
    }

    for (unsigned int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** FILLING R_by_rank for CPU%d ***************** \n", i);
        for (unsigned int j = 0; j < subdomains.size(); j++) {
            subdomains[i]->fill_R_by_rank(subdomains[j]->O, j); 
        }
    }
#endif
    //return subdomains;
}
//----------------------------------------------------------------------



#if 0
int DomainNoMPI::send(int my_rank, int receiver_rank) {

    sendSTL(&id, my_rank, receiver_rank);  

    MPI_Send(&global_num_nodes, 1, MPI::UNSIGNED, receiver_rank, TAG, MPI_COMM_WORLD);

    double buff[6] = { xmin, xmax, ymin, ymax, zmin, zmax };
    MPI_Send(&buff, 6, MPI::DOUBLE, receiver_rank, TAG, MPI_COMM_WORLD);

    sendSTL(&Q, my_rank, receiver_rank); // All stencil centers in this CPUs QUEUE
    sendSTL(&D, my_rank, receiver_rank); // Set of stencil centers DEPEND on nodes in R before evaluation
    sendSTL(&O, my_rank, receiver_rank); // Centers that are OUTPUT to other Domains
    sendSTL(&B, my_rank, receiver_rank); // Centers on BOUNDARY (in O and D or both)
    sendSTL(&QmD, my_rank, receiver_rank); // Interior centers (computed without communication)
    sendSTL(&QmB, my_rank, receiver_rank); // Interior centers (computed without communication)
    sendSTL(&BmO, my_rank, receiver_rank); // Interior centers (computed without communication)
    sendSTL(&R, my_rank, receiver_rank); // Nodes REQUIRED from other Domains.

    sendSTL(&stencil_map, my_rank, receiver_rank); // All stencils receiver_rank is responsible for

    sendSTL(&node_list, my_rank, receiver_rank); // Q_centers + R_centers = node_list (all nodes necessary on receiver_rank CPU)

    sendSTL(&avg_stencil_radii, my_rank, receiver_rank); // Average distances (possibly stencil radii)
    sendSTL(&max_stencil_radii, my_rank, receiver_rank); // Average distances (possibly stencil radii)
    sendSTL(&min_stencil_radii, my_rank, receiver_rank); // Average distances (possibly stencil radii)

    sendSTL(&loc_to_glob, my_rank, receiver_rank); // l2g
    sendSTL(&glob_to_loc, my_rank, receiver_rank); // g2l

    sendSTL(&O_by_rank, my_rank, receiver_rank); // Subsets of O that this DomainNoMPI will send out to each other DomainNoMPI
    sendSTL(&R_by_rank, my_rank, receiver_rank); // Subsets of R that this DomainNoMPI will receive from every other DomainNoMPI
    sendSTL(&boundary_indices, my_rank, receiver_rank);
    sendSTL(&max_st_size, my_rank, receiver_rank);  

    cout << "RANK " << my_rank << " REPORTS: sent DomainNoMPI object" << endl;
    return 0;           // FIXME: return bytes sent (in case we need to monitor this)
}

int DomainNoMPI::receive(int my_rank, int sender_rank, int _comm_size) {

    MPI_Status stat;

    // Start by identifying the subdomain ID
    recvSTL(&id, my_rank, sender_rank);  

    MPI_Recv(&global_num_nodes, 1, MPI::UNSIGNED, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // Get the subdomain bounds 
    double buff[6];
    MPI_Recv(&buff, 6, MPI::DOUBLE, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    xmin = buff[0];
    xmax = buff[1];
    ymin = buff[2];
    ymax = buff[3];
    zmin = buff[4]; 
    zmax = buff[5];

    recvSTL(&Q, my_rank, sender_rank); // All stencil centers in this CPUs QUEUE
    recvSTL(&D, my_rank, sender_rank); // Set of stencil centers DEPEND on nodes in R before evaluation
    recvSTL(&O, my_rank, sender_rank); // Centers that are OUTPUT to other Domains
    recvSTL(&B, my_rank, sender_rank); // Centers on BOUNDARY (in O and D or both)
    recvSTL(&QmD, my_rank, sender_rank); // Interior centers (computed without communication)
    recvSTL(&QmB, my_rank, sender_rank); // Interior centers (computed without communication)
    recvSTL(&BmO, my_rank, sender_rank); // Interior centers (computed without communication)
    recvSTL(&R, my_rank, sender_rank); // Nodes REQUIRED from other Domains.

    recvSTL(&stencil_map, my_rank, sender_rank); // All stencils sender_rank is responsible for

    recvSTL(&node_list, my_rank, sender_rank); // Q_centers + R_centers = node_list (all nodes necessary on sender_rank CPU)

    recvSTL(&avg_stencil_radii, my_rank, sender_rank); // Average distances (possibly stencil radii)
    recvSTL(&max_stencil_radii, my_rank, sender_rank); // Average distances (possibly stencil radii)
    recvSTL(&min_stencil_radii, my_rank, sender_rank); // Average distances (possibly stencil radii)

    recvSTL(&loc_to_glob, my_rank, sender_rank); // l2g
    recvSTL(&glob_to_loc, my_rank, sender_rank); // g2l

    recvSTL(&O_by_rank, my_rank, sender_rank); // Subsets of O that this DomainNoMPI will send out to each other DomainNoMPI
    recvSTL(&R_by_rank, my_rank, sender_rank); 
    recvSTL(&boundary_indices, my_rank, sender_rank);
    recvSTL(&max_st_size, my_rank, sender_rank);  

    this->nb_nodes = node_list.size();

    set_union(Q.begin(), Q.end(), R.begin(), R.end(), inserter(G, G.end()));
    this->comm_size = _comm_size; 
    cout << "RANK " << my_rank << " of " << comm_size << " REPORTS: received DomainNoMPI object" << endl;
    return 0;           // FIXME: return bytes sent (in case we need to monitor this)
}
#endif

void DomainNoMPI::printVerboseDependencyGraph() {
    // Verify we everything passed correctly.
    cout << "********************* Stencil dependency on R *********************" << endl;
    printStencilNodesIn(stencil_map, R, "R"); // Reveal stencils dependent on R
    cout << "********************* Node Membership (LOCAL NODES) *********************" << endl;
    printCenterMemberships(G, "G ");

#if 0
    printVector(avg_stencil_radii, "Q_AVG_DISTS");
    for (int i = 0; i < stencil_map.size(); i++) {
        printStencil(stencil_map[i], "Q_STENCIL: ");
    }
    printCenters(node_list, "G_CENTERS");
#else 
    std::cout << "-----------------------------\n";
    std::cout << "See stencils_" << max_st_size << "_" << this->getFilename() 
        << " for average stencil indices\n";
    std::cout << "-----------------------------\n";
    std::cout << "See avg_radii_" << this->getFilename() << " for average stencil radii\n"; 
    std::cout << "-----------------------------\n";
#endif
}

// Append to O_by_rank (find what subset of O is needed by rank subdomain_rank)
void DomainNoMPI::fill_O_by_rank(std::set<int>& subdomain_R, int subdomain_rank) {
    set<int>::iterator qit;

    if (O_by_rank.size() == 0) {
       O_by_rank.resize(comm_size);
    }

    // subdomain_R contains the R for one subdomain. 
    // We take the intersection to find what elements of R on that subdomain
    // are sent by this processor and use the builtin STL inserter for efficiency
    printf("before set_intersection\n");
    printf("subdomain_R size: %d\n", subdomain_R.size());
    printf("this_O size: %d\n", this->O.size());
    printf("rank: %d\n", subdomain_rank);
    printf("O_by_rank size: %d\n", O_by_rank.size());
    set_intersection(subdomain_R.begin(), subdomain_R.end(), this->O.begin(), this->O.end(), inserter(O_by_rank[subdomain_rank], O_by_rank[subdomain_rank].end())); 
    printf("after set_intersection\n");
    return;
}

// Append to R_by_rank (find what subset of O is needed by rank subdomain_rank)
void DomainNoMPI::fill_R_by_rank(std::set<int>& subdomain_O, int subdomain_rank) {
    set<int>::iterator qit;
    if (R_by_rank.size() == 0) {
        R_by_rank.resize(comm_size);
    }
    // subdomain_O contains the O for one subdomain. 
    // We take the intersection to find what elements of R on that subdomain
    // are sent by processor indicated by subdomain_rank and use the builtin
    // STL inserter for efficiency
    set_intersection(subdomain_O.begin(), subdomain_O.end(), this->R.begin(), this->R.end(), inserter(R_by_rank[subdomain_rank], R_by_rank[subdomain_rank].end())); 
    return;
}

// TODO: remove boundary parameter
// NOTE: should iterate over ALL rbf_centers
void DomainNoMPI::fillCenterSets(vector<NodeType>& rbf_centers, vector<StencilType>& stencils) {
    //********************************
    //  STENCIL MEMBERSHIP SETS
    //********************************
    // NEED TO FILL THESE:
    // std::set<int> Q;			// All stencil centers in this CPUs QUEUE
    // std::set<int> D;			// Set of stencil centers DEPEND on nodes in R before evaluation
    // std::set<int> O;			// Centers that are OUTPUT to other Domains
    // std::set<int> B; 		// Centers on BOUNDARY (in O and D or both)
    // std::set<int> QmD; 		// Interior centers (computed without RECEIVING)
    // std::set<int> QmB; 		// Interior centers without dependence on R
    // std::set<int> BmO; 		// 
    // std::set<int> R;			// Nodes REQUIRED from other Domains.
    //
    set<int>::iterator qit;

    printf("... enter fillCenterSets\n");
    printf("DomainNoMPI %d, xmin/max= %f, %f, ymin/max= %f, %f, zmin/max= %f, %f (Checking %dD)\n", id, xmin, xmax,
            ymin, ymax, zmin, zmax, dim_num);
    //    printf("NB_NODES: %d\n", rbf_centers.size());
    //   printf("Q_NODES: %d\n", Q.size());
    //   printf("NB_STENCILS: %d\n", stencils.size());

#if 1
// TODO: 
// This adds all nodes to Q, then gets all nodes associated with stencils in Q
// the only thing is that rbf_centers[i] => stencils[i]. We need rbf_centers[i] => stencils[l2g(rbf_centers[i])]


    // Generate sets Q and D
    // Q represents number of nodes in the domain. The {Qs} form a partition of the entire domain
    printf("fillCenterSets, nb rbf_centers: %d\n", rbf_centers.size());
    for (unsigned int i = 0; i < rbf_centers.size(); i++) {
        NodeType& pt = rbf_centers[i];
        if (this->isInsideSubdomain(pt, i)) {
            Q.insert(i);
        }
    } 
    //printf("****** Q size: %d *****\n", Q.size());
    int depR = 0; 
    for (qit = Q.begin(); qit != Q.end(); qit++) {
        StencilType& st = stencils[*qit]; 

        // Now, if the center is in Q but it depends on nodes in R then we need to distinguish
        depR = 0;
        //        std::cout << "begin stencil" << std::endl;
        for (unsigned int j = 0; j < st.size(); j++) { // Check all nodes in corresponding stencil
            unsigned int indx = st[j];
            NodeType& pt2 = rbf_centers[indx];
            //            std::cout << *qit << ": " << pt2 << "==>"<< this->isInsideSubdomain(pt2) << std::endl;
            // If any stencil node is outside the domain, then set this to true
            if (!this->isInsideSubdomain(pt2, j)) {
                depR = 1;  
            }
        }
        //        std::cout << "end stencil" << std::endl;
        if (depR) {
            D.insert(*qit);
        }
    }
#endif 


    std::cout << "Q size before set operations: " << Q.size() << std::endl;
    std::cout << "D size before set operations: " << D.size() << std::endl;

    // Each Q (a set) is, by construction, sorted

    //Create set of stencil points of all elements of Q (ineffecient since there are repeats)
    set<int> SQ; 
    std::cout << "SQ.size before difference: " << SQ.size() << std::endl;
    stencilSet(Q, stencils, SQ);
    std::cout << "SQ.size before difference: " << SQ.size() << std::endl;

    // Set of nodes from stencils that are not in Q (i.e. not on the DomainNoMPI)
    // compute set R = S(Q) \ Q
    std::cout << "R.size before difference: " << R.size() << std::endl;
    set_difference(SQ.begin(), SQ.end(), Q.begin(), Q.end(), inserter(R, R.end()));

    std::cout << "R.size after difference: " << R.size() << std::endl;
    // Determine set O = pts in Q that are in stencils on other Domains
    // O: S(A\Q) \ (A\Q) = S(A\Q) intersect Q

    // ALL STENCILS "set A"
    std::set<int> A;
    for (unsigned int i = 0; i < rbf_centers.size(); i++) {
        A.insert((int)i);
    }
    std::cout << "A.size after difference: " << A.size() << std::endl;

    // A\Q
    set<int> AmQ;
    set_difference(A.begin(), A.end(), Q.begin(), Q.end(), inserter(AmQ,
                AmQ.end()));

    std::cout << "AmQ.size after difference: " << AmQ.size() << std::endl;
    // S(A\Q)
    set<int> SAmQ;
    std::cout << "SAmQ.size before stencilSet: " << SAmQ.size() << std::endl;
    stencilSet(AmQ, stencils, SAmQ);
    std::cout << "SAmQ.size after stencilSet: " << SAmQ.size() << std::endl;

    std::cout << "O.size before stencilSet: " << O.size() << std::endl;
    // O = S(A\Q) intersect Q
    set_intersection(SAmQ.begin(), SAmQ.end(), Q.begin(), Q.end(), inserter(O,O.end()));
    std::cout << "O.size after stencilSet: " << O.size() << std::endl;

    // (QmD D R) = (Q\B B\O O R)  (NOTE: O and D overlap and are NOT guaranteed to be the same, so we maintain B\O
    //
    // QmD contains all nodes we can operate on without RECEIVING 
    // D   contains all nodes we must wait to operate on after RECEIVING R
    // R   contains all nodes we receive
    // 
    // NOTE: by using QmD we can operate on stencils in kernels that do not break to RECEIVE

    //      SENDING can happen without kernels breaking, so we dont need to worry about that.
    // B = O U D, this used to be B=O U, But B\O is NOT_GUARANTEED_EQUAL to Dep, so we
    // do an additional subtraction to guarantee that B\O == D
    //printf("... before set_union\n");
    set_union(D.begin(), D.end(), O.begin(), O.end(), inserter(B, B.end()));
    //printf("... after set_union\n");
#if 0
    // D* = B\O (so we can arrange G = { Q\B O\D* D* R }
    // By ordering nodes this way, we have the OPTION of operating on {Q\B O\D} 
    // while communication happens for O and R. However, if we are on the GPU, O\D STILL require
    // GPU to CPU transfer before it can be completed. That is why we concentrate kernels on 
    // Q\B and B.
    set_difference(O.begin(), O.end(), D.begin(), D.end(), inserter(OmD, OmD.end()));
#endif 
    // QmD != Q\B
    set_difference(Q.begin(), Q.end(), D.begin(), D.end(), inserter(QmD, QmD.end()));
    set_difference(Q.begin(), Q.end(), B.begin(), B.end(), inserter(QmB, QmB.end()));
    set_difference(B.begin(), B.end(), O.begin(), O.end(), inserter(BmO, BmO.end()));

    set_union(Q.begin(), Q.end(), R.begin(), R.end(), inserter(G, G.end()));

    printf("Q size= %d\n", (int) Q.size());
    printf("O.size= %d\n", (int) O.size());
    printf("D.size= %d\n", (int) D.size());
    //    printf("OmD.size= %d\n", (int) OmD.size());
    printf("B.size= %d\n", (int) B.size());
    printf("QmD.size= %d\n", (int) QmD.size());
    printf("QmB.size= %d\n", (int) QmB.size());
    printf("BmO.size= %d\n", (int) BmO.size());
    printf("R.size= %d\n", (int) R.size());
    printf("G.size = %d\n", (int) G.size());
}

//----------------------------------------------------------------------
void DomainNoMPI::fillLocalData(vector<NodeType>& rbf_centers, vector<StencilType>& stencil, vector<unsigned int>& boundary, vector<double>& avg_dist, vector<double>& max_dist, vector<double>& min_dist) {

    // Generate stencil membership lists (i.e., which set each stencil center belongs to)
    //printf("just entered fillLocalData\n"); // Not entering for last subdomain (7)
    //{
        //printf("enter fillLocalData, before fillCenterSets, nb rbf_centers: %d\n", rbf_centers.size()); 
        //return;
    //}

    this->fillCenterSets(rbf_centers, stencil);
    //printf("after fillCenterSets, nb rbf_centers: %d\n", rbf_centers.size()); return;
    //printf("... enter fillLocalData\n"); return;

    //******************************** 
    // GEN MAPPINGS local/global and full stencil sets based on membership.
    //********************************
    set<int>::iterator qit;
    int i = 0;

    // generate local to global map.
    // Index of local map corresponds to position in G (list of all centers). 
    // The local map elements map G[i] back to global domain indices

    // We want these maps in order: (Q\B B\O O R) where B=(D O)
    // to make it more convenient when we work on memory management
    for (qit = QmB.begin(); qit != QmB.end(); qit++, i++) {
        //printf("*qit= %d\n", *qit);
        loc_to_glob.push_back(*qit);
        node_list.push_back(rbf_centers[*qit]); // In order to compute we need the physical locations of all function values
        stencil_map.push_back(stencil[*qit]); // We also need to push the connectivity to evaluate stencils
        avg_stencil_radii.push_back(avg_dist[*qit]);
        max_stencil_radii.push_back(max_dist[*qit]);
        min_stencil_radii.push_back(min_dist[*qit]);
    }
    for (qit = BmO.begin(); qit != BmO.end(); qit++, i++) {
        loc_to_glob.push_back(*qit);
        node_list.push_back(rbf_centers[*qit]);
        stencil_map.push_back(stencil[*qit]); // in global coordinates?
        avg_stencil_radii.push_back(avg_dist[*qit]);
        max_stencil_radii.push_back(max_dist[*qit]);
        min_stencil_radii.push_back(min_dist[*qit]);
    }
    for (qit = O.begin(); qit != O.end(); qit++, i++) {
        loc_to_glob.push_back(*qit);
        node_list.push_back(rbf_centers[*qit]);
        stencil_map.push_back(stencil[*qit]);
        avg_stencil_radii.push_back(avg_dist[*qit]);
        max_stencil_radii.push_back(max_dist[*qit]);
        min_stencil_radii.push_back(min_dist[*qit]);
    }
    for (qit = R.begin(); qit != R.end(); qit++, i++) {
        loc_to_glob.push_back(*qit);
        node_list.push_back(rbf_centers[*qit]); // Assume non-moving node problem so we can store positions at initialization
        // HOWEVER, NO CONNECTIVITY REQUIRED FOR R (THESE ARE ON OTHER CPUs)
        avg_stencil_radii.push_back(avg_dist[*qit]);
        max_stencil_radii.push_back(max_dist[*qit]);
        min_stencil_radii.push_back(min_dist[*qit]);
    }

#if 0
    printf("*** inside fillLocalData\n");
    for (int i=0; i < 5; i++) {
        for (int j=0; j < 5; j++) {
           printf("(dom) stencil[%d][%d]= %d\n", i, j, stencil[i][j]);
           printf("  (subdom) stencil_map[%d][%d]= %d\n", i, j, stencil_map[i][j]);
        }
    }
#endif


    printf("*** diff of sets = %d\n", G.size()-R.size()-BmO.size()-O.size()-QmB.size());

    // global to local map
    for (int i = 0; i < (int)loc_to_glob.size(); i++) {
        // We offset the global local indices by 1. Since the map returns 0 for
        // anything not in the key list, we can subtract 1 from any return
        // index and see -1 for elements not in list and the true local index
        // for all others
        glob_to_loc[l2g(i)] = i + 1;
    }

    std::cout << "DomainNoMPI stencils size = " << stencil_map.size() << std::endl;
    // Convert all stencils to local indexing.
    printf("0\n");
    printf("stencil map size: %d\n", stencil_map.size());
    for (int i = 0; i < (int)stencil_map.size(); i++) {
        int sz = stencil_map[i].size();
        for (int j = 0; j < sz; j++) {
            stencil_map[i][j] = g2l(stencil_map[i][j]); // in local coordinates
        }
        std::sort(&stencil_map[i][0], &stencil_map[i][sz]); // GE
    }

    // create an R_row array: R_row[i] points to the first element that is in R
    // if there is no element of R on that row, R_row[i] points to the first element
    // of the next row

    // only works for stencils of constant size
    printf("*** stencil map.size = %d\n", stencil_map.size());
    printf("*** Q size: %d\n", Q.size());
    printf("*** R size: %d\n", R.size());
    //for (set<int>::iterator qit=R.begin(); qit != R.end(); qit++) {
        //printf("R= glob %d, loc %d\n", *qit, g2l(*qit)); // work ok
    //}
    //if (R.size() != 0) exit(0); // REMOVE WHEN DONE
    int qrows = Q.size();
    int count = 0;
    int countn = 0;
    Qbeg_rows.resize(stencil_map.size());
    Qend_rows.resize(stencil_map.size());
    std::fill(Qend_rows.begin(), Qend_rows.end(), -1);
    for (int i=0; i < (int) stencil_map.size(); i++) {
        Qbeg_rows[i] = stencil_map[i].size()*i;
        //Qend_rows[i] = -1;
        for (int j=0; j < stencil_map[i].size(); j++) {
            int s = stencil_map[i][j];
            if (s >= Q.size()) {
                Qend_rows[i] = Qbeg_rows[i] + j;
                break;
            }
        }
        if (Qend_rows[i] == -1) {
            Qend_rows[i] = Qbeg_rows[i] + stencil_map[i].size();
            count++;
        } /* else {
            if (countn++ > 10) continue;
            printf("\nstencil row %d, beg: %d, end: %d: \n", i, Qbeg_rows[i], Qend_rows[i]);
            for (int j=0; j < (int) stencil_map[i].size(); j++) {
                printf("%d,", stencil_map[i][j]-Q.size()); // negative if in Q, else positive
            }
        }
        */
    }
    printf("\n");
    printf("*** nb rows without elements in R: %d\n", count);

#if 0
    for (int i=0; i < (int) stencil_map.size(); i++) {
    for (int i=0; i < 10; i++) {
        printf("stencil row %d: \n", i, Qbeg_rows[i], Qend_rows[i]);
        for (int j=0; j < (int) stencil_map[i].size(); j++) {
            printf("%d,", stencil_map[i][j]);
        }
        printf("\n");
    }
#endif

    // This forms the boundary set (needed)

    printf("MASTER BOUNDARY.size= %d\n", (int) boundary.size());

    set_intersection(Q.begin(), Q.end(), boundary.begin(), boundary.end(),
            inserter(boundary_indices, boundary_indices.end()));
    printf("SUBDOMAIN BOUNDARY.size= %d\n", (int)boundary_indices.size());
    for (int i = 0; i < (int)boundary_indices.size(); i++) {
        //EVAN: 
#if 0
        cout << "Subdomain Adding Boundary Node[" << i << "] = " << boundary_indices[i]
            << " (local index: " << g2l(boundary_indices[i]) << ")"
            << endl;
#endif 
        boundary_indices[i] = g2l(boundary_indices[i]);
    }
    printf("GLOBAL_BOUNDARY.size= %d\n", (int) boundary_indices.size());

    // Finally: update the number of known nodes: 
    nb_nodes = node_list.size();


    printf("l2g size= %d\n", (int) loc_to_glob.size());
    printf("g2l size= %d\n", (int) glob_to_loc.size());
    printf("node_list size= %d (nb_nodes=%d)\n", (int) node_list.size(), (int) nb_nodes);
    printf("avg_stencil_radii size= %d\n", (int) avg_stencil_radii.size());
    printf("max_stencil_radii size= %d\n", (int) max_stencil_radii.size());
    printf("min_stencil_radii size= %d\n", (int) min_stencil_radii.size());
}
//----------------------------------------------------------------------
void DomainNoMPI::stencilSet(set<int>& s, vector<StencilType>& stencil, set<int>& Sset) {
    //set<int>* Sset = new set<int> ;
    set<int>::iterator qit;

    //set<int>::iterator qit;
    for (qit = s.begin(); qit != s.end(); qit++) {
        int qi = *qit;
        StencilType& si = stencil[qi];
        //        std::cout << "Working on stencil: " << qi << std::endl;
        for (unsigned int j = 0; j < si.size(); j++) {
            //         std::cout << si[j] << " "; 
            Sset.insert(si[j]);
        }
        //       std::cout << std::endl;
    }
    //return *Sset;
}

//----------------------------------------------------------------------
void DomainNoMPI::printStencilNodesIn(const vector<StencilType>& stencils, const set<int>& center_set, std::string display_char) {
    for (int i = 0; i < (int)stencils.size(); i++) {
        cout << "Stencil[local:" << i << " (global: " << l2g(i) << ")] = ";
        for (int j = 0; j < (int)stencils[i].size(); j++) {
            cout << " ";
            if (isInSet(l2g(stencils[i][j]), center_set)) {
                cout << display_char;
            } else {
                cout << '.';
            }
        }
        cout << endl;
    }
}

// HERE: center_set is in local index
void DomainNoMPI::printCenterMemberships(const set<int>& center_set, std::string display_name) {
    // Center[ ID ] =     [Q|.]  [D|.]  [O|.]  [R][+]
    // NOTE: [a|b] --> if (true) then a else b. 
    // 	ID --> global node id
    // CONDITIONS: 
    //  Q  --> in set Q? 
    //  D  --> in set D? 
    //  O  --> in set O? 
    //  R  --> depends on nodes in R? 
    //  +  --> is the center in R?
    cout << "\t" << display_name
        << "[ global_index | local_index ] = \t[Q|.]   [D|.]   [Q|.]   [R][+]   [B|.]"
        << endl;
    cout << "\tCONDITIONS: " << endl;
    cout << "\t\tQ  --> in set Q?" << endl;
    cout << "\t\tD  --> in set D?" << endl;
    cout << "\t\tO  --> in set O?" << endl;
    cout << "\t\tR  --> depends on set R (i.e., nodes in other subdomains)?" << endl;
    cout << "\t\t+  --> is the node in R (i.e., inside another subdomain)?" << endl;
    cout << "\t\tB*  --> is the node on the global Boundary?" << endl;
    cout << "\tGaps in global indices are indicated with [... GAP ...]" << endl;
    cout << "\t-------------------------------------------------" << endl;
    int i = 0;
    int j = 0; 
    for (set<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++, i++, j++) {
        if (j != *setiter) {
            //            cout << "\tGAP\t------------------------------------------" << std::endl;
            cout << "[... GAP ...]" << std::endl;
            j = *setiter; 
        }

        cout << i << "\t" << display_name 
            << "[ global:" << (*setiter)
            << " | local:" << g2l(*setiter) 
            << " ] =\t\t";

        if (isInSet(*setiter, Q)) {
            cout << "Q";
        } else {
            cout << ".";
        }
        cout << "   ";
        if (isInSet(*setiter, D)) {
            cout << "D";
        } else {
            cout << ".";
        }
        cout << "   ";
        if (isInSet(*setiter, O)) {
            cout << "O";
        } else {
            cout << ".";
        }
        cout << "   ";//<< stencil_map.size() << ":" << g2l(*setiter);
        if (dependsOnSet(*setiter, R)) {
            cout << "R";
        } else {
            cout << ".";
        }
        // IF A NODE IS IN SET R WE SHOULD MARK IT FOR NOTIFICATION
        if (isInSet(*setiter, R)) {
            cout << "+";
        }
        cout << "   ";
        if (isInVector(g2l(*setiter), this->boundary_indices)) {
            cout << "B*";
        } else {
            cout << ".";
        }
        cout << endl;
    }
}

bool DomainNoMPI::isInSet(const int center, const set<int>& center_set) const {
    //bool inSet = false;
    for (set<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++) {
        // True -> stencil[i][j] is in center set
        if (center == *setiter) {
            return true;
        }
    }
    return false;
}

bool DomainNoMPI::isInVector(const unsigned int center, const vector<unsigned int>& center_set) const {
    //bool inSet = false;
    for (vector<unsigned int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++) {
        // True -> stencil[i][j] is in center set
        if (center == *setiter) {
            return true;
        }
    }
    return false;
}

bool DomainNoMPI::dependsOnSet(const int local_stencil_id, const set<int>& center_set) {
    // stencil_map are in local indices
    if (local_stencil_id >= (int)stencil_map.size()) {
        return true;
    }
    StencilType& stencil = stencil_map[local_stencil_id];
    for (int i = 0; i < (int)stencil.size(); i++) {
        if (isInSet(stencil[i], center_set)) {
            return true;
        } // short circuit return
    }
    return false;
}

void DomainNoMPI::printSetL2G(const set<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    for (set<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++) {
        // True -> stencil[i][j] is in center set
        int i = *setiter;
        cout << "\tindx:" << i << " (l2g: " << l2g(i) << ")" << endl;
    }
    cout << "}" << endl;
}


void DomainNoMPI::printSetG2L(const set<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    for (set<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++) {
        // True -> stencil[i][j] is in center set
        int i = *setiter;
        cout << "\t" << i /*<< " (l2g: " << l2g(i) << ")"*/ << " (g2l: " << g2l(i) << ")" << endl;
    }
    cout << "}" << endl;
}

void DomainNoMPI::printVector(const vector<double>& stencil_radii, std::string set_label) {
    cout << set_label << " = {" << endl;
    int i = 0;
    for (vector<double>::const_iterator setiter = stencil_radii.begin(); setiter
            != stencil_radii.end(); setiter++, i++) {
        // True -> stencil[i][j] is in center set
        if (loc_to_glob.size() > 0) {
            cout << "\t[" << i << " (global:" << l2g(i) << ")] = " << *setiter
                << endl;
        } else {
            cout << "\t[" << i << " (" << i << ")] = " << *setiter << endl;
        }
    }
    cout << "}" << endl;
}

void DomainNoMPI::printVector(const vector<unsigned int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    int i = 0;
    for (vector<unsigned int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++, i++) {
        // True -> stencil[i][j] is in center set
        if (loc_to_glob.size() > 0) {
            cout << "\t[" << i << " (" << loc_to_glob[i] << ")] = " << *setiter
                << endl;
        } else {
            cout << "\t[" << i << " (" << i << ")] = " << *setiter << endl;
        }
    }
    cout << "}" << endl;
}

void DomainNoMPI::printVectorL2G(const vector<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    unsigned int i = 0;
    for (vector<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++, i++) {
        // True -> stencil[i][j] is in center set
        cout << "\t" << *setiter << " (l2g=" << l2g(*setiter) << ") " << endl;
    }
    cout << "}" << endl;
}
void DomainNoMPI::printVectorG2L(const vector<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    unsigned int i = 0;
    for (vector<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++, i++) {
        // True -> stencil[i][j] is in center set
        cout << "\t" << *setiter << " (g2l=" << g2l(*setiter) << ") " << endl;
    }
    cout << "}" << endl;
}

void DomainNoMPI::printCenters(const std::vector<NodeType>& centers, std::string center_label) {
    cout << center_label << " = {" << endl;
    int i = 0;
    for (vector<NodeType>::const_iterator setiter = centers.begin(); setiter
            != centers.end(); setiter++, i++) {
        // True -> stencil[i][j] is in center set
        cout << "\t" << i;
        if (loc_to_glob.size() > 0) { // We MIGHT be in original code if this is 0
            cout << "  (" << loc_to_glob[i] << ")";
        } else {
            cout << "  (" << i << ")";
        }
        cout << "\t[" << (*setiter).x() << ", " << (*setiter).y() << ", "
            << (*setiter).z() << "]" << endl;
    }
    cout << "}" << endl;
}

void DomainNoMPI::printStencil(const StencilType& stencil, std::string stencil_label) {
    cout << stencil_label << " = " << "\t";
    int i = 0;
    int index_sum = 0; 
    if (loc_to_glob.size() > 0) {
        for (StencilType::const_iterator setiter = stencil.begin(); setiter
                != stencil.end(); setiter++, i++) {
            // True -> stencil[i][j] is in center set
            cout << " [" << *setiter << " (" << loc_to_glob[*setiter] << ")] ";
            index_sum += *setiter; 
        }
    } else { // WE MIGHT BE IN THE ORIGINAL CODE
        for (StencilType::const_iterator setiter = stencil.begin(); setiter
                != stencil.end(); setiter++, i++) {
            // True -> stencil[i][j] is in center set
            cout << " [" << *setiter << " (" << *setiter << ")] ";
            index_sum += *setiter; 
        }
    }
    cout << " SUM: " << index_sum;
    cout << endl;
}

void DomainNoMPI::printStencilPlus(const StencilType& stencil, const std::vector<
        double>& function_values, std::string stencil_label) {
    cout << stencil_label << " = " << "\t";
    int i = 0;
    if (loc_to_glob.size() > 0) {
        for (StencilType::const_iterator setiter = stencil.begin(); setiter
                != stencil.end(); setiter++, i++) {
            // True -> stencil[i][j] is in center set
            cout << " [" << *setiter << " (" << loc_to_glob[*setiter] << ")] {"
                << function_values[*setiter] << "} ";
        }
    } else { // WE MIGHT BE IN THE ORIGINAL CODE
        for (StencilType::const_iterator setiter = stencil.begin(); setiter
                != stencil.end(); setiter++, i++) {
            // True -> stencil[i][j] is in center set
            cout << " [" << *setiter << " (" << *setiter << ")] {"
                << function_values[*setiter] << "} ";
        }
    }
    cout << endl;
}

//----------------------------------------------------------------------

void DomainNoMPI::writeG2LToFile(std::string filename) {

    std::string fname = "g2lmap_"; 
    fname.append(filename); 
    std::ofstream fout(fname.c_str()); 
    if (fout.is_open()) {
        std::map<int, int>::iterator mit;  
        for (mit = glob_to_loc.begin(); mit != glob_to_loc.end(); mit++) {
            // Subtract 1 because all indices are offset by 1. When an element
            // doesnt exist its mapped to 0. By subtracting 1 off everything we
            // get -1 when an index is not in the map 
            fout << (*mit).first << " " << (*mit).second - 1 << std::endl; 
        }
    } else {
        printf("Error opening file to write\n"); 
        exit(EXIT_FAILURE); 
    }
    fout.close();
    std::cout << "[DomainNoMPI] \tWrote " << glob_to_loc.size() << " global to local index map elements to \t" << fname << std::endl;

}

//----------------------------------------------------------------------

void DomainNoMPI::writeL2GToFile(std::string filename) {

    std::string fname = "l2gmap_"; 
    fname.append(filename); 
    std::ofstream fout(fname.c_str()); 
    if (fout.is_open()) {
        std::vector<int>::iterator mit;  
        int i = 0; 
        for (mit = loc_to_glob.begin(); mit != loc_to_glob.end(); mit++, i++) {
            fout << i << " " << (*mit) << std::endl; 
        }
    } else {
        printf("Error opening file to write\n"); 
        exit(EXIT_FAILURE); 
    }
    fout.close();
    std::cout << "[DomainNoMPI] \tWrote " << loc_to_glob.size() << " local to global index map elements to \t" << fname << std::endl;

}
//----------------------------------------------------------------------
void DomainNoMPI::writeToEllpackBinaryFile(std::string filename, std::vector<DomainNoMPI*>& subdomains)
{
    // write several subdomains to a single file
    // top of the file is a list of the characteristics of all subdomains. 
    // Only store col_id file (rowise). 

    printf("write to Ellpack binary file\n");
    FILE *fd;
    fd = fopen(filename.c_str(), "w");
    fprintf(fd, "#nb_rows  nb_nonzeros, extended domain size (for each subdomain)\n");
    fprintf(fd, "%d   # number of subdomains\n", subdomains.size());

    for (int i=0; i < subdomains.size(); i++) {
        DomainNoMPI& domain = *subdomains[i];
        std::vector<StencilType>& stencil = domain.getStencils();
        int nb_rows = stencil.size();
        int nb_nonzeros = stencil[0].size(); // assume all stencils have the same size
        fprintf(fd, "%d %d %d\n", nb_rows, nb_nonzeros, domain.G.size());
    }

    for (int i=0; i < subdomains.size(); i++) {
        DomainNoMPI& dom = *subdomains[i];
        writeToEllpackBinaryFile(fd, dom);
    }

    fclose(fd);
}
//----------------------------------------------------------------------
void DomainNoMPI::writeToEllpackBinaryFile(FILE* fd, DomainNoMPI& domain)
{
    // write stencil array: where are the non-zeros. 
    // nb rows: stencil.size()
    // nb of nonzeros: stencil[0].size()
    //
        std::vector<StencilType>& stencils = domain.getStencils();
        printf("*** writeToEllpack: stencil size: %d\n", stencils.size());
        int nb_rows = stencils.size();
        int nb_nonzeros = stencils[0].size(); // assume all stencils have the same size

        std::vector<int> col_id(nb_rows*nb_nonzeros);
        //std::string ell_filename = "ell_" + filename;
        //printf("1 write to %s\n", ell_filename.c_str());
        //printf("nb_rows= %d\n", nb_rows);
        //printf("nb_nonzeros= %d\n", nb_nonzeros);
        //printf("Extended domain size= %d\n", G.size());

    for (int i=0; i < 5; i++) {
        for (int j=0; j < 5; j++) {
           printf("(writeToEll) stencil[%d][%d]= %d\n", i, j, stencils[i][j]);
        }
    }

        for (int i = 0; i < nb_rows; i++) {
            for (int j = 0; j < nb_nonzeros; j++) {
                col_id[j+i*nb_nonzeros] = stencils[i][j];
                //if (j < 5 and i < 5) printf("stencils[%d][%d]= %d\n", i, j, stencils[i][j]);
                //printf("stencils[%d][%d]= %d\n", i, j, stencils[i][j]);
            }
        }

        // number of rows of col_id should be the number of derivatives to be be computed
        // The vector of values (not dealt with here), is the number of indices in the G set
        // which is the Q set (the nodes in the subdomain) + all nodes required for the stencil
        // So I must encode the domain size in the header of the Ellpack file. 

        //FILE *fd;
        //fd = fopen(ell_filename.c_str(), "w");
        //fprintf(fd, "#nb_rows  nb_nonzeros, extended domain size\n");
        //fprintf(fd, "%d %d %d\n", nb_rows, nb_nonzeros, G.size());
        fwrite(&col_id[0], sizeof(int), col_id.size(), fd);
        // just in case the number of nonzeros per row is not constant (for future perhaps)
        fwrite(&Qbeg_rows[0], sizeof(int), Qbeg_rows.size(), fd);
        fwrite(&Qend_rows[0], sizeof(int), Qend_rows.size(), fd);
        //fclose(fd);
}
//----------------------------------------------------------------------
