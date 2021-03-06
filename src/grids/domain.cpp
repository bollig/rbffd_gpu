#define REMOVE_SOLUTION_FROM_DOMAIN

#include "utils/comm/communicator.h"
#include "domain.h"

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>

//#include <set>

using namespace std;
//using namespace boost;


Domain::Domain(const Domain& subdomain) {
    cout << "INSIDE COPY CONSTRUCTOR" << endl;
    exit(EXIT_FAILURE);
}

Domain::Domain(int dimension, Grid* grid, int _comm_size)
    :
        dim_num(dimension),
        id(0), comm_size(_comm_size),
        inclMX(true),inclMY(true),inclMZ(true)
{
    xmin = grid->xmin;
    xmax = grid->xmax;
    ymin = grid->ymin;
    ymax = grid->ymax;
    zmin = grid->zmin;
    zmax = grid->zmax;

    // We might need to know how many nodes are in the domain globally for things like Hyperviscosity
    this->global_num_nodes = grid->getNodeListSize();

    // Generate stencil membership lists (i.e., which set each stencil center belongs to)
    this->fillCenterSets(grid->getNodeList(), grid->getStencils());
    std::cout << "INSIDE ODDBALL\n";

    // Forms sets (Q,O,R) and l2g/g2l maps
    fillLocalData(grid->getNodeList(), grid->getStencils(), grid->getBoundaryIndices(), grid->getStencilRadii(), grid->getMaxStencilRadii(), grid->getMinStencilRadii());
    this->max_st_size = grid->getMaxStencilSize();
}


// Construct a new Domain object
Domain::Domain(int dimension, unsigned int global_nb_nodes,
        double _xmin, double _xmax, double _ymin, double _ymax, double _zmin, double _zmax,
        int _comm_rank, int _comm_size) :
    dim_num(dimension),
    id(_comm_rank), comm_size(_comm_size),
    inclMX(false),inclMY(false),inclMZ(false)
{

    // These are props of Grid inherited by Domain
    global_num_nodes = global_nb_nodes;
    xmin = _xmin;
    xmax = _xmax;
    ymin = _ymin;
    ymax = _ymax;
    zmin = _zmin;
    zmax = _zmax;
}


void Domain::generateDecomposition(std::vector<Domain*>& subdomains, int x_divisions, int y_divisions, int z_divisions)
{
    int gx = x_divisions;
    int gy = y_divisions;
    int gz = z_divisions;
    //    std::vector<Domain*> subdomains;
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
        subdomains[id] = new Domain(dim_num, global_num_nodes, left_bound, right_bound, bottom_bound, top_bound, front_bound, back_bound, id, comm_size);
        subdomains[id]->setMaxStencilSize(this->max_st_size);
        subdomains[id]->setInclusiveMaxBoundary(igx == gx-1, igy == gy-1, igz == gz-1);
    }

    printf("nb subdomains: %d\n", (int) subdomains.size());
    for (unsigned int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** CPU %d ***************** \n", i);
        // Generate stencil membership lists (i.e., which set each stencil center belongs to)
        subdomains[i]->fillCenterSets(this->node_list, this->stencil_map);
    }

    for (unsigned int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** FILLING R_by_rank for CPU%d ***************** \n", i);
        for (unsigned int j = 0; j < subdomains.size(); j++) {
            subdomains[i]->fill_R_by_rank(subdomains[j]->O, j);
        }
    }

    for (unsigned int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** Local Ordering %d ***************** \n", i);
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



    //return subdomains;
}



int Domain::send(int my_rank, int receiver_rank) {

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

    sendSTL(&O_by_rank, my_rank, receiver_rank); // Subsets of O that this Domain will send out to each other Domain
    sendSTL(&R_by_rank, my_rank, receiver_rank); // Subsets of R that this Domain will receive from every other Domain
    sendSTL(&boundary_indices, my_rank, receiver_rank);
    sendSTL(&max_st_size, my_rank, receiver_rank);

    cout << "RANK " << my_rank << " REPORTS: sent Domain object" << endl;
    return 0;           // FIXME: return bytes sent (in case we need to monitor this)
}

int Domain::receive(int my_rank, int sender_rank, int _comm_size) {

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

    recvSTL(&O_by_rank, my_rank, sender_rank); // Subsets of O that this Domain will send out to each other Domain
    recvSTL(&R_by_rank, my_rank, sender_rank);
    recvSTL(&boundary_indices, my_rank, sender_rank);
    recvSTL(&max_st_size, my_rank, sender_rank);

    this->nb_nodes = node_list.size();

    set_union(Q.begin(), Q.end(), R.begin(), R.end(), inserter(G, G.end()));
    this->comm_size = _comm_size;
    cout << "RANK " << my_rank << " of " << comm_size << " REPORTS: received Domain object" << endl;
    return 0;           // FIXME: return bytes sent (in case we need to monitor this)
}

void Domain::printVerboseDependencyGraph() {
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
void Domain::fill_O_by_rank(std::set<int>& subdomain_R, int subdomain_rank) {
    set<int>::iterator qit;

    if (O_by_rank.size() == 0) {
       O_by_rank.resize(comm_size);
    }

    // subdomain_R contains the R for one subdomain.
    // We take the intersection to find what elements of R on that subdomain
    // are sent by this processor and use the builtin STL inserter for efficiency
    set_intersection(subdomain_R.begin(), subdomain_R.end(), this->O.begin(), this->O.end(), inserter(O_by_rank[subdomain_rank], O_by_rank[subdomain_rank].end()));
    return;
}

// Append to R_by_rank (find what subset of O is needed by rank subdomain_rank)
void Domain::fill_R_by_rank(std::set<int>& subdomain_O, int subdomain_rank) {
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
void Domain::fillCenterSets(vector<NodeType>& rbf_centers, vector<StencilType>& stencils) {
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

    printf("Domain %d, xmin/max= %f, %f, ymin/max= %f, %f, zmin/max= %f, %f (Checking %dD)\n", id, xmin, xmax,
            ymin, ymax, zmin, zmax, dim_num);
    //    printf("NB_NODES: %d\n", rbf_centers.size());
    //   printf("Q_NODES: %d\n", Q.size());
    //   printf("NB_STENCILS: %d\n", stencils.size());

#if 1
// TODO:
// This adds all nodes to Q, then gets all nodes associated with stencils in Q
// the only thing is that rbf_centers[i] => stencils[i]. We need rbf_centers[i] => stencils[l2g(rbf_centers[i])]


    // Generate sets Q and D
    for (unsigned int i = 0; i < rbf_centers.size(); i++) {
        NodeType& pt = rbf_centers[i];
        if (this->isInsideSubdomain(pt, i)) {
            // Insert the global index
            Q.insert(i);
        }
    }
    int depR = 0;
    for (qit = Q.begin(); qit != Q.end(); qit++) {
        StencilType& st = stencils[*qit];

        // Now, if the center is in Q but it depends on nodes in R then we need to distinguish
        depR = 0;
        //        std::cout << "begin stencil" << std::endl;
        for (unsigned int j = 1; j < st.size(); j++) { // Check all nodes in corresponding stencil
            unsigned int indx = st[j];
            NodeType& pt2 = rbf_centers[indx];
            // std::cout << *qit << ": " << pt2 << "==>"<< this->isInsideSubdomain(pt2, indx) << std::endl;
            // If any stencil node is outside the domain, then set this to true
            if (!isInsideSubdomain(pt2, indx)) {
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

    // Set of nodes from stencils that are not in Q (i.e. not on the Domain)
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
    set_union(D.begin(), D.end(), O.begin(), O.end(), inserter(B, B.end()));
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

    printf("QmB.size= %d\n", (int) QmB.size());
    this->QmB_size = QmB.size();
    printf("BmO.size= %d\n", (int) BmO.size());
    this->BmO_size = BmO.size();
    printf("O.size= %d\n", (int) O.size());
    this->O_size = O.size();
    printf("R.size= %d\n", (int) R.size());
    this->R_size = R.size();

    printf("G.size = %d\n", (int) G.size());
    this->G_size = G.size();
    printf("Q size= %d\n", (int) Q.size());
    this->Q_size = Q.size();
    printf("QmD.size= %d\n", (int) QmD.size());
    this->QmD_size = QmD.size();
    printf("D.size= %d\n", (int) D.size());
    this->D_size = D.size();
    printf("B.size= %d\n", (int) B.size());
    this->B_size = B.size();
    //    printf("OmD.size= %d\n", (int) OmD.size());
}

//----------------------------------------------------------------------
void Domain::fillLocalData(vector<NodeType>& rbf_centers, vector<StencilType>& stencil, vector<unsigned int>& boundary, vector<double>& avg_dist, vector<double>& max_dist, vector<double>& min_dist) {


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
        stencil_map.push_back(stencil[*qit]);
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
#if 1
    if (R.size() && !isRfull) {
        std::cout << "Missing R_by_rank info: " << isRfull << "\n";
        exit(-1);
    }
    vector<int>::iterator vit;
    for (unsigned int ri = 0; ri < R_by_rank.size(); ri++) {
        if (R_by_rank[ri].size()) {
            for (vit = R_by_rank[ri].begin(); vit != R_by_rank[ri].end(); vit++) {
                loc_to_glob.push_back(*vit);
                node_list.push_back(rbf_centers[*vit]); // Assume non-moving node problem so we can store positions at initialization
                // HOWEVER, NO CONNECTIVITY REQUIRED FOR R (THESE ARE ON OTHER CPUs)
                avg_stencil_radii.push_back(avg_dist[*vit]);
                max_stencil_radii.push_back(max_dist[*vit]);
                min_stencil_radii.push_back(min_dist[*vit]);
            }
        }
    }
#else
    for (qit = R.begin(); qit != R.end(); qit++, i++) {
        loc_to_glob.push_back(*qit);
        node_list.push_back(rbf_centers[*qit]); // Assume non-moving node problem so we can store positions at initialization
        // HOWEVER, NO CONNECTIVITY REQUIRED FOR R (THESE ARE ON OTHER CPUs)
        avg_stencil_radii.push_back(avg_dist[*qit]);
        max_stencil_radii.push_back(max_dist[*qit]);
        min_stencil_radii.push_back(min_dist[*qit]);
    }
#endif

    // global to local map
    for (int i = 0; i < (int)loc_to_glob.size(); i++) {
        // We offset the global local indices by 1. Since the map returns 0 for
        // anything not in the key list, we can subtract 1 from any return
        // index and see -1 for elements not in list and the true local index
        // for all others
        glob_to_loc[l2g(i)] = i + 1;
    }

    std::cout << "Domain stencils size = " << stencil_map.size() << std::endl;
    // Convert all stencils to local indexing.
    for (int i = 0; i < (int)stencil_map.size(); i++) {
        for (int j = 0; j < (int)stencil_map[i].size(); j++) {
            stencil_map[i][j] = g2l(stencil_map[i][j]);
        }
    }

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
void Domain::stencilSet(set<int>& s, vector<StencilType>& stencil, set<int>& Sset) {
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


void Domain::printStencilNodesIn(const vector<StencilType>& stencils, const set<int>& center_set, std::string display_char) {
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
void Domain::printCenterMemberships(const set<int>& center_set, std::string display_name) {
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

bool Domain::isInSet(const int center, const set<int>& center_set) const {
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

bool Domain::isInVector(const unsigned int center, const vector<unsigned int>& center_set) const {
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

bool Domain::dependsOnSet(const int local_stencil_id, const set<int>& center_set) {
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

void Domain::printSetL2G(const set<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    for (set<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++) {
        // True -> stencil[i][j] is in center set
        int i = *setiter;
        cout << "\tindx:" << i << " (l2g: " << l2g(i) << ")" << endl;
    }
    cout << "}" << endl;
}


void Domain::printSetG2L(const set<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    for (set<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++) {
        // True -> stencil[i][j] is in center set
        int i = *setiter;
        cout << "\t" << i /*<< " (l2g: " << l2g(i) << ")"*/ << " (g2l: " << g2l(i) << ")" << endl;
    }
    cout << "}" << endl;
}

void Domain::printVector(const vector<double>& stencil_radii, std::string set_label) {
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

void Domain::printVector(const vector<unsigned int>& center_set, std::string set_label) {
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

void Domain::printVectorL2G(const vector<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    unsigned int i = 0;
    for (vector<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++, i++) {
        // True -> stencil[i][j] is in center set
        cout << "\t" << *setiter << " (l2g=" << l2g(*setiter) << ") " << endl;
    }
    cout << "}" << endl;
}
void Domain::printVectorG2L(const vector<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    unsigned int i = 0;
    for (vector<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++, i++) {
        // True -> stencil[i][j] is in center set
        cout << "\t" << *setiter << " (g2l=" << g2l(*setiter) << ") " << endl;
    }
    cout << "}" << endl;
}

void Domain::printCenters(const std::vector<NodeType>& centers, std::string center_label) {
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

void Domain::printStencil(const StencilType& stencil, std::string stencil_label) {
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

void Domain::printStencilPlus(const StencilType& stencil, const std::vector<
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

void Domain::writeG2LToFile(std::string filename) {

    std::string fname = "g2lmap_";
    fname.append(filename);
    std::ofstream fout(fname.c_str());
    if (fout.is_open()) {
        std::map<int, int>::iterator mit;
        for (mit = glob_to_loc.begin(); mit != glob_to_loc.end(); mit++) {
            // NOTE: the map is already offset by -1 so that all the elements
            // that have no index map to -1
            fout << (*mit).first << " " << (*mit).second << std::endl;
        }
    } else {
        printf("Error opening file to write\n");
        exit(EXIT_FAILURE);
    }
    fout.close();
    std::cout << "[Domain] \tWrote " << glob_to_loc.size() << " global to local index map elements to \t" << fname << std::endl;

}

//----------------------------------------------------------------------

void Domain::writeL2GToFile(std::string filename) {

    std::string fname = "l2gmap_";
    fname.append(filename);
    std::ofstream fout(fname.c_str());
    if (fout.is_open()) {
        fout << G_size << " " << Q_size << " " << B_size << std::endl;
        fout << QmB_size << " " << BmO_size << " " << O_size << " " << R_size << std::endl;
        fout << QmD_size << " " << D_size << " " << R_size << std::endl;

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
    std::cout << "[Domain] \tWrote " << loc_to_glob.size() << " local to global index map elements to \t" << fname << std::endl;

}

//----------------------------------------------------------------------

void Domain::writeOMapToFile(std::string filename) {

    std::string fname = "O_by_rank_";
    fname.append(filename);
    std::ofstream fout(fname.c_str());
    int obr_count = 0;
    if (fout.is_open()) {
        std::vector< std::vector<int> >::iterator mit;
        std::vector<int>::iterator nit;
        int i = 0;
        for (mit = O_by_rank.begin(); mit != O_by_rank.end(); mit++, i++) {
            for (nit = (*mit).begin(); nit != (*mit).end(); nit++, obr_count++) {
                // We write the <rank> <node_indx> to file.
                fout << i << " " << (*nit) << std::endl;
            }
        }
    } else {
        printf("Error opening file to write\n");
        exit(EXIT_FAILURE);
    }
    fout.close();
    std::cout << "[Domain] \tWrote " << obr_count << " O_by_rank map elements to \t" << fname << std::endl;


}

//----------------------------------------------------------------------

void Domain::writeRMapToFile(std::string filename) {

    std::string fname = "R_by_rank_";
    fname.append(filename);
    std::ofstream fout(fname.c_str());
    int rbr_count = 0;
    if (fout.is_open()) {
        std::vector< std::vector<int> >::iterator mit;
        std::vector<int>::iterator nit;
        int i = 0;
        for (mit = R_by_rank.begin(); mit != R_by_rank.end(); mit++, i++) {
            for (nit = (*mit).begin(); nit != (*mit).end(); nit++, rbr_count++) {
                // We write the <rank> <node_indx> to file.
                fout << i << " " << (*nit) << std::endl;
            }
        }
    } else {
        printf("Error opening file to write\n");
        exit(EXIT_FAILURE);
    }
    fout.close();
    std::cout << "[Domain] \tWrote " << rbr_count << " R_by_rank map elements to \t" << fname << std::endl;

}

//----------------------------------------------------------------------

Grid::GridLoadErrType Domain::loadG2LFromFile(std::string filename) {
	std::string fname = "g2lmap_";
	fname.append(filename);
	std::cout << "[" << this->className() << "] \treading global to local (g2lmap) file: " << fname << std::endl;

	std::ifstream fin;
	fin.open(fname.c_str());

	if (fin.is_open()) {
		glob_to_loc.clear();
		while (fin.good()) {
			unsigned int glob_indx, loc_indx;
			fin >> glob_indx >> loc_indx;
			if (!fin.eof()) {
				glob_to_loc[glob_indx] = loc_indx;
			}
		}
	} else {
		printf("Error opening g2lmap file to read\n");
		return NO_EXTRA_FILES;
	}

	fin.close();

	std::cout << "[" << this->className() << "] \tLoaded " << glob_to_loc.size() << " global to local mappings from \t" << fname << std::endl;
	return GRID_AND_STENCILS_LOADED;
}

//----------------------------------------------------------------------

Grid::GridLoadErrType Domain::loadL2GFromFile(std::string filename) {
	std::string fname = "l2gmap_";
	fname.append(filename);
	std::cout << "[" << this->className() << "] \treading local to global (l2gmap) file: " << fname << std::endl;

	std::ifstream fin;
	fin.open(fname.c_str());

	if (fin.is_open()) {
        fin >> G_size >> Q_size >> B_size;
        fin >> QmB_size >> BmO_size >> O_size >> R_size;
        fin >> QmD_size >> D_size >> R_size;

        // TODO:
        // So we can load the file and know that the nodes are ordered QmB BmO O
        // and R. Then we also know the nodes are ordered: QmD D R so we can
        // refill each of the vectors/sets as we read in files

        this->G.clear();
        this->Q.clear();
        this->B.clear();
        this->QmB.clear();
        this->BmO.clear();
        this->O.clear();
        this->R.clear();
        this->QmD.clear();
        this->D.clear();

        int kk = 0;
		loc_to_glob.clear();
		while (fin.good()) {
			unsigned int glob_indx, loc_indx;
			fin >> loc_indx >> glob_indx;
			if (!fin.eof()) {
				loc_to_glob.push_back(glob_indx);

                // Populate index sets (for legacy code)
                G.insert(glob_indx);
                if (kk < Q_size) {
                    Q.insert(glob_indx);
                }
                // QmB BmO O R
                if (kk < QmB_size) {
                    QmB.insert(glob_indx);
                } else {
                    if (kk < QmB_size + BmO_size) {
                        BmO.insert(glob_indx);
                    } else {
                        if (kk < QmB_size + BmO_size + O_size) {
                            O.insert(glob_indx);
                        } else {
                            R.insert(glob_indx);
                        }
                    }
                    // B = BmO + O
                    if (kk < QmB_size + BmO_size + O_size) {
                        B.insert(glob_indx);
                    }
                }
                // QmD D R
                if (kk < QmD_size) {
                    QmD.insert(glob_indx);
                } else {
                    if (kk < QmD_size + D_size) {
                        D.insert(glob_indx);
                    }
                }
			}
		}
	} else {
		printf("Error opening l2gmap file to read\n");
		return NO_EXTRA_FILES;
	}

	fin.close();

	std::cout << "[" << this->className() << "] \tLoaded " << loc_to_glob.size() << " global to local mappings from \t" << fname << std::endl;
	return GRID_AND_STENCILS_LOADED;
}

//----------------------------------------------------------------------

Grid::GridLoadErrType Domain::loadOMapFromFile(std::string filename) {
	std::string fname = "O_by_rank_";
	fname.append(filename);
	std::cout << "[" << this->className() << "] \treading O_by_rank map file: " << fname << std::endl;

	std::ifstream fin;
	fin.open(fname.c_str());
    O_by_rank.resize(comm_size);

    for (int i = 0 ; i < comm_size; i++) {
        O_by_rank[i].clear();
    }

    int obr_count = 0;

	if (fin.is_open()) {
		while (fin.good()) {
			unsigned int O_indx, lrank;
			fin >> lrank >> O_indx;
			if (!fin.eof()) {
				O_by_rank[lrank].push_back(O_indx);
                obr_count++;
			}
		}
	} else {
		printf("Error opening O_by_rank file to read\n");
		return NO_EXTRA_FILES;
	}

	fin.close();

	std::cout << "[" << this->className() << "] \tLoaded " << obr_count << " mappings from \t" << fname << std::endl;
	return GRID_AND_STENCILS_LOADED;
}

//----------------------------------------------------------------------

Grid::GridLoadErrType Domain::loadRMapFromFile(std::string filename) {
	std::string fname = "R_by_rank_";
	fname.append(filename);
	std::cout << "[" << this->className() << "] \treading R_by_rank map file: " << fname << std::endl;

	std::ifstream fin;
	fin.open(fname.c_str());
    R_by_rank.resize(comm_size);

    for (int i = 0 ; i < comm_size; i++) {
        R_by_rank[i].clear();
    }

    int obr_count = 0;

	if (fin.is_open()) {
		while (fin.good()) {
			unsigned int R_indx, lrank;
			fin >> lrank >> R_indx;
			if (!fin.eof()) {
				R_by_rank[lrank].push_back(R_indx);
                obr_count++;
			}
		}
	} else {
		printf("Error opening R_by_rank file to read\n");
		return NO_EXTRA_FILES;
	}

	fin.close();

	std::cout << "[" << this->className() << "] \tLoaded " << obr_count << " mappings from \t" << fname << std::endl;
	return GRID_AND_STENCILS_LOADED;
}



//----------------------------------------------------------------------

void Domain::writeStencilsToFile(std::string filename) {
	if (max_st_size > 0) {
#if 1
		std::ostringstream prefix;
		prefix << "stencils_maxsz" << this->max_st_size << "_";

		std::string fname = prefix.str();
		fname.append(filename);
#endif
		std::cout << "[" << this->className() << "] \t writing stencils to file: " << fname << std::endl;

		std::ofstream fout(fname.c_str());

		if (fout.is_open()) {
			for (unsigned int i = 0; i < stencil_map.size(); i++) {
				fout << stencil_map[i].size();
				for (unsigned int j=0; j < stencil_map[i].size(); j++) {
					fout << " " << stencil_map[i][j];
				}
				fout << std::endl;
			}
		} else {
			printf("Error opening file to write\n");
			exit(EXIT_FAILURE);
		}
		fout.close();
		std::cout << "[" << this->className() << "] \tWrote " << stencil_map.size() << " stencils to \t" << fname << std::endl;
	} else {
		std::cout << "[" << this->className() << "] \tMax stencil size not set. No stencils to write to disk" << std::endl;
	}
}

//----------------------------------------------------------------------
