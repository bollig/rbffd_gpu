#define REMOVE_SOLUTION_FROM_DOMAIN

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "domain.h"

//#include <set>
#include "utils/comm/communicator.h"

using namespace std;
//using namespace boost;


Domain::Domain(const Domain& subdomain) {
    cout << "INSIDE COPY CONSTRUCTOR" << endl;
    exit(EXIT_FAILURE);
}

    Domain::Domain(Grid* grid, int _comm_size)
: comm_size(_comm_size), id(0), 
    xmin(grid->xmin), xmax(grid->xmax),
    ymin(grid->ymin), ymax(grid->ymax),
    zmin(grid->zmin), zmax(grid->zmax)
{
    // Forms sets (Q,O,R) and l2g/g2l maps
    fillLocalData(grid->getNodeList(), grid->getStencils(), grid->getBoundaryIndices(), grid->getStencilRadii()); 
    fillVarData(grid->getNodeList());
    this->max_st_size = grid->getMaxStencilSize();
}


// Construct a new Domain object
Domain::Domain(double _xmin, double _xmax, double _ymin, double _ymax, double _zmin, double _zmax, 
        int _comm_rank, int _comm_size) :
    xmin(_xmin), xmax(_xmax), ymin(_ymin), ymax(_ymax), zmin(_zmin), zmax(_zmax),
    id(_comm_rank), comm_size(_comm_size) {

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
        int igz = id / gx*gy;
        // 2) Find the row within the slice
        int igy = (id - igz * (gx*gy)) / gx;
        // 3) Find the column within the row
        int igx = (id - igz * (gx*gy)) - igy * gx;

        //        printf("igx = %d, igy = %d, igz = %d\n", igx, igy, igz);
        double xm = xmin + igx * deltax;
        double ym = ymin + igy * deltay;
        double zm = zmin + igz * deltaz;
        printf("Subdomain[%d (%d of %d)] Extents = (%f, %f) x (%f, %f) x (%f, %f)\n",id, id+1, comm_size, xm, xm+deltax, ym, ym+deltay, zm, zm+deltaz);
        printf("Tile (ix, iy, iz) = (%d, %d, %d)\n", igx, igy, igz); 
        subdomains[id] = new Domain(xm, xm + deltax, ym, ym + deltay,  zm, zm + deltaz, id, comm_size);
        subdomains[id]->setMaxStencilSize(this->max_st_size);
    }

    // Figure out the sets Bi, Oi Qi

    printf("nb subdomains: %d\n", (int) subdomains.size());
    for (int i = 0; i < subdomains.size(); i++) {
        printf("\n ***************** CPU %d ***************** \n", i);
        // Forms sets (Q,O,R) and l2g/g2l maps
        subdomains[i]->fillLocalData( this->node_list, this->stencil_map, this->boundary_indices, this->avg_stencil_radii); 
        subdomains[i]->fillVarData(this->node_list); // Sets function values in U
    }

    for (int i = 0; i < subdomains.size(); i++) {
        printf(
                "\n ***************** FILLING O_by_rank for CPU%d ***************** \n",
                i);
        for (int j = 0; j < subdomains.size(); j++) {
            subdomains[i]->fillDependencyList(subdomains[j]->R, j); // appends to O_by_rank any nodes required by subdomain[j]
        }
    }

    //return subdomains;
}



int Domain::send(int my_rank, int receiver_rank) {

    sendSTL(&id, my_rank, receiver_rank);  

    double buff[6] = { xmin, xmax, ymin, ymax, zmin, zmax };
    MPI_Send(&buff, 6, MPI::DOUBLE, receiver_rank, TAG, MPI_COMM_WORLD);

    sendSTL(&Q, my_rank, receiver_rank); // All stencil centers in this CPUs QUEUE
    sendSTL(&D, my_rank, receiver_rank); // Set of stencil centers DEPEND on nodes in R before evaluation
    sendSTL(&O, my_rank, receiver_rank); // Centers that are OUTPUT to other Domains
    sendSTL(&B, my_rank, receiver_rank); // Centers on BOUNDARY (in O and D or both)
    sendSTL(&QmB, my_rank, receiver_rank); // Interior centers (computed without communication)
    sendSTL(&R, my_rank, receiver_rank); // Nodes REQUIRED from other Domains.

    sendSTL(&stencil_map, my_rank, receiver_rank); // All stencils receiver_rank is responsible for

    sendSTL(&node_list, my_rank, receiver_rank); // Q_centers + R_centers = node_list (all nodes necessary on receiver_rank CPU)

#ifndef REMOVE_SOLUTION_FROM_DOMAIN
    sendSTL(&U_G, my_rank, receiver_rank); // Initial function values of all centers in G
#endif 
    sendSTL(&avg_stencil_radii, my_rank, receiver_rank); // Average distances (possibly stencil radii)

    sendSTL(&loc_to_glob, my_rank, receiver_rank); // l2g
    sendSTL(&glob_to_loc, my_rank, receiver_rank); // g2l

    sendSTL(&O_by_rank, my_rank, receiver_rank); // Subsets of O that this Domain will send out to each other Domain
    sendSTL(&boundary_indices, my_rank, receiver_rank);
    sendSTL(&max_st_size, my_rank, receiver_rank);  

    cout << "RANK " << my_rank << " REPORTS: sent Domain object" << endl;
}

int Domain::receive(int my_rank, int sender_rank) {

    MPI_Status stat;

    // Start by identifying the subdomain ID
    recvSTL(&id, my_rank, sender_rank);  

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
    recvSTL(&QmB, my_rank, sender_rank); // Interior centers (computed without communication)
    recvSTL(&R, my_rank, sender_rank); // Nodes REQUIRED from other Domains.

    recvSTL(&stencil_map, my_rank, sender_rank); // All stencils sender_rank is responsible for

    recvSTL(&node_list, my_rank, sender_rank); // Q_centers + R_centers = node_list (all nodes necessary on sender_rank CPU)

#ifndef REMOVE_SOLUTION_FROM_DOMAIN
    recvSTL(&U_G, my_rank, sender_rank); // Initial Function values of all centers in G
#endif
    recvSTL(&avg_stencil_radii, my_rank, sender_rank); // Average distances (possibly stencil radii)

    recvSTL(&loc_to_glob, my_rank, sender_rank); // l2g
    recvSTL(&glob_to_loc, my_rank, sender_rank); // g2l

    recvSTL(&O_by_rank, my_rank, sender_rank); // Subsets of O that this Domain will send out to each other Domain
    recvSTL(&boundary_indices, my_rank, sender_rank);
    recvSTL(&max_st_size, my_rank, sender_rank);  

    this->nb_nodes = node_list.size();

    // cout << "EVAN YOURE WRONG HERE!" <<endl;
    set_union(Q.begin(), Q.end(), R.begin(), R.end(), inserter(G, G.end()));

    cout << "RANK " << my_rank << " REPORTS: received Domain object" << endl;
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
void Domain::fillDependencyList(std::set<int>& subdomain_R, int subdomain_rank) {
    set<int>::iterator qit;
    int i = 0;

    if (O_by_rank.size() == 0) {
        cout << "RESIZING O_by_rank" << endl;
        O_by_rank.resize(comm_size);
        printSet(this->O, "Original O");
    }

    for (qit = subdomain_R.begin(); qit != subdomain_R.end(); qit++, i++) {
        std::cout << "R[" << i << "] = " << *qit << std::endl;
        if (isInSet(*qit, this->O)) {
            this->O_by_rank[subdomain_rank].push_back(*qit);
        }
    }
    char label[256];
    sprintf(label, "Rank %d O_by_rank[%d]", id, subdomain_rank);
    printVector(this->O_by_rank[subdomain_rank], label);

    return;
}

// TODO: remove boundary parameter
void Domain::fillCenterSets(vector<NodeType>& rbf_centers, vector<StencilType>& stencils) {
    //********************************
    //  STENCIL MEMBERSHIP SETS
    //********************************
    // NEED TO FILL THESE:
    // std::set<int> Q;			// All stencil centers in this CPUs QUEUE
    // std::set<int> D;			// Set of stencil centers DEPEND on nodes in R before evaluation
    // std::set<int> O;			// Centers that are OUTPUT to other Domains
    // std::set<int> B; 		// Centers on BOUNDARY (in O and D or both)
    // std::set<int> QmB; 		// Interior centers (computed without communication)
    // std::set<int> R;			// Nodes REQUIRED from other Domains.
    //
    set<int>::iterator qit;

    printf("Domain %d, xmin/max= %f, %f, ymin/max= %f, %f, zmin/max= %f, %f\n", id, xmin, xmax,
            ymin, ymax, zmin, zmax);

    // Generate sets Q and D
    for (int i = 0; i < rbf_centers.size(); i++) {
        NodeType& pt = rbf_centers[i];
        if (this->isInsideSubdomain(pt))
            continue; // Do not add to Q or D
        // If we dont continue then it is a center in Q.
        Q.insert(i);

        // Now, if the center is in Q but it depends on nodes in R then we need to distinguish
        bool depR = false;
        for (int j = 0; j < stencils[i].size(); j++) { // Check all nodes in corresponding stencil
            NodeType& pt2 = rbf_centers[stencils[i][j]];
            if (this->isInsideSubdomain(pt2)) {
                depR = true;
            }
        }
        if (depR) {
            D.insert(i);
        }
    }

    // Each Q (a set) is, by construction, sorted

    //Create set of stencil points of all elements of Q (ineffecient since there are repeats)
    set<int>& SQ = stencilSet(Q, stencils);

    // Set of nodes from stencils that are not in Q (i.e. not on the Domain)
    // compute set R = S(Q) \ Q
    set_difference(SQ.begin(), SQ.end(), Q.begin(), Q.end(), inserter(R,
                R.end()));

    // Determine set O = pts in Q that are in stencils on other Domains
    // O: S(A\Q) \ (A\Q) = S(A\Q) intersect Q

    // ALL STENCILS "set A"
    vector<int> A;
    for (int i = 0; i < rbf_centers.size(); i++) {
        A.push_back(i);
    }

    // A\Q
    set<int> AmQ;
    set_difference(A.begin(), A.end(), Q.begin(), Q.end(), inserter(AmQ,
                AmQ.end()));

    // S(A\Q)
    set<int>& SAmQ = stencilSet(AmQ, stencils);
    // O = S(A\Q) intersect Q
    set_intersection(SAmQ.begin(), SAmQ.end(), Q.begin(), Q.end(), inserter(O,
                O.end()));

    // B = O U D, But B\O is NOT_EQUAL to D
    set_union(D.begin(), D.end(), O.begin(), O.end(), inserter(B, B.end()));

    // QmB = Q\B
    set_difference(Q.begin(), Q.end(), B.begin(), B.end(), inserter(QmB,
                QmB.end()));

    set_union(Q.begin(), Q.end(), R.begin(), R.end(), inserter(G, G.end()));

    printf("Q size= %d\n", (int) Q.size());
    printf("O.size= %d\n", (int) O.size());
    printf("D.size= %d\n", (int) D.size());
    printf("B.size= %d\n", (int) B.size());
    printf("QmB.size= %d\n", (int) QmB.size());
    printf("R.size= %d\n", (int) R.size());
    printf("G.size = %d\n", (int) G.size());

    delete &SAmQ;
    delete &SQ;
}

//----------------------------------------------------------------------
void Domain::fillLocalData(vector<NodeType>& rbf_centers, vector<StencilType>& stencil, vector<size_t>& boundary, vector<double>& avg_dist) {
    // Generate stencil membership lists (i.e., which set each stencil center belongs to)
    this->fillCenterSets(rbf_centers, stencil);

    //******************************** 
    // GEN MAPPINGS local/global and full stencil sets based on membership.
    //********************************
    set<int>::iterator qit;
    int i = 0;

    // generate local to global map.
    // Index of local map corresponds to position in G (list of all centers). 
    // The local map elements map G[i] back to global domain indices

    // We want these maps in order: (Q\B B R) 
    // to make it more convenient when we work on memory management
    for (qit = QmB.begin(); qit != QmB.end(); qit++, i++) {
        loc_to_glob.push_back(*qit);
        node_list.push_back(rbf_centers[*qit]); // In order to compute we need the physical locations of all function values
        stencil_map.push_back(stencil[*qit]); // We also need to push the connectivity to evaluate stencils
        avg_stencil_radii.push_back(avg_dist[*qit]);
    }
    for (qit = B.begin(); qit != B.end(); qit++, i++) {
        loc_to_glob.push_back(*qit);
        node_list.push_back(rbf_centers[*qit]);
        stencil_map.push_back(stencil[*qit]);
        avg_stencil_radii.push_back(avg_dist[*qit]);
    }
    for (qit = R.begin(); qit != R.end(); qit++, i++) {
        loc_to_glob.push_back(*qit);
        node_list.push_back(rbf_centers[*qit]); // Assume non-moving node problem so we can store positions at initialization
        // HOWEVER, NO CONNECTIVITY REQUIRED FOR R (THESE ARE ON OTHER CPUs)
        avg_stencil_radii.push_back(avg_dist[*qit]);
    }

    // global to local map
    for (int i = 0; i < loc_to_glob.size(); i++) {
        glob_to_loc[l2g(i)] = i;
    }

    // Convert all stencils to local indexing.
    for (int i = 0; i < stencil_map.size(); i++) {
        for (int j = 0; j < stencil_map[i].size(); j++) {
            stencil_map[i][j] = g2l(stencil_map[i][j]);
        }
    }

    // This forms the boundary set (needed)

    printf("MASTER BOUNDARY.size= %d\n", (int) boundary.size());
    set_intersection(Q.begin(), Q.end(), boundary.begin(), boundary.end(),
            inserter(boundary_indices, boundary_indices.end()));
    printf("SUBDOMAIN BOUNDARY.size= %d\n", (int)boundary_indices.size());
    for (int i = 0; i < boundary_indices.size(); i++) {
        //EVAN: 
#if 0
        cout << "Subdomain Adding Boundary Node[" << i << "] = " << boundary_indices[i]
            << " (local index: " << g2l(boundary_indices[i]) << ")"
            << endl;
#endif 
        boundary_indices[i] = g2l(boundary_indices[i]);
    }
    //printf("GLOBAL_BOUNDARY.size= %d\n", (int) boundary_indices.size());

    // Finally: update the number of known nodes: 
    nb_nodes = node_list.size();


    printf("l2g size= %d\n", (int) loc_to_glob.size());
    printf("g2l size= %d\n", (int) glob_to_loc.size());
    printf("node_list size= %d (nb_nodes=%d)\n", (int) node_list.size(), (int) nb_nodes);
    printf("avg_stencil_radii size= %d\n", (int) avg_stencil_radii.size());
}
//----------------------------------------------------------------------
set<int>& Domain::stencilSet(set<int>& s, vector<StencilType>& stencil) {
    set<int>* Sset = new set<int> ;
    set<int>::iterator qit;

    //set<int>::iterator qit;
    for (qit = s.begin(); qit != s.end(); qit++) {
        int qi = *qit;
        StencilType& si = stencil[qi];

        for (int j = 0; j < si.size(); j++) {
            Sset->insert(si[j]);
        }
    }

    return *Sset;
}
//----------------------------------------------------------------------
void Domain::fillVarData(vector<NodeType>& rbf_centers) {
#ifndef REMOVE_SOLUTION_FROM_DOMAIN
    // Initial condition: linear data: x + 2 y + 3 z
    // NOTE: the loops here should be structured similar to Domain::fillLocalData
    // to ensure our l2g and g2l mappings apply correctly.
    set<int>::iterator qit;

    for (qit = QmB.begin(); qit != QmB.end(); qit++) {
        NodeType& v = rbf_centers[*qit];
        double s = *qit; //v.x() + 2.*v.y() + 3.*v.z();
        U_G.push_back(s);
    }

    for (qit = B.begin(); qit != B.end(); qit++) {
        NodeType& v = rbf_centers[*qit];
        double s = *qit;//v.x() + 2.*v.y() + 3.*v.z();
        U_G.push_back(s);
    }

    // Initial R values (these will be updated at each iteration)
    for (qit = R.begin(); qit != R.end(); qit++) {
        NodeType& v = rbf_centers[*qit];
        double s = *qit;//v.x() + 2.*v.y() + 3.*v.z();
        U_G.push_back(s);
    }
#endif 
}
//----------------------------------------------------------------------


void Domain::printStencilNodesIn(const vector<StencilType>& stencils, const set<int>& center_set, std::string display_char) {
    for (int i = 0; i < stencils.size(); i++) {
        cout << "Stencil[local:" << i << " (global: " << l2g(i) << ")] = ";
        for (int j = 0; j < stencils[i].size(); j++) {
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

bool Domain::isInVector(const size_t center, const vector<size_t>& center_set) const {
    //bool inSet = false;
    for (vector<size_t>::const_iterator setiter = center_set.begin(); setiter
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
    if (local_stencil_id >= stencil_map.size()) {
        return true;
    }
    StencilType& stencil = stencil_map[local_stencil_id];
    for (int i = 0; i < stencil.size(); i++) {
        if (isInSet(stencil[i], center_set)) {
            return true;
        } // short circuit return
    }
    return false;
}

void Domain::printSet(const set<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    for (set<int>::const_iterator setiter = center_set.begin(); setiter
            != center_set.end(); setiter++) {
        // True -> stencil[i][j] is in center set
        int i = *setiter;
        cout << "\t" << i << " (global:" << g2l(i) << ")" << endl;
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

void Domain::printVector(const vector<int>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    int i = 0;
    for (vector<int>::const_iterator setiter = center_set.begin(); setiter
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


void Domain::printVector(const vector<size_t>& center_set, std::string set_label) {
    cout << set_label << " = {" << endl;
    size_t i = 0;
    for (vector<size_t>::const_iterator setiter = center_set.begin(); setiter
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
    std::cout << "[Domain] \tWrote " << glob_to_loc.size() << " local to global index map elements to \t" << fname << std::endl;

}



//----------------------------------------------------------------------
