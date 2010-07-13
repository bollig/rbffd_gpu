#include <algorithm>
#include <iostream>
#include <fstream>
#include "gpu.h"

//#include <set>
#include "communicator.h"

using namespace std;
//using namespace boost;


GPU::GPU(const GPU& subdomain) {
	cout << "INSIDE COPY CONSTRUCTOR" << endl;
	exit(EXIT_FAILURE);

}

// Construct a new GPU object
GPU::GPU(double _xmin, double _xmax, double _ymin, double _ymax, double _zmin, double _zmax, double _dt,
		int _comm_rank, int _comm_size) :
	xmin(_xmin), xmax(_xmax), ymin(_ymin), ymax(_ymax), zmin(_zmin), zmax(_zmax), dt(_dt),
			id(_comm_rank), comm_size(_comm_size) {

}

int GPU::send(int my_rank, int receiver_rank) {
	double buff[7] = { xmin, xmax, ymin, ymax, zmin, zmax, dt };

	MPI_Send(&buff, 7, MPI::DOUBLE, receiver_rank, TAG, MPI_COMM_WORLD);

	sendSTL(&Q, my_rank, receiver_rank); // All stencil centers in this CPUs QUEUE
	sendSTL(&D, my_rank, receiver_rank); // Set of stencil centers DEPEND on nodes in R before evaluation
	sendSTL(&O, my_rank, receiver_rank); // Centers that are OUTPUT to other GPUs
	sendSTL(&B, my_rank, receiver_rank); // Centers on BOUNDARY (in O and D or both)
	sendSTL(&QmB, my_rank, receiver_rank); // Interior centers (computed without communication)
	sendSTL(&R, my_rank, receiver_rank); // Nodes REQUIRED from other GPUs.

	sendSTL(&Q_stencils, my_rank, receiver_rank); // All stencils receiver_rank is responsible for

	sendSTL(&G_centers, my_rank, receiver_rank); // Q_centers + R_centers = G_centers (all nodes necessary on receiver_rank CPU)

	sendSTL(&U_G, my_rank, receiver_rank); // Initial function values of all centers in G
	sendSTL(&Q_avg_dists, my_rank, receiver_rank); // Average distances (possibly stencil radii)

	sendSTL(&loc_to_glob, my_rank, receiver_rank); // l2g
	sendSTL(&globmap, my_rank, receiver_rank); // g2l

	sendSTL(&O_by_rank, my_rank, receiver_rank); // Subsets of O that this GPU will send out to each other GPU
	sendSTL(&global_boundary_nodes, my_rank, receiver_rank);

	cout << "RANK " << my_rank << " REPORTS: sent GPU object" << endl;
}

int GPU::receive(int my_rank, int sender_rank) {

	MPI_Status stat;
	
	// Get the subdomain bounds and dt
	double buff[7];
	MPI_Recv(&buff, 7, MPI::DOUBLE, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	xmin = buff[0];
	xmax = buff[1];
	ymin = buff[2];
	ymax = buff[3];
	zmin = buff[4]; 
	zmax = buff[5];
	dt = buff[6];

	recvSTL(&Q, my_rank, sender_rank); // All stencil centers in this CPUs QUEUE
	recvSTL(&D, my_rank, sender_rank); // Set of stencil centers DEPEND on nodes in R before evaluation
	recvSTL(&O, my_rank, sender_rank); // Centers that are OUTPUT to other GPUs
	recvSTL(&B, my_rank, sender_rank); // Centers on BOUNDARY (in O and D or both)
	recvSTL(&QmB, my_rank, sender_rank); // Interior centers (computed without communication)
	recvSTL(&R, my_rank, sender_rank); // Nodes REQUIRED from other GPUs.

	recvSTL(&Q_stencils, my_rank, sender_rank); // All stencils sender_rank is responsible for

	recvSTL(&G_centers, my_rank, sender_rank); // Q_centers + R_centers = G_centers (all nodes necessary on sender_rank CPU)

	recvSTL(&U_G, my_rank, sender_rank); // Initial Function values of all centers in G
	recvSTL(&Q_avg_dists, my_rank, sender_rank); // Average distances (possibly stencil radii)

	recvSTL(&loc_to_glob, my_rank, sender_rank); // l2g
	recvSTL(&globmap, my_rank, sender_rank); // g2l

	recvSTL(&O_by_rank, my_rank, sender_rank); // Subsets of O that this GPU will send out to each other GPU
	recvSTL(&global_boundary_nodes, my_rank, sender_rank);

        // cout << "EVAN YOURE WRONG HERE!" <<endl;
	set_union(Q.begin(), Q.end(), R.begin(), R.end(), inserter(G, G.end()));

	// Verify we everything passed correctly.
	cout
			<< "********************* Stencil dependency on R *********************"
			<< endl;
	printStencilNodesIn(Q_stencils, R, "R"); // Reveal stencils dependent on R
	cout
			<< "********************* Node Membership (LOCAL NODES) *********************"
			<< endl;
	printCenterMemberships(G, "G ");

	printVector(Q_avg_dists, "Q_AVG_DISTS");
	for (int i = 0; i < Q_stencils.size(); i++) {
		printStencil(Q_stencils[i], "Q_STENCIL: ");
	}
	printCenters(G_centers, "G_CENTERS");

	cout << "RANK " << my_rank << " REPORTS: received GPU object" << endl;
}

int GPU::sendUpdate(int my_rank, int receiver_rank) {

	if (my_rank != receiver_rank) {
		vector<int>::iterator qit;
		// vector<set<int> > O; gives us the list of (global) indices which we
		// are sending to receiver_rank		

		// We send a list of node values 
		vector<double> U_O;

		if (O_by_rank.size() > 0) { // Many segfault issues are caused by empty sets.
			//	cout << "O_by_rank is NOT size(0)" << endl;
			int i;

			for (qit = O_by_rank[receiver_rank].begin(); qit
					!= O_by_rank[receiver_rank].end(); qit++, i++) {
				// Elements in O are in global indices 
				// so we need to first convert to local to index our U_G
				U_O.push_back(U_G[g2l(*qit)]);
				cout << "SENDING CPU" << receiver_rank << " U_G[" << *qit
						<< "]: " << U_G[g2l(*qit)] << endl;
			}
		}
		cout << "O_by_rank[" << receiver_rank << "].size = "
				<< O_by_rank[receiver_rank].size() << endl;

		sendSTL(&O_by_rank[receiver_rank], my_rank, receiver_rank);
		sendSTL(&U_O, my_rank, receiver_rank);
		cout << "RANK " << my_rank << " REPORTS: sent update" << endl;
	}
}

int GPU::receiveUpdate(int my_rank, int sender_rank) {
	if (my_rank != sender_rank) {
		vector<int>::iterator qit;
		int i = 0;
		// vector<set<int> > R;
		// gives us the list of (global) indices which we are receiving from sender_rank		
		// We send a list of node values 
		vector<double> U_R;
		// Also need to tell what subset of O was sent.
		vector<int> R_sub;

		recvSTL(&R_sub, my_rank, sender_rank);
		recvSTL(&U_R, my_rank, sender_rank);

		cout << endl << "Received Update from CPU" << sender_rank << " ("
				<< R_sub.size() << " centers)" << endl;
		// Then we integrate the values as an update: 
		for (qit = R_sub.begin(); qit != R_sub.end(); qit++, i++) {
			cout << "\t(Global Index): " << *qit << "\t (Local Index:" << g2l(
					*qit) << ")\tOld U_G[" << g2l(*qit) << "]: " << U_G[g2l(
					*qit)] << "\t New U_G[" << g2l(*qit) << "]: " << U_R[i]
					<< endl;
			// Global to local mapping required
			U_G[g2l(*qit)] = U_R[i]; // Overwrite with new values
		}

		cout << "RANK " << my_rank << " REPORTS: received update" << endl;
	}
}

int GPU::sendFinal(int my_rank, int receiver_rank) {

	if (my_rank != receiver_rank) {
		set<int>::iterator qit;
		// vector<set<int> > O; gives us the list of (global) indices which we
		// are sending to receiver_rank

		// We send a list of node values
		vector<double> U_Q;

		if (O_by_rank.size() > 0) { // Many segfault issues are caused by empty sets.
			//	cout << "O_by_rank is NOT size(0)" << endl;
			int i;

			// TODO the set U_G should be in order (Q, R) so we dont need this full copy (maybe...)
			for (qit = Q.begin(); qit != Q.end(); qit++, i++) {
				// Elements in Q are in global indices
				// so we need to first convert to local to index our U_G
				U_Q.push_back(U_G[g2l(*qit)]);
				cout << "SENDING CPU" << receiver_rank << " U_G[" << *qit
						<< "]: " << U_G[g2l(*qit)] << endl;
			}
		}

		sendSTL(&Q, my_rank, receiver_rank);
		sendSTL(&U_Q, my_rank, receiver_rank);
		cout << "RANK " << my_rank << " REPORTS: sent final" << endl;
	}
}

int GPU::receiveFinal(int my_rank, int sender_rank) {
	// A couple options for this routine:
	// Receive the stl set<int> for Q and the vector<double> for U_Q
	// do a linear merge by hand:
	// 		- The set<int> Q is pre-sorted
	//		- Therefore we do linear loop:
	//			if (remoteQ[i] < localQ[j]) && (remoteQ[i] > localQ[j-1])
	//				localQ[j].insert(remoteQ[i], j)
	// 				localU_G[j].insert(remoteU_G[i], j)
	// OR since we have set<int>:KEY = set<double>:VALUE we can use a map and
	// say: map<int, double> solution;
	// 		for i...
	//			solution[remoteQ[i]] = U_G[i];
	// 	Then use an iterator to copy content into global U_G vector

	set<int> remoteQ;
	vector<double> remoteU_Q;

	recvSTL(&remoteQ, my_rank, sender_rank);
	recvSTL(&remoteU_Q, my_rank, sender_rank);

	//pair<map<char,int>::iterator,bool> ret;
	set<int>::iterator qit;
	int i = 0;
	for (qit = remoteQ.begin(); qit != remoteQ.end(); qit++, i++) {
		global_U_G.insert(pair<int, double> (*qit, remoteU_Q[i]));
	}

	cout << "RECEIVED FINAL FROM CPU " << sender_rank << endl;
	cout << "NEW FINAL_U_G: " << endl;
	map<int, double>::iterator it;
	for (it = global_U_G.begin(); it != global_U_G.end(); it++) {
		cout << "\tU_G[" << (*it).first << "] = " << (*it).second << endl;
	}
}

int GPU::initFinal() {
	//pair<map<char,int>::iterator,bool> ret;
	set<int>::iterator qit;
	int i = 0;
	for (qit = Q.begin(); qit != Q.end(); qit++, i++) {
		global_U_G.insert(pair<int, double> (*qit, U_G[g2l(*qit)]));
	}
}

void GPU::getFinal(std::vector<double> *final) {
	map<int, double>::iterator it;
	final->clear();
	for (it = global_U_G.begin(); it != global_U_G.end(); it++) {
		final->push_back((*it).second);
	}
}

int GPU::writeFinal(std::vector<Vec3> nodes, char* filename) {
	//ofstream fout;
	//fout.open(filename);
	FILE* fdsol;
	fdsol = fopen(filename, "w");
	map<int, double>::iterator it;
	int i = 0;
	for (it = global_U_G.begin(); it != global_U_G.end(); it++, i++) {
		// TODO 3D print
		//fout << nodes[(*it).first].x() << " " << nodes[(*it).first].y() << " " << (*it).second << endl;
		fprintf(fdsol, "%f %f %f %f\n", nodes[(*it).first].x(), nodes[(*it).first].y(), nodes[(*it).first].z(), (*it).second);
	}
	//fout.close();
	fclose(fdsol);
}

// Append to O_by_rank (find what subset of O is needed by rank subdomain_rank)
void GPU::fillDependencyList(std::set<int> subdomain_R, int subdomain_rank) {
	set<int>::iterator qit;
	int i = 0;

	if (O_by_rank.size() == 0) {
		cout << "RESIZING O_by_rank" << endl;
		O_by_rank.resize(comm_size);
		printSet(this->O, "Original O");
	}

	for (qit = subdomain_R.begin(); qit != subdomain_R.end(); qit++, i++) {
		if (isInSet(*qit, this->O)) {
			this->O_by_rank[subdomain_rank].push_back(*qit);
		}
	}
	char label[256];
	sprintf(label, "Rank %d O_by_rank[%d]", id, subdomain_rank);
	printVector(this->O_by_rank[subdomain_rank], label);

	return;
}

void GPU::fillCenterSets(vector<Vec3>& rbf_centers, vector<int> boundary,
		vector<vector<int> >& stencils) {
	//********************************
	//  STENCIL MEMBERSHIP SETS
	//********************************
	// NEED TO FILL THESE:
	// std::set<int> Q;			// All stencil centers in this CPUs QUEUE
	// std::set<int> D;			// Set of stencil centers DEPEND on nodes in R before evaluation
	// std::set<int> O;			// Centers that are OUTPUT to other GPUs
	// std::set<int> B; 		// Centers on BOUNDARY (in O and D or both)
	// std::set<int> QmB; 		// Interior centers (computed without communication)
	// std::set<int> R;			// Nodes REQUIRED from other GPUs.
	//
	set<int>::iterator qit;

	printf("gpu %d, xmin/max= %f, %f, ymin/max= %f, %f, zmin/max= %f, %f\n", id, xmin, xmax,
			ymin, ymax, zmin, zmax);

	// Generate sets Q and D
	for (int i = 0; i < rbf_centers.size(); i++) {
		Vec3& pt = rbf_centers[i];
		if (this->isInsideSubdomain(pt))
			continue; // Do not add to Q or D
		// If we dont continue then it is a center in Q.
		Q.insert(i);

		// Now, if the center is in Q but it depends on nodes in R then we need to distinguish
		bool depR = false;
		for (int j = 0; j < stencils[i].size(); j++) { // Check all nodes in corresponding stencil
			Vec3& pt2 = rbf_centers[stencils[i][j]];
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

	// Set of nodes from stencils that are not in Q (i.e. not on the GPU)
	// compute set R = S(Q) \ Q
	set_difference(SQ.begin(), SQ.end(), Q.begin(), Q.end(), inserter(R,
			R.end()));

	// Determine set O = pts in Q that are in stencils on other GPUs
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
void GPU::fillLocalData(vector<Vec3>& rbf_centers,
		vector<vector<int> >& stencil, vector<int>& boundary,
		vector<double> avg_dist) {
	// Generate stencil membership lists (i.e., which set each stencil center belongs to)
	this->fillCenterSets(rbf_centers, boundary, stencil);

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
		G_centers.push_back(rbf_centers[*qit]); // In order to compute we need the physical locations of all function values
		Q_stencils.push_back(stencil[*qit]); // We also need to push the connectivity to evaluate stencils
		Q_avg_dists.push_back(avg_dist[*qit]);
	}
	for (qit = B.begin(); qit != B.end(); qit++, i++) {
		loc_to_glob.push_back(*qit);
		G_centers.push_back(rbf_centers[*qit]);
		Q_stencils.push_back(stencil[*qit]);
		Q_avg_dists.push_back(avg_dist[*qit]);
	}
	for (qit = R.begin(); qit != R.end(); qit++, i++) {
		loc_to_glob.push_back(*qit);
		G_centers.push_back(rbf_centers[*qit]); // Assume non-moving node problem so we can store positions at initialization
		// HOWEVER, NO CONNECTIVITY REQUIRED FOR R (THESE ARE ON OTHER CPUs)
		Q_avg_dists.push_back(avg_dist[*qit]);
	}

	// global to local map
	for (int i = 0; i < loc_to_glob.size(); i++) {
		globmap[loc_to_glob[i]] = i;
	}

	// Convert all stencils to local indexing.
	for (int i = 0; i < Q_stencils.size(); i++) {
		for (int j = 0; j < Q_stencils[i].size(); j++) {
			Q_stencils[i][j] = globmap[Q_stencils[i][j]];
		}
	}

        // This forms the boundary set (needed)
	printf("BOUNDARY.size= %d\n", (int) boundary.size());
	set_intersection(Q.begin(), Q.end(), boundary.begin(), boundary.end(),
			inserter(global_boundary_nodes, global_boundary_nodes.end()));
	for (int i = 0; i < global_boundary_nodes.size(); i++) {
		cout << "Boundary Node[" << i << "] = " << global_boundary_nodes[i]
				<< endl;
		global_boundary_nodes[i] = globmap[global_boundary_nodes[i]];
	}
	printf("GLOBAL_BOUNDARY.size= %d\n", (int) global_boundary_nodes.size());

	printf("l2g size= %d\n", (int) loc_to_glob.size());
	printf("g2l size= %d\n", (int) globmap.size());
	printf("G_centers size= %d\n", (int) G_centers.size());
        printf("Q_avg_dists size= %d\n", (int) Q_avg_dists.size());
}
//----------------------------------------------------------------------
set<int>& GPU::stencilSet(set<int>& s, vector<vector<int> >& stencil) {
	set<int>* Sset = new set<int> ;
	set<int>::iterator qit;

	//set<int>::iterator qit;
	for (qit = s.begin(); qit != s.end(); qit++) {
		int qi = *qit;
		vector<int>& si = stencil[qi];

		for (int j = 0; j < si.size(); j++) {
			Sset->insert(si[j]);
		}
	}

	return *Sset;
}
//----------------------------------------------------------------------
void GPU::fillVarData(vector<Vec3>& rbf_centers) {
	// Initial condition: linear data: x + 2 y + 3 z
	// NOTE: the loops here should be structured similar to GPU::fillLocalData
	// to ensure our l2g and g2l mappings apply correctly.
	set<int>::iterator qit;

	for (qit = QmB.begin(); qit != QmB.end(); qit++) {
		Vec3& v = rbf_centers[*qit];
		double s = *qit; //v.x() + 2.*v.y() + 3.*v.z();
		U_G.push_back(s);
	}

	for (qit = B.begin(); qit != B.end(); qit++) {
		Vec3& v = rbf_centers[*qit];
		double s = *qit;//v.x() + 2.*v.y() + 3.*v.z();
		U_G.push_back(s);
	}

	// Initial R values (these will be updated at each iteration)
	for (qit = R.begin(); qit != R.end(); qit++) {
		Vec3& v = rbf_centers[*qit];
		double s = *qit;//v.x() + 2.*v.y() + 3.*v.z();
		U_G.push_back(s);
	}
}
//----------------------------------------------------------------------


void GPU::printStencilNodesIn(const vector<vector<int> > stencils, const set<
		int> center_set, const char* display_char) {
	for (int i = 0; i < stencils.size(); i++) {
		cout << "Stencil[" << i << "] = ";
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

// HERE: center_set is global index
void GPU::printCenterMemberships(const set<int> center_set,
		const char* display_name) {
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
			<< "[ local_index | global_index ] = \t[Q|.]   [D|.]   [Q|.]   [R][+]   [B|.]"
			<< endl;
	cout << "\tCONDITIONS: " << endl;
	cout << "\t\tQ  --> in set Q?" << endl;
	cout << "\t\tD  --> in set D?" << endl;
	cout << "\t\tO  --> in set O?" << endl;
	cout << "\t\tR  --> depends on set R?" << endl;
	cout << "\t\t+  --> is the center in R?" << endl;
	cout << "\t\tB*  --> is the center on the global Boundary?" << endl;
	int i = 0;
	for (set<int>::const_iterator setiter = center_set.begin(); setiter
			!= center_set.end(); setiter++, i++) {
		cout << i << "\t" << display_name << "[ " << g2l(*setiter) << " | "
				<< *setiter << " ] =\t\t";
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
		cout << "   ";//<< Q_stencils.size() << ":" << globmap[*setiter];
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
                if (isInVector(*setiter, this->global_boundary_nodes)) {
			cout << "B*";
		} else {
			cout << ".";
		}
		cout << endl;
	}
}

bool GPU::isInSet(const int center, const set<int> center_set) const {
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

bool GPU::isInVector(const int center, const vector<int> center_set) const {
	//bool inSet = false;
	for (vector<int>::const_iterator setiter = center_set.begin(); setiter
			!= center_set.end(); setiter++) {
		// True -> stencil[i][j] is in center set
		if (center == *setiter) {
			return true;
		}
	}
	return false;
}

bool GPU::dependsOnSet(const int local_stencil_id, const set<int> center_set) {
	// Q_stencils are in local indices
	if (local_stencil_id >= Q_stencils.size()) {
		return true;
	}
	vector<int> stencil = Q_stencils[local_stencil_id];
	for (int i = 0; i < stencil.size(); i++) {
		if (isInSet(stencil[i], center_set)) {
			return true;
		} // short circuit return
	}
	return false;
}

void GPU::printSet(const set<int> center_set, const char* set_label) {
	cout << set_label << " = {" << endl;
	for (set<int>::const_iterator setiter = center_set.begin(); setiter
			!= center_set.end(); setiter++) {
		// True -> stencil[i][j] is in center set
		int i = *setiter;
		cout << "\t" << i << " (" << globmap[i] << ")" << endl;
	}
	cout << "}" << endl;
}

void GPU::printVector(const vector<double> stencil_radii, const char* set_label) {
	cout << set_label << " = {" << endl;
	int i = 0;
	for (vector<double>::const_iterator setiter = stencil_radii.begin(); setiter
			!= stencil_radii.end(); setiter++, i++) {
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

void GPU::printVector(const vector<int> center_set, const char* set_label) {
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

void GPU::printCenters(const std::vector<Vec3> centers,
		const char* center_label) {
	cout << center_label << " = {" << endl;
	int i = 0;
	for (vector<Vec3>::const_iterator setiter = centers.begin(); setiter
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

void GPU::printStencil(const std::vector<int> stencil,
		const char* stencil_label) {
	cout << stencil_label << " = " << "\t";
	int i = 0;
	if (loc_to_glob.size() > 0) {
		for (vector<int>::const_iterator setiter = stencil.begin(); setiter
				!= stencil.end(); setiter++, i++) {
			// True -> stencil[i][j] is in center set
			cout << " [" << *setiter << " (" << loc_to_glob[*setiter] << ")] ";
		}
	} else { // WE MIGHT BE IN THE ORIGINAL CODE
		for (vector<int>::const_iterator setiter = stencil.begin(); setiter
				!= stencil.end(); setiter++, i++) {
			// True -> stencil[i][j] is in center set
			cout << " [" << *setiter << " (" << *setiter << ")] ";
		}
	}
	cout << endl;
}

void GPU::printStencilPlus(const std::vector<int> stencil, const std::vector<
		double> function_values, const char* stencil_label) {
	cout << stencil_label << " = " << "\t";
	int i = 0;
	if (loc_to_glob.size() > 0) {
		for (vector<int>::const_iterator setiter = stencil.begin(); setiter
				!= stencil.end(); setiter++, i++) {
			// True -> stencil[i][j] is in center set
			cout << " [" << *setiter << " (" << loc_to_glob[*setiter] << ")] {"
					<< function_values[*setiter] << "} ";
		}
	} else { // WE MIGHT BE IN THE ORIGINAL CODE
		for (vector<int>::const_iterator setiter = stencil.begin(); setiter
				!= stencil.end(); setiter++, i++) {
			// True -> stencil[i][j] is in center set
			cout << " [" << *setiter << " (" << *setiter << ")] {"
					<< function_values[*setiter] << "} ";
		}
	}
	cout << endl;
}
//----------------------------------------------------------------------
