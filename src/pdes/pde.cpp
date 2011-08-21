#include "pde.h"

#include "utils/norms.h"

#include <algorithm>
#include <cmath>

void PDE::setupTimers()
{
        tm["sendrecv"] = new EB::Timer("[PDE] MPI Communicate PDE CPU to CPU"); 
}
 
int PDE::send(int my_rank, int receiver_rank) {
    // Initially we have nothing to send.
    return 0; 
}

int PDE::receive(int my_rank, int sender_rank) {
    // Initially we have nothing to receive 
    return 0;
}

//----------------------------------------------------------------------------
std::string PDE::getFileDetailString() {
    std::stringstream ss(std::stringstream::out); 
    ss << "pde_" << grid_ref.getFileDetailString();  
    return ss.str();
}

//----------------------------------------------------------------------------
std::string PDE::getFilename(std::string base_filename, int iter) {
    std::stringstream ss(std::stringstream::out);
#if 0
    if (iter < 0) {
        ss << base_filename << "_" << this->getFileDetailString() << "_final.ascii";  
    } else if (iter == 0) {
        ss << base_filename << "_" << this->getFileDetailString() << "_initial.ascii";  
    } else {
        ss << base_filename << "_" << this->getFileDetailString() << "_" << iter << "iters.ascii";  
    }
#endif 
    ss << base_filename << "_" << grid_ref.getFilename(iter); 
    std::string filename = ss.str();
    return filename;
}

//----------------------------------------------------------------------------
std::string PDE::getFilename(int iter) {
    return this->getFilename(this->className(), iter); 
}

//----------------------------------------------------------------------

void PDE::writeLocalSolutionToFile(std::string filename) {

    std::string fname = "sol_"; 
    fname.append(filename); 
    std::ofstream fout(fname.c_str()); 
    if (fout.is_open()) {
        std::vector<double>::iterator sit;  
        for (sit = U_G.begin(); sit != U_G.end(); sit++) {
            fout << (*sit) << std::endl; 
        }
    } else {
        printf("Error opening file to write\n"); 
        exit(EXIT_FAILURE); 
    }
    fout.close();
    std::cout << "[PDE] \tWrote " << U_G.size() << " local solution values to \t" << fname << std::endl;

}

//----------------------------------------------------------------------

void PDE::writeGlobalSolutionToFile(std::string filename) {

    if (comm_ref.getRank() == Communicator::MASTER) {

        std::string fname = "globalsol_";
        fname.append(filename); 
#if 0
        char nstr[256]; 
        std::string fname = "globalsol_"; 
        sprintf(nstr,"%lunodes",global_U_G.size()); 
        fname.append(nstr); 

        if (iter < 0) {
            fname.append("_final.ascii");  
        } else if (iter == 0) {
            fname.append("_initial.ascii");  
        } else {
            char iterstr[256]; 
            sprintf(iterstr, "%d", iter); 
            fname.append("_"); 
            fname.append(iterstr); 
            fname.append("iters.ascii");  
        }
#endif 
        std::ofstream fout(fname.c_str()); 
        if (fout.is_open()) {
            for (int i = 0; i < global_U_G.size(); i++) {
                fout << global_U_G[i] << std::endl; 
            }
        } else {
            printf("Error opening file to write\n"); 
            exit(EXIT_FAILURE); 
        }
        fout.close();
        std::cout << "[PDE] \tWrote " << global_U_G.size() << " global solution values to \t" << fname << std::endl;
    } else {
        std::cout << "[PDE] \tDeferring global solution write to master process" << std::endl;
    }
}

// By default we send updates for the SOLUTION of our PDE. 
int PDE::sendUpdate(int my_rank, int receiver_rank) {
    return this->sendUpdate(this->U_G, my_rank, receiver_rank, "U_G");
}

// Alternatively we can specify a vector to update
// NOTE: the vector we update MUST be of size n_nodes and NOT of size n_stencils
int PDE::sendUpdate(std::vector<SolutionType>& vec, int my_rank, int receiver_rank, std::string label) {

    if (my_rank != receiver_rank) {
        //vector<int>::iterator oit;
        // vector<set<int> > O; gives us the list of (global) indices which we
        // are sending to receiver_rank		

        // FIXME: domain should not make these public. hide them behind accessors
        std::vector<std::vector<int> >& O_by_rank = grid_ref.O_by_rank;
        std::vector<int>::iterator oit;

        // We send a list of node values 
        vector<double> U_O(O_by_rank[receiver_rank].size());

        if (O_by_rank.size() > 0) { // Many segfault issues are caused by empty sets.
            //	cout << "O_by_rank is NOT size(0)" << endl;
            unsigned int i=0;

            for (oit = O_by_rank[receiver_rank].begin(); oit
                    != O_by_rank[receiver_rank].end(); oit++, i++) {
                int g_indx = *oit; 
                int l_indx = grid_ref.g2l(g_indx);
                // Elements in O are in global indices 
                // so we need to first convert to local to index our U_G
                U_O[i] = vec[l_indx];
#if 0
                cout << "SEND "<< label << "\t(Targeting Global Index): " << g_indx << "\t (Local Index:" << l_indx << ")" << std::endl;
                cout << receiver_rank << " vec[" << l_indx << "] = " << vec[l_indx] << "\tU_O[" << i << "] = " << U_O[i] << std::endl;

#endif 
            }
        } else {
            cout << "O_BY_RANK for " << receiver_rank << " is 0" << std::endl;
            exit(-1);
        }

#if 0
        cout << "O_by_rank[" << receiver_rank << "].size = "
            << O_by_rank[receiver_rank].size() << endl;
#endif 
        sendSTL(&O_by_rank[receiver_rank], my_rank, receiver_rank);
        sendSTL(&U_O, my_rank, receiver_rank);
//        cout << "RANK " << my_rank << " REPORTS: sent update to RANK " << receiver_rank << endl;
    }
    return 0;           // FIXME: return bytes sent (in case we need to monitor this)
}

int PDE::receiveUpdate(int my_rank, int sender_rank) {
    return this->receiveUpdate(this->U_G, my_rank, sender_rank, "U_G");
}

int PDE::receiveUpdate(std::vector<SolutionType>& vec, int my_rank, int sender_rank, std::string label) {
    if (my_rank != sender_rank) {
        vector<int>::iterator rit;
        int i = 0;
        // We receive a list of (global) indices which we are receiving from sender_rank		
        vector<double> U_R;
        // Also need to tell what subset of O was received.
        vector<int> R_sub;

        recvSTL(&R_sub, my_rank, sender_rank);
        recvSTL(&U_R, my_rank, sender_rank);

        // EFB06112011: 
        // FIXME: we need to improve communication so we are not making
        // connections for 0 transfers
#if 0
        cout << "Received Update from CPU" << sender_rank << " (" << R_sub.size() << " centers)" << endl;
        if (!(R_sub.size() > 0)) {
            std::cout << "[PDE] error. did not receive any updates\n";
            exit(EXIT_FAILURE);
        }
#endif 

        // Then we integrate the values as an update: 
        for (rit = R_sub.begin(); rit != R_sub.end(); rit++, i++) {
            int g_indx = *rit; 
            int l_indx = grid_ref.g2l(*rit);
#if 0 
            cout << label << "\t(Global Index): " << g_indx << "\t (Local Index:" << l_indx << ")\tOld vec[" << l_indx << "]: " << vec[l_indx] << "\t New U_G[" << l_indx << "]: " << U_R[i] << endl;
#endif 
            // Global to local mapping required
            vec[l_indx] = U_R[i]; // Overwrite with new values
        }

//        cout << "RANK " << my_rank << " REPORTS: received update from RANK " << sender_rank << endl;
    }

    return 0;  // FIXME: return number of bytes received in case we want to monitor this 
}

int PDE::sendrecvUpdates(std::vector<SolutionType>& vec, std::string label) 
{
    tm["sendrecv"]->start();
    if (comm_ref.getSize() > 1) {
//        std::cout << "[PDE] SEND/RECEIVE\n";
        vector<int> receiver_list; 

        // This is BAD: we should only send to CPUs in need. 
        for (int i = 0; i < comm_ref.getSize(); i++) {
            // FIXME: we need an R_by_rank to knwo when to receive from another CPU
            // unless we use some sort of BCAST mechanism in MPI and MPI determines the details. 
            // For now we allow 0 byte transfers but this should be improved
            if (i != comm_ref.getRank()) 
            {
                receiver_list.push_back(i); 	
            }
        }

        // Round robin: each CPU takes a turn at sending message to CPUs that need nodes
        for (int j = 0; j < comm_ref.getSize(); j++) {
            if (comm_ref.getRank() == j) {		// My turn
                for (int i = 0; i < receiver_list.size(); i++) {
                    this->sendUpdate(vec, comm_ref.getRank(), receiver_list[i], label);
                }
            } else {						// All CPUs listen
                this->receiveUpdate(vec, comm_ref.getRank(), j, label);
            }
        }
    }
    tm["sendrecv"]->stop();
    return 0;  // FIXME: return number of bytes received in case we want to monitor this 
}

int PDE::sendFinal(int my_rank, int receiver_rank) {
    
    if (my_rank != receiver_rank) {
        this->syncCPUtoGPU();
        // This should match grid_ref.Q.size():
        unsigned int nb_stencils = grid_ref.getStencilsSize();

        // We send a list of node values
        vector<double> U_Q(nb_stencils);

        if (grid_ref.O_by_rank.size() > 0) { // Many segfault issues are caused by empty sets.
            //	cout << "O_by_rank is NOT size(0)" << endl;
            int i = 0;

            std::set<int>::iterator qit;
            // TODO the set U_G should be in order (Q, R) so we dont need this full copy (maybe...)
            for (qit = grid_ref.Q.begin(); qit != grid_ref.Q.end(); qit++, i++) {
                // Elements in Q are in global indices but out of order
                // so we need to first convert to local to index our U_G
                U_Q[i] = U_G[grid_ref.g2l(*qit)];
            }

#if 0    
                cout << "SENDING CPU" << receiver_rank << " U_G[" << *qit
                    << "]: " << U_G[grid_ref.g2l(*qit)] << endl;
#endif 
        }

        // This Q is already in global indexing
        sendSTL(&grid_ref.Q, my_rank, receiver_rank);
        sendSTL(&U_Q, my_rank, receiver_rank);
    //    cout << "RANK " << my_rank << " REPORTS: sent final" << endl;
    }
    return 0;  // FIXME: return number of bytes received in case we want to monitor this 
}



int PDE::receiveFinal(int my_rank, int sender_rank) {
    // A couple options for this routine:
    // Receive the stl set<int> for Q and the vector<double> for U_Q
    // do a linear merge by hand:
    // 		- The set<int> Q is pre-sorted and in GLOBAL index
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
        int l_indx = i; 
        int g_indx = *qit;
        global_U_G[g_indx] = remoteU_Q[l_indx]; 
    }

    cout << "RECEIVED FINAL (size=" << remoteQ.size() << " FROM CPU " << sender_rank << endl;
#if 0
    cout << "NEW FINAL_U_G: " << endl;
    map<int, double>::iterator it;
    for (it = global_U_G.begin(); it != global_U_G.end(); it++) {
        cout << "\tU_G[" << (*it).first << "] = " << (*it).second << endl;
    }
#endif 
    return 0;  // FIXME: return number of bytes received in case we want to monitor this 
}

int PDE::initFinal() {
    std::cout << "FINAL INITIALIZED\n";
    //pair<map<char,int>::iterator,bool> ret;
    set<int>::iterator qit;
    int i = 0;
    for (qit = grid_ref.Q.begin(); qit != grid_ref.Q.end(); qit++, i++) {
        unsigned int l_indx = grid_ref.g2l(*qit); 
        unsigned int g_indx = *qit;
        global_U_G[g_indx] = U_G[l_indx]; 
    }
    return 0;  // FIXME: return number of bytes received in case we want to monitor this 
}

int PDE::updateFinal() {

    if (!initCount) {
        this->initFinal();
        initCount ++;
    }

    this->syncCPUtoGPU();

    //pair<map<char,int>::iterator,bool> ret;
    set<int>::iterator qit;
    int i = 0;
    for (qit = grid_ref.Q.begin(); qit != grid_ref.Q.end(); qit++, i++) {
        unsigned int l_indx = grid_ref.g2l(*qit); 
        unsigned int g_indx = *qit;
        global_U_G[g_indx] = U_G[l_indx]; 
    }
    std::cout << "GLOBAL_U_G.size() = " << global_U_G.size() << std::endl;
    return 0;  // FIXME: return number of bytes received in case we want to monitor this 
}

void PDE::printSolution(std::string set_label) {
    cout << set_label << " = {" << endl;
    int i = 0;
    for (vector<SolutionType>::const_iterator setiter = U_G.begin(); setiter
            != U_G.end(); setiter++, i++) {
        // True -> stencil[i][j] is in center set
        if (grid_ref.loc_to_glob.size() > 0) {
            cout << "\t[" << i << " (global:" << grid_ref.l2g(i) << ")] = " << *setiter
                << endl;
        } else {
            cout << "\t[" << i << " (" << i << ")] = " << *setiter << endl;
        }
    }
    cout << "}" << endl;
}


void PDE::getGlobalSolution(std::vector<double> *final) {
    // The global_U_G map is sorted according to node index. All we need is to fill the 
    // output vector and we're done.
    map<int, double>::iterator it;
    final->resize(global_U_G.size());
    unsigned int i = 0; 
    for (it = global_U_G.begin(); it != global_U_G.end(); it++, i++) {
        (*final)[i] = (*it).second;
    }
}

int PDE::writeGlobalGridAndSolutionToFile(std::vector<NodeType>& nodes, std::string filename) {
    //ofstream fout;
    //fout.open(filename);
    FILE* fdsol;
    fdsol = fopen(filename.c_str(), "w");
    map<int, double>::iterator it;
    int i = 0;
    for (it = global_U_G.begin(); it != global_U_G.end(); it++, i++) {
        // TODO 3D print
        //fout << nodes[(*it).first].x() << " " << nodes[(*it).first].y() << " " << (*it).second << endl;
        fprintf(fdsol, "%f %f %f %f\n", nodes[(*it).first].x(), nodes[(*it).first].y(), nodes[(*it).first].z(), (*it).second);
    }
    //fout.close();
    fclose(fdsol);
    return i; 
}


//----------------------------------------------------------------------

    struct ltclass {
        bool operator() (unsigned int i, unsigned int j) { return (i<j); }
    } srtobject; 


void PDE::checkError(std::vector<SolutionType>& sol_exact, std::vector<SolutionType>& sol_vec, Grid& grid, double rel_err_max)
{
    std::set<unsigned int>& b_indices = grid.getSortedBoundarySet();
    std::set<unsigned int>& i_indices = grid.getSortedInteriorSet(); 
    unsigned int nb_bnd = b_indices.size();
    unsigned int nb_int = i_indices.size();
    int nb_centers = grid.getNodeListSize();
    int nb_stencils = grid.getStencilsSize();


#if 0
    if (rel_err_max < 0) { 
        rel_err_max = rel_err_tol; 
    }
#endif 
    vector<double> sol_error(nb_stencils);

    std::vector<double> sol_vec_bnd(nb_bnd);
    std::vector<double> sol_exact_bnd(nb_bnd); 

    std::vector<double> sol_vec_int(nb_int);
    std::vector<double> sol_exact_int(nb_int);
    
    std::vector<double> sol_vec_int_no_bnd;
    std::vector<double> sol_exact_int_no_bnd;

    std::set<unsigned int>::iterator it;
    int i = 0;
    for (it = b_indices.begin(); it != b_indices.end(); it++, i++) { 
        int j = *it;
        sol_vec_bnd[i] = sol_vec[j]; 
        sol_exact_bnd[i] = sol_exact[j]; 
    }
    int k = 0;
    int l = 0;
    for (it = i_indices.begin(); it != i_indices.end(); it++, k++) { 
        int j = *it;
        sol_vec_int[k] = sol_vec[j]; 
        sol_exact_int[k] = sol_exact[j]; 

        // Assume that the nodes at the tail of the sol_vec are not part of
        // the subdomain (i.e., that they're ghost nodes) 
        if (j < nb_stencils) {
            // Now, what if the stencil contains boundary nodes? Most error
            // accumulates where stencils are unbalanced 
            StencilType& st = grid.getStencil(j); 
#if 0
            std::cout << "ST[" << j << "].size= " << st.size() << std::endl;
            std::cout << "bindices.size= " << bindices.size() << std::endl;
#endif 
            // does the stencil contain any nodes that are on the boundary?
            bool dep_boundary = false; 
            for (unsigned int sz = 0; sz < st.size(); sz ++) {
                for(std::set<unsigned int>::iterator bit = b_indices.begin(); bit != b_indices.end(); bit++) 
                {
                    if (st[sz] == *bit){
                        dep_boundary=true;
                        //break;
                    }
                }
            }
            if (!dep_boundary) {
                sol_vec_int_no_bnd.push_back(sol_vec[j]); 
                sol_exact_int_no_bnd.push_back(sol_exact[j]); 
                l++;
            }
        }
    }

    calcSolNorms(sol_vec_bnd, sol_exact_bnd, "Boundary", rel_err_max);  // Boundary only

    calcSolNorms(sol_vec_int_no_bnd, sol_exact_int_no_bnd, "Interior Stencils w/o Boundary", rel_err_max);  // Interior only (excludes unbalanced stencils)

    calcSolNorms(sol_vec_int, sol_exact_int, "All Interior", rel_err_max);  // Interior only

    calcSolNorms(sol_vec, sol_exact, "Interior & Boundary", rel_err_max);  // Full domain
}

void PDE::calcSolNorms(std::vector<double>& sol_vec, std::vector<double>& sol_exact, std::string label, double rel_err_max) {
    // We want: || x_exact - x_approx ||_{1,2,inf} 
    // and  || x_exact - x_approx ||_{1,2,inf} / || x_exact ||_{1,2,inf}

    int nb_pts = sol_vec.size();
    double l1fabs = l1norm(sol_vec, sol_exact, 0, nb_pts); 
    double l1denom = l1norm(sol_exact, 0, nb_pts);
    double l1rel = (l1denom > 1e-10) ? l1fabs/l1denom : 0.;
    double l2fabs = l2norm(sol_vec, sol_exact, 0, nb_pts); 
    double l2denom = l2norm(sol_exact, 0, nb_pts);
    double l2rel = (l2denom > 1e-10) ? l2fabs/l2denom : 0.;
    double lifabs = linfnorm(sol_vec, sol_exact, 0, nb_pts); 
    double linfdenom = linfnorm(sol_exact, 0, nb_pts); 
    double lirel = (linfdenom > 1e-10) ? lifabs/linfdenom : 0.;
#define COMPONENTWISE_ERR 0
#if COMPONENTWISE_ERR
    double comp_sum; 
    for (unsigned int i=0; i < nb_pts; i++) {
        double abserr = fabs(sol_vec[i] - sol_exact[i]); 
        double relerr = (fabs(sol_exact[i]) > 1e-10) ? abserr / fabs(sol_exact[i]) : 0.;
        std::cout <<  "AbsErr[" << i << "] = " << abserr << "\t" << sol_exact[i] << "\t";
        std::cout <<  "RelErr[" << i << "] = " << relerr << std::endl;
        comp_sum += abserr; 
    }
    std::cout << "SUM OF ABS ERR: " << comp_sum << std::endl;
#endif 

    // Only print this when we're looking at the global norms
    if (!label.compare("")) {
        printf("========= Norms For Current Solution ========\n"); 
        printf("Absolute =>  || x_exact - x_approx ||_p                    ,  where p={1,2,inf}\n"); 
        printf("Relative =>  || x_exact - x_approx ||_p / || x_exact ||_p  ,  where p={1,2,inf}\n"); 
    }

    printf("%s l1 error (%d nodes):   Absolute = %le,    Relative = %le \n", label.c_str(), nb_pts, l1fabs, l1rel);
    printf("%s l2 error (%d nodes):   Absolute = %le,    Relative = %le \n", label.c_str(), nb_pts, l2fabs, l2rel);
    printf("%s linf error (%d nodes): Absolute = %le,    Relative = %le \n", label.c_str(), nb_pts, lifabs, lirel);

#if 0
    if (l1rel > rel_err_max) {
        printf("[PDE] Error! l1 relative error (=%f%%) is too high to continue. We require %f%% or less.\n", 100.*l1rel, 100.*rel_err_max); 
        exit(EXIT_FAILURE);
    }
#endif 
    if (l2rel > rel_err_max) {
        printf("[PDE] Error! l2 relative error (=%f%%) is too high to continue. We require %f%% or less.\n", 100.*l2rel, 100.*rel_err_max); 
        exit(EXIT_FAILURE);
    }
#if 0
    if (lirel > rel_err_max) {
        printf("[PDE] Error! linf relative error (=%f%%) is too high to continue. We require %f%% or less.\n", 100.*lirel, 100.*rel_err_max); 
        exit(EXIT_FAILURE);
    }
#endif 

#define CATCH_NANS 1
#if CATCH_NANS
    if ((l1fabs != l1fabs) ||
        (l2fabs != l2fabs) || 
            (lifabs != lifabs)) {
        std::cout << "Caught NaNs in error!\n";
        exit(EXIT_FAILURE);
    }

#endif 
}


void PDE::checkNorms(double max_l2_norm) {
        this->syncCPUtoGPU();
    std::vector<SolutionType>& sol_vec = this->U_G;
    int nb_pts = sol_vec.size();
    double l1fabs = l1norm(sol_vec, 0, nb_pts); 
    double l2fabs = l2norm(sol_vec, 0, nb_pts); 
    double lifabs = linfnorm(sol_vec, 0, nb_pts); 

    printf("\tApprox Solution  l1  norm (%d nodes):  %le\n", nb_pts, l1fabs);
    printf("\tApprox Solution  l2  norm (%d nodes):  %le\n", nb_pts, l2fabs);
    printf("\tApprox Solution linf norm (%d nodes):  %le\n", nb_pts, lifabs);

    if (max_l2_norm > 0) {
        if (l2fabs > max_l2_norm) {
            printf("Approx Solution l2 norm exceeds %le!", max_l2_norm); 
            exit(EXIT_FAILURE);
        }
    }

    if (max_l2_norm < 0) {

        if (l2fabs > 1e100) {
            printf("Approx Solution l2 norm exceeds 1.0e100. Assuming instability!"); 
            exit(EXIT_FAILURE);
        }

    }

    // NaN is the only thing that satisfies this and we should DEFINITELY exit
    // if its encountered
    if (l2fabs != l2fabs) {
        printf("NaN encountered. Modify support parameter or node distribution to improve stability"); 
        exit(EXIT_FAILURE);
    }

}
