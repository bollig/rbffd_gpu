#include "pde.h"

#include "utils/norms.h"

#include <algorithm>
#include <cmath>

int PDE::send(int my_rank, int receiver_rank) {
    // Initially we have nothing to send.
    ; 
}

int PDE::receive(int my_rank, int sender_rank) {
    // Initially we have nothing to receive 
    ;
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


int PDE::sendUpdate(int my_rank, int receiver_rank) {

    if (my_rank != receiver_rank) {
        vector<int>::iterator oit;
        // vector<set<int> > O; gives us the list of (global) indices which we
        // are sending to receiver_rank		

        // FIXME: domain should not make these public. hide them behind accessors
        std::vector<std::vector<int> >& O_by_rank = grid_ref.O_by_rank;

        // We send a list of node values 
        vector<double> U_O;

        if (O_by_rank.size() > 0) { // Many segfault issues are caused by empty sets.
            //	cout << "O_by_rank is NOT size(0)" << endl;
            int i;

            for (oit = O_by_rank[receiver_rank].begin(); oit
                    != O_by_rank[receiver_rank].end(); oit++, i++) {
                // Elements in O are in global indices 
                // so we need to first convert to local to index our U_G
                U_O.push_back(U_G[grid_ref.g2l(*oit)]);
#if 0
                cout << "SENDING CPU" << receiver_rank << " U_G[" << *oit
                    << " (local index: " << grid_ref.g2l(*oit) << ")"
                    << "]: " << U_G[grid_ref.g2l(*oit)] << endl;
#endif 
            }
        }

        cout << "O_by_rank[" << receiver_rank << "].size = "
            << O_by_rank[receiver_rank].size() << endl;

        sendSTL(&O_by_rank[receiver_rank], my_rank, receiver_rank);
        sendSTL(&U_O, my_rank, receiver_rank);
        cout << "RANK " << my_rank << " REPORTS: sent update" << endl;
    }
}

int PDE::receiveUpdate(int my_rank, int sender_rank) {
    if (my_rank != sender_rank) {
        vector<int>::iterator rit;
        int i = 0;
        // We receive a list of (global) indices which we are receiving from sender_rank		
        vector<double> U_R;
        // Also need to tell what subset of O was received.
        vector<int> R_sub;

        recvSTL(&R_sub, my_rank, sender_rank);
        recvSTL(&U_R, my_rank, sender_rank);

        cout << endl << "Received Update from CPU" << sender_rank << " ("
            << R_sub.size() << " centers)" << endl;
        // Then we integrate the values as an update: 
        for (rit = R_sub.begin(); rit != R_sub.end(); rit++, i++) {
#if 0 
            cout << "\t(Global Index): " << *rit << "\t (Local Index:" << grid_ref.g2l(
                        *rit) << ")\tOld U_G[" << grid_ref.g2l(*rit) << "]: " << U_G[grid_ref.g2l(
                        *rit)] << "\t New U_G[" << grid_ref.g2l(*rit) << "]: " << U_R[i]
                                             << endl;
#endif 
            // Global to local mapping required
            U_G[grid_ref.g2l(*rit)] = U_R[i]; // Overwrite with new values
        }

        cout << "RANK " << my_rank << " REPORTS: received update" << endl;
    }
}

int PDE::sendFinal(int my_rank, int receiver_rank) {

    if (my_rank != receiver_rank) {
        set<int>::iterator qit;
        // vector<set<int> > O; gives us the list of (global) indices which we
        // are sending to receiver_rank

        // We send a list of node values
        vector<double> U_Q;

        if (grid_ref.O_by_rank.size() > 0) { // Many segfault issues are caused by empty sets.
            //	cout << "O_by_rank is NOT size(0)" << endl;
            int i;

            // TODO the set U_G should be in order (Q, R) so we dont need this full copy (maybe...)
            for (qit = grid_ref.Q.begin(); qit != grid_ref.Q.end(); qit++, i++) {
                // Elements in Q are in global indices
                // so we need to first convert to local to index our U_G
                U_Q.push_back(U_G[grid_ref.g2l(*qit)]);
#if 0    
                cout << "SENDING CPU" << receiver_rank << " U_G[" << *qit
                    << "]: " << U_G[grid_ref.g2l(*qit)] << endl;
#endif 
            }
        }

        sendSTL(&grid_ref.Q, my_rank, receiver_rank);
        sendSTL(&U_Q, my_rank, receiver_rank);
        cout << "RANK " << my_rank << " REPORTS: sent final" << endl;
    }
}



int PDE::receiveFinal(int my_rank, int sender_rank) {
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
#if DEBUG
    cout << "NEW FINAL_U_G: " << endl;
    map<int, double>::iterator it;
    for (it = global_U_G.begin(); it != global_U_G.end(); it++) {
        cout << "\tU_G[" << (*it).first << "] = " << (*it).second << endl;
    }
#endif 
}

int PDE::initFinal() {
    //pair<map<char,int>::iterator,bool> ret;
    set<int>::iterator qit;
    int i = 0;
    for (qit = grid_ref.Q.begin(); qit != grid_ref.Q.end(); qit++, i++) {
        global_U_G.insert(pair<int, double> (*qit, U_G[grid_ref.g2l(*qit)]));
    }
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
    size_t i = 0; 
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
}


//----------------------------------------------------------------------

    struct ltclass {
        bool operator() (size_t i, size_t j) { return (i<j); }
    } srtobject; 


void PDE::checkError(std::vector<SolutionType>& sol_exact, std::vector<SolutionType>& sol_vec, std::vector<NodeType>& nodes, double rel_err_max)
{
    // Get a COPY of the indices because we want to sort them
    std::vector<size_t> bindices = grid_ref.getBoundaryIndices(); 

    size_t nb_nodes = grid_ref.getNodeListSize(); 
#if 0
    if (rel_err_max < 0) { 
        rel_err_max = rel_err_tol; 
    }
#endif 
    vector<double> sol_error(sol_vec.size());
#if 0
    vector<double> sol_exact(sol_vec.size());

    //std::cout << "======= TIME: " << cur_time << " =======\n"; 
    for (int i = 0; i < sol_vec.size(); i++) {
        Vec3& v = nodes[i];
        sol_exact[i] = exactSolution->at(v, cur_time);
        //  sol_error[i] = sol_exact[i] - sol_vec[i];
        //  printf("sol_error[%d] = %f\n", i, sol_error[i]); 
    }
#endif 
    std::sort(bindices.begin(), bindices.end(), srtobject); 

    std::vector<double> sol_vec_bnd(bindices.size()); 
    std::vector<double> sol_exact_bnd(bindices.size()); 

    std::vector<double> sol_vec_int(nb_nodes - bindices.size()); 
    std::vector<double> sol_exact_int(nb_nodes - bindices.size()); 

    int i = 0;  // Index on boundary
    int k = 0;  // index on interior
    //for (int j = 0; j < sol_vec.size(); j++) {
    for (int j = 0; j < nb_nodes; j++) {
        // Skim off the boundary
        if (j == bindices[i]) {
            sol_vec_bnd[i] = sol_vec[j]; 
            sol_exact_bnd[i] = sol_exact[j]; 
            i++; 
            //  std::cout << "BOUNDARY: " << i << " / " << j << std::endl;
        } else {
            sol_vec_int[k] = sol_vec[j]; 
            sol_exact_int[k] = sol_exact[j]; 
            k++; 
            // std::cout << "INTERIOR: " << k << " / " << j <<  std::endl;
        }
    }

    //    writeErrorToFile(sol_error);

    calcSolNorms(sol_vec, sol_exact, "Interior & Boundary", rel_err_max);  // Full domain
    calcSolNorms(sol_vec_int, sol_exact_int, "Interior", rel_err_max);  // Interior only
    calcSolNorms(sol_vec_bnd, sol_exact_bnd, "Boundary", rel_err_max);  // Boundary only
}

void PDE::calcSolNorms(std::vector<double>& sol_vec, std::vector<double>& sol_exact, std::string label, double rel_err_max) {
    // We want: || x_exact - x_approx ||_{1,2,inf} 
    // and  || x_exact - x_approx ||_{1,2,inf} / || x_exact ||_{1,2,inf}

    double l1fabs = l1norm(sol_vec, sol_exact); 
    double l1rel = (l1norm(sol_exact) > 1e-10) ? l1fabs/l1norm(sol_exact) : 0.;
    double l2fabs = l2norm(sol_vec, sol_exact); 
    double l2rel = (l2norm(sol_exact) > 1e-10) ? l2fabs/l2norm(sol_exact) : 0.;
    double lifabs = linfnorm(sol_vec, sol_exact); 
    double lirel = (linfnorm(sol_exact) > 1e-10) ? lifabs/linfnorm(sol_exact) : 0.;

    // Only print this when we're looking at the global norms
    if (!label.compare("")) {
        printf("========= Norms For Current Solution ========\n"); 
        printf("Absolute =>  || x_exact - x_approx ||_p                    ,  where p={1,2,inf}\n"); 
        printf("Relative =>  || x_exact - x_approx ||_p / || x_exact ||_p  ,  where p={1,2,inf}\n"); 
    }

    printf("%s l1 error : Absolute = %f, Relative = %f\n", label.c_str(), l1fabs, l1rel );
    printf("%s l2 error : Absolute = %f, Relative = %f\n", label.c_str(), l2fabs, l2rel );
    printf("%s linf error : Absolute = %f, Relative = %f\n", label.c_str(), lifabs, lirel);

    if (l1rel > rel_err_max) {
        printf("[PDE] Error! l1 relative error (=%f%%) is too high to continue. We require %f%% or less.\n", 100.*l1rel, 100.*rel_err_max); 
        exit(EXIT_FAILURE);
    }
    if (l2rel > rel_err_max) {
        printf("[PDE] Error! l2 relative error (=%f%%) is too high to continue. We require %f%% or less.\n", 100.*l2rel, 100.*rel_err_max); 
        exit(EXIT_FAILURE);
    }
    if (lirel > rel_err_max) {
        printf("[PDE] Error! linf relative error (=%f%%) is too high to continue. We require %f%% or less.\n", 100.*lirel, 100.*rel_err_max); 
        exit(EXIT_FAILURE);
    }


}
