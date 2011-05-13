#include "pde.h"

int PDE::send(int my_rank, int receiver_rank) {

}

int PDE::receive(int my_rank, int sender_rank) {;}
int PDE::sendUpdate(int my_rank, int receiver_rank){ ;} 
int PDE::receiveUpdate(int my_rank, int sender_rank){;}

int PDE::sendFinal(int my_rank, int receiver_rank){;}
int PDE::receiveFinal(int my_rank, int sender_rank){;}
int PDE::initFinal(){;}


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


#if 0
int PDE::sendUpdate(int my_rank, int receiver_rank) {

    if (my_rank != receiver_rank) {
        vector<int>::iterator oit;
        // vector<set<int> > O; gives us the list of (global) indices which we
        // are sending to receiver_rank		

        // We send a list of node values 
        vector<double> U_O;

        if (O_by_rank.size() > 0) { // Many segfault issues are caused by empty sets.
            //	cout << "O_by_rank is NOT size(0)" << endl;
            int i;

            for (oit = O_by_rank[receiver_rank].begin(); oit
                    != O_by_rank[receiver_rank].end(); oit++, i++) {
                // Elements in O are in global indices 
                // so we need to first convert to local to index our U_G
                U_O.push_back(U_G[g2l(*oit)]);
                if (DEBUG) {
                    cout << "SENDING CPU" << receiver_rank << " U_G[" << *oit
                        << " (local index: " << g2l(*oit) << ")"
                        << "]: " << U_G[g2l(*oit)] << endl;
                }
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
            if (DEBUG) {
                cout << "\t(Global Index): " << *rit << "\t (Local Index:" << g2l(
                            *rit) << ")\tOld U_G[" << g2l(*rit) << "]: " << U_G[g2l(
                            *rit)] << "\t New U_G[" << g2l(*rit) << "]: " << U_R[i]
                                                 << endl;
            }
            // Global to local mapping required
            U_G[g2l(*rit)] = U_R[i]; // Overwrite with new values
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

        if (O_by_rank.size() > 0) { // Many segfault issues are caused by empty sets.
            //	cout << "O_by_rank is NOT size(0)" << endl;
            int i;

            // TODO the set U_G should be in order (Q, R) so we dont need this full copy (maybe...)
            for (qit = Q.begin(); qit != Q.end(); qit++, i++) {
                // Elements in Q are in global indices
                // so we need to first convert to local to index our U_G
                U_Q.push_back(U_G[g2l(*qit)]);
                if (DEBUG) {
                    cout << "SENDING CPU" << receiver_rank << " U_G[" << *qit
                        << "]: " << U_G[g2l(*qit)] << endl;
                }
            }
        }

        sendSTL(&Q, my_rank, receiver_rank);
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
    if (DEBUG) {
        cout << "NEW FINAL_U_G: " << endl;
        map<int, double>::iterator it;
        for (it = global_U_G.begin(); it != global_U_G.end(); it++) {
            cout << "\tU_G[" << (*it).first << "] = " << (*it).second << endl;
        }
    }
}

int PDE::initFinal() {
    //pair<map<char,int>::iterator,bool> ret;
    set<int>::iterator qit;
    int i = 0;
    for (qit = Q.begin(); qit != Q.end(); qit++, i++) {
        global_U_G.insert(pair<int, double> (*qit, U_G[g2l(*qit)]));
    }
}

void PDE::getFinal(std::vector<double> *final) {
    map<int, double>::iterator it;
    final->clear();
    for (it = global_U_G.begin(); it != global_U_G.end(); it++) {
        final->push_back((*it).second);
    }
}

int PDE::writeFinal(std::vector<NodeType>& nodes, std::string filename) {
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
#endif 
