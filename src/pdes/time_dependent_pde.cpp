#include "time_dependent_pde.h"

#include <iostream>

void TimeDependentPDE::fillInitialConditions() {
    vector<SolutionType>& s = U_G;

    std::set<int>& Q = grid_ref.Q;			// All stencil centers in this CPU's QUEUE							

    // Only fill the solution values under this procs control 
    // NOTE: any remaining values will be 0 initially

    printf("=========== Initial Conditions ===========\n");

    std::set<int>::iterator Q_iter; 

    for (Q_iter = Q.begin(); Q_iter != Q.end(); Q_iter++) {
        NodeType& v = grid_ref.getNode(*Q_iter); 
        s[grid_ref.g2l(*Q_iter)] = *Q_iter; 
    }

#if 0
    printf("Using ExactSolution to fill initial conditions for interior\n");
    for (int i = 0; i < s.size(); i++) {
        Vec3& v = rbf_centers[i];
        //s[i] = exp(-alpha*v.square());
        //s[i] = 1. - v.square();
        s[i] = exactSolution->at(v, 0.);
        // printf("interior: %f %f %f ==> %f\n", v.x(), v.y(), v.z(),s[i]);
        //s[i] = 1.0;
        //printf("s[%d]= %f\n", i, s[i]);
    }
    //exit(0);

    vector<size_t>& bnd_index = boundary_set; //grid.getBoundary();
    int sz = bnd_index.size();
    bnd_sol.resize(sz);


    printf("Copying solution to bnd_sol (boundary solution buffer)\n"); 
    for (int i = 0; i < sz; i++) {
        bnd_sol[i] = s[bnd_index[i]];
        Vec3& v = rbf_centers[bnd_index[i]];
        // printf("boundary: %f %f %f ==> %f\n", v.x(), v.y(), v.z(),bnd_sol[i]);
    }

    if (solution != NULL) {
        cout << "Copying initial conditions to output buffer" << endl;
        for (int i = 0; i < s.size(); i++) {
            Vec3& v = rbf_centers[i];
            (*solution)[i] = s[i];
            // printf("interior: %f %f %f ==> %f\n", v.x(), v.y(), v.z(),s[i]);
        }
    }
#endif 
    printf("============ End Initial Conditions ===========\n");
}

// Advancing requires: 
//  - computing an update to the current solution (i.e., calling applyWeightsForDerivs(currentSolution)) 
//  - applying the updates to the current solution (i.e., RK45 weighted summation of intermediate updates).
void TimeDependentPDE::advance(TimeScheme& which) {
    switch (which) 
    {
        case FIRST_EULER: 
            advanceFirstEuler(); 
            break; 
        case SECOND_EULER: 
            advanceSecondEuler();
            break;  
        default: 
            std::cout << "[TimeDependentPDE] Invalid TimeScheme specified. Bailing...\n";
            exit(EXIT_FAILURE); 
            break; 
    };
}

void TimeDependentPDE::advanceFirstEuler() {;}
void TimeDependentPDE::advanceSecondEuler() {;}



