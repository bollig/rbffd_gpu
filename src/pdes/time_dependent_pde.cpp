#include "time_dependent_pde.h"

#include <iostream>

void TimeDependentPDE::fillInitialConditions() {
    ;
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



