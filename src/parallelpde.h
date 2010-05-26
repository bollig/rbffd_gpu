#ifndef __PARALLEL_PDE__
#define __PARALLEL_PDE__

#include <vector>

#include "communicator.h"

class ParallelPDE {
    //--------------------------------------------------------------
    // PROPERTIES
    //--------------------------------------------------------------
protected:
    double dt;
    std::vector<double> solution[2]; // Double buffered solution (current(0) and intermediate(1))

    Domain* myDomain;



    //--------------------------------------------------------------
    // PURE VIRTUAL METHODS
    //--------------------------------------------------------------
public:

    // NOTE: globaldomain is NULL (or ignored) for all processors, except the comm_unit::MASTER processor.
    ParallelPDE(Domain* globaldomain, Communicator* comm_unit) {
        printf("Inside ParallelPDE Constructor\n");
        
        // Start with an empty domain. This will change below
        this->myDomain = new Domain();

        // Update the empty domain by receiving data from the master processor
        if (comm_unit->isMaster()) {
            // Decompose the domain evenly across all processors
            std::vector<Domain*> subdomains = globaldomain->decomposeDomain(comm_unit->getSize());

            // Send subdomains to processors and update myDomain with information for this processor
            comm_unit->distributeObjects(subdomains, &myDomain);
        } else {
            // Receive a subdomain from the MASTER node
            comm_unit->receiveObject(&myDomain, Communicator::MASTER);
        }
        comm_unit->barrier();
    }

    // Setup: initial conditions, boundary conditions, etc. 
    virtual void Initialize() = 0;

    // Advance all subdomains by timestep dt. 
    virtual void AdvanceTimestep(double dt) = 0;

    // Reduce all norms and return the global norm value
    virtual double CheckNorm() = 0;

    // Reduce all solution parts to the globaldomain object on the master processor
    virtual void Consolidate(Domain* globaldomain) = 0;

    //--------------------------------------------------------------
    // IMPLEMENTED (BUT EXTENDABLE) METHODS
    //--------------------------------------------------------------
public:
    //----------------------------------------------------------------------

    virtual double maxNorm() {
        return this->maxNorm(sol[0]);
    }

    //----------------------------------------------------------------------

    virtual double maxNorm(vector<double> sol) {
        double nrm = 0.;
        for (int i = 0; i < sol.size(); i++) {
            double s = abs(sol[i]);
            if (s > nrm)
                nrm = s;
        }
        return nrm;
    }


    //--------------------------------------------------------------
    // IMPLEMENTED (AND NON-VIRTUAL) METHODS
    //--------------------------------------------------------------
public:
    // set the time step

    void setDt(double _dt) {
        this->dt = _dt;
    }


};

#endif //__PARALLEL_PDE__