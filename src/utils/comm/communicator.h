#ifndef __COMMUNICATOR_H__
#define __COMMUNICATOR_H__

#include <mpi.h>
#include <vector> 
#include "Vec3.h"
#include "utils/comm/mpisendable.h"

class Communicator {
public:
    Communicator(int argc, char** argv);
    virtual ~Communicator();

    MPI_Comm getComm() const { 
        return MPI_COMM_WORLD; 
    }

    // Return unique ID for this compute node
    int getRank() const;

    // Return number of compute nodes registered with MPI
    int getSize() const;

    // Setup MPI_Send for CPU0
    int sendObject(MPISendable* object, int reciever_rank);

    // Setup MPI_Recv for any CPUs expecting subdomain from CPU0
    int receiveObject(MPISendable* object, int sender_rank);

    // As a CPU, loop through CPUs and send updates required by each.
    // Round Robin style posting as ALL CPUs will need opportunity to broadcast
    void broadcastObjectUpdates(MPISendable* object);

    // When this is called, all subdomains must pass their contribution in
    // the solution to the master CPU. The master CPU resizes its buffers
    // to receives the solution. NOTE: each CPU must send both solution data
    // AND the global indices for said data so the master can organize it correctly
    void consolidateObjects(MPISendable* object);

    //Cause barrier for all CPUs
    void barrier();

    // True if this instance matches Communicator::MASTER
    bool isMaster() const { return (this->comm_rank == Communicator::MASTER); }

public:
    static const int MASTER = 0;  // The rank associated with the master processor

private:
    int comm_size;
    int comm_rank;
};

#endif
