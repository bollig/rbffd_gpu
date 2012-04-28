#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdlib.h> 

using namespace std; 

#include "communicator.h"

extern "C" {
    int comm_destruct_called =0;
}
void closeAllMPI(void) {
    std::cout << "[Communicator] Comm Unit Destroyed?: " << comm_destruct_called << std::endl;
    if (!comm_destruct_called) {
	char* myhostname; 
	myhostname = getenv("HOSTNAME"); 	

        std::cout << "[Communicator] calling MPI_Abort on host: " << myhostname << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    } else {
        std::cout << "[Communicator] already finalized. Nothing to do atexit" << std::endl;
    }
}

Communicator::Communicator(int argc, char** argv) {
 	MPI_Init(&argc, &argv); 

	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    atexit(closeAllMPI);
}


Communicator::~Communicator() {
    std::cout << "[Communicator] Rank " << this->getRank() << " calling MPI_finalize\n";
    MPI_Finalize();
    std::cout << "[Communicator] destroying communicator on rank " << this->getRank() << ", delaying MPI_Finalize within atexit()." << std::endl;
    comm_destruct_called = 1;
}


int Communicator::getSize() const {
	return comm_size;
}


int Communicator::getRank() const {
	return comm_rank;
}


int Communicator::sendObject(MPISendable* object, int receiver_rank)
{
	return object->send(this->getRank(), receiver_rank); 
}

int Communicator::receiveObject(MPISendable* object, int sender_rank)
{
	return object->receive(this->getRank(), sender_rank); 
}

// As a CPU, loop through CPUs and send updates required by each. 
// Round Robin style posting as ALL CPUs will need opportunity to broadcast
void Communicator::broadcastObjectUpdates(MPISendable* object)
{
	vector<int> receiver_list; 
	
	// This is BAD: we should only send to CPUs in need. 
	for (int i = 0; i < this->getSize(); i++) {
		if (i != this->getRank()) 
		{
				receiver_list.push_back(i); 	
		}
	}
	
	// Round robin: each CPU takes a turn at sending message to CPUs that need nodes
	for (int j = 0; j < this->getSize(); j++) {
		if (this->getRank() == j) {		// My turn
			for (size_t i = 0; i < receiver_list.size(); i++) {
				object->sendUpdate(this->getRank(), receiver_list[i]);
			}
		} else {						// All CPUs listen
			object->receiveUpdate(this->getRank(), j);
		}
	}
    this->barrier();
}

void Communicator::consolidateObjects(MPISendable* object)
{
	// Consolidate all objects into a master object on CPU0.
	// (TODO) We could also imagine conslidating the a local master and then
	// further consolidating the local masters into the global master to reduce
	// communication overhead.
	if (this->getRank() == 0) {			// TODO Make this a constant for MASTER_CPU
        // Add master CPU contribution to the final solution
		object->updateFinal();
		for (int i = 1; i < this->getSize(); i++) {
			object->receiveFinal(0, i);
		}
	} else {
		// Send to CPU0
		// 1) convert indices to global
		// 2) Send indices
		// 3) Send U_G[0 .. l2g(Q.size)]
		object->sendFinal(this->getRank(), 0);
	}
    this->barrier();
}


void Communicator::barrier() 
{
	MPI_Barrier(MPI_COMM_WORLD);
}
