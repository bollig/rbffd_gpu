#include <mpi.h>
#include <cmath>
#include <iostream>

using namespace std; 

#include "communicator.h"

Communicator::Communicator(int argc, char** argv) {
 	MPI_Init(&argc, &argv); 

	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
}


Communicator::~Communicator() {
	MPI_Finalize();
}


int Communicator::getSize() {
	return comm_size;
}


int Communicator::getRank() {
	return comm_rank;
}


void Communicator::sendMessage(StencilMessage* message) {
	int tag = 0; 
	double send_buffer[512]; 

	// This class should be inherited and this method overridden for various comm tests
	// Message structure; 
	// 	1) # of nodes  		= not needed
	// 	2) receiver list	= one SEND/RECV for each
	// 	3) node list 		= Unique to each SEND/RECV (array of doubles x3)
	// 	4) offset list 		= not needed
	
	// Round robin: each CPU takes a turn at sending message to CPUs that need nodes
	for (int j = 0; j < comm_size; j++) {
		if (comm_rank == j) {
			// Warning! Sending full list of nodes to all receiving units. This is bloated
			// and we should consider reducing the size
			for (int i = 0; i < message->receiver_list.size(); i++) {

				int msg_sz = message->node_list.size() * 3; 

 				// this should be done at code init so we anticipate how many to receive from each CPU.   
				MPI::COMM_WORLD.Send(&msg_sz, 1, MPI::INT, message->receiver_list[i], tag);
				int count = 0; 
				for (int k = 0; k < message->node_list.size(); k++) {
					send_buffer[count++] = message->node_list[k]->x(); 
					send_buffer[count++] = message->node_list[k]->y(); 
					send_buffer[count++] = message->node_list[k]->z(); 
				}
				MPI::COMM_WORLD.Send(&send_buffer, message->node_list.size()*3, MPI::DOUBLE, message->receiver_list[i], tag);	
				cout << "CPU " << j << " SEND " << message->node_list_size << " DOUBLES COMPLETE TO CPU " << message->receiver_list[i] << endl;
			}
		} else {
			int receive_size = 0; 
			// The needy CPUs
			MPI::COMM_WORLD.Recv(&receive_size, 1, MPI::INT, j, tag);  
			MPI::COMM_WORLD.Recv(&send_buffer, receive_size, MPI::DOUBLE, j, tag);  
			if (comm_rank == 0) {
				cout << "\tCPU " << comm_rank << " RECEIVED " << receive_size << " DOUBLES FROM CPU " << j << endl;
				for (int p = 0; p < receive_size; p+=3) {
					cout << "\tCPU " << comm_rank << ": " << send_buffer[p] << ", " << send_buffer[p+1] << ", " << send_buffer[p+2] <<  endl;
				}
			}
		}
	}
}

void Communicator::sendObject(MPISendable* object, int receiver_rank)
{
	object->send(this->getRank(), receiver_rank); 
}

int Communicator::receiveObject(MPISendable* object, int sender_rank)
{
	object->receive(this->getRank(), sender_rank); 
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
			for (int i = 0; i < receiver_list.size(); i++) {
				object->sendUpdate(this->getRank(), receiver_list[i]);
			}
		} else {						// All CPUs listen
			object->receiveUpdate(this->getRank(), j);
		}
	}
}

void Communicator::consolidateObjects(MPISendable* object)
{
	// Consolidate all objects into a master object on CPU0.
	// (TODO) We could also imagine conslidating the a local master and then
	// further consolidating the local masters into the global master to reduce
	// communication overhead.
	if (this->getRank() == 0) {			// TODO Make this a constant for MASTER_CPU
		// Add master CPU contribution to the final solution
		object->initFinal();
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
}


void Communicator::barrier() 
{
	MPI_Barrier(MPI_COMM_WORLD);
}
