#ifndef __COMMUNICATOR_H__
#define __COMMUNICATOR_H__

#include <mpi.h>
#include <vector> 
#include "Vec3.h"
#include "gpu.h"

class StencilMessage {
    public: 
	int node_list_size; 		// Number of nodes in message
	std::vector<int> receiver_list; 	// Set of GPUs receiving nodes
	std::vector<Vec3*> node_list; 	// List of nodes and values
	std::vector<int> offset_list; 	// Offsets into reference node_list;
};


class Communicator {
	public: 
		Communicator(int argc, char** argv); 
		~Communicator(); 

		// Return unique ID for this compute node
		int getRank();
		
		// Return number of compute nodes registered with MPI
		int getSize();

		// Send our special message format (incomplete)
		void sendMessage(StencilMessage*); 
		
		// Receive our special message format (incomplete)
		StencilMessage* receiveMessage(); 

		// Setup MPI_Send for CPU0
		void sendObject(MPISendable* object, int reciever_rank);
		
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
		
		
	private: 
		int comm_size; 
		int comm_rank;
};

#endif
