#include <iostream>
#include <vector> 

#include <communicator.h>

using namespace std; 

int main (int argc, char** argv) {
	Communicator* comm_unit = new Communicator(argc, argv); 


	cout << " Got Rank: " << comm_unit->getRank() << endl;
	cout << " Got Size: " << comm_unit->getSize() << endl;

	StencilMessage* sm = new StencilMessage(); 


	// Receiving GPUs
	for (int i=0; i < comm_unit->getSize(); i++) {
		if (i != comm_unit->getRank()) {
			sm->receiver_list.push_back(i); 
		}
	}

	Vec3* v; 
	// Set R
	sm->size = 0; 


	for (int i=0; i < comm_unit->getSize(); i++) {
		if (i != comm_unit->getRank()) {
			cout << "CPU " << comm_unit->getRank() << " adding offset: " << sm->size << " for unit " << i << endl;
			sm->offset_list.push_back(sm->size); 
			// Nodes specific to each receiver
		 	for (int j = 0; j < comm_unit->getRank(); j++) {	
				// Test: verify communication works right and that we get all offsets for each CPU
				v = new Vec3((double)comm_unit->getRank(),(double)sm->size,(double)sm->offset_list.size()); 
				sm->node_list.push_back(v); 
				sm->size++; 
			}


		}

	}

	comm_unit->sendMessage(sm);

	return EXIT_SUCCESS; 
}
