#ifndef __MPI_SENDABLE__
#define __MPI_SENDABLE__

#include <mpi.h>
#include <vector> 
#include <set> 
#include <map>
#include "Vec3.h"
 
class MPISendable
{
protected:
 	static const int TAG = 0; 		// All MPISendable messages will have tag=0
	
public: 
	// [Pure virtual] Pack and send an instance of MPISendable to the receiver_rank CPU
	virtual int send(int my_rank, int receiver_rank)  =0; 
	
	// [Pure virtual] Receive and unpack an instance of MPISendable from the sender_rank CPU
	virtual int receive(int my_rank, int sender_rank) =0; 
	
	// [Pure virtual] Pack and send updates to an MPISendable object to the receiver_rank CPU
	virtual int sendUpdate(int my_rank, int receiver_rank)  =0; 
	
	// [Pure virtual] Receive and unpack updates to an MPISendable object
	virtual int receiveUpdate(int my_rank, int sender_rank) =0;
	

	// [Pure virtual] Pack and send a finalized MPISendable object to the receiver_rank CPU
	// Finalized implies that the code is near completion and no more whatever distributed
	// computation/values need to be reduced/reassembled.
	virtual int sendFinal(int my_rank, int receiver_rank)  =0;

	// [Pure virtual] receive and unpack a finalized MPISendable object
	virtual int receiveFinal(int my_rank, int sender_rank) =0;

	// [Pure virtual] initize buffers for finalization. For example if finalizing implies
	// consolidating a solution vector onto the master CPU then the master must initialize
	// the vector by populating it with its own solution contribution.
	virtual int initFinal() =0;


public: 
	// Generic sends/receives for different STL types. These should accept template types in the future
	// Send a STL::Vector (pack as double array and send via MPI)	
	int sendSTL(const std::vector<double> *origin, int myrank, int recv_rank) const ; 
	// Send a STL::Vector (pack as int array and send via MPI)
	int sendSTL(const std::vector<int> *origin, int myrank, int recv_rank) const ; 
	
    int sendSTL(const std::vector<size_t> *origin, int myrank, int recv_rank) const ; 
	
    // Send a STL::Map (pack as int[2] array and send via MPI)
	int sendSTL(const std::map<int, int> *origin, int myrank, int recv_rank) const ; 
	// Send a STL::Set (pack as int array and send via MPI)
	int sendSTL(const std::set<int> *origin, int myrank, int recv_rank) const ; 
	// Send a STL::Set (pack as int array and send via MPI)
	int sendSTL(const std::vector<Vec3> *origin, int myrank, int recv_rank) const ; 

	int sendSTL(const std::vector<std::vector<size_t> > *origin, int myrank, int recv_rank) const ; 

	int sendSTL(const std::vector<std::vector<int> > *origin, int myrank, int recv_rank) const ; 
	int sendSTL(const std::set<std::vector<Vec3> > *origin, int myrank, int recv_rank) const ; 

    // Single element of type size_t (multiple would be "const size_t **origin"
    int sendSTL(const size_t *origin, int myrank, int recv_rank) const ; 
    int sendSTL(const int *origin, int myrank, int recv_rank) const ; 

	// Recv double array and pack as STL::Vector.
	int recvSTL(std::vector<double> *destination, int myrank, int sender_rank); 
	// Recv int array and pack as STL::Vector.
	int recvSTL(std::vector<int> *destination, int myrank, int sender_rank); 
	int recvSTL(std::vector<size_t> *destination, int myrank, int sender_rank); 
	// Recv int[2] array and pack as STL::Map.
	int recvSTL(std::map<int, int> *destination, int myrank, int sender_rank); 	
	// Recv int array and pack as STL::Set.
	int recvSTL(std::set<int> *destination, int myrank, int sender_rank); 
	// Recv double[3] array and pack as STL::Set.
	int recvSTL(std::vector<Vec3> *destination, int myrank, int sender_rank); 
	
	int recvSTL(std::vector<std::vector<int> > *destination, int myrank, int sender_rank); 
	int recvSTL(std::vector<std::vector<size_t> > *destination, int myrank, int sender_rank); 
	int recvSTL(std::set<std::vector<Vec3> > *destination, int myrank, int sender_rank); 
	
    int recvSTL(size_t *destination, int myrank, int sender_rank); 
    int recvSTL(int *destination, int myrank, int sender_rank); 
	
};

#endif
