#include "mpisendable.h"

// TODO: make sure size_t send/recv works
// TODO: send vector data without copying to buffer http://stackoverflow.com/questions/2546298/vector-usage-in-mpic
// TODO: consider using boost serialization

using namespace std; 
//----------------------------------------------------------------------------
// 						ALL SENDS
//----------------------------------------------------------------------------


// Send a STL::Vector (pack as int array and send via MPI)
int MPISendable::sendSTL(const std::vector<double> *origin, int myrank, int recv_rank) const 
{
	int sz = origin->size(); 	// All nodes for GPU
	//cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
	MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

	if (sz > 0) {
		double buff[sz]; 
		for (int i = 0; i < sz; i++) {
			buff[i] = (*origin)[i]; 
		}

		MPI_Send(&buff, sz, MPI::DOUBLE, recv_rank, TAG, MPI_COMM_WORLD);
	}
	cout << "RANK " << myrank << " REPORTS: finished sending std::vector<double> to RANK " << recv_rank << endl;
	return sz; 
}

int MPISendable::sendSTL(const std::vector<int> *origin, int myrank, int recv_rank) const 
{
	int sz = origin->size(); 	// All nodes for GPU
	//cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
	MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

	if (sz > 0) {
		int buff[sz]; 
		for (int i = 0; i < sz; i++) {
			buff[i] = (*origin)[i]; 
		}

		MPI_Send(&buff, sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);	
	}
	cout << "RANK " << myrank << " REPORTS: finished sending std::vector<int> to RANK " << recv_rank << endl;
	return sz; 
}

int MPISendable::sendSTL(const std::vector<size_t> *origin, int myrank, int recv_rank) const 
{
	int sz = origin->size(); 	// All nodes for GPU
	//cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
	MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

	if (sz > 0) {
        MPI_Datatype type; 
        // We have to check the system size_t to know what it maps to in MPI
        if (sizeof(size_t) == sizeof(unsigned long)) {   // This is on my laptop (Evan)
            type = MPI::UNSIGNED_LONG;
        } else if (sizeof(size_t) == sizeof(unsigned int)) {
            type = MPI::UNSIGNED;  
        } else if (sizeof(size_t) == sizeof(unsigned short)) {
            type = MPI::UNSIGNED_SHORT; 
        } 
		size_t buff[sz]; 
		for (int i = 0; i < sz; i++) {
			buff[i] = (*origin)[i]; 
		}

		MPI_Send(&buff, sz, type, recv_rank, TAG, MPI_COMM_WORLD);	
	}
	cout << "RANK " << myrank << " REPORTS: finished sending std::vector<int> to RANK " << recv_rank << endl;
	return sz; 
}

// Send a STL::Set (pack as int array and send via MPI)
int MPISendable::sendSTL(const std::set<int> *origin, int myrank, int recv_rank) const 
{
	int sz = origin->size(); 	// All nodes for GPU
	//cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
	MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

	// Static allocations are needed for message passing. 
	int buff[sz]; 

	int i = 0;
	for (set<int>::const_iterator it=origin->begin(); it != origin->end(); it++, i++) {
		buff[i] = *it;
	}

	// Send stencil set Q (note: no values sent yet)
	MPI_Send(&buff, sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);
	cout << "RANK " << myrank << " REPORTS: finished sending std::set<int> to RANK " << recv_rank << endl;
}

// Send a STL::Set (pack as int array and send via MPI)
int MPISendable::sendSTL(const std::vector<Vec3> *origin, int myrank, int recv_rank) const 
{
	int sz = origin->size(); 	// All nodes for GPU
	//cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
	MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

	// Static allocations are needed for message passing. 
	double buff[sz][3]; 

	int i = 0;
	for (std::vector<Vec3>::const_iterator it=origin->begin(); it != origin->end(); it++, i++) {
		buff[i][0] = (*it).x();
		buff[i][1] = (*it).y();
		buff[i][2] = (*it).z();	
	}

	// Send stencil set Q (note: no values sent yet)
	MPI_Send(&buff, sz*3, MPI::DOUBLE, recv_rank, TAG, MPI_COMM_WORLD);
	cout << "RANK " << myrank << " REPORTS: finished sending std::set<Vec3> to RANK " << recv_rank << endl;
}


// Send a STL::Map (pack as int[2] array and send via MPI)
int MPISendable::sendSTL(const std::map<int, int> *origin, int myrank, int recv_rank) const 
{	
	int sz = origin->size();
	
	//cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
	MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

	// 0  =  KEY; 1 = VALUE
	int buff[sz][2]; 

	int i = 0;
	for (map<int, int>::const_iterator miter=origin->begin(); miter != origin->end(); miter++, i++) {
		buff[i][0] = (*miter).first;
		buff[i][1] = (*miter).second;
	}

	// Send stencil set Q (note: no values sent yet)
	MPI_Send(&buff, sz*2, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);
	cout << "RANK " << myrank << " REPORTS: finished sending std::map<int,int> to RANK " << recv_rank << endl;
}

int MPISendable::sendSTL(const std::vector<std::vector<size_t> > *origin, int myrank, int recv_rank) const 
{
	int sz = origin->size(); 	// All nodes for GPU
	
	//cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
	MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

	// Static allocations are needed for message passing. 
	int buff[sz]; 

	int totsize = 0;
	int i = 0;
	for (std::vector<std::vector<size_t> >::const_iterator it=origin->begin(); it != origin->end(); it++, i++) {
		const std::vector<size_t> *vint = &(*it);
		buff[i] = vint->size();
		totsize += vint->size(); 
	}

	// Send offsets into stencil buffer
	MPI_Send(&buff, sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);
	
	size_t buff2[totsize]; 
	int offset = 0; 
	for (std::vector<std::vector<size_t> >::const_iterator it=origin->begin(); it != origin->end(); it++) {
		const std::vector<size_t> *vint = &(*it);
		for (int i = 0; i < vint->size(); i++) {
			buff2[offset + i] = (*vint)[i];
		}
		offset += vint->size(); 
	}
	      MPI_Datatype type; 
        // We have to check the system size_t to know what it maps to in MPI
        if (sizeof(size_t) == sizeof(unsigned long)) {   // This is on my laptop (Evan)
            type = MPI::UNSIGNED_LONG;
        } else if (sizeof(size_t) == sizeof(unsigned int)) {
            type = MPI::UNSIGNED;  
        } else if (sizeof(size_t) == sizeof(unsigned short)) {
            type = MPI::UNSIGNED_SHORT; 
        } 

	// Send stencil buffer
	MPI_Send(&buff2, totsize, type, recv_rank, TAG, MPI_COMM_WORLD);
	
	cout << "RANK " << myrank << " REPORTS: finished sending std::set< std::vector<size_t> > to RANK " << recv_rank << endl;
    cout << "WARNING! size_t passing is not verified YET.\n";
}

int MPISendable::sendSTL(const std::vector<std::vector<int> > *origin, int myrank, int recv_rank) const 
{
	int sz = origin->size(); 	// All nodes for GPU
	
	//cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
	MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

	// Static allocations are needed for message passing. 
	int buff[sz]; 

	int totsize = 0;
	int i = 0;
	for (std::vector<std::vector<int> >::const_iterator it=origin->begin(); it != origin->end(); it++, i++) {
		const std::vector<int> *vint = &(*it);
		buff[i] = vint->size();
		totsize += vint->size(); 
	}

	// Send offsets into stencil buffer
	MPI_Send(&buff, sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);
	
	int buff2[totsize]; 
	int offset = 0; 
	for (std::vector<std::vector<int> >::const_iterator it=origin->begin(); it != origin->end(); it++) {
		const std::vector<int> *vint = &(*it);
		for (int i = 0; i < vint->size(); i++) {
			buff2[offset + i] = (*vint)[i];
		}
		offset += vint->size(); 
	}
	
	// Send stencil buffer
	MPI_Send(&buff2, totsize, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);
	
	cout << "RANK " << myrank << " REPORTS: finished sending std::set< std::vector<int> > to RANK " << recv_rank << endl;
}
 
int MPISendable::sendSTL(const std::set<std::vector<Vec3> > *origin, int myrank, int recv_rank) const
{
	int sz = origin->size(); 	// All nodes for GPU
	
	// length of set
	MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

	int buff[sz]; 
	
	int totsize = 0;
	int i = 0; 
	for (set<std::vector<Vec3> >::const_iterator it=origin->begin(); it != origin->end(); it++) {
		const std::vector<Vec3> *vint = &(*it);
		buff[i] = vint->size();
		totsize += vint->size(); 
	}

	// Offsets into into stencil buffer
	MPI_Send(&buff, sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);
	
	double buff2[totsize][3]; 
	int offset = 0; 
	for (set<std::vector<Vec3> >::const_iterator it=origin->begin(); it != origin->end(); it++) {
		const std::vector<Vec3> *vint = &(*it);
		for (int i = 0; i < vint->size(); i++) {
			buff2[offset + i][0] = (*vint)[i].x();
			buff2[offset + i][1] = (*vint)[i].y();
			buff2[offset + i][2] = (*vint)[i].z();
			cout << "COUNTING: " << offset + i << " of " << totsize << endl;
		}
		offset += vint->size(); 
	}
	
	// Raw data
	MPI_Send(&buff2, totsize * 3, MPI::DOUBLE, recv_rank, TAG, MPI_COMM_WORLD);
	
	cout << "RANK " << myrank << " REPORTS: finished sending std::set< std::vector<Vec3> > to RANK " << recv_rank << endl;
}



//----------------------------------------------------------------------------
// 						ALL RECEIVES
//----------------------------------------------------------------------------

// Recv double array and pack as STL::Vector.
int MPISendable::recvSTL(std::vector<double> *destination, int myrank, int sender_rank)
{
	MPI_Status stat; 

	int sz;
	MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// WARNING! THIS ERASES ALL ELEMENTS IN destination
	destination->clear();
	
	if (sz > 0) {
		double buff[sz];
		MPI_Recv(buff, sz, MPI::DOUBLE, sender_rank, TAG, MPI_COMM_WORLD, &stat);

		for (int i=0; i < sz; i++) {
			destination->push_back(buff[i]);
		}	
	}
	cout << "RANK " << myrank << " REPORTS: received std::vector<double> (size: " << sz << ") from RANK " << sender_rank << endl;
	return sz; 
}
// Recv int array and pack as STL::Vector.
int MPISendable::recvSTL(std::vector<int> *destination, int myrank, int sender_rank)
{
	MPI_Status stat; 

	int sz;
	MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// WARNING! THIS ERASES ALL ELEMENTS IN destination
	destination->clear(); 

	if (sz > 0) {
		int buff[sz];
		MPI_Recv(buff, sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

		for (int i=0; i < sz; i++) {
			destination->push_back(buff[i]);
		}	
	}
	cout << "RANK " << myrank << " REPORTS: received std::vector<int> (size: " << sz << ") from RANK " << sender_rank << endl;
	return sz; 
}

// Recv int array and pack as STL::Vector.
int MPISendable::recvSTL(std::vector<size_t> *destination, int myrank, int sender_rank)
{
	MPI_Status stat; 

	int sz;
	MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// WARNING! THIS ERASES ALL ELEMENTS IN destination
	destination->clear(); 

	if (sz > 0) {
        MPI_Datatype type; 
        // We have to check the system size_t to know what it maps to in MPI
        if (sizeof(size_t) == sizeof(unsigned long)) {   // This is on my laptop (Evan)
            type = MPI::UNSIGNED_LONG;
        } else if (sizeof(size_t) == sizeof(unsigned int)) {
            type = MPI::UNSIGNED;  
        } else if (sizeof(size_t) == sizeof(unsigned short)) {
            type = MPI::UNSIGNED_SHORT; 
        } 

		size_t buff[sz];
		MPI_Recv(buff, sz, type, sender_rank, TAG, MPI_COMM_WORLD, &stat);

		for (int i=0; i < sz; i++) {
			destination->push_back(buff[i]);
		}	
	}
	cout << "RANK " << myrank << " REPORTS: received std::vector<int> (size: " << sz << ") from RANK " << sender_rank << endl;
	return sz; 
}


// Recv int[2] array and pack as STL::Map.
int MPISendable::recvSTL(std::map<int, int> *destination, int myrank, int sender_rank)
{
	MPI_Status stat; 

	int sz;
	MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	int buff[sz][2];
	MPI_Recv(buff, sz*2, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// WARNING! THIS ERASES ALL ELEMENTS IN destination
	destination->clear(); 
	
	for (int i=0; i < sz; i++) {
		(*destination)[buff[i][0]] = buff[i][1];
	}

	cout << "RANK " << myrank << " REPORTS: received std::map<int,int> (size: " << sz << ") from RANK " << sender_rank << endl;	
}

// Recv int array and pack as STL::Set.
int MPISendable::recvSTL(std::set<int> *destination, int myrank, int sender_rank)
{
	MPI_Status stat; 

	int sz;
	MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	int buff[sz];
	MPI_Recv(buff, sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// WARNING! THIS ERASES ALL ELEMENTS IN destination
	destination->clear(); 
	
	for (int i=0; i < sz; i++) {
		destination->insert(buff[i]);
	}	
	cout << "RANK " << myrank << " REPORTS: received std::set<int> (size: " << sz << ") from RANK " << sender_rank << endl;
}

// Recv int array and pack as STL::Set.
int MPISendable::recvSTL(std::vector<Vec3> *destination, int myrank, int sender_rank)
{
	MPI_Status stat; 

	int sz;
	MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	double buff[sz][3];
	MPI_Recv(buff, sz*3, MPI::DOUBLE, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// WARNING! THIS ERASES ALL ELEMENTS IN destination
	destination->clear(); 
	
	for (int i=0; i < sz; i++) {
		Vec3 v(buff[i][0], buff[i][1], buff[i][2]);
		destination->push_back(v);
	}	
	cout << "RANK " << myrank << " REPORTS: received std::set<Vec3> (size: " << sz << ") from RANK " << sender_rank << endl;
}

int MPISendable::recvSTL(std::vector<std::vector<size_t> > *destination, int myrank, int sender_rank)
{
	MPI_Status stat; 

	// Length of set
	int sz;
	MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// Offsets into vector
	int buff[sz];
	MPI_Recv(buff, sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);
	
	int totsize = 0;
	for (int i=0; i < sz; i++) {
		totsize += buff[i]; 
	}

        MPI_Datatype type; 
        // We have to check the system size_t to know what it maps to in MPI
        if (sizeof(size_t) == sizeof(unsigned long)) {   // This is on my laptop (Evan)
            type = MPI::UNSIGNED_LONG;
        } else if (sizeof(size_t) == sizeof(unsigned int)) {
            type = MPI::UNSIGNED;  
        } else if (sizeof(size_t) == sizeof(unsigned short)) {
            type = MPI::UNSIGNED_SHORT; 
        } 


	// Raw data
	size_t buff2[totsize];
	MPI_Recv(buff2, totsize, type, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// WARNING! THIS ERASES ALL ELEMENTS IN destination
	destination->clear(); 
	
	int offset = 0; 
	for (int i=0; i < sz; i++) {
		std::vector<size_t> temp; 
		for (int j=0; j < buff[i]; j++) {
			temp.push_back(buff2[offset+j]);
		}
		offset += buff[i]; 
		destination->push_back(temp);
	}	
	cout << "RANK " << myrank << " REPORTS: received std::set<int> (size: " << sz << ") from RANK " << sender_rank << endl;	
    cout << "WARNING! size_t sending is not verified YET.\n"; 
}


int MPISendable::recvSTL(std::vector<std::vector<int> > *destination, int myrank, int sender_rank)
{
	MPI_Status stat; 

	// Length of set
	int sz;
	MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// Offsets into vector
	int buff[sz];
	MPI_Recv(buff, sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);
	
	int totsize = 0;
	for (int i=0; i < sz; i++) {
		totsize += buff[i]; 
	}

	// Raw data
	int buff2[totsize];
	MPI_Recv(buff2, totsize, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// WARNING! THIS ERASES ALL ELEMENTS IN destination
	destination->clear(); 
	
	int offset = 0; 
	for (int i=0; i < sz; i++) {
		std::vector<int> temp; 
		for (int j=0; j < buff[i]; j++) {
			temp.push_back(buff2[offset+j]);
		}
		offset += buff[i]; 
		destination->push_back(temp);
	}	
	cout << "RANK " << myrank << " REPORTS: received std::set<int> (size: " << sz << ") from RANK " << sender_rank << endl;	
}

int MPISendable::recvSTL(std::set<std::vector<Vec3> > *destination, int myrank, int sender_rank)
{
	MPI_Status stat; 

	// Length of set
	int sz;
	MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// Offsets into vector
	int buff[sz];
	MPI_Recv(buff, sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);
	
	int totsize = 0;
	for (int i=0; i < sz; i++) {
		totsize += buff[i]; 
	}

	// Raw data
	double buff2[totsize][3];
	MPI_Recv(buff2, totsize*3, MPI::DOUBLE, sender_rank, TAG, MPI_COMM_WORLD, &stat);

	// WARNING! THIS ERASES ALL ELEMENTS IN destination
	destination->clear(); 
	
	int offset = 0; 
	for (int i=0; i < sz; i++) {
		std::vector<Vec3> temp; 
		for (int j=0; j < buff[i]; j++) {
			Vec3 v(buff2[offset+j][0], buff2[offset+j][1], buff2[offset+j][2]); // cast to ensure correct constructor
			temp.push_back(v);
		}
		offset += buff[i]; 
		destination->insert(temp);
	}	
	cout << "RANK " << myrank << " REPORTS: received std::set<int> (size: " << sz << ") from RANK " << sender_rank << endl;
}
