#include "mpisendable.h"
#include <stdlib.h> 

// TODO: make sure unsigned int send/recv works
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
        double* buff = new double[sz]; 
        for (int i = 0; i < sz; i++) {
            buff[i] = (*origin)[i]; 
        }

        MPI_Send(&buff[0], sz, MPI::DOUBLE, recv_rank, TAG, MPI_COMM_WORLD);
        delete [] buff; 
    }
//    cout << "RANK " << myrank << " REPORTS: finished sending std::vector<double> to RANK " << recv_rank << endl;
    return sz; 
}

//----------------------------------------------------------------------------

int MPISendable::sendSTL(const std::vector<int> *origin, int myrank, int recv_rank) const 
{
    int sz = origin->size(); 	// All nodes for GPU
    //cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
    MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    if (sz > 0) {
        int* buff = new int[sz]; 
        for (int i = 0; i < sz; i++) {
            buff[i] = (*origin)[i]; 
        }

        MPI_Send(&buff[0], sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);
        delete [] buff; 
    }
//    cout << "RANK " << myrank << " REPORTS: finished sending std::vector<int> to RANK " << recv_rank << endl;
    return sz; 
}

//----------------------------------------------------------------------------

int MPISendable::sendSTL(const std::vector<unsigned int> *origin, int myrank, int recv_rank) const 
{
    int sz = origin->size(); 	// All nodes for GPU
    //cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
    MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    if (sz > 0) {
        MPI_Datatype type = MPI::UNSIGNED; 

        unsigned int* buff = new unsigned int[sz]; 
        for (int i = 0; i < sz; i++) {
            buff[i] = (*origin)[i]; 
        }

        MPI_Send(&buff[0], sz, type, recv_rank, TAG, MPI_COMM_WORLD);	
        delete [] buff; 
    }
 //   cout << "RANK " << myrank << " REPORTS: finished sending std::vector<int> to RANK " << recv_rank << endl;
    return sz; 
}

//----------------------------------------------------------------------------

// Send a STL::Set (pack as int array and send via MPI)
int MPISendable::sendSTL(const std::set<int> *origin, int myrank, int recv_rank) const 
{
    int sz = origin->size(); 	// All nodes for GPU
    //cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
    MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    // Static allocations are needed for message passing. 
    int* buff = new int[sz]; 

    int i = 0;
    for (set<int>::const_iterator it=origin->begin(); it != origin->end(); it++, i++) {
        buff[i] = *it;
    }

    // Send stencil set Q (note: no values sent yet)
    MPI_Send(&buff[0], sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    //  cout << "RANK " << myrank << " REPORTS: finished sending std::set<int> to RANK " << recv_rank << endl;
    delete [] buff; 
    //  cout << "RANK " << myrank << " buff was freed\n";  
}

//----------------------------------------------------------------------------

// Send a STL::Set (pack as int array and send via MPI)
int MPISendable::sendSTL(const std::vector<Vec3> *origin, int myrank, int recv_rank) const 
{
    int sz = origin->size(); 	// All nodes for GPU
    //cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
    MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    // Static allocations are needed for message passing. 
    double* buff = new double[sz*3]; 

    int i = 0;
    for (std::vector<Vec3>::const_iterator it=origin->begin(); it != origin->end(); it++, i++) {
        buff[i*3 + 0] = (*it).x();
        buff[i*3 + 1] = (*it).y();
        buff[i*3 + 2] = (*it).z();	
    }

    // Send stencil set Q (note: no values sent yet)
    MPI_Send(&buff[0], sz*3, MPI::DOUBLE, recv_rank, TAG, MPI_COMM_WORLD);
    //  cout << "RANK " << myrank << " REPORTS: finished sending std::set<Vec3> to RANK " << recv_rank << endl;
    delete [] buff; 
}

//----------------------------------------------------------------------------


// Send a STL::Map (pack as int[2] array and send via MPI)
int MPISendable::sendSTL(const std::map<int, int> *origin, int myrank, int recv_rank) const 
{	
    int sz = origin->size();

    //cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
    MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    // 0  =  KEY; 1 = VALUE
    int* buff = new int[sz*2]; 

    int i = 0;
    for (map<int, int>::const_iterator miter=origin->begin(); miter != origin->end(); miter++, i++) {
        buff[i*2 + 0] = (*miter).first;
        buff[i*2 + 1] = (*miter).second;
    }

    // Send stencil set Q (note: no values sent yet)
    MPI_Send(&buff[0], sz*2, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);
    //  cout << "RANK " << myrank << " REPORTS: finished sending std::map<int,int> to RANK " << recv_rank << endl;
    delete [] buff; 
}

//----------------------------------------------------------------------------

int MPISendable::sendSTL(const std::vector<std::vector<unsigned int> > *origin, int myrank, int recv_rank) const 
{
    int sz = origin->size(); 	// All nodes for GPU

    //cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
    MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    // Static allocations are needed for message passing. 
    int* buff = new int[sz]; 

    int totsize = 0;
    int i = 0;
    for (std::vector<std::vector<unsigned int> >::const_iterator it=origin->begin(); it != origin->end(); it++, i++) {
        const std::vector<unsigned int> *vint = &(*it);
        buff[i] = vint->size();
        totsize += vint->size(); 
    }

    // Send offsets into stencil buffer
    MPI_Send(&buff[0], sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    unsigned int* buff2 = new unsigned int[totsize]; 
    int offset = 0; 
    for (std::vector<std::vector<unsigned int> >::const_iterator it=origin->begin(); it != origin->end(); it++) {
        const std::vector<unsigned int> *vint = &(*it);
        for (int i = 0; i < vint->size(); i++) {
            buff2[offset + i] = (*vint)[i];
        }
        offset += vint->size(); 
    }
    MPI_Datatype type = MPI::UNSIGNED; 
    // Send stencil buffer
    MPI_Send(&buff2[0], totsize, type, recv_rank, TAG, MPI_COMM_WORLD);

    //  cout << "RANK " << myrank << " REPORTS: finished sending std::set< std::vector<unsigned int> > to RANK " << recv_rank << endl;
    //  cout << "WARNING! unsigned int passing is not verified YET.\n";
    delete [] buff;
    delete [] buff2;
}

//----------------------------------------------------------------------------

int MPISendable::sendSTL(const std::vector<std::vector<int> > *origin, int myrank, int recv_rank) const 
{
    int sz = origin->size(); 	// All nodes for GPU

    //cout << "RANK " << my_rank << ": sending GPU size " << sz << " to RANK " << receiver_rank << endl;
    MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    // Static allocations are needed for message passing. 
    int* buff = new int[sz]; 

    int totsize = 0;
    int i = 0;
    for (std::vector<std::vector<int> >::const_iterator it=origin->begin(); it != origin->end(); it++, i++) {
        const std::vector<int> *vint = &(*it);
        buff[i] = vint->size();
        totsize += vint->size(); 
    }

    // Send offsets into stencil buffer
    MPI_Send(&buff[0], sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    int* buff2 = new int[totsize]; 
    int offset = 0; 
    for (std::vector<std::vector<int> >::const_iterator it=origin->begin(); it != origin->end(); it++) {
        const std::vector<int> *vint = &(*it);
        for (int i = 0; i < vint->size(); i++) {
            buff2[offset + i] = (*vint)[i];
        }
        offset += vint->size(); 
    }

    // Send stencil buffer
    MPI_Send(&buff2[0], totsize, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    //  cout << "RANK " << myrank << " REPORTS: finished sending std::set< std::vector<int> > to RANK " << recv_rank << endl;
    delete [] buff; 
    delete [] buff2; 
    //  cout << "RANK " << myrank << " REPORTS: freed buff and buff2\n"; 
}

//----------------------------------------------------------------------------

int MPISendable::sendSTL(const std::set<std::vector<Vec3> > *origin, int myrank, int recv_rank) const
{
    int sz = origin->size(); 	// All nodes for GPU

    // length of set
    MPI_Send(&sz, 1, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    int* buff = new int[sz]; 

    int totsize = 0;
    int i = 0; 
    for (set<std::vector<Vec3> >::const_iterator it=origin->begin(); it != origin->end(); it++) {
        const std::vector<Vec3> *vint = &(*it);
        buff[i] = vint->size();
        totsize += vint->size(); 
    }

    // Offsets into into stencil buffer
    MPI_Send(&buff[0], sz, MPI::INT, recv_rank, TAG, MPI_COMM_WORLD);

    double* buff2 = new double[totsize*3]; 
    int offset = 0; 
    for (set<std::vector<Vec3> >::const_iterator it=origin->begin(); it != origin->end(); it++) {
        const std::vector<Vec3> *vint = &(*it);
        for (int i = 0; i < vint->size(); i++) {
            buff2[(offset + i)*3 + 0] = (*vint)[i].x();
            buff2[(offset + i)*3 + 1] = (*vint)[i].y();
            buff2[(offset + i)*3 + 2] = (*vint)[i].z();
            //  cout << "COUNTING: " << offset + i << " of " << totsize << endl;
        }
        offset += vint->size(); 
    }

    // Raw data
    MPI_Send(&buff2[0], totsize * 3, MPI::DOUBLE, recv_rank, TAG, MPI_COMM_WORLD);

    //  cout << "RANK " << myrank << " REPORTS: finished sending std::set< std::vector<Vec3> > to RANK " << recv_rank << endl;
    delete [] buff; 
    delete [] buff2; 
}
//----------------------------------------------------------------------------

int MPISendable::sendSTL(const unsigned int *destination, int myrank, int recv_rank) const
{
    MPI_Status stat; 

    MPI_Datatype type = MPI::UNSIGNED; 
    unsigned int buf = (*destination); 

    // Length of set
    MPI_Send(&buf, 1, type, recv_rank, TAG, MPI_COMM_WORLD);

    //  cout << "RANK " << myrank << " REPORTS: finished sending unsigned int (size: 1) to RANK " << recv_rank << endl;
}

//----------------------------------------------------------------------------

int MPISendable::sendSTL(const int *destination, int myrank, int recv_rank) const
{
    MPI_Status stat; 

    MPI_Datatype type = MPI::INT; 
    int buf = (*destination); 

    // Length of set
    MPI_Send(&buf, 1, type, recv_rank, TAG, MPI_COMM_WORLD);

    //  cout << "RANK " << myrank << " REPORTS: finished sending int (size: 1) to RANK " << recv_rank << endl;
}

//----------------------------------------------------------------------------


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
        double* buff = new double[sz];
        MPI_Recv(&buff[0], sz, MPI::DOUBLE, sender_rank, TAG, MPI_COMM_WORLD, &stat);

        for (int i=0; i < sz; i++) {
            destination->push_back(buff[i]);
        }	
        delete [] buff; 
    }
    //  cout << "RANK " << myrank << " REPORTS: received std::vector<double> (size: " << sz << ") from RANK " << sender_rank << endl;

    return sz; 
}

//----------------------------------------------------------------------------

// Recv int array and pack as STL::Vector.
int MPISendable::recvSTL(std::vector<int> *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    int sz;
    MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // WARNING! THIS ERASES ALL ELEMENTS IN destination
    destination->clear(); 

    if (sz > 0) {
        int* buff = new int[sz];
        MPI_Recv(&buff[0], sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

        for (int i=0; i < sz; i++) {
            destination->push_back(buff[i]);
        }	
        delete [] buff; 
    }
    //  cout << "RANK " << myrank << " REPORTS: received std::vector<int> (size: " << sz << ") from RANK " << sender_rank << endl;
    return sz; 
}

//----------------------------------------------------------------------------

// Recv int array and pack as STL::Vector.
int MPISendable::recvSTL(std::vector<unsigned int> *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    int sz;
    MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // WARNING! THIS ERASES ALL ELEMENTS IN destination
    destination->resize(sz); 

    if (sz > 0) {
        MPI_Datatype type = MPI::UNSIGNED; 
        unsigned int *buff = new unsigned int[sz];
        MPI_Recv(&buff[0], sz, type, sender_rank, TAG, MPI_COMM_WORLD, &stat);

        for (int i=0; i < sz; i++) {
            (*destination)[i] = buff[i];
        }	
        delete [] buff; 
    }
    //  cout << "RANK " << myrank << " REPORTS: received std::vector<int> (size: " << sz << ") from RANK " << sender_rank << endl;

    return sz; 
}


//----------------------------------------------------------------------------

// Recv int[2] array and pack as STL::Map.
int MPISendable::recvSTL(std::map<int, int> *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    int sz;
    MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    int* buff = new int[sz*2];
    MPI_Recv(&buff[0], sz*2, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // WARNING! THIS ERASES ALL ELEMENTS IN destination
    // But we cant avoid it since we want the map to contain EXACTLY what is passed in, not more, not less. 
    destination->clear(); 

    for (int i=0; i < sz; i++) {
        (*destination)[buff[i*2 + 0]] = buff[i*2 + 1];
    }

    //  cout << "RANK " << myrank << " REPORTS: received std::map<int,int> (size: " << sz << ") from RANK " << sender_rank << endl;	
    delete [] buff; 
}

//----------------------------------------------------------------------------

// Recv int array and pack as STL::Set.
int MPISendable::recvSTL(std::set<int> *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    int sz;
    MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    int* buff = new int[sz];
    MPI_Recv(&buff[0], sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // WARNING! THIS ERASES ALL ELEMENTS IN destination
    destination->clear(); 

    for (int i=0; i < sz; i++) {
        destination->insert(buff[i]);
    }	
    //  cout << "RANK " << myrank << " REPORTS: received std::set<int> (size: " << sz << ") from RANK " << sender_rank << endl;
    delete [] buff; 
    //  cout << "RANK " << myrank << " REPORTS: buff freed\n";
}

//----------------------------------------------------------------------------

// Recv int array and pack as STL::Set.
int MPISendable::recvSTL(std::vector<Vec3> *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    int sz;
    MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    double* buff = new double[sz*3]; // I reallly want double[sz][3]
    MPI_Recv(&buff[0], sz*3, MPI::DOUBLE, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // WARNING! THIS ERASES ALL ELEMENTS IN destination
    destination->clear(); 

    for (int i=0; i < sz; i++) {
        Vec3 v(buff[i*3 + 0], buff[i*3 + 1], buff[i*3 + 2]);
        destination->push_back(v);
    }	
    //  cout << "RANK " << myrank << " REPORTS: received std::set<Vec3> (size: " << sz << ") from RANK " << sender_rank << endl;
    delete [] buff; 
}

//----------------------------------------------------------------------------

int MPISendable::recvSTL(std::vector<std::vector<unsigned int> > *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    // Length of set
    int sz;
    MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // Offsets into vector
    int* buff = new int[sz];
    MPI_Recv(&buff[0], sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    int totsize = 0;
    for (int i=0; i < sz; i++) {
        totsize += buff[i]; 
    }

    MPI_Datatype type = MPI::UNSIGNED; 
    // Raw data
    unsigned int* buff2 = new unsigned int[totsize];
    MPI_Recv(&buff2[0], totsize, type, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // WARNING! THIS ERASES ALL ELEMENTS IN destination
    destination->clear(); 

    int offset = 0; 
    for (int i=0; i < sz; i++) {
        //FIXME: this is a potential source of error. what if linux deletes this memory and its not
        // accessible after leaving this routine? the push_back is supposed to call copy constructors
        // right? Just to be sure we better allocate, push the derefed mem, and delete. if we ever
        // get a double free from this then we know its not handled properly. 
        std::vector<unsigned int>* temp = new std::vector<unsigned int>; 
        for (int j=0; j < buff[i]; j++) {
            temp->push_back(buff2[offset+j]);
        }
        offset += buff[i]; 
        destination->push_back(*temp);
        delete(temp);
    }	
    //  cout << "RANK " << myrank << " REPORTS: received std::set<int> (size: " << sz << ") from RANK " << sender_rank << endl;	
    //  cout << "WARNING! unsigned int sending is not verified YET.\n"; 
    delete [] buff; 
    delete [] buff2;
}

//----------------------------------------------------------------------------


int MPISendable::recvSTL(std::vector<std::vector<int> > *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    // Length of set
    int sz;
    MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // Offsets into vector
    int* buff = new int[sz];
    MPI_Recv(&buff[0], sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    int totsize = 0;
    for (int i=0; i < sz; i++) {
        totsize += buff[i]; 
    }

    // Raw data
    int* buff2 = new int[totsize];
    MPI_Recv(&buff2[0], totsize, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

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
    //  cout << "RANK " << myrank << " REPORTS: received std::set<int> (size: " << sz << ") from RANK " << sender_rank << endl;	
    delete [] buff; 
    delete [] buff2; 
    //  cout << "RANK " << myrank << " REPORTS: freed buff and buff2\n";
}

//----------------------------------------------------------------------------

int MPISendable::recvSTL(std::set<std::vector<Vec3> > *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    // Length of set
    int sz;
    MPI_Recv(&sz, 1, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // Offsets into vector
    int* buff = new int[sz];
    MPI_Recv(&buff[0], sz, MPI::INT, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    int totsize = 0;
    for (int i=0; i < sz; i++) {
        totsize += buff[i]; 
    }

    // Raw data
    double* buff2 = new double[totsize*3];
    MPI_Recv(&buff2[0], totsize*3, MPI::DOUBLE, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    // WARNING! THIS ERASES ALL ELEMENTS IN destination
    destination->clear(); 

    int offset = 0; 
    for (int i=0; i < sz; i++) {
        std::vector<Vec3>* temp = new std::vector<Vec3>; 
        for (int j=0; j < buff[i]; j++) {
            Vec3* v = new Vec3(buff2[(offset+j) + 0], buff2[(offset+j) + 1], buff2[(offset+j) + 2]); // cast to ensure correct constructor
            temp->push_back(*v);
            delete(v); 
        }
        offset += buff[i]; 
        destination->insert(*temp);
        delete(temp);
    }	
    //  cout << "RANK " << myrank << " REPORTS: received std::set<int> (size: " << sz << ") from RANK " << sender_rank << endl;
    delete [] buff; 
    delete [] buff2; 
}

//----------------------------------------------------------------------------

int MPISendable::recvSTL(unsigned int *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    MPI_Datatype type = MPI::UNSIGNED; 

    // Length of set
    MPI_Recv(destination, 1, type, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    //  cout << "RANK " << myrank << " REPORTS: received unsigned int (size: 1) from RANK " << sender_rank << endl;
}

//----------------------------------------------------------------------------

int MPISendable::recvSTL(int *destination, int myrank, int sender_rank)
{
    MPI_Status stat; 

    MPI_Datatype type = MPI::INT; 

    // Length of set
    MPI_Recv(destination, 1, type, sender_rank, TAG, MPI_COMM_WORLD, &stat);

    //  cout << "RANK " << myrank << " REPORTS: received unsigned int (size: 1) from RANK " << sender_rank << endl;
}

//----------------------------------------------------------------------------
