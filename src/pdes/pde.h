#ifndef __PDE_H__
#define __PDE_H__

#include "grids/grid_interface.h"
#include "rbffd/rbffd.h"
#include "utils/comm/communicator.h"
#include "utils/comm/mpisendable.h"

#include "timer_eb.h"

#include "common_typedefs.h"

// Base interface class
class PDE : public MPISendable
{
    protected: 
        EB::TimerList tm; 

        Domain& grid_ref;
        RBFFD& der_ref; 
        Communicator& comm_ref; 

        // The solution for our PDE. Might be spatial and/or temporal 
        // Each solution type could be a scalar or vector
        std::vector<SolutionType> U_G; 



    private:
        // A map for global INDEX=VALUE storage of the final solution
        // recvFinal will populate this and then we can call (TODO) getFinal()
        // to get the values as a vector.
        // SOLUTION of all nodes in global domain (valid only on master)
        std::map<int,double> global_U_G;

#if 0
        void printSolution(std::string label) {
            printVector(this->U_G, label); 
        }


        std::vector<double>& getU() { return U_G; };

        virtual void getFinal(std::vector<double>* final);

        // Dump the final solution to a file along with the vector of nodes that
        // the values correspond to.
        virtual int writeFinal(std::vector<NodeType>& nodes, std::string filename);

#endif 


    public: 
        PDE(Domain* grid, RBFFD* der, Communicator* comm) 
            : grid_ref(*grid), der_ref(*der), comm_ref(*comm)
        {;}

        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble() =0; 
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve() = 0;

        // Document the solutions on disk
        void writeLocalSolutionToFile(std::string filename); 
        void writeGlobalSolutionToFile(std::string filename);

        void writeLocalSolutionToFile(int iter=0) { this->writeLocalSolutionToFile(this->getFilename(iter)); }  
        void writeGlobalSolutionToFile(int iter=0) { this->writeGlobalSolutionToFile(this->getFilename(iter)); }

    protected:
        // ******** BEGIN MPISENDABLE ************
        // The following seven routines are required by MPISendable inheritence.
        virtual int send(int my_rank, int receiver_rank); 
        virtual int receive(int my_rank, int sender_rank);
        virtual int sendUpdate(int my_rank, int receiver_rank); 
        virtual int receiveUpdate(int my_rank, int sender_rank);

        virtual int sendFinal(int my_rank, int receiver_rank);
        virtual int receiveFinal(int my_rank, int sender_rank);
        virtual int initFinal();
        // ******** END MPISENDABLE ************

        virtual void setupTimers() = 0;

    protected: 
        // FIXME: put these in another pure virtual interface class

        // =====================================================================
        // Convert a basic filename like "output_file" to something more
        // descriptive and appropriate to the pde like
        // "output_file_ncar_poisson1_iteration_10.ascii" 
        std::string getFilename(std::string base_filename, int iter=0);

        // Get a filename appropriate for output from this class
        // same as getFilename(std::string, int) however it uses 
        // the class's internal name instead of a user specified string. 
        std::string getFilename(int iter=0); 

        // Get a string that gives some detail about the grid (used by
        // expandFilename(...)) 
        // NOTE: replace spaces with '_'
        virtual std::string getFileDetailString(); 

	    virtual std::string className() = 0;
        // =====================================================================
};

#endif // __PDE_H__

// Lu = f
// Lu = lapl(u)
// dt = u + del(t) lapl(u)
