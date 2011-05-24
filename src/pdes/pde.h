#ifndef __PDE_H__
#define __PDE_H__

#include "grids/grid_interface.h"
#include "rbffd/rbffd.h"
#include "utils/comm/communicator.h"
#include "utils/comm/mpisendable.h"

#include "exact_solutions/exact_solution.h"

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
        std::vector<double>& getU() { return U_G; };
#endif 


    public: 
        PDE(Domain* grid, RBFFD* der, Communicator* comm) 
            : grid_ref(*grid), der_ref(*der), comm_ref(*comm)
        {
            // We want our solution to match the number of nodes
            U_G.resize(grid_ref.getNodeListSize());
        }

        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble() =0; 
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y, std::vector<SolutionType>* f_out) = 0;


        // Print the current solution to STDOUT
        void printSolution(std::string label="Solution");

        // Document the solutions on disk
        void writeLocalSolutionToFile(std::string filename); 
        void writeGlobalSolutionToFile(std::string filename);

        void writeLocalSolutionToFile(int iter=0) { this->writeLocalSolutionToFile(this->getFilename(iter)); }  
        void writeGlobalSolutionToFile(int iter=0) { this->writeGlobalSolutionToFile(this->getFilename(iter)); }


        // Dump the final solution to a file along with the vector of nodes that
        // the values correspond to.
        // NOTE: the nodes from the GLOBAL grid must be passed here because this class
        //       is only aware of the LOCAL subgrid prior to this point.
        virtual int writeGlobalGridAndSolutionToFile(std::vector<NodeType>& nodes, std::string filename);

        // Fill the passed vector with the global solution
        virtual void getGlobalSolution(std::vector<double>* final);

        // Check the error locally
        void checkLocalError(ExactSolution* exact, double rel_err_max=-1.) { 
            std::vector<SolutionType> exactSolution;
            this->getExactSolution(exact, this->grid_ref.getNodeList(), &exactSolution); 
            this->checkError(exactSolution, this->U_G, this->grid_ref.getNodeList(), this->grid_ref.getBoundaryIndices(), rel_err_max); 
        }

        void checkGlobalError(ExactSolution* exact, Grid* global_grid, double rel_err_max=-1.) {
            std::vector<SolutionType> sol;
            std::vector<SolutionType> exactSolution;
            this->getGlobalSolution(&sol);
            //this->getExactSolution(exact, global_grid->getNodeList(), &exactSolution); 
            this->checkError(exactSolution, sol, global_grid->getNodeList(), global_grid->getBoundaryIndices(), rel_err_max); 
        }


        SolutionType getLocalSolution(size_t indx) { return U_G[indx]; }

    protected: 

        // Fill vector with exact solution at provided nodes.
        // NOTE: override in time dependent PDE to leverage time-based solutions
        virtual void getExactSolution(ExactSolution* exact, std::vector<NodeType>& nodes, std::vector<SolutionType>* exact_vec) {
            std::vector<NodeType>::iterator it; 
            exact_vec->resize(nodes.size());
            size_t i = 0; 
            for (it = nodes.begin(); it != nodes.end(); it++, i++) {
                (*exact_vec)[i] = exact->at(*it); 
            }
        }


        // Check that the error in the solution is 
        void checkError(std::vector<SolutionType>& exactSolution, std::vector<SolutionType>& solution, std::vector<NodeType>& nodes, std::vector<size_t> boundary_indx, double rel_err_max=-1.);
        void calcSolNorms(std::vector<double>& sol_vec, std::vector<double>& sol_exact, std::string label, double rel_err_max=1.);

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
