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

        ExactSolution* exact_ptr;

        // The solution for our PDE. Might be spatial and/or temporal 
        // Each solution type could be a scalar or vector
        std::vector<SolutionType> U_G; 

        int initCount; 

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
            : grid_ref(*grid), der_ref(*der), comm_ref(*comm), 
            initCount(0)
        {
            // We want our solution to match the number of nodes
            U_G.resize(grid_ref.getNodeListSize());
            setupTimers();
        }


        // This should assemble a matrix L of weights which can be used to solve the PDE
        virtual void assemble() =0; 
        virtual void solve()=0;
        // This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
        virtual void solve(std::vector<SolutionType>& y, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes) = 0;
        virtual void solve(std::vector<SolutionType>& y, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t) {};


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


//        virtual double getMaxVelocity(double at_time) =0; 

        // Check the error locally
        void checkLocalError(ExactSolution* exact, double rel_err_max=-1.) { 
            exact_ptr = exact;
            std::vector<SolutionType> exactSolution;
            // Should synchronize the U_G on CPU and GPU
            this->syncCPUtoGPU(); 
            this->getExactSolution(exact, this->grid_ref.getNodeList(), &exactSolution); 
            this->checkError(exactSolution, this->U_G, this->grid_ref, rel_err_max); 
        }

        void checkGlobalError(ExactSolution* exact, Grid* global_grid, double rel_err_max=-1.) {
            exact_ptr = exact;
            std::vector<NodeType>& nodes = global_grid->getNodeList();
            //std::vector<unsigned int>& bounds = global_grid->getBoundaryIndices();

            std::vector<SolutionType> sol(nodes.size(), 0.);
            std::vector<SolutionType> exactSolution(nodes.size());
            
            // TODO: call for all subdomains to send final to master

            // Should synchronize the U_G on CPU and GPU
            this->syncCPUtoGPU(); 

            this->getGlobalSolution(&sol);
            this->getExactSolution(exact, nodes, &exactSolution); 
   
#if 0
            std::cout << "Global Grid nodelist size is " << global_grid->getNodeListSize() << std::endl;
            for (unsigned int i = 0; i < global_grid->getNodeListSize(); i++) {
                std::cout << i << "\t" << sol[i] << "\t" << exactSolution[i] << std::endl;
            }
#endif 
            this->checkError(exactSolution, sol, *global_grid, rel_err_max); 
        }


        // Check the L1, L2 and Linf norms of our approximate solution.
        // This does not break the code, but will allow us to monitor decay and perhaps
        // notice when the solution explodes due to instability. 
        void checkNorms(double rel_err_max=-1.);

        SolutionType getLocalSolution(unsigned int indx) { return U_G[indx]; }

    protected: 
        // This is intended to be overridden by GPU based classes. when called,
        // its time to synchronize our solution with the results on the GPU
        virtual void syncCPUtoGPU() { 
            //std::cout << "NOT DOING ANYTHING\n";
        } 

        // Fill vector with exact solution at provided nodes.
        // NOTE: override in time dependent PDE to leverage time-based solutions
        virtual void getExactSolution(ExactSolution* exact, std::vector<NodeType>& nodes, std::vector<SolutionType>* exact_vec) {
            std::cout << "Getting master solution from PDE.h\n";
            std::vector<NodeType>::iterator it; 
            (*exact_vec).resize(nodes.size());
            unsigned int i = 0; 
            for (it = nodes.begin(); it != nodes.end(); it++, i++) {
                //exit(-1);
                (*exact_vec)[i] = exact->at(*it); 
            }
        }

        // Check our approximate solution against and exact solution. 
        void checkError(std::vector<SolutionType>& exactSolution, std::vector<SolutionType>& solution, Grid& grid, double rel_err_max=-1.);
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
        virtual int updateFinal();
        // ******** END MPISENDABLE ************

        // By default (send/recvUpdate above) we send elements from U_G. These
        // two routines allow us to specify a mat/vector to transmit
        int receiveUpdate(std::vector<SolutionType>& vec, int my_rank, int sender_rank, std::string label="");
        int sendUpdate(std::vector<SolutionType>& vec, int my_rank, int sender_rank, std::string label="");
        int sendrecvUpdates(std::vector<SolutionType>& vec, std::string label=""); 

        void setupTimers();
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
