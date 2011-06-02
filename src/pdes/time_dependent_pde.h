#ifndef __TIME_DEPENDENT_PDE_H__
#define __TIME_DEPENDENT_PDE_H__

#include "pdes/pde.h"

#include "utils/norms.h"

// Interface class
class TimeDependentPDE : public PDE 
{
    protected: 
        // The current time of our solution (typ. # of iterations * dt)
        double cur_time;    

    // This count should match the number of TimeScheme types
    public:
#define NUM_TIME_SCHEMES 3
        enum TimeScheme {EULER=0, MIDPOINT, RK4};

    public: 
        TimeDependentPDE(Domain* grid, RBFFD* der, Communicator* comm) 
            : PDE(grid, der, comm), cur_time(0.) 
        {
            tm["advance"] = new EB::Timer("[TimeDependentPDE] Advance timestep");
        }

        // Fill in the initial conditions of the PDE. (overwrite the solution)
        virtual void fillInitialConditions(ExactSolution* exact=NULL);

        // Advancing requires: 
        //  - computing an update to the current solution (i.e., calling
        //  applyWeightsForDerivs(currentSolution)) 
        //  - applying the updates to the current solution (i.e., RK45 weighted
        //  summation of intermediate updates).
        //  NOTE: at the end of the advance routine the PDE::solution should
        //  contain the advanced solution. If intermediate steps are required (i.e.
        //  in 2nd order or RK45, intermediate solutions and ghost node broadcasts
        //  are required), then archive the original solution and any subsequent
        //  buffers and overwrite the final solution at the end of the routine.
        virtual void advance(TimeScheme which, double delta_t);
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, size_t n, double t)=0;

        // If we use an explicit scheme we can enforce boundaries here
        // Else, leave this routine empty for implicit and enforce boundaries
        // in the assembled matrix L.
        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t) = 0;

        void setTime(double current_time) { cur_time = current_time; }
        double getTime() { return cur_time; }

        virtual SolutionType getExactSolution(size_t indx) {
            NodeType& pt = grid_ref.getNode(indx); 
            SolutionType ex = exact_ptr->at(pt, cur_time); 
            return ex;
        }

        virtual SolutionType getAbsoluteError(size_t indx) {
            NodeType& pt = grid_ref.getNode(indx); 
            SolutionType ex = exact_ptr->at(pt, cur_time); 
            SolutionType app = this->getLocalSolution(indx);

            return l2norm(ex - app);
        }

        virtual SolutionType getRelativeError(size_t indx) {
            NodeType& pt = grid_ref.getNode(indx); 
            SolutionType ex = exact_ptr->at(pt, cur_time); 
            SolutionType app = this->getLocalSolution(indx);
            return l2norm(ex - app) / l2norm(ex);
        }
 
    protected: 
        virtual void advanceFirstOrderEuler(double dt);
        virtual void advanceSecondOrderMidpoint(double dt);
        virtual void advanceRungeKutta4(double dt);

        // Fill vector with exact solution at provided nodes.
        // NOTE: override in time dependent PDE to leverage time-based solutions
        virtual void getExactSolution(ExactSolution* exact, std::vector<NodeType>& nodes, std::vector<SolutionType>* exact_vec) {
            std::vector<NodeType>::iterator it; 
            exact_vec->resize(nodes.size());
            size_t i = 0; 
            for (it = nodes.begin(); it != nodes.end(); it++, i++) {
                (*exact_vec)[i] = exact->at(*it, cur_time); 
            }
        }

       // We'll hide this routine because we want one based on time (see above)
        virtual void solve(std::vector<SolutionType>& y_t, std::vector<SolutionType>* f_out, size_t n) { std::cout << "ERROR! SHOULD CALL THE TIME BASE SOLVE\n"; exit(EXIT_FAILURE); } 


};
#endif // __TIME_DEPENDENT_PDE_H__
