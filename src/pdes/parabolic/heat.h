#ifndef _HEAT_H_ 
#define _HEAT_H_

#include <vector> 
#include <ArrayT.h> 
#include "utils/comm/communicator.h"
#include "grids/domain.h" 
#include "exact_solutions/exact_solution.h" 
#include "grids/grid_interface.h" 
#include "timer_eb.h" 
#include "rbffd/derivative.h"

//class Derivative;

class Heat { 
    
    private:
        // Lookup our timers with a short string string keyword description
        std::map<std::string, EB::Timer*> tm; 

        // time step
        double dt; 

        // initial solution and forcing term
        double PI; 
        double freq; 
        double decay;

        // solution to heat equation How to create an array of vector<double> ?
        std::vector<double> sol[2];


        std::vector<NodeType>* rbf_centers;

        // The indices of rbf_centers that correspond to global domain boundary
        // nodes (i.e. boundaries of the PDE)
        std::vector<size_t>* boundary_set; 		



        Domain* subdomain; Derivative* der; 
        std::vector<double> lapl_deriv;
        std::vector<double> x_deriv; 
        std::vector<double> y_deriv;
        std::vector<double> xx_deriv; 
        std::vector<double> yy_deriv;
        // derivate based on derivative operator
        std::vector<double> cart_laplace; 
        std::vector<double> diffusion;
        std::vector<double> diff_x;
        std::vector<double> diff_y;

        double time; 

        // boundary values (in the same order as boundary_index)
        std::vector<double> bnd_sol; 

        // total number of rbfs
        int nb_rbf; int nb_stencils;

        ExactSolution* exactSolution; 

        //double major, maji2, maji4; double minor, mini2, mini4;

        int id; 		// Comm rank or comm id

    public: 

        Heat(ExactSolution* _solution, std::vector<Vec3>* rb_centers_, int
                stencil_size, std::vector<size_t>* global_boundary_nodes_,
                Derivative* der_, int rank);

        // Constructor #2:
        Heat(ExactSolution* _solution, Domain* subdomain_, Derivative* der_,
                int rank); 
        
        Heat(ExactSolution* _solution, Grid& grid_, Derivative& der_); 
        
        ~Heat();




        void setupTimers();

        // set the time step
        void setDt(double dt) { this->dt = dt; }

        // Advance the equation one time step using the Domain class to perform
        // communication Depends on Constructor #2 to be used so that a Domain
        // class exists within this class.  This is on the CPU. We need to
        // reimplement this routine on the GPU 
        void advanceOneStepWithComm(Communicator* comm_unit);

        // Only update the updated_solution vector if it is non-null (i.e. we
        // actually pass something to the routine)
        void advanceOneStep(std::vector<double>* updated_solution = NULL); 
        
        void advanceOneStepDivGrad();

        // div(D grad)T =  grad(D).grad(T) + D lapl(T)
        void advanceOneStepTwoTerms();

        // Only update the updated_solution vector if it is non-null (i.e. we
        // actually pass something to the routine)
        void initialConditions(std::vector<double>* solution = NULL);

        std::vector<Vec3>* getRbfCenters() { return rbf_centers; }

        double maxNorm(); 
        
        double maxNorm(std::vector<double> sol);

        void computeDiffusion(std::vector<double>& sol);

        //	double exactSolution(Vec3& v, double t);

        // forcing term to force an exact solution we are solving: dT/dt =
        // lapl(T) + force(x,t) T (x, t) = f(x,t) dT/dt = lapl(T) + (df/dt -
        // lapl(f)) = T(x,0) = f(x,0) so each iteration we 
        double force(Vec3& pt, double t);

        std::vector<double> getSolution() { return sol[0]; }


        void checkError(std::vector<double>& solution);
        void writeErrorToFile(std::vector<double>& error); 
};

#endif
