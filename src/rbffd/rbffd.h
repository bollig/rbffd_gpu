#ifndef _RBFFD_H_
#define _RBFFD_H_

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "Vec3.h"
#include "rbffd/rbfs/rbf_gaussian.h"
#include "rbffd/rbfs/rbf_mq.h"
#include "utils/conf/projectsettings.h"
#include "timer_eb.h"
#include "common_typedefs.h"
#include <armadillo> 

#include "grids/domain.h"

//typedef RBF_Gaussian IRBF;
typedef RBF_MQ IRBF;
// TODO: put this in rbf.h and have all rbf types available
//enum RBF_Type {MQ=0, GA, IMQ, TPS, W2};

// Should match how many DerTypes we have below
#define NUM_DERIV_TYPES 4

class RBFFD
{
    public:
        enum DerType {X, Y, Z, LAPL};

    protected: 
        EB::TimerList tm; 

        Domain& grid_ref;

        // Weight array. Each element is associated with one DerType (see above). 
        std::vector<double*> weights[NUM_DERIV_TYPES]; 

        // 0/1 (false/true) are the weights for the associated stencil computed yet? 
        // NOTE: each vector is associate with one DerType (see above), and each element
        // of the vector corresponds to a stencil 
        std::vector<int> stWeightsComputed; 

        // A list of support parameters for each RBF centered at each node
        std::vector<double> var_epsilon;

        // 0/1 (false/true) should we assume our support parameter is
        // heterogenous (true) or homogenous (false)?
        int use_var_eps; 

        // Number of dimensions over which we compute derivatives (can be
        // independent of dimension for grid). 
        int dim_num;

        // Have any parameters been modified?
        int modified; 

        //TODO: add choice for RBF (only one option at the moment)

    public: 

        // Note: dim_num here is the desired dimensions for which we calculate derivatives
        // (up to 3 right now) 
        RBFFD(Domain& grid, int dim_num);
        // , RBF_Type rbf_choice=MQ); 

        virtual ~RBFFD(); 

        // NOTE: we compute all weights for a stencil at the same time,
        // because we can do this for (nearly) free. Its a multiple RHS
        // solve---hard to justify NOT doing this! 
        //
        void computeAllWeightsForAllStencils();
        void computeAllWeightsForStencil(int st_indx);
        void computeWeightsForStencil(DerType, int st_indx);

        // Use a direct solver on to calculate weights.
        virtual int computeStencilWeights(std::vector<Vec3>& rbf_centers,
                StencilType& stencil, int irbf, double* weights);

            // Apply weights to an input solution vector and get the corresponding derivatives out
            void applyWeightsForDeriv(DerType which, std::vector<double>& u, std::vector<double>& deriv) { 
                deriv.resize(u.size()); 
                applyWeightsForDeriv(which, u.size(), &u[0], &deriv[0]);
            }

        virtual void applyWeightsForDeriv(DerType which, int npts, double* u, double* deriv);


        void setEpsilon(double eps) { 
            modified = 1;
            use_var_eps = 1; 
            std::cout << "DERIVATIVE:: SET UNIFORM EPSILON = " << eps << std::endl;
            for (int i = 0; i < var_epsilon.size(); i++) {
                this->var_epsilon[i] = eps; 
            }
        }

        void setEpsilon(std::vector<double>& lst_of_epsilon) {
            modified = 1; 
            use_var_eps = 1; 
            if (lst_of_epsilon.size() != var_epsilon.size()) {
                std::cout << "ERROR! length of support params list does not match" << std::endl;
                exit(EXIT_FAILURE); 
            }
            for (int i = 0; i < var_epsilon.size(); i++) {
                this->var_epsilon[i] = lst_of_epsilon[i]; 
            }
        }

        void setVariableEpsilon(double alpha=1.0f, double beta=1.0f) {
           this->setVariableEpsilon(grid_ref.getStencilRadii(), alpha, beta);
        }
        void setVariableEpsilon(std::vector<double>& avg_radius_, double alpha=1.0f, double beta=1.0f) {
            //modified = 1;
            use_var_eps = 1;
            std::cout << "DERIVATIVE:: SET VARIABLE EPSILON = " << alpha << "/(avg_st_radius^" << beta << ")" << std::endl;
            std::vector<double>& avg_stencil_radius = avg_radius_;

            var_epsilon.resize(avg_stencil_radius.size());
            for (int i=0; i < var_epsilon.size(); i++) {
                var_epsilon[i] = alpha / std::pow(avg_stencil_radius[i], beta);
                printf("avg_stencil_radius(%d) = %10.10f\n", i , avg_stencil_radius[i]); 
                printf("var_epsilon(%d) = %10.10f\n", i, var_epsilon[i]);
            }
        }

        // Accessors for weight data: 
        std::vector<double*>& getXWeights() { return weights[this->X]; }
        std::vector<double*>& getYWeights() { return weights[this->Y]; }
        std::vector<double*>& getZWeights() { return weights[this->Z]; }
        std::vector<double*>& getLaplWeights() { return weights[this->LAPL]; }
        double* getXWeights(int indx) { return (getXWeights())[indx]; }
        double* getYWeights(int indx) { return (getYWeights())[indx]; }
        double* getZWeights(int indx) { return (getZWeights())[indx]; }
        double* getLaplWeights(int indx) { return (getLaplWeights())[indx]; }

        std::vector<double*>& getWeights(DerType choice) { return weights[choice]; }
        double*& getStencilWeights(DerType choice, int st_indx) { return weights[choice][st_indx]; } 

        void writeToFile(DerType which, std::string filename="weights.mtx");
        void loadWeightsFromFile(std::string filename) {;} 


    protected: 
        void setupTimers(); 

        void distanceMatrix(std::vector<NodeType>& rbf_centers, StencilType& stencil, int dim_num, arma::mat& d_matrix); 


        void getStencilMultiRHS(std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& rhs);
        void getStencilRHS(DerType which, std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& rhs);
        void getStencilLHS(std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& lhs);

};

#endif 
