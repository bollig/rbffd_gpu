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

//typedef RBF_Gaussian IRBF;
typedef RBF_MQ IRBF;

// Temporary:
#define RBFFD Derivative
class RBFFD
{
    public:
        enum DerType {X, Y, Z, LAPL};

    protected: 
        TimerList tm; 

        Grid& grid_ref;

        // The weights calculated by this class
        // NOTE: we could put these in a single vector or a map and have each
        // element looked up by a key "dxdy", "dzdy", etc.
#if 0
        std::vector<double*> x_weights;
        std::vector<double*> y_weights;
        std::vector<double*> z_weights;
        std::vector<double*> lapl_weights;
#else 
        // Weight array. Each element is associated with one DerType (see above). 
        std::vector<double*>[4] weights; 
#endif 
        // 0/1 (false/true) are the weights for the associated stencil computed yet? 
        //
        // NOTE: each vector is associate with one DerType (see above), and each element
        // of the vector corresponds to a stencil 
        std::vector<int>[4] stWeightsComputed; 

        // A list of support parameters for each RBF centered at each node
        std::vector<double> var_epsilon;

        // 0/1 (false/true) should we assume our support parameter is
        // heterogenous (true) or homogenous (false)?
        int use_var_epsilon; 

        // Number of dimensions over which we compute derivatives (can be
        // independent of dimension for grid). 
        int dim_num;

        // Have any parameters been modified?
        int modified; 

    public: 

        // Note: dim_num here is the desired dimensions for which we calculate derivatives
        // (up to 3 right now) 
        RBFFD(const Grid& grid, int dim_num); 

        ~RBFFD(); 

        // Compute the full set of derivative weights for all stencils 
        int computeAllWeightsForAllDerivs();
        // Compute the full set of weights for a derivative type
        int computeAllWeightsForDeriv(DerType which); 
        // Compute the full set of derivative weights for a stencil
//TODO:        int computeAllWeightsForStencil(int st_indx); 

        // Use a direct solver on to calculate weights.
        virtual int computeStencilWeights(std::vector<Vec3>& rbf_centers, StencilType& stencil, int irbf, double* weights)

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
            std::cout << "DERIVATIVE:: SET UNIFORM EPSILON = " << eps << std::endl;
            if (lst_of_epsilon.size() != var_epsilon.size()) {
                std::cout << "ERROR! length of support params list does not match" << std::endl;
                exit(EXIT_FAILURE); 
            }
            for (int i = 0; i < var_epsilon.size(); i++) {
                this->var_epsilon[i] = lst_of_epsilon[i]; 
            }
        }

        void setVariableEpsilon(double alpha=1.0f, double beta=1.0f) {
            this->setVariableEpsilon(grid_ref->getStencilRadii(), alpha, beta);
        }
        void setVariableEpsilon(std::vector<double>& avg_radius_, double alpha=1.0f, double beta=1.0f) {
            modified = 1;
            use_var_eps = 1;
            std::cout << "DERIVATIVE:: SET VARIABLE EPSILON = " << alpha << "/(avg_st_radius^" << beta << ")" << std::endl;
            avg_stencil_radius = avg_radius_;
            var_epsilon.resize(avg_stencil_radius.size());
            for (int i=0; i < var_epsilon.size(); i++) {
                var_epsilon[i] = alpha / std::pow(avg_stencil_radius[i], beta);
                printf("avg_stencil_radius(%d) = %10.10f\n", i , avg_stencil_radius[i]); 
                printf("var_epsilon(%d) = %10.10f\n", i, var_epsilon[i]);
            }
        }

#if 0
        // Accessors for weight data: 
        std::vector<double*>& getXWeights() { return x_weights; }
        std::vector<double*>& getYWeights() { return y_weights; }
        std::vector<double*>& getZWeights() { return z_weights; }
        std::vector<double*>& getRWeights() { return r_weights; }
        std::vector<double*>& getLaplWeights() { return lapl_weights; }
        double* getXWeights(int indx) { return x_weights[indx]; }
        double* getYWeights(int indx) { return y_weights[indx]; }
        double* getZWeights(int indx) { return z_weights[indx]; }
        double* getRWeights(int indx) { return r_weights[indx]; }
        double* getLaplWeights(int indx) { return lapl_weights[indx]; }
#else 
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
#endif 

        void writeWeightsToFile(std::string filename); 
        void loadWeightsFromFile(std::string filename); 


    protected: 
        void setupTimers(); 

        arma::mat distanceMatrix(std::vector<NodeType>& rbf_centers,
                StencilType& stencil, size_t stencil_center_indx, int dim_num); 

        // TEMPORARY: 
#undef RBFFD
};

#endif 
