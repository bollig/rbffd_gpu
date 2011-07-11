#ifndef _RBFFD_H_
#define _RBFFD_H_

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "Vec3.h"
#include "rbffd/rbfs/rbf_gaussian.h"
#include "rbffd/rbfs/rbf_multiquadric.h"
#include "rbffd/rbfs/rbf_inv_multiquadric.h"

#include "utils/conf/projectsettings.h"
#include "timer_eb.h"
#include "common_typedefs.h"
#include <armadillo> 

#include "grids/domain.h"

#if 1
typedef RBF_Gaussian IRBF;
#else 
#if 0
typedef RBF_Multiquadric IRBF;
#else 
typedef RBF_InvMultiquadric IRBF;
#endif 
#endif 

// TODO: put this in rbf.h and have all rbf types available
//enum RBF_Type {MQ=0, GA, IMQ, TPS, W2};

// Should match how many DerTypes we have below
#define NUM_DERIV_TYPES 8

class RBFFD
{
    public:
        // INTERP are for NO derivatives, but for interpolating functions in
        // those cases we avoid using monomial terms
        // R is the radial deriv (dPhi/dr)
        enum DerType {LAMBDA, X, Y, Z, LAPL, HV2, R, INTERP};
        std::string derTypeStr[NUM_DERIV_TYPES]; 

        enum WeightType {Direct, ContourSVD};
        std::string weightTypeStr[2]; 

        typedef struct e_val_output {
            double max_pos_eig; 
            double min_pos_eig;
            double max_neg_eig; 
            double min_neg_eig;
            int nb_positive; 
            int nb_negative;
            int nb_zero;
        } EigenvalueOutput;


    protected: 
        EB::TimerList tm; 

        WeightType weightMethod; 

        Grid& grid_ref;

        std::string eps_string;

        // Weight array. Each element is associated with one DerType (see above). 
        std::vector<double*> weights[NUM_DERIV_TYPES]; 

        // 0/1 (false/true) are the weights for the associated stencil computed yet? 
        // NOTE: each vector is associate with one DerType (see above), and each element
        // of the vector corresponds to a stencil 
        // TODO:        std::vector<int> stWeightsComputed; 

        // A list of support parameters for each RBF centered at each node
        std::vector<double> var_epsilon;

        // FIXME: has no effect
        // 0/1 (false/true) should we assume our support parameter is
        // heterogenous (true) or homogenous (false)?
        int use_var_eps; 

        // Number of dimensions over which we compute derivatives (can be
        // independent of dimension for grid). 
        int dim_num;

        // Potentially useful if we want to know the rank of our MPI process
        // for assigning procs or gpus?
        int rank; 


        // FIXME: has no effect
        // Have any parameters been modified?
        int modified; 
        bool weightsModified;


        bool computeCondNums; 

        std::vector<CMPLX> condNums; 

        EigenvalueOutput cachedEigenvalues;
        bool eigenvalues_computed;

        //TODO: add choice for RBF (only one option at the moment)

    public: 

        // Note: dim_num here is the desired dimensions for which we calculate derivatives
        // (up to 3 right now) 
        RBFFD(Grid* grid, int dim_num, int rank=0);
        // , RBF_Type rbf_choice=MQ); 

        virtual ~RBFFD(); 

        // NOTE: we compute all weights for a stencil at the same time,
        // because we can do this for (nearly) free. Its a multiple RHS
        // solve---hard to justify NOT doing this! 
        //
        void computeAllWeightsForAllStencils();
        void computeAllWeightsForStencil(int st_indx);
        void computeWeightsForStencil(DerType deriv_type, int st_indx);
        void computeWeightsForAllStencils(DerType deriv_type);

        // Compute Multiple RHS Direct
        void computeAllWeightsForStencil_Direct(int st_indx);
        // Single RHS Direct
        void computeWeightsForStencil_Direct(DerType, int st_indx);

        // Single RHS ContourSVD
        void computeWeightsForStencil_ContourSVD(DerType, int st_indx);

        void setComputeConditionNumber(bool tf) {
            computeCondNums = tf; 
        }

        void setWeightType(RBFFD::WeightType type) {
            weightMethod = type;
        }


        // Apply weights to an input solution vector and get the corresponding derivatives out
        virtual void applyWeightsForDeriv(DerType which, std::vector<double>& u, std::vector<double>& deriv, bool isChangedU=true) { 
//            std::cout << "CPU: ";
            deriv.resize(u.size()); 
            RBFFD::applyWeightsForDeriv(which, u.size(), &u[0], &deriv[0], isChangedU);
        }

        // Can be CPU or GPU depending on Subclasses
        // NOTE: We apply the necessary weights to get the FULL derivative of u(x,y,z).
        //       That is, we see L{u}(x,y,z) evaluated all ALL points
        //       Also, if isChangedU==false then we can avoid overwriting a
        //       local u with the one passed in (e.g., when using the GPU)
        // TODO: npts is unused at the momement. Could prove useful if we had a
        //      applyWeightsForDerivAtNode(i) routine
        virtual void applyWeightsForDeriv(DerType which, int npts, double* u, double* deriv, bool isChangedU=true);


        // Returns MAXIMUM negative eigenvalue
        // Use output to obtain MINIMUM and MAXIMUM
        double computeEigenvalues(DerType which, bool exit_on_fail, EigenvalueOutput* output=NULL);
        
        // Compute Eigenvalues of DM + Hyperviscosity term. 
        // hyperviscosity = - gamma * N_nodes^-k * (lapl)^-k
        double computeHyperviscosityEigenvalues(DerType which, int k, double gamma, EigenvalueOutput* output=NULL);


        void setEpsilon(double eps) { 
            modified = 1;
            use_var_eps = 1; 
            std::cout << "DERIVATIVE:: SET UNIFORM EPSILON = " << eps << std::endl;
            for (int i = 0; i < var_epsilon.size(); i++) {
                this->var_epsilon[i] = eps; 
            }
            std::stringstream ss(std::stringstream::out); 
            ss << "uniform_epsilon_" << eps;
            eps_string = ss.str();
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
            eps_string = "manual_epsilon";
        }

        void setVariableEpsilon(double alpha=1.0f, double beta=1.0f) {
            this->setVariableEpsilon(grid_ref.getStencilRadii(), alpha, beta);
        }

        void setVariableEpsilon(std::vector<double>& avg_radius_, double alpha=1.0f, double beta=1.0f); 

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



        void writeAllWeightsToFile(); 
        void writeToFile(DerType which, std::string filename);
        void writeToFile(DerType which) { this->writeToFile(which, this->getFilename(which)); }
        int  loadAllWeightsFromFile();
        int  loadFromFile(DerType which, std::string filename);
        int  loadFromFile(DerType which){ return this->loadFromFile(which, this->getFilename(which)); }

        EigenvalueOutput getEigenvalues() {
            if (!eigenvalues_computed) {
                this->computeEigenvalues(LAPL, false); 
            }
            return cachedEigenvalues;
        }

    protected: 
        void setupTimers(); 

        void distanceMatrix(std::vector<NodeType>& rbf_centers, StencilType& stencil, int dim_num, arma::mat& d_matrix, double h); 

        void getStencilMultiRHS(std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& rhs, double h);
        void getStencilRHS(DerType which, std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& rhs, double h);
        void getStencilLHS(std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& lhs, double h);

        // =====================================================================
        // Convert a basic filename like "output_file" to something more
        // descriptive and appropriate to the pde like
        // "output_file_ncar_poisson1_iteration_10.ascii" 
        std::string getFilename(DerType which, std::string base_filename);

        // Get a filename appropriate for output from this class
        // same as getFilename(std::string, int) however it uses 
        // the class's internal name instead of a user specified string. 
        std::string getFilename(DerType which); 

        // Get a string that gives some detail about the grid (used by
        // expandFilename(...)) 
        // NOTE: replace spaces with '_'
        virtual std::string getFileDetailString(DerType which); 

        virtual std::string getEpsString() {
            return eps_string; 
        }

        virtual std::string className() { return "rbffd"; }
        // =====================================================================

};

#endif 
