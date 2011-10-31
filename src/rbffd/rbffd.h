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
#define NUM_DERIV_TYPES 13

class RBFFD
{
    public:
        // INTERP are for NO derivatives, but for interpolating functions in
        // those cases we avoid using monomial terms
        // R is the radial deriv (dPhi/dr)
        // LAMBDA is longitude
        // THETA is latitude
        // SPH_LAPL is spherical laplacian (Laplace-Beltrami Operator)

        enum DerTypeID {
            X       = 0x1,
            Y       = 0x2,
            Z       = 0x4,
            LAPL    = 0x8,
            R       = 0x10,
            HV      = 0x20,
            LAMBDA  = 0x40,
            THETA   = 0x80, 
            LSFC    = 0x100,
            XSFC    = 0x200, 
            YSFC    = 0x400, 
            ZSFC    = 0x800, 
            INTERP  = 0x1000
        };

        enum DerTypeIndx {
            X_i       = 0,
            Y_i       ,
            Z_i      ,
            LAPL_i  ,
            R_i    ,
            INTERP_i,
            LAMBDA_i,
            THETA_i ,
            LSFC_i   ,
            XSFC_i   ,
            YSFC_i   ,
            ZSFC_i  ,  
            HV_i     
        };

        typedef unsigned int DerTypes; 

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
        // This stores a list of weights we need to compute or have computed
        DerTypes computedTypes; 
        unsigned int numSelectedDerTypes;

        EB::TimerList tm; 

        WeightType weightMethod; 

        Grid& grid_ref;

        std::string eps_string;

        // Weight array. Each element is associated with one DerTypeID (see above). 
        // But not all elements of this vector will be allocated. Only when a deriv type
        // is computed will the element point to something valid 
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


        // Parameters for hyperviscosity: 
        // recall that we get the hyperviscosity at -(gamma * N^{-k} * (lapl)^{k})
        double hv_gamma;
        double hv_k; 

        int useHyperviscosity;

    protected: 
        inline DerTypeID getDerType(int i) {
            return getDerType((DerTypeID)i); 
        }

        inline DerTypeID getDerType(DerTypeIndx i) {
            return (DerTypeID) (0x1 << i);
        }

        inline DerTypeIndx getDerTypeIndx(DerTypeID dt) {
            int iterator = dt; 
            int i = -1; 
            // Iterate until we get all 0s.
            while (iterator) {
                iterator >>= 1; 
                i+= 1; 
            }
            return (DerTypeIndx) i;
        }

    public:

        // Note: dim_num here is the desired dimensions for which we calculate derivatives
        // (up to 3 right now) 
       // RBFFD(Grid* grid, int dim_num, int rank=0);
        RBFFD(DerTypes typesToCompute, Grid* grid, int dim_num, int rank=0); 

        virtual ~RBFFD(); 

        // NOTE: we compute all weights for a stencil at the same time,
        // because we can do this for (nearly) free. Its a multiple RHS
        // solve---hard to justify NOT doing this! 
        //
        void computeAllWeightsForAllStencils();
        void computeAllWeightsForStencil(int st_indx);
        void computeWeightsForStencil(DerTypeID deriv_type, int st_indx);
        void computeWeightsForAllStencils(DerTypeID deriv_type);

        // Compute Multiple RHS Direct
        void computeAllWeightsForStencil_Direct(int st_indx);
        // Single RHS Direct
        void computeWeightsForStencil_Direct(DerTypeID dt, int st_indx);

        // Single RHS ContourSVD
        void computeWeightsForStencil_ContourSVD(DerTypeID dt, int st_indx);

        void setUseHyperviscosity(int tf) {
            std::cout << "USE " << tf << std::endl;
            useHyperviscosity = tf; 
        }

        int getUseHyperviscosity() {
            return useHyperviscosity;
        }

        double getHVScalar() {
            double scale = -hv_gamma * pow((double)grid_ref.getGlobalNodeListSize(), -hv_k);
            static int printed = 0; 
            if (!printed) {
                std::cout << "HVSCALAR = " << scale << " (HV_K=" << hv_k << ", HV_GAMMA=" << hv_gamma << ")" << std::endl;
                printed = 1; 
            }
            return scale; 
        }

        // Allow me to override the defaults. Right now, the scalars are set for cosine bell
        double setHVScalars(int k, double gamma_c) {
            hv_gamma = gamma_c; 
            hv_k = k;
            return hv_gamma;
        }

        void setComputeConditionNumber(bool tf) {
            computeCondNums = tf; 
        }

        void setWeightType(RBFFD::WeightType type) {
            weightMethod = type;
        }


        // Apply weights to an input solution vector and get the corresponding derivatives out
        virtual void applyWeightsForDeriv(DerTypeID which, std::vector<double>& u, std::vector<double>& deriv, bool isChangedU=true) { 
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
        virtual void applyWeightsForDeriv(DerTypeID which, int npts, double* u, double* deriv, bool isChangedU=true);


        // Returns MAXIMUM negative eigenvalue
        // Use output to obtain MINIMUM and MAXIMUM
        double computeEigenvalues(DerTypeID which, bool exit_on_fail, EigenvalueOutput* output=NULL);


        // Same as next routine below, but this allows manual override fo the c1, c2 parameters
        void setEpsilonByParameters(double c1, double c2) {
            // Epsilon as a function of condition number is a linear function: 
            double eps = c1 * sqrt(grid_ref.getGlobalNodeListSize()) - c2; 
            this->setEpsilon(eps); 
        }



        // Support parameter type 3 (variable depends on stencil size and might
        // ONLY work on a sphere) These will be available as part of Natasha
        // and Erik's new paper on shallow water.  Given a stencil size (right
        // now we only have a few candidates), we know the support parameter
        // can scale linearly to produce a constant average condition number.
        // So we take the stencil size and automatically adjust the support. 
        void setEpsilonByStencilSize(unsigned int st_size) {
            double c1, c2; 

            // NOTE: these are specific to the cosine bell. use
            // "setHVScalars(k, gamma)" to specify the scalars for vortex roll
            switch(st_size) {
                case 17: 
                    c1 = 0.026; 
                    c2 = 0.08; 
                    hv_k = 2; 
                    // NOTE: After testing this, I found that this gamma includes N^{-k} within it. 
                    //      This is very VERY important
                    hv_gamma = 8e-4;
                    break; 
                case 31: 
                    c1 = 0.035; 
                    c2 = 0.1; 
                    hv_k = 4; 
                    hv_gamma = 5e-2;
                    break; 
                case 50: 
                    c1 = 0.044; 
                    c2 = 0.14; 
                    hv_k = 6; 
                    hv_gamma = 5e-1;
                    break;
                case 101:
                    c1 = 0.058; 
                    c2 = 0.16; 
                    // These params get me hvscalar approx equal to 1e-39 at N=7225
                    hv_k = 8; hv_gamma=5e-2;
                    //hv_k = 10; hv_gamma=0.38;

                    // These dont work yet: 
                    //NO: hv_k = 6; hv_gamma=1.43e-12;
                    //hv_k = 8; hv_gamma=7.43e-9;
                    //NO: hv_k = 12; hv_gamma=2.03e7;
                    //hv_gamma = 1e-39;
                    // hv_gamma = 0.39;
                    break; 
                default:
                    std::cout << "[RBFFD] Error: setEpsilonByStencilSize does not support stencil size " << st_size << " at this time. Try using 17, 31, 50, and 101\n"; 
                    exit(EXIT_FAILURE);
            };

            // Epsilon as a function of condition number is a linear function: 
            double eps = c1 * sqrt(grid_ref.getGlobalNodeListSize()) - c2; 
            this->setEpsilon(eps); 
        }


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

        std::vector<double*>& getWeights(DerTypeID choice) { return weights[getDerTypeIndx(choice)]; }
        std::vector<NodeType> getNodes() { return grid_ref.getNodeList(); }
        double*& getStencilWeights(DerTypeID choice, int st_indx) { return weights[getDerTypeIndx(choice)][st_indx]; }



        void writeAllWeightsToFile(); 
        void writeToFile(DerTypeID which, std::string filename);
        void writeToFile(DerTypeID which) { this->writeToFile(which, this->getFilename(which)); }
        int  loadAllWeightsFromFile();
        int  loadFromFile(DerTypeID which, std::string filename);
        int  loadFromFile(DerTypeID which){ return this->loadFromFile(which, this->getFilename(which)); }

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
        void getStencilRHS(DerTypeID which, std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& rhs, double h);
        void getStencilLHS(std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& lhs, double h);

        // =====================================================================
        // Convert a basic filename like "output_file" to something more
        // descriptive and appropriate to the pde like
        // "output_file_ncar_poisson1_iteration_10.ascii" 
        std::string getFilename(DerTypeID which, std::string base_filename);

        // Get a filename appropriate for output from this class
        // same as getFilename(std::string, int) however it uses 
        // the class's internal name instead of a user specified string. 
        std::string getFilename(DerTypeID which); 

        // Get a string that gives some detail about the grid (used by
        // expandFilename(...)) 
        // NOTE: replace spaces with '_'
        virtual std::string getFileDetailString(DerTypeID which); 

        virtual std::string getEpsString() {
            return eps_string; 
        }

        // String for hyperviscosity
        virtual std::string getHVString() {
            std::stringstream ss(std::stringstream::out); 
            if (this->useHyperviscosity) {
                ss << "hv_k_" << hv_k << "_hv_gamma_" << hv_gamma; 
            } else {
                ss << "no_hv"; 
            }
            return ss.str(); 
        }

        virtual std::string className() { return "rbffd"; }
        // =====================================================================

        // ================= These control derivative type bits ================
    public: 
        void printCalculated() {
            // Iterate through flags and prints which ones we are going to calculate
            for (int i = 0; i < NUM_DERIV_TYPES; i++) {
                DerTypeID dt = getDerType((DerTypeIndx)i); 
                if (isSelected(dt)) {
                    cout << "Will compute DerType: 0x" << hex << dt << endl;
                }
            }
        }        

        bool isSelected(DerTypeID dt) {
            return computedTypes & dt; 

        }

        // Set a collection of derivative types (only updates the existing)
        void setDerTypes(DerTypes derTypeList) {
            computedTypes = derTypeList;
        }

        // Set a collection of derivative types (only updates the existing)
        void appendDerTypes(DerTypes& derTypeList) {
            computedTypes |= derTypeList;
        }

        DerTypes getDerTypes() {
            return computedTypes; 
        }

        // Flip a single bit for a specific dertype 
        void switchDerType (DerTypeID& dt) {
            computedTypes ^= dt;
        }

        void removeDerType(DerTypeID& dt) {
            if ((computedTypes & dt) == dt) {
                computedTypes ^= dt;
            }
        }

};

#endif 
