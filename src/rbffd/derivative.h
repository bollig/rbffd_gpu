#ifndef _DERIVATIVE_H_
#define _DERIVATIVE_H_

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "ArrayT.h"
#include "Vec3.h"
//#include <armadillo>
#include "rbffd/rbfs/rbf_gaussian.h"
#include "rbffd/rbfs/rbf_mq.h"
#include "utils/conf/projectsettings.h"
#include "timer_eb.h"
#include "common_typedefs.h" 

//typedef RBF_Gaussian IRBF;
typedef RBF_MQ IRBF;

typedef std::vector<IRBF*> BasesType;
typedef std::vector< BasesType > BasesListType;

class Derivative
{
    public:
        enum DerType {X, Y, Z, LAPL};

    protected:
        typedef ArrayT<double> AF;
        //AF arr;
        size_t nb_rbfs;
        size_t nb_stencils;
        double maxint;

        //BasesListType rbfs;
        double* weights_p;
        // Make sure these are deleted inside the destructor
        std::vector<double*> x_weights;
        std::vector<double*> y_weights;
        std::vector<double*> z_weights;
        std::vector<double*> lapl_weights;

        std::vector<Vec3>& rbf_centers;
        std::vector<StencilType >& stencil;
        std::vector<double> avg_stencil_radius;

        // Two choices of epsilon (support parameter) for RBFs: 
        //  0) Static: 
        //  1) Variable: 
        //  (fill this buffer with uniform values if static)
        std::vector<double> var_epsilon;
        // specify selection for epsilon type in case subsets of the code depend on it
        int use_var_eps;

        //    int nb_bnd; // number of points on the boundary (EB: is this the boundary of subdomain or PDE?)

        // Configurable option from projectSettings
        int debug_mode;		// optional
        int dim_num; 		// required

        EB::TimerList tm; 

    public:
        //Derivative(int nb_rbfs);
        Derivative(std::vector<Vec3>& rbf_centers_, std::vector<StencilType>& stencil_, int nb_bnd_pts, int dim_num);
        ~Derivative();

        void setupTimers(); 

        AF& cholesky_cpu(AF& arr);
        AF& cholesky(AF& arr);
        double rowDot(AF& a, int row_a, AF& b, int row_b, int ix);

        // use an SVD decomposition on the distance matrix to simplify the direct solve.
        void computeWeightsSVD_Direct(std::vector<Vec3>& rbf_centers, StencilType& stencil, int irbf);

        // Use a direct solver on teh distance matrix.
        int computeWeights(std::vector<Vec3>& rbf_centers, StencilType& stencil, int irbf);

        // Use Grady's contourSVD (my c++) version to compute the weights
        // choice: compute either "lapl", "x" or "y" derivative stencils
        void computeWeightsSVD(std::vector<Vec3>& rbf_centers, StencilType& stencil, int irbf, const char* choice);

        // Fill distance_matrix with nrows-by-ncols values for a distance matrix where each row i corresponds 
        // to the distances from rbf_centers in stencil i to the ith rbf_center. 
        void distanceMatrix(std::vector<Vec3>& rbf_centers, StencilType& stencil, int irbf, double* distance_matrix, int nrows, int ncols, int dim_num);

        AF& solve(AF& l, AF& b);
        AF& matmul(AF& arr, AF& x);

        // u : take derivative of this scalar variable (already allocated)
        // deriv : resulting derivative (already allocated)
        // which : which derivative (X, Y, LAPL)
        // NOTE: these are on the CPU. we need GPU equivalents to perform update

        // Developers note: the second of these routines WAS the primary and did all the work
        //          the first was just a redirect to it. However, I need to override the virtual
        //          routine for a GPU version of the class. Apparently if I override a virtual
        //          function, any other functions with the same name must be overridden or will
        //          not be available when calling from the derived class. To work around this
        //          now BOTH are redirects to the computeDerivatives(..) routine.
    public: 
        void computeDeriv(DerType which, std::vector<double>& u, std::vector<double>& deriv);
        void computeDeriv(DerType which, double* u, double* deriv, int npts);

    // Hide this one so we force the use of computeDeriv which redirects to it (FIXME)
    protected:  
        virtual void computeDerivatives(DerType which, double* u, double* deriv, int npts);

    public: 
        void computeDerivCPU(DerType which, std::vector<double>& u, std::vector<double>& deriv);
        void computeDerivativesCPU(DerType which, double* u, double* deriv, int npts);


        std::vector<double*>& getXWeights() { return x_weights; }
        std::vector<double*>& getYWeights() { return y_weights; }
        std::vector<double*>& getZWeights() { return z_weights; }
        std::vector<double*>& getLaplWeights() { return lapl_weights; }

        double* getXWeights(int indx) { return x_weights[indx]; }
        double* getYWeights(int indx) { return y_weights[indx]; }
        double* getZWeights(int indx) { return z_weights[indx]; }
        double* getLaplWeights(int indx) { return lapl_weights[indx]; }

        //    BasesType& getRBFList(int stencil_indx) { return rbfs[stencil_indx]; }
        //    BasesListType& getRBFList() { return rbfs; }

        double computeEig();

        // Update the derivative class with a set of avg stencil radii. 
        // Other parameters are 
        //      alpha  -> scaling on numerator for (eps = alpha/(r^beta)) where r is the stencil radius
        //      beta   -> power on r for (eps = alpha/r^beta). 
        void setAvgStencilRadius(std::vector<double>& avg_radius_) {
            this->setVariableEpsilon(avg_radius_); 
        }
        void setVariableEpsilon(std::vector<double>& avg_radius_) {
            // FIXME: stencil is class param, but avg_radius is not
            this->setVariableEpsilon(avg_radius_, stencil); 
        }
        void setVariableEpsilon(std::vector<double>& avg_radius_, std::vector<StencilType>& stencils, double alpha=1.0f, double beta=1.0f) {
            use_var_eps = 1;
            //std::cout << "DERIVATIVE:: SET VARIABLE EPSILON = " << alpha << "/(avg_st_radius^" << beta << ")" << std::endl;
            std::cout << "NOTE: We are computing the variable support parameter using the avg radius of a stencil.\n"; 
            std::cout << "Every node in the stencil will use the same support!\n"; 
            std::cout << "If you want unique support for each node in the stencil you will need to modify Derivative.h and make sure the Domain class passes additional stencil information for ghost nodes (right now ghost nodes are contained within stencils, but we are unaware of what their own stencils (i.e., stencils for which they are the centers) look like\n"; 

            std::cout << "AVG_RADIUS.size() = " << avg_radius_.size() << "\t STENCILS.size() = " << stencils.size() << std::endl;
            avg_stencil_radius = avg_radius_;
            var_epsilon.resize(avg_stencil_radius.size());
            for (int i=0; i < stencils.size(); i++) {
                // var_epsilon[i] = alpha / std::pow(avg_stencil_radius[i], beta);
                //            var_epsilon[i] = alpha * avg_stencil_radius[i] / sqrt(beta);

                // Hardy 1972: 
                //var_epsilon[i] = 1.0 / (0.815 * avg_stencil_radius[i]);

                // Franke 1982: 
                // TODO: franke actually had max_stencil_radius in denom
                //var_epsilon[i] = 0.8 * sqrt(stencils[i].size()) / max_stencil_radius[i] ;

                // Note: for 24x24, alpha = 0.04. For 64x64, alpha = 0.05; for 1000x1000, alpha = 0.07
                // we use stencils[i][0] to get the index for the stencil center and its corresponding "avg_radius" 
                std::cout << "var_epsilon[" << i << "] = " << alpha << " * sqrt( " << stencils[i].size() << " / avg_radius_[ " << stencils[i][0] << " ] " << std::endl; 
                // the indx on var_epsilon should be linear 0->stencils.size(), but just in case we have random access based on stencil center index
                var_epsilon[stencils[i][0]] = (alpha * sqrt(stencils[i].size())) / avg_radius_[stencils[i][0]] ;

                //   printf("var_epsilon(%d) = %f (%f, %f, %f)\n", i, var_epsilon[i], alpha, sqrt(stencils[i].size()), avg_stencil_radius[i]);
            }
        }
#if 0
        // Update the variable epsilons without using stencil radii
        void setVariableEpsilon(std::vector<double>& var_eps_) {
            this->var_eps = var_eps_;
        }
#endif 
        double minimum(std::vector<double>& vec);

        double setEpsilon(double eps) { 
            use_var_eps = 1; 
            std::cout << "DERIVATIVE:: SET UNIFORM EPSILON = " << eps << std::endl;
            for (int i = 0; i < var_epsilon.size(); i++) {
                this->var_epsilon[i] = eps; 
            }
            return this->var_epsilon[0];
        }
};

#endif
