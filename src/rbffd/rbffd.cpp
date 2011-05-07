#include "rbffd.h"
// For writing weights in (sparse) matrix market format
#include "mmio.h"

// Note: dim_num here is the desired dimensions for which we calculate derivatives
// (up to 3 right now) 
    RBFFD::RBFFD(Domain& grid, int dim_num_)//, RBF_Type rbf_choice) 
: grid_ref(grid), dim_num(dim_num_)
{
    int nb_rbfs = grid.getNodeListSize(); 

    for (int i = 0; i < NUM_DERIV_TYPES; i++) {
        // All weights should be NULL initially so we can tell if they're
        // already allocated
        this->weights[i].resize(nb_rbfs, NULL); 
    }

    // each stencil has a support specified at its center
    // but more importantly, the stencil nodes (neighbors) 
    // also have their own supports 
    // TODO: verify that the Domain class passes all nb_rbf support params to each subdomain
    var_epsilon.resize(nb_rbfs); 

    this->setupTimers(); 
}
//--------------------------------------------------------------------

RBFFD::~RBFFD() {
    for (int j = 0; j < NUM_DERIV_TYPES; j++) {
        for (int i = 0; i < weights[j].size(); i++) {
            if (weights[j][i] != NULL) {
                delete [] weights[j][i];
            }
        }
    }
}
//--------------------------------------------------------------------

// Compute the full set of derivative weights for all stencils 
void RBFFD::computeAllWeightsForAllStencils() {

    std::vector<StencilType>& st_map = grid_ref.getStencils(); 
    size_t nb_st = st_map.size(); 

    for (size_t i = 0; i < nb_st; i++) {
        this->computeAllWeightsForStencil(i); 
    }
}

//--------------------------------------------------------------------
void RBFFD::getStencilMultiRHS(std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& rhs) {
    size_t nn = stencil.size()+num_monomials; 
    for (int i = 0; i < NUM_DERIV_TYPES; i++) {
        arma::mat col(nn,1); 
        // Fill by column
       this->getStencilRHS((DerType)i, rbf_centers, stencil, num_monomials, col); 
       // we want a( : , i ) where : is a vec of length nn
       rhs.submat(0,i,nn-1,i) = col; 
    }
//    return rhs; 
}
//--------------------------------------------------------------------


void RBFFD::getStencilLHS(std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& d_matrix) {

    int n = stencil.size();
    int np = num_monomials;
    // Generate a distance matrix and find the SVD of it.
    // n+4 = 1 + dim(3) for x,y,z
//    arma::mat d_matrix(n+np, n+np);
//    d_matrix.zeros(n+np,n+np);
    d_matrix.zeros(); 

    // value 0 => stencil center is at index 0 in "stencil"
    // dim_num required for RBF
    this->distanceMatrix(rbf_centers, stencil, dim_num, d_matrix);

    // Fill the polynomial part
    for (int i=0; i < n; i++) {
        d_matrix(n, i) = 1.0;
        d_matrix(i, n) = 1.0;
    }
    if (np > 1) {
        for (int i=0; i < n; i++) {
            d_matrix(n+1, i) = rbf_centers[stencil[i]].x();
            d_matrix(i, n+1) = rbf_centers[stencil[i]].x();

            if (np > 2) {
                d_matrix(n+2, i) = rbf_centers[stencil[i]].y();
                d_matrix(i, n+2) = rbf_centers[stencil[i]].y();
            }

            if (np == 4) {
                d_matrix(n+3, i) = rbf_centers[stencil[i]].z();
                d_matrix(i, n+3) = rbf_centers[stencil[i]].z();
            }
        }
    }

   // d_matrix.print("DISTANCE MATRIX BEFORE: ");
}

//--------------------------------------------------------------------

// Compute the full set of weights for a derivative type
void RBFFD::computeAllWeightsForStencil(int st_indx) {
    // Same as computeAllWeightsForStencil, but we dont leverage multiple RHS solve
    tm["computeAllWeightsOne"]->start(); 
    StencilType& stencil = grid_ref.getStencil(st_indx); 
    int n = stencil.size();
    int np = 1+dim_num; // +3 for the x,y,z monomials

    std::vector<NodeType>& rbf_centers = grid_ref.getNodeList(); 

    // Stencil center
    Vec3& x0v = rbf_centers[stencil[0]];

    arma::mat rhs = arma::mat(n+np, NUM_DERIV_TYPES); 
    arma::mat lhs = arma::mat(n+np, n+np); 

    this->getStencilMultiRHS(rbf_centers, stencil, np, rhs);
    this->getStencilLHS(rbf_centers, stencil, np, lhs);

    lhs.print("LHS="); 
    rhs.print("RHS="); 


    // Remember: b*(A^-1) = (b*(A^-1))^T = (A^-T) * b^T = (A^-1) * b^T
    // because A is symmetric. Rather than compute full inverse we leverage
    // the solver for increased efficiency
    // We dont need the transpose of the RHS because we fill it in column form already
    // Also we use the multiple rhs solver for improved efficiency (BLAS3).
    arma::mat weights_new = arma::solve(lhs, rhs); //bx*Ainv;
    int irbf = st_indx;
    weights_new.print("weights");
    // FIXME: do not save the extra NP coeffs
    for (int i = 0; i < NUM_DERIV_TYPES; i++) {
        if (this->weights[i][irbf] == NULL) {
            this->weights[i][irbf] = new double[n+np];
        }
        for (int j = 0; j < n+np; j++) {
            this->weights[i][irbf][j] = weights_new(j, i);
        }

#if 1
        double sum_nodes_only = 0.;
        double sum_nodes_and_monomials = 0.;
        for (int j = 0; j < n; j++) {
            sum_nodes_only += weights_new(j,i);
        }
        sum_nodes_and_monomials = sum_nodes_only;
        for (int j = n; j < n+np; j++) {
            sum_nodes_and_monomials += weights_new(j,i);
        }

        cout << "(Stencil: " << irbf << ", DerivType: " << i << ") ";
        //weights_new.print("lapl_weights");
        cout << "Sum of Stencil Node Weights: " << sum_nodes_only << endl;
        cout << "Sum of Node and Monomial Weights: " << sum_nodes_and_monomials << endl;
        if (sum_nodes_only > 1e-7) {
            cout << "WARNING! SUM OF WEIGHTS FOR LAPL NODES IS NOT ZERO: " << sum_nodes_only << endl;
            exit(EXIT_FAILURE);
        }
#endif // DEBUG
    }

    tm["computeAllWeightsOne"]->end();
}

//--------------------------------------------------------------------

void RBFFD::getStencilRHS(DerType which, std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& rhs) { 

    int np = num_monomials; 
    int n = stencil.size(); 

    // Assume single RHS
//    rhs.zeros(n);

    NodeType& x0v = rbf_centers[stencil[0]];

    // NOTE: we want to evaluate the analytic derivs of phi at the stencil center point
    // using every stencil RBF: 
    //  RHS: 
    //  | phi_0(x_0)  phi_1(x_0) ... phi_N(x_0) |^T
    //  LHS: 
    //  DMat(stencil) 
    // CONSISTENCY: index x_"j" indices stencil center in all papers
    for (int j=0; j < n; j++) {
        // We want to evaluate every basis function. Each stencil node xjv has
        // its own basis function and we evaluate them with the distance to the
        // stencil center node x0v. This is B_j(||x0v - xjv||)
        // NOTE: when all basis functions are the same (eqn and support) then
        // B_c(||x0v - xjv||) = B_j(||x0v - xjv||).
        IRBF rbf(var_epsilon[stencil[j]], dim_num); 

        // printf("%d\t%d\n", j, stencil[j]);
        Vec3& xjv = rbf_centers[stencil[j]];
        //xjv.print("xjv = ");
        switch (which) {
            case X:
                rhs(j,0) = rbf.xderiv(x0v, xjv);
                break; 
            case Y: 
                rhs(j,0) = rbf.yderiv(x0v, xjv);
                break; 
            case Z:
                rhs(j,0) = rbf.zderiv(x0v, xjv);
                break; 
            case LAPL: 
                rhs(j,0) = rbf.lapl_deriv(x0v, xjv);
                break; 
            default:
                std::cout << "[RBFFD] ERROR: deriv type " << which << " is not supported yet\n" << std::endl;
                exit(EXIT_FAILURE); 
        }
    }

    if (np > 0) {
        // REQUIRE at least np = 1 (will ERR out if not valid)
        rhs(n) = 0.0; 

        switch (which) {
            case X: 
                if (np > 1) {
                    rhs(n+1) = 1.; 
                }
                break; 
            case Y: 
                if (np > 2) {
                    rhs(n+2) = 1.; 
                }
                break; 
            case Z:
                if (np > 3) {
                    rhs(n+3) = 1.; 
                }
                break; 
            case LAPL: 
                // all 3 dims for LAPL are 0.0
                break; 
            default:
                break; 
        }
    }
    rhs.print("RHS before");
}


//--------------------------------------------------------------------

void RBFFD::computeWeightsForStencil(DerType which, int st_indx) {
    // Same as computeAllWeightsForStencil, but we dont leverage multiple RHS solve
    tm["computeOneWeights"]->start(); 



    StencilType stencil = grid_ref.getStencil(st_indx); 
    std::vector<NodeType>& rbf_centers = grid_ref.getNodeList(); 

    int np = 1+dim_num; // +3 for the x,y,z monomials
    int n = stencil.size();

    // Stencil center
    Vec3& x0v = rbf_centers[stencil[0]];

    arma::mat rhs(n+np, 1); 
    rhs.zeros();
    arma::mat lhs(n+np, n+np); 
    lhs.zeros();

    this->getStencilRHS(which, rbf_centers, stencil, np, rhs);
    this->getStencilLHS(rbf_centers, stencil, np, lhs); 

    rhs.print("RHS="); 
    lhs.print("LHS=");

    // Remember: b*(A^-1) = (b*(A^-1))^T = (A^-T) * b^T = (A^-1) * b^T
    // because A is symmetric. Rather than compute full inverse we leverage
    // the solver for increased efficiency
    arma::mat weights_new = arma::solve(lhs, trans(rhs)); //bx*Ainv;
    int irbf = st_indx;

    if (this->weights[which][irbf] == NULL) {
        this->weights[which][irbf] = new double[n+np];
    }
    for (int j = 0; j < n+np; j++) {
        this->weights[which][irbf][j] = weights_new[j];
    }

#if DEBUG
    double sum_nodes_only = 0.;
    double sum_nodes_and_monomials = 0.;
    for (int j = 0; j < n; j++) {
        sum_nodes_only += weights_new[j];
    }
    sum_nodes_and_monomials = sum_nodes_only;
    for (int j = n; j < n+np; j++) {
        sum_nodes_and_monomials += weights_new[j];
    }

    cout << "(" << irbf << ") ";
    weights_new.print("lapl_weights");
    cout << "Sum of Stencil Node Weights: " << sum_nodes_only << endl;
    cout << "Sum of Node and Monomial Weights: " << sum_nodes_and_monomials << endl;
    if (sum_nodes_only > 1e-7) {
        cout << "WARNING! SUM OF WEIGHTS FOR LAPL NODES IS NOT ZERO: " << sum_nodes_only << endl;
        exit(EXIT_FAILURE);
    }
#endif // DEBUG

    tm["computeOneWeights"]->end();
}

//--------------------------------------------------------------------

// Use a direct solver on to calculate weights.
int RBFFD::computeStencilWeights(std::vector<Vec3>& rbf_centers, StencilType& stencil, int irbf, double* weights) {

}

//--------------------------------------------------------------------

void RBFFD::applyWeightsForDeriv(DerType which, int npts, double* u, double* deriv) {
    ;
}

//--------------------------------------------------------------------

void RBFFD::setupTimers() {
    tm["computeAllWeightsAll"] = new EB::Timer("[RBFFD] Compute All Weights For ALL Stencils (CPU)"); 
    tm["computeAllWeightsOne"] = new EB::Timer("[RBFFD] Compute All Weights For One Stencil (CPU)"); 
    tm["computeOneWeights"] = new EB::Timer("[RBFFD] Compute One Weights For One Stencil (CPU)"); 
    tm["fillDMat"] = new EB::Timer("[RBFFD] Fill Distance Matrix"); 
    tm["apply"] = new EB::Timer("[RBFFD] Apply Weights for a single derivative type of u"); 
    tm["applyAll"] = new EB::Timer("[RBFFD] Apply Weights for all derivative types of u"); 
}

//--------------------------------------------------------------------

void RBFFD::distanceMatrix(std::vector<NodeType>& rbf_centers, StencilType& stencil, int dim_num, arma::mat& ar) {
    // We assume that all stencils are centered around the first node in the Stencil node list.
    size_t irbf = stencil[0]; 
    Vec3& c = rbf_centers[irbf];
    size_t n = stencil.size();
    //printf("stencil size= %d\n", n);

    //printf("stencil size(%d): n= %d\n", irbf, n);
    //printf("n= %d\n", n);

    //arma::mat &ar = *distance_matrix;
    // mat NewMat(memspace, rowdim, coldim, reuseMemSpace?)


    // Assume ar is pre-allocated
#if 0
    arma::mat& ar(distance_matrix,nrows,ncols,false);

    //mat ar(n,n);
    // Derivative of a constant should be zero
    // Derivative of a linear function should be constant
    //
    if ((ar.n_cols > n)||(ar.n_rows > n)) {
        ar.submat(0,0,n-1,n-1) = zeros<mat>(n,n);
    } else {
        ar.zeros(n,n);
    }
#endif 
    int st_center = -1;

    // which stencil point is irbf
    for (int i=0; i < n; i++) {
        if (irbf == stencil[i]) {
            st_center = i;
            break;
        }
    }
    if (st_center == -1) {
        printf("inconsistency with global rbf map (stencil should contain center: %d)\n", irbf);
        exit(0);
    }

    // DMat: (note: phi_0(x_N) is the 0th RBF evaluated at x_N) 
    //
    //  | phi_0(x_0)   phi_0(x_1)   ... phi_0(x_N) |
    //  | phi_1(x_0)   phi_1(x_1)   ... phi_1(x_N) |
    //  |     ...         ...             ...      | 
    //  | phi_N(x_0)   phi_N(x_1)   ... phi_N(x_N) |
    //
    // stencil includes the point itself
    for (int j=0; j < n; j++) {
        // We allow a unique var_epsilon at each RBF. within a stencil.
        // TODO: test when its a unique var_epsilon by stencil, not by RBF *WITHIN* a stencil
        IRBF rbf(var_epsilon[stencil[j]], dim_num);
        // rbf centered at xj
        Vec3& xjv = rbf_centers[stencil[j]];
        for (int i=0; i < n; i++) {
            // by row
            Vec3& xiv = rbf_centers[stencil[i]];
            // Center is on right:
            ar(j,i) = rbf(xiv, xjv);
        }
    }

//    ar.print("INSIDE DMATRIX");

}

//--------------------------------------------------------------------

void RBFFD::writeToFile(DerType which, std::string filename) {

    // number of non-zeros (should be close to max_st_size*num_stencils)
    size_t nz = 0;

    std::vector<StencilType>& stencil = grid_ref.getStencils(); 
    std::vector<double*>* deriv_choice_ptr = &(weights[which]); 
#if 0
    switch (which) {
        case X: 
            deriv_choice_ptr = &x_weights; 
            break; 
        case Y: 
            deriv_choice_ptr = &y_weights; 
            break; 
        case Z: 
            deriv_choice_ptr = &z_weights; 
            break; 
        case LAPL: 
            deriv_choice_ptr = &lapl_weights; 
            break; 
        default: 
            std::cout << "[Derivative] ERROR! INVALID CHOICE INSIDE writeToFile()\n"; 
            exit(EXIT_FAILURE);
            break; 
    }
#endif 
    for (size_t i = 0; i < stencil.size(); i++) {
        nz += stencil[i].size();
    }
    fprintf(stdout, "Writing %d weights to %s\n", nz, filename.c_str()); 

    // Num stencils (x_weights.size())
    const size_t M = (*deriv_choice_ptr).size();
    // We have a square MxN matrix
    const size_t N = M;

    // Value obtained from mm_set_* routine
    MM_typecode matcode;                        
    
  //  int I[nz] = { 0, 4, 2, 8 };
  //  int J[nz] = { 3, 8, 7, 5 };
  //  double val[nz] = {1.1, 2.2, 3.2, 4.4};

    int err = 0; 
    FILE *f; 
    f = fopen(filename.c_str(), "w"); 
    err += mm_initialize_typecode(&matcode);
    err += mm_set_matrix(&matcode);
    err += mm_set_coordinate(&matcode);
    err += mm_set_real(&matcode);

    err += mm_write_banner(f, matcode); 
    err += mm_write_mtx_crd_size(f, M, N, nz);

    /* NOTE: matrix market files use 1-based indices, i.e. first element
       of a vector has index 1, not 0.  */
//    fprintf(stdout, "Writing file contents: \n"); 
    for (size_t i = 0; i < stencil.size(); i++) {
        for (size_t j = 0; j < stencil[i].size(); j++) {
            // Add 1 because matrix market assumes we index 1:N instead of 0:N-1
            fprintf(f, "%d %d %lg\n", stencil[i][0]+1, stencil[i][j]+1, (*deriv_choice_ptr)[i][j]); 
           // fprintf(stdout, "%d %d %lg\n", stencil[i][0]+1, stencil[i][j]+1, (*deriv_choice_ptr)[i][j]); 
            //fprintf(stdout, "%d %d %lg\n", i, j, (*deriv_choice_ptr)[i][j]); 
        }
    }

    fclose(f);
}

//----------------------------------------------------------------------