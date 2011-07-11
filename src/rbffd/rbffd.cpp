#define ONE_MONOMIAL 1
#define SCALE_BY_H 0 
#define SCALE_OUT_BY_H 0 

#include "rbffd/stencils.h"
#include "utils/geom/cart2sph.h"

#include "rbffd.h"
// For writing weights in (sparse) matrix market format
#include "mmio.h"

// Note: dim_num here is the desired dimensions for which we calculate derivatives
// (up to 3 right now) 
    RBFFD::RBFFD(Grid* grid, int dim_num_, int rank_)//, RBF_Type rbf_choice) 
: grid_ref(*grid), dim_num(dim_num_), rank(rank_), 
    weightsModified(false), weightMethod(RBFFD::Direct), 
    eigenvalues_computed(false)
{
    int nb_rbfs = grid_ref.getNodeListSize(); 

    for (int i = 0; i < NUM_DERIV_TYPES; i++) {
        // All weights should be NULL initially so we can tell if they're
        // already allocated
        this->weights[i].resize(nb_rbfs, NULL); 
    }

    derTypeStr[X] = "x";
    derTypeStr[Y] = "y";
    derTypeStr[Z] = "z";
    derTypeStr[LAPL] = "lapl";
    derTypeStr[HV2] = "hv2";
    derTypeStr[R] = "r";   // radial deriv 
    derTypeStr[LAMBDA] = "lambda";   // Longitude
    derTypeStr[INTERP] = "interp";


    weightTypeStr[0] = "direct"; 
    weightTypeStr[1] = "contoursvd"; 

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
    unsigned int nb_st = st_map.size(); 

    for (unsigned int i = 0; i < nb_st; i++) {
        this->computeAllWeightsForStencil(i); 
    }

    weightsModified = true;
}

//--------------------------------------------------------------------
void RBFFD::getStencilMultiRHS(std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& rhs, double h) {
    unsigned int nn = stencil.size()+num_monomials; 
    for (int i = 0; i < NUM_DERIV_TYPES; i++) {
        arma::mat col(nn,1); 
        // Fill by column
        this->getStencilRHS((DerType)i, rbf_centers, stencil, num_monomials, col, h); 
        // we want a( : , i ) where : is a vec of length nn
        rhs.submat(0,i,nn-1,i) = col; 
    }
    //    return rhs; 
}
//--------------------------------------------------------------------


void RBFFD::getStencilLHS(std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& d_matrix, double h) {

    int n = stencil.size();
    int np = num_monomials;
    // Generate a distance matrix and find the SVD of it.
    // n+4 = 1 + dim(3) for x,y,z
    //    arma::mat d_matrix(n+np, n+np);
    //    d_matrix.zeros(n+np,n+np);
    d_matrix.zeros(); 

    // value 0 => stencil center is at index 0 in "stencil"
    // dim_num required for RBF
    this->distanceMatrix(rbf_centers, stencil, dim_num, d_matrix, h);
 //   d_matrix.print("DMAT=");
 //   arma::mat re = d_matrix.submat(0,0,n,0);
//    re.print("RE=");
//    this->rbf(d_matrix);

    if (np > 0) {
        // Fill the polynomial part
        for (int i=0; i < n; i++) {
            d_matrix(n, i) = 1.0;
            d_matrix(i, n) = 1.0;
        }
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

    switch (weightMethod)
    {
        case RBFFD::Direct:
            this->computeAllWeightsForStencil_Direct(st_indx);
            break;
        case RBFFD::ContourSVD: 
            for (int i = 0; i < NUM_DERIV_TYPES; i++) {
                this->computeWeightsForStencil_ContourSVD((RBFFD::DerType)i, st_indx);
            }
            break;
        default:
            std::cout << "Unknown method for computing weights" << std::endl;
            exit(EXIT_FAILURE);
    }
}

void RBFFD::computeAllWeightsForStencil_Direct(int st_indx) {
    // Same as computeAllWeightsForStencil, but we dont leverage multiple RHS solve
    tm["computeAllWeightsOne"]->start(); 
    StencilType& stencil = grid_ref.getStencil(st_indx); 
    int n = stencil.size();
#if ONE_MONOMIAL
    int np = 1;//+dim_num; // +3 for the x,y,z monomials
#else 
    int np = 1+dim_num; // +3 for the x,y,z monomials
#endif 
    std::vector<NodeType>& rbf_centers = grid_ref.getNodeList(); 

    // Stencil center
    Vec3& x0v = rbf_centers[stencil[0]];

    arma::mat rhs = arma::mat(n+np, NUM_DERIV_TYPES); 
    arma::mat lhs = arma::mat(n+np, n+np); 

    // We pass h (the minimum dist to nearest node) so we can potentially factor it out
    double h = 1.;
#if SCALE_BY_H
    h = grid_ref.getMaxStencilRadius(st_indx); 
#endif 

    this->getStencilMultiRHS(rbf_centers, stencil, np, rhs, h);
    this->getStencilLHS(rbf_centers, stencil, np, lhs, h);

    // Remember: b*(A^-1) = (b*(A^-1))^T = (A^-T) * b^T = (A^-1) * b^T
    // because A is symmetric. Rather than compute full inverse we leverage
    // the solver for increased efficiency
    // We dont need the transpose of the RHS because we fill it in column form already
    // Also we use the multiple rhs solver for improved efficiency (BLAS3).
    arma::mat weights_new = arma::solve(lhs, rhs); //bx*Ainv;
    int irbf = st_indx;

#if 0
    char buf[256]; 
    sprintf(buf, "LHS(%d)=", st_indx); 
    lhs.print(buf); 
    sprintf(buf, "RHS(%d)=", st_indx); 
    rhs.print(buf); 
    weights_new.print("weights");
#endif 

    // FIXME: do not save the extra NP coeffs
    for (int i = 0; i < NUM_DERIV_TYPES; i++) {
        // X,Y,Z weights should scale by 1/h
        // FIXME: for 3D, h^2
        double scale = 1.;// grid_ref.getStencilRadius(irbf);
 
#if SCALE_OUT_BY_H
        scale = 1./h;
        // LAPL should scale by 1/h^2
        if (i == LAPL) {
            scale *= scale; 
        }
        if (i == INTERP) {
            scale = 1.; 
        }
#endif 
        if (this->weights[i][irbf] == NULL) {
            this->weights[i][irbf] = new double[n+np];
        }

        for (int j = 0; j < n+np; j++) {
            this->weights[i][irbf][j] = weights_new(j, i) * scale;
            //            this->weights[i][irbf][j] = scale;
        }

#if DEBUG
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

    weightsModified = true;
}

//--------------------------------------------------------------------

void RBFFD::getStencilRHS(DerType which, std::vector<NodeType>& rbf_centers, StencilType& stencil, int num_monomials, arma::mat& rhs, double h) { 

    int np = num_monomials; 
    int n = stencil.size(); 

    // Assume single RHS
    rhs.zeros(n+np,1);

    // RHS we evaluate all RBF translates at the stencil center
    NodeType& x0v = rbf_centers[stencil[0]];

    // To test interpolation on sphere
#if 0
    if ((which == INTERP) && false) {
        std::cout << "Changing: " << x0v << " to "; 
        x0v = x0v + Vec3(h, 0., 0.); 
       // x0v.normalize();
        std::cout << x0v << std::endl;
    }
#endif 
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
        //
        // FIXME: allow var_epsilon[stencil[j]] (we need to pass stencil info for ghost nodes if this is to be allowed)
#if SCALE_BY_H
        double eps = var_epsilon[stencil[0]] * h; 
#else 
        double eps = var_epsilon[stencil[0]]; 
#endif 

        IRBF rbf(eps, dim_num); 

        // printf("%d\t%d\n", j, stencil[j]);
        Vec3 xjv = rbf_centers[stencil[j]];

#if SCALE_BY_H
        //xjv.print("xjv = ");
        // This scales the stencil to the unit disk. This seems to improve the conditioning
        // when solving for weights in domains with high numbers of nodes
        Vec3 diff = (x0v - xjv) * (1./h);
#else 
        Vec3 diff = (x0v - xjv);
#endif 
        switch (which) {
            case X:
                rhs(j,0) = rbf.xderiv(diff);
                break; 
            case Y: 
                rhs(j,0) = rbf.yderiv(diff);
                break; 
            case Z:
                rhs(j,0) = rbf.zderiv(diff);
                break; 
            case LAPL: 
                rhs(j,0) = rbf.lapl_deriv(diff);
                break; 
            case INTERP: 
                rhs(j,0) = rbf.eval(diff);
                break; 
            case R: 
                rhs(j,0) = rbf.rderiv(diff);
                break; 
            case LAMBDA:
                {
                    // LAMBDA is: dr/dlambda * dphi/dr = dphi/dlambda
                    // yes, i know: we're swapping theta and phi here. This is consistent with 
                    // Natashas paper.
                    sph_coords_type spherical_coords_j = cart2sph(xjv.x(), xjv.y(), xjv.z());
                    double thetaP_j = spherical_coords_j.phi; 
                    double lambdaP_j = spherical_coords_j.theta; 
                    double r_j = spherical_coords_j.r; 

                    sph_coords_type spherical_coords_i = cart2sph(x0v.x(), x0v.y(), x0v.z());
                    double thetaP_i = spherical_coords_i.phi; 
                    double lambdaP_i = spherical_coords_i.theta; 
                    double r_i = spherical_coords_i.r; 
                    
                    double dr_dlambda = cos(thetaP_i) * cos(thetaP_j) * sin(lambdaP_i - lambdaP_j);

                    double r2 = (x0v - xjv).square();
                    double r = sqrt(r2);

                    if (fabs(dr_dlambda) > 0.) {
                        // See equation below eq 20 in Flyer Wright, Transport schemes paper
                        rhs(j,0) = dr_dlambda * (1./r) * rbf.rderiv(diff); 
                    } else {
                        rhs(j,0) = 0.; 
                    }
                }   // Note: use the {'s to allow new declarations inside case
                break; 
            case HV2: 
                // FIXME: this is only for 2D right now
                rhs(j,0) = rbf.lapl2_deriv2D(diff);
                break; 
            default:
                std::cout << "[RBFFD] ERROR: deriv type " << which << " is not supported yet\n" << std::endl;
                exit(EXIT_FAILURE); 
        }
   }


    if (np > 0) {
        rhs(n) = 0.0; 
    }

    if (np > 1) {
        // Analytic derivs of monomial terms
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
                std::cout << "[RBFFD] warning. np > 1 not supported for deriv type: " << which << std::endl;
                break; 
        }
    }
//        rhs.print("RHS before");
}

//--------------------------------------------------------------------
//
void RBFFD::computeWeightsForAllStencils(DerType which) {
    unsigned int nb_stencils = grid_ref.getStencilsSize(); 
    for (unsigned int i = 0; i < nb_stencils; i++) {
        this->computeWeightsForStencil(which, i);
    }
}

//--------------------------------------------------------------------
//
void RBFFD::computeWeightsForStencil(DerType which, int st_indx) {
    switch (weightMethod) {
        case RBFFD::Direct:
            this->computeWeightsForStencil_Direct(which, st_indx);
            break; 
        case RBFFD::ContourSVD:
            this->computeWeightsForStencil_ContourSVD(which, st_indx);
            break;
        default: 
            std::cout << "Unknown method to compute weights for single stencil" << std::endl;
            exit(EXIT_FAILURE);
    }
}

void RBFFD::computeWeightsForStencil_Direct(DerType which, int st_indx) {
    // Same as computeAllWeightsForStencil, but we dont leverage multiple RHS solve
    tm["computeOneWeights"]->start(); 

    StencilType stencil = grid_ref.getStencil(st_indx); 
    std::vector<NodeType>& rbf_centers = grid_ref.getNodeList(); 

#if ONE_MONOMIAL
    int np = 1;//+dim_num; // +3 for the x,y,z monomials
#else 
    int np = 1+dim_num; // +3 for the x,y,z monomials
#endif 

    int n = stencil.size();

    // Stencil center
    Vec3& x0v = rbf_centers[stencil[0]];

    arma::mat rhs(n+np, 1); 
    rhs.zeros();
    arma::mat lhs(n+np, n+np); 
    lhs.zeros();

    double h = 1.;
#if SCALE_BY_H
    h = grid_ref.getMaxStencilRadius(st_indx);
#endif 

    this->getStencilRHS(which, rbf_centers, stencil, np, rhs, h);
    this->getStencilLHS(rbf_centers, stencil, np, lhs, h); 

    // Remember: b*(A^-1) = (b*(A^-1))^T = (A^-T) * b^T = (A^-1) * b^T
    // because A is symmetric. Rather than compute full inverse we leverage
    // the solver for increased efficiency
    // We avoid the transpose of RHS by allocating a colvec with armadillo
    arma::mat weights_new = arma::solve(lhs, rhs); //bx*Ainv;
    int irbf = st_indx;

#if 0
    char buf[256]; 
    sprintf(buf, "LHS(%d)=", st_indx); 
    lhs.print(buf); 
    sprintf(buf, "RHS(%d)=", st_indx); 
    rhs.print(buf); 
    weights_new.print("weights");
#endif 

    // X,Y,Z weights should scale by 1/h
    double scale = 1.;

#if SCALE_OUT_BY_H
    scale = 1./h;
    // LAPL should scale by 1/h^2
    if (which == LAPL) {
        for (int i = 1; i < dim_num; i++) {
            scale *= scale; 
        }
    }
    if (which == INTERP) {
        scale = 1.; 
    }
#endif 

    if (this->weights[which][irbf] == NULL) {
        this->weights[which][irbf] = new double[n+np];
    }
    for (int j = 0; j < n+np; j++) {
        this->weights[which][irbf][j] = weights_new[j] * scale;
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

    weightsModified = true;
}

//--------------------------------------------------------------------
// NOTE: ignore isChangedU because we are on the CPU
void RBFFD::applyWeightsForDeriv(DerType which, int npts, double* u, double* deriv, bool isChangedU) {
//    std::cout << "CPU VERSION OF APPLY WEIGHTS FOR DERIVATIVES: " << which << std::endl;
    tm["applyAll"]->start(); 
    unsigned int nb_stencils = grid_ref.getStencilsSize(); 
    double der;

    // TODO: this if we took advantage of a sparse matrix container, we might be able to
    // improve this Mat-Vec multiply. We could also do it on the GPU.
    for (unsigned int i=0; i < nb_stencils; i++) {
        double* w = this->weights[which][i]; 
        StencilType& st = grid_ref.getStencil(i);
        der = 0.0;
        unsigned int n = st.size();
        for (unsigned int s=0; s < n; s++) {
            der += w[s] * u[st[s]]; 
        }
        deriv[i] = der;
    }
    tm["applyAll"]->stop();
}

//----------------------------------------------------------------------
//
// Fills a large matrix with the weights for interior nodes as we would for an implicit system
// and puts 1's on the diagonal for boundary nodes.
// Then computes the eigenvalue decomposition using Armadillo
// Then computes the maximum and minimum abs(eigenvalue.real())
// Then counts the number of eigenvalues > 0, number < 0 and number within a
// tolerance of 0, and stops the computation if exit_on_fail is true and the
// number of positive and negative are both > 0 or number of zero is greater
// than 1. The logic for this break is based on Parabolic PDEs having ALL
// positive save for one eigenvalue equal to 0, or ALL negative save for one
// equal to 0. If we have a mix of positive and negative we might have an
// unstable laplace operator 
double RBFFD::computeEigenvalues(DerType which, bool exit_on_fail, EigenvalueOutput* output) 
{
    std::vector<double*>& weights_r = this->weights[which];
  
    // Use a std::set because it auto sorts as we insert
    std::set<unsigned int>& b_indices = grid_ref.getSortedBoundarySet();
    std::set<unsigned int>& i_indices = grid_ref.getSortedInteriorSet(); 
    unsigned int nb_bnd = b_indices.size();
    unsigned int nb_int = i_indices.size();
    int nb_centers = grid_ref.getNodeListSize();
    int nb_stencils = grid_ref.getStencilsSize();

    // compute eigenvalues of derivative operator
    unsigned int sz = weights_r.size();

#if 0
    printf("sz= %lu\n", sz);
    printf("weights.size= %d\n", (int) weights_r.size());
#endif 

    printf("Generating matrix of weights for interior nodes ONLY\n");
    arma::mat eigmat(sz, sz);
    eigmat.zeros();

    // Boundary nodes first with diagonal 1's.
    std::set<unsigned int>::iterator it; 
    int i = 0;
    for (it = b_indices.begin(); it != b_indices.end(); it++, i++) {
        eigmat(*it,*it) = 1.0;
    }
   
    // We put our interior nodes after the boundary nodes in matrix
    for (it = i_indices.begin(); it != i_indices.end(); it++, i++) {
        double* w = weights_r[*it];
        StencilType& st = grid_ref.getStencil(*it);
        for (int j=0; j < st.size(); j++) {
            eigmat(*it,st[j])  = w[j];
            // 	printf ("eigmat(%d, st[%d]) = w[%d] = %g\n", i, j, j, eigmat(i,st[j]));
        }
    }

    printf("sz= %u, nb_bnd= %u\n", sz, nb_bnd);

    arma::cx_colvec eigval;
    arma::cx_mat eigvec;
    printf("Computing Eigenvalues of Laplacian Operator on Interior Nodes\n");
    eig_gen(eigval, eigvec, eigmat);
    //eigval.print("eigval");

    int pos_count=0;
    int neg_count=0;
    int zero_count=0;

    double max_pos_eig= fabs(real(eigval(0)));
    double min_pos_eig = fabs(real(eigval(0)));

    double max_neg_eig = fabs(real(eigval(0)));
    double min_neg_eig = fabs(real(eigval(0)));

    
    int err = 0; 
    FILE *f; 
    std::string filename = "eigenvalues_"; 
    filename.append(this->getFileDetailString(which));
    filename.append(".ascii");
    f = fopen(filename.c_str(), "w");
    
    // Compute number of unstable modes
    // Also compute the largest and smallest (in magnitude) eigenvalue
    //for (int i=0; i < (sz-nb_bnd); i++) {
    for (int i=0; i < sz; i++) {
        double e = real(eigval(i));
        if (e > 1.e-8) {
            pos_count++;
            if (fabs(e) > max_pos_eig) {
                max_pos_eig = fabs(e);
            }
            if (fabs(e) < min_pos_eig) {
                min_pos_eig = fabs(e);
            }
        } else if (e < -1.e-8) {
            neg_count++;
            if (fabs(e) > max_neg_eig) {
                max_neg_eig = fabs(e);
            }
            if (fabs(e) < min_neg_eig) {
                min_neg_eig = fabs(e);
            }
        } else {
            zero_count++;
        }
        fprintf(f, "%+lf%+lfi\n",real(eigval(i)),imag(eigval(i))); 
    }

    fclose(f);
#if 0
    count -= nb_bnd; // since we know at least nb_bnd eigenvalues are are (1+0i)

#endif 
    printf("\n[RBFFD] ****** Number positive eigenvalues: %d (nb_bnd: %d) *******\n\n", pos_count, nb_bnd);
    printf("\n[RBFFD] ****** Number negative eigenvalues: %d *******\n\n", neg_count);
    printf("\n[RBFFD] ****** Number zero (-1e-8 <= eval <= 1e-8) eigenvalues: %d *******\n\n", zero_count);

    printf("min abs(real(eig)) (among negative reals): %f\n", min_neg_eig);
    printf("max abs(real(eig)) (among negative reals): %f\n", max_neg_eig);

    printf("min abs(real(eig)) (among positive reals): %f\n", min_pos_eig);
    printf("max abs(real(eig)) (among positive reals): %f\n", max_pos_eig);

    if (output) {
        output->max_pos_eig = max_pos_eig; 
        output->min_pos_eig = min_pos_eig; 
        output->max_neg_eig = max_neg_eig; 
        output->min_neg_eig = min_neg_eig; 
        output->nb_positive = pos_count; 
        output->nb_negative = neg_count; 
        output->nb_zero     = zero_count; 
    }

    eigenvalues_computed = true;
    cachedEigenvalues.max_pos_eig = max_pos_eig; 
    cachedEigenvalues.min_pos_eig = min_pos_eig;
    cachedEigenvalues.max_neg_eig = max_neg_eig; 
    cachedEigenvalues.min_neg_eig = min_neg_eig;
    cachedEigenvalues.nb_positive = pos_count; 
    cachedEigenvalues.nb_negative = neg_count; 
    cachedEigenvalues.nb_zero     = zero_count; 

    // If both are greater than 0 then we have a problem
    if ((neg_count - nb_bnd > 0) && (pos_count - nb_bnd > 0)){
        std::cout << "[RBFFD] Error: we expect either all positive or all negative eigenvalues (except for the number of boundary nodes).\n";
        if (exit_on_fail) {
            exit(EXIT_FAILURE);
        }
    }
#if 0
    if ((neg_count > 0) && (pos_count > 0)) {
        std::cout << "[RBFFD] Error: parabolic PDEs have either all positive or all negative eigenvalues (except one that is zero).\n";
        if (exit_on_fail) {
            exit(EXIT_FAILURE);
        }
    }
#endif 
    if (zero_count > 1) {
        std::cout << "[RBFFD] Error: parabolic PDEs have either all positive or all negative eigenvalues (except one that is zero).\n";
        if (exit_on_fail) {
            exit(EXIT_FAILURE);
        }
    }

    return max_neg_eig;
}


double RBFFD::computeHyperviscosityEigenvalues(DerType which, int k, double gamma, EigenvalueOutput* output) 
{

    std::vector<double*>& weights_r = this->weights[which];
  
    // Use a std::set because it auto sorts as we insert
    std::set<unsigned int>& b_indices = grid_ref.getSortedBoundarySet();
    std::set<unsigned int>& i_indices = grid_ref.getSortedInteriorSet(); 
    unsigned int nb_bnd = b_indices.size();
    unsigned int nb_int = i_indices.size();
    int nb_centers = grid_ref.getNodeListSize();
    int nb_stencils = grid_ref.getStencilsSize();

    // compute eigenvalues of derivative operator
    unsigned int sz = weights_r.size();

#if 0
    printf("sz= %lu\n", sz);
    printf("weights.size= %d\n", (int) weights_r.size());
#endif 

    printf("Generating matrix of weights for interior nodes ONLY\n");
    arma::mat eigmat(sz, sz);
    eigmat.zeros();

    // Boundary nodes first with diagonal 1's.
    std::set<unsigned int>::iterator it; 
    int i = 0;
    for (it = b_indices.begin(); it != b_indices.end(); it++, i++) {
        eigmat(*it,*it) = 1.0;
    }
   
    // We put our interior nodes after the boundary nodes in matrix
    for (it = i_indices.begin(); it != i_indices.end(); it++, i++) {
        double* w = weights_r[*it];
        StencilType& st = grid_ref.getStencil(*it);
        for (int j=0; j < st.size(); j++) {
            eigmat(*it,st[j])  = w[j];
            // 	printf ("eigmat(%d, st[%d]) = w[%d] = %g\n", i, j, j, eigmat(i,st[j]));
        }
    }

    printf("sz= %u, nb_bnd= %u\n", sz, nb_bnd);

    arma::cx_colvec eigval;
    arma::cx_mat eigvec;
    printf("Computing Eigenvalues of Laplacian Operator on Interior Nodes\n");
    eig_gen(eigval, eigvec, eigmat);
    //eigval.print("eigval");

    int pos_count=0;
    int neg_count=0;
    int zero_count=0;

    double max_pos_eig= fabs(real(eigval(0)));
    double min_pos_eig = fabs(real(eigval(0)));

    double max_neg_eig = fabs(real(eigval(0)));
    double min_neg_eig = fabs(real(eigval(0)));

    // Compute number of unstable modes
    // Also compute the largest and smallest (in magnitude) eigenvalue
    //for (int i=0; i < (sz-nb_bnd); i++) {
    for (int i=0; i < sz; i++) {
        double e = real(eigval(i));
        if (e > 1.e-8) {
            pos_count++;
            if (fabs(e) > max_pos_eig) {
                max_pos_eig = fabs(e);
            }
            if (fabs(e) < min_pos_eig) {
                min_pos_eig = fabs(e);
            }
        } else if (e < -1.e-8) {
            neg_count++;
            if (fabs(e) > max_neg_eig) {
                max_neg_eig = fabs(e);
            }
            if (fabs(e) < min_neg_eig) {
                min_neg_eig = fabs(e);
            }
        } else {
            zero_count++;
        }
    }
#if 0
    count -= nb_bnd; // since we know at least nb_bnd eigenvalues are are (1+0i)

#endif 
    printf("\n[RBFFD] ****** Number positive eigenvalues: %d *******\n\n", pos_count);
    printf("\n[RBFFD] ****** Number negative eigenvalues: %d *******\n\n", neg_count);
    printf("\n[RBFFD] ****** Number zero (-1e-8 <= eval <= 1e-8) eigenvalues: %d *******\n\n", zero_count);

    printf("min abs(real(eig)) (among negative reals): %f\n", min_neg_eig);
    printf("max abs(real(eig)) (among negative reals): %f\n", max_neg_eig);

    printf("min abs(real(eig)) (among positive reals): %f\n", min_pos_eig);
    printf("max abs(real(eig)) (among positive reals): %f\n", max_pos_eig);

    if (output) {
        output->max_pos_eig = max_pos_eig; 
        output->min_pos_eig = min_pos_eig; 
        output->max_neg_eig = max_neg_eig; 
        output->min_neg_eig = min_neg_eig; 
        output->nb_positive = pos_count; 
        output->nb_negative = neg_count; 
        output->nb_zero     = zero_count; 
    }

    eigenvalues_computed = true;
    cachedEigenvalues.max_pos_eig = max_pos_eig; 
    cachedEigenvalues.min_pos_eig = min_pos_eig;
    cachedEigenvalues.max_neg_eig = max_neg_eig; 
    cachedEigenvalues.min_neg_eig = min_neg_eig;
    cachedEigenvalues.nb_positive = pos_count; 
    cachedEigenvalues.nb_negative = neg_count; 
    cachedEigenvalues.nb_zero     = zero_count; 

    return max_neg_eig;
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

void RBFFD::distanceMatrix(std::vector<NodeType>& rbf_centers, StencilType& stencil, int dim_num, arma::mat& ar, double h) {
    // We assume that all stencils are centered around the first node in the Stencil node list.
    unsigned int irbf = stencil[0]; 
    Vec3& c = rbf_centers[irbf];
    unsigned int n = stencil.size();
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
        printf("inconsistency with global rbf map (stencil should contain center: %u)\n", irbf);
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
        // TODO: test when its a uniq var_epsilon by stencil, not by RBF *WITHIN* a stencil
        // FIXME: allow var_epsilon[stencil[j]] (we need to pass stencil info for ghost nodes if this is to be allowed)
#if SCALE_BY_H
        double eps = var_epsilon[stencil[0]] * h; 
#else 
        double eps = var_epsilon[stencil[0]]; 
#endif 

        IRBF rbf(eps, dim_num);
        // rbf centered at xj
        Vec3& xjv = rbf_centers[stencil[j]];
        for (int i=0; i < n; i++) {
            // by row
            Vec3 xiv = rbf_centers[stencil[i]]; 
#if SCALE_BY_H
            // Center is on right:
            // NOTE we scale this by h so we are computing weights of the
            // stencil spanning the unit disk
            Vec3 diff = (xiv - xjv) * (1./h); 
#else 
            Vec3 diff = (xiv - xjv);
#endif 
    //        std::cout << xiv << "\t" << xjv << "\t" << diff <<  std::endl;
            ar(j,i) = rbf.eval(diff);
        }
    }
//    std::cout << "H = " << h << std::endl;
//       ar.print("INSIDE DMATRIX");

}

//--------------------------------------------------------------------

int RBFFD::loadFromFile(DerType which, std::string filename) {
    int ret_code; 
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i, *I, *J;
    double *val;    
    int err; 

    if ((f = fopen(filename.c_str(), "r")) == NULL)
    {
        std::cout << "File not found: " << filename << std::endl;
        return 1; 
    }
    if (mm_read_banner(f, &matcode) != 0) 
    {
        std::cout << "Could not process MatrixMarket Banner in " << filename << std::endl;
        return 2; 
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
    {
        std::cout << "Error! failed to parse file contents" << std::endl;
        return 4; 
    }

    std::vector<StencilType>& stencil = grid_ref.getStencils(); 
    std::vector<double*>* deriv_choice_ptr = &(weights[which]); 
    // number of non-zeros (should be close to max_st_size*num_stencils)
    unsigned int expected_nz = 0;
    // Num stencils (x_weights.size())
    const unsigned int expected_M = (*deriv_choice_ptr).size();


    for (unsigned int i = 0; i < stencil.size(); i++) {
        expected_nz += stencil[i].size();
    }
    fprintf(stdout, "Attempting to read %u weights (%u, %u) from %s\n", expected_nz, M, N, filename.c_str()); 

    if (M != expected_M) {
        std::cout << "Error! not enough stencils in the file" << std::endl;
        return 5;
    }

    if (nz != expected_nz) {
        std::cout << "Error! not enough weights in the file" << std::endl;
        return 5;
    }

    /* reseve memory for matrices */

    I = new int[nz]; 
    J = new int[nz];
    val = new double[nz];

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        //        fscanf(f, "%d %d %24le\n", &I[i], &J[i], &val[i]);
        fscanf(f, "%d %d %le\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    //if (f !=stdin) 
    fclose(f);

    // Convert to our weights: 
    for (unsigned int irbf = 0; irbf < N; irbf++) {
        this->weights[which][irbf] = new double[M]; 
    }
    unsigned int j = 0; 
    unsigned int local_i = 0; 
    for (i = 0; i < nz; i++) {
        if (local_i != I[i]) {
            j = 0; 
            local_i = I[i]; 
        }
        this->weights[which][local_i][j] = val[i];
        //std::cout << "Placing: weights[" << which << "][ " << local_i << "(" << I[i] << ") ][ " << j << "(" << J[i] << ") ] = " << val[i] << std::endl;;
        j++;
    }

    delete [] I; 
    delete [] J; 
    delete [] val;

    // let routines like RBFFD_CL::updateWeights(..) know that its time for
    // them to do some work
    weightsModified = true;
    return 0;

}
//--------------------------------------------------------------------

int RBFFD::loadAllWeightsFromFile() {
    
    int err = 0;
    for (int i = 0; i < NUM_DERIV_TYPES; i++) {
        err += this->loadFromFile((DerType)i);
    }
    return err;
}

//--------------------------------------------------------------------

void RBFFD::writeAllWeightsToFile() {
    
    for (int i = 0; i < NUM_DERIV_TYPES; i++) {
        this->writeToFile((DerType)i);
    }

}


//--------------------------------------------------------------------

void RBFFD::writeToFile(DerType which, std::string filename) {

    // number of non-zeros (should be close to max_st_size*num_stencils)
    unsigned int nz = 0;

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
    for (unsigned int i = 0; i < stencil.size(); i++) {
        nz += stencil[i].size();
    }
    fprintf(stdout, "Writing %u weights to %s\n", nz, filename.c_str()); 

    // Num stencils (x_weights.size())
    const unsigned int M = (*deriv_choice_ptr).size();
    // We have a square MxN matrix
    const unsigned int N = M;

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
    for (unsigned int i = 0; i < stencil.size(); i++) {
        for (unsigned int j = 0; j < stencil[i].size(); j++) {
            // Add 1 because matrix market assumes we index 1:N instead of 0:N-1
            fprintf(f, "%d %d %24.16le\n", stencil[i][0]+1, stencil[i][j]+1, (*deriv_choice_ptr)[i][j]); 
            // fprintf(f, "%d %d %24.16le\n", stencil[i][0]+1, stencil[i][j]+1, (*deriv_choice_ptr)[i][j]); 
        }
    }

    fclose(f);
}

//----------------------------------------------------------------------
void RBFFD::setVariableEpsilon(std::vector<double>& avg_radius_, double alpha, double beta) {
    modified = 1;
    use_var_eps = 1;
    std::cout << "DERIVATIVE:: SET VARIABLE EPSILON = " << alpha << "/(avg_st_radius^" << beta << ")" << std::endl;
    std::vector<double>& avg_stencil_radius = avg_radius_;

    unsigned int nb_stencils = grid_ref.getStencilsSize(); 
    unsigned int nb_radii = avg_stencil_radius.size(); 

    std::cout << "NB_STENCILS: " << nb_stencils << ", NB_RADII: " << nb_radii << std::endl;
// Types: 0 : Mine. 1 : Sarler2006 
#define VARIABLE_EPS_TYPE 1
    var_epsilon.resize(nb_stencils);
    for (int i=0; i < nb_stencils; i++) {
        StencilType& stencil = grid_ref.getStencil(i);
#if 0
        var_epsilon[i] = alpha / std::pow(avg_stencil_radius[i], beta);
        printf("avg_stencil_radius(%d) = %10.10f\n", i , avg_stencil_radius[i]); 
        printf("var_epsilon(%d) = %10.10f\n", i, var_epsilon[i]);
#endif                
        // var_epsilon[i] = alpha / std::pow(avg_stencil_radius[i], beta);
        //            var_epsilon[i] = alpha * avg_stencil_radius[i] / sqrt(beta);

        // Hardy 1972: 
        //var_epsilon[stencil[i][0]] = 1.0 / (0.815 * avg_stencil_radius[stencil[i][0]]);

        // Franke 1982: 
        // TODO: franke actually had max_stencil_radius in denom
        //var_epsilon[i] = 0.8 * sqrt(stencils[i].size()) / max_stencil_radius[i] ;

        // Note: for 24x24, alpha = 0.04. For 64x64, alpha = 0.05; for 1000x1000, alpha = 0.07
        // we use stencils[i][0] to get the index for the stencil center and its corresponding "avg_radius" 
        //std::cout << "var_epsilon[" << i << "] = " << alpha << " * sqrt( " << stencils[i].size() << " / avg_radius_[ " << stencils[i][0] << " ] " << std::endl; 
        // the indx on var_epsilon should be linear 0->stencils.size(), but just in case we have random access based on stencil center index
#if (VARIABLE_EPS_TYPE == 0)
        var_epsilon[stencil[0]] = (alpha * sqrt(stencil.size())) / avg_radius_[stencil[0]] ;
#else 
        // alpha = c. Sarler2006 (Meshfree Explicit Local Radial Basis Function ...) 
        // cr = c * max(r_i) where r_i is the distance to each node in the stencil
        // Of course we have eps = 1/(cr)
        var_epsilon[stencil[0]] = 1./ ( alpha * grid_ref.getMaxStencilRadius(stencil[0]) ); 
#endif 
        //printf("var_epsilon(%d) = %f (%f, %f, %f)\n", i, var_epsilon[i], alpha, sqrt(stencils[i].size()), avg_stencil_radius[i]);
    }
    std::stringstream ss(std::stringstream::out); 
    ss << "variable_epsilon_" << alpha << "_" << beta;
    eps_string = ss.str();
}



void RBFFD::computeWeightsForStencil_ContourSVD(DerType which, int st_indx) {
    //----------------------------------------------------------------------
    //void Derivative::computeWeightsSVD(vector<Vec3>& rbf_centers, StencilType& stencil, int irbf, const char* choice)
    {
        unsigned int irbf = st_indx;
        StencilType& stencil = grid_ref.getStencil(st_indx); 

        std::vector<NodeType>& rbf_centers = grid_ref.getNodeList();

        //printf("Computing Weights for Stencil %d Using ContourSVD\n", st_indx);

        int st_center = -1;
        // which stencil point is irbf
        for (int i=0; i < stencil.size(); i++) {
            if (st_indx == stencil[i]) {
                st_center = i;
                break;
            }
        }
        if (st_center == -1) {
            printf("inconsistency with global rbf map (stencil should contain center: %d)\n", st_indx);
            exit(0);
        }

        //printf("st_center= %d\n", st_center);
        if (st_center != 0) {
            printf("st_center should be the first element of the stencil!\n");
            exit(0);
        }

        // estimate radius for contour-svd method
        // distance matrix: each entry is the square of the internode distance
        arma::mat xd(stencil.size(), 3);
        for (int i=0; i < stencil.size(); i++) {
            Vec3& rc = rbf_centers[stencil[i]];
            xd(i,0) = rc[0];
            xd(i,1) = rc[1];
            xd(i,2) = rc[2];
        }

#if 0
        vector<double> epsv(nb_rbfs);
        for (int i=0; i < nb_rbfs; i++) {
            var_eps[i] = 1. / avg_stencil_radius[i];
            //printf("avg rad(%d) = %f\n", i, avg_stencil_radius[i]);
        }
        double mm = minimum(avg_stencil_radius);
        printf("min avg_stencil_radius= %f\n", mm);
        //exit(0);
#endif

#if 0
        var_eps[irbf] = 1.; // works
        var_eps[irbf] *= .07; // TEMP Does not work
#endif

#if 0
        double rad = 1.1;              // rad should also be proportional to (1/avg_stencil_radius)
        double eps = 1.0; // * var_eps[irbf]; // variable epsilon (for 300 pts)
        //double eps = 1.1; // * var_eps[irbf]; // variable epsilon (for 300 pts)
        //double eps = 1.5 * var_eps[irbf]; // variable epsilon (for 1000 pts)
#else 
        // FIXME: Gordons comments above indicate a desire to vary the rad with avg_stencil_radii. 
        //
        // Rad is the Radius for ContourSVD method. (???) No paper on ContourSVD yet.
        // For both options below, a grid 41x41 over [-10,10] x [-10,10]
        // Up to 933 for st=13 and eps = 0.01
        // double rad = 1. / ( 0.185 * grid_ref.getMaxStencilRadius(st_indx));  
        //      -- better: --
        // up to 1063 for st=13 and eps=0.01 (eps=0.1 gives 1061)
        //        double rad = 1. / ( 0.100 * grid_ref.getStencilRadius(st_indx));  
        //NOTE: the 1063 above is reduced to 178 when grid is 41x41 [-1,1]x[-1,1]
        //NOTE: the 1063 above is reduced to 1059 when grid is 41x41 [-100,100]x[-100,100]
        // That leads me to believe out [-1,1]x[-1,1] domain may have been
        // causing things to die a lot easier. perhaps related to conditioning?
        // Stencil size 9 gives 850 iters (eps=0.01)
        // Stencil size 28 gives 800 iters (eps=0.01); 799 iters (eps=0.1)


        //EFB060111
        // By choosing rad = 1 we are NOT normalizing or scaling any of the inputs to ContourSVD and/or the output coefs
#if SCALE_BY_H
        double rad = 1./grid_ref.getMaxStencilRadius(st_indx);
      //  double rad = 1.;
      //  double h = grid_ref.getMaxStencilRadius(st_indx);
        double h = 1.;
#else 
        double rad = 1.;
        double h = 1.;
#endif 
        double eps = var_epsilon[st_indx]; 
#endif 
        //printf("var_eps[%d]= %f\n", irbf, var_eps[irbf]);
        //cout << "CHOICE: " << choice << endl;

        // NOTE: this was 3 and caused the original heat problem to fail. WHY? 
        // Answer: the laplacian is not dimensionless. It increases as the
        // dimension changes. Not only do we need to double check out laplacian
        // in the RBFs, but also in the ExactSolution 
        IRBF rbf(eps, dim_num);

        std::string choice = derTypeStr[which];  
        
        // This is a 
        Stencils sten(&rbf, rad, h, eps, &xd, choice.c_str());
        //arma::mat rd2 = sten.computeDistMatrix2(xd,xd);

        int N = 128; // Why can't I increase N?
        arma::mat weights_new = sten.execute(N);
        //weights_new * rad*rad;

        // There should be a better way of doing this
        unsigned int n = weights_new.n_rows;
        //printf("choice= %s, outsize: %d\n", choice, n);
        unsigned int np = 0;

        if (this->weights[which][irbf] == NULL) {
            this->weights[which][irbf] = new double[n+np];
        }
        for (int j = 0; j < n+np; j++) {
            this->weights[which][irbf][j] = weights_new[j];
        }

#if 0
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
    }
    ;
}


//----------------------------------------------------------------------------
std::string RBFFD::getFileDetailString(DerType which) {
    std::stringstream ss(std::stringstream::out); 
    ss << derTypeStr[which] << "_weights_" << weightTypeStr[weightMethod] << "_" << this->getEpsString() << "_" << grid_ref.getStencilDetailString() << "_" << dim_num << "d" << "_" << grid_ref.getFileDetailString();  
    return ss.str();
}

//----------------------------------------------------------------------------
std::string RBFFD::getFilename(DerType which, std::string base_filename) {
    std::stringstream ss(std::stringstream::out);
    ss << this->getFileDetailString(which) << ".mtx";
    std::string filename = ss.str();
    return filename;
}

//----------------------------------------------------------------------------
std::string RBFFD::getFilename(DerType which) {
    std::stringstream ss(std::stringstream::out); 
    ss << "weights_" << this->className();
    return this->getFilename(which, ss.str()); 
}

