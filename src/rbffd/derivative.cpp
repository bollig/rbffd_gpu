#include <stdlib.h>
#include <math.h>
#include "derivative.h"
#include "contour_svd.h"
#include <armadillo>
#include "stencils.h"

using namespace std;
using namespace arma;

//----------------------------------------------------------------------
//Derivative::Derivative(int nb_rbfs) : arr(1,1,1), maxint(maxint = (1 << 31) - 1)
Derivative::Derivative(vector<Vec3>& rbf_centers_, vector<vector<int> >& stencil_, int nb_bnd_) :
        rbf_centers(rbf_centers_), stencil(stencil_), maxint((1 << 31) - 1.), debug_mode(0)
{
    this->nb_bnd = nb_bnd_;
    this->nb_rbfs = rbf_centers.size();

    x_weights.resize(nb_rbfs);
    y_weights.resize(nb_rbfs);
    z_weights.resize(nb_rbfs);
    r_weights.resize(nb_rbfs);
    lapl_weights.resize(nb_rbfs);
    rbfs.resize(nb_rbfs);

    // derivative depends strongly on epsilon!
    // there must be a range of epsilon for which the derivative is approx. constant!
    epsilon = 0.1; // laplacian returns zero (related to SVD perhaps)
    //epsilon = 3.; // Pretty accurate
    epsilon = 2.; // Pretty accurate
    //epsilon = 1.;
    //epsilon = 0.5; // Pretty accurate

    printf("nb_rbfs= %d\n", nb_rbfs); // ok
}
//----------------------------------------------------------------------
Derivative::Derivative(ProjectSettings* settings, vector<Vec3>& rbf_centers_, vector<vector<int> >& stencil_, int nb_bnd_) :
        rbf_centers(rbf_centers_), stencil(stencil_), maxint((1 << 31) - 1.)
{
    this->nb_bnd = nb_bnd_;
    this->nb_rbfs = rbf_centers.size();

    x_weights.resize(nb_rbfs);
    y_weights.resize(nb_rbfs);
    z_weights.resize(nb_rbfs);
    r_weights.resize(nb_rbfs);
    lapl_weights.resize(nb_rbfs);
    rbfs.resize(nb_rbfs);

    // derivative depends strongly on epsilon!
    // there must be a range of epsilon for which the derivative is approx. constant!
    epsilon = 0.1; // laplacian returns zero (related to SVD perhaps)
    //epsilon = 3.; // Pretty accurate
    epsilon = 2.; // Pretty accurate
    //epsilon = 1.;
    //epsilon = 0.5; // Pretty accurate

    printf("nb_rbfs= %d\n", nb_rbfs); // ok

    debug_mode = settings->GetSettingAs<int>("DEBUG_MODE");
}
//----------------------------------------------------------------------
Derivative::~Derivative()
{
    for (int i = 0; i < x_weights.size(); i++) {
        if (x_weights[i] != NULL) {
            delete [] x_weights[i];
        }
    }
    for (int i = 0; i < y_weights.size(); i++) {
        if (y_weights[i] != NULL) {
            delete [] y_weights[i];
        }
    }
    for (int i = 0; i < z_weights.size(); i++) {
        if (z_weights[i] != NULL) {
            delete [] z_weights[i];
        }
    }
    for (int i = 0; i < lapl_weights.size(); i++) {
        if (lapl_weights[i] != NULL) {
            delete [] lapl_weights[i];
        }
    }
    for (int i = 0; i < r_weights.size(); i++) {
        if (r_weights[i] != NULL) {
            delete [] r_weights[i];
        }
    }
    for (int i = 0; i < rbfs.size(); i++) {
        for (int j = 0; j < rbfs[i].size(); j++) {
            if (rbfs[i][j] != NULL) {
                delete [] rbfs[i][j];
            }
        }
    }
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Solve for x, Ax = b
// Assume that arr is non-singular
// Assume that arr is symmetric
// Only need to store 1/2 the array
// Assume that dimensions of arr, b and x are compatible
// A(m,m), x(m), b(m)
// comes from program on bones that should work properly
Derivative::AF& Derivative::cholesky_cpu(AF& arr)
{

    const int* dims = arr.getDims();
    int m = dims[0];
    int n = dims[1];

    //	printf("n= %d\n", n); exit(0);

    if (n != m) {
        printf("arr not square\n");
        exit(0);
    }

    printf("matrix size: %d\n", n);

#if 0
    for (int j=0; j < n; j++) {
	for (int i=0; i < n; i++) {
            //printf("arr(%d,%d)= %f\n", i,j, arr(i,j));
	}}
    printf("\n");
#endif

    // Cholesky decomposition (save on memory)
    // A = L^T L

    AF* L = new AF(m, m);
    L->setTo(0.);
    AF& Lr = *L;


    int nb_runs = 1;
    //chol_cpu.start();
    for (int nn = 0; nn < nb_runs; nn++) {

	for (int i=0; i < n; i++) {
            double d = rowDot(Lr, i, Lr, i, i);
            Lr(i,i) = sqrt(arr(i,i)-d);
            for (int j=i+1; j < n; j++) {
                double dot = rowDot(Lr, i, Lr, j, i);
                Lr(j,i) = (arr(j,i) - dot) / Lr(i,i);
            }
	}


#if 0
	// Original matrix should be Lr*Lr^T
	// Perform multiplication

	AF res(n, n);
	res.setTo(0.);

	for (int j=0; j < n; j++) {
            for (int i=0; i < n; i++) {
		for (int k=i+1; k < n; k++) {
		}
		for (int k=0; k < n; k++) {
                    // Lr^T(i,j) = Lr(j,i)
                    res(i,j) += Lr(i,k)*Lr(j,k);
		}
		printf("(res-arr)(%d,%d)= %f\n", i,j, res(i,j)-arr(i,j));
            }}
#endif

    }
    //chol_cpu.end();

#if 0
    for (int j=0; j < n; j++) {
	for (int i=0; i < n; i++) {
            printf("CPU: L(%d,%d) = %f\n", i, j, Lr(i,j));
	}}
#endif


    return Lr;
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
Derivative::AF& Derivative::cholesky(AF& arr)
{
    const int* dims = arr.getDims();
    int m = dims[0];
    int n = dims[1];

    if (n != m) {
        printf("arr not square\n");
        exit(0);
    }

    printf("\n");

    for (int j=0; j < n; j++) {
	for (int i=0; i < n; i++) {
            //printf("arr(%d,%d)= %f\n", i,j, arr(i,j));
	}}
    printf("\n");

    // Cholesky decomposition (save on memory)
    // A = L^T L

    AF* L = new AF(m, m);
    L->setTo(0.);
    AF& Lr = *L;

    for (int i=0; i < n; i++) {
        double d = rowDot(Lr, i, Lr, i, i);
        Lr(i,i) = sqrt(arr(i,i)-d);
        // What if Lr(i,i) == 0? Matrix not invertible? Appears to be happening
        printf("arr(i,i),d= %21.14g, %21.14g\n", arr(i,i), d);
        printf("i= %d, Lr(i,i)= %f\n", i, Lr(i,i));
        for (int j=i+1; j < n; j++) {
            double dot = rowDot(Lr, i, Lr, j, i);
            Lr(j,i) = (arr(j,i) - dot) / Lr(i,i);
        }
    }

#if 0
    for (int j=0; j < n; j++) {
	for (int i=0; i < n; i++) {
            printf("L(%d,%d) = %f\n", i, j, Lr(i,j));
	}}

    // Original matrix should be Lr*Lr^T
    // Perform multiplication

    AF res(n, n);
    res.setTo(0.);

    for (int j=0; j < n; j++) {
	for (int i=0; i < n; i++) {
            for (int k=i+1; k < n; k++) {
            }
            //for (int k=0; k <= i; k++) {
            for (int k=0; k < n; k++) {
                // Lr^T(i,j) = Lr(j,i)
                res(i,j) += Lr(i,k)*Lr(j,k);
            }
            printf("(res-arr)(%d,%d)= %f\n", i,j, res(i,j)-arr(i,j));
	}}
#endif

    return Lr;
}
//----------------------------------------------------------------------
double Derivative::rowDot(AF& a, int row_a, AF& b, int row_b, int ix)
{
    double dot = 0.;
    for (int i=0; i < ix; i++) {
        dot += a(row_a, i)*b(row_b, i);
    }

    return dot;
}

#if 0
//----------------------------------------------------------------------
// Generate a distance matrix and compute the SVD of it.
int Derivative::distanceMatrixSVD(vector<Vec3>& rbf_centers, vector<int>& stencil,int irbf, int nb_eig)
{

    int n = stencil.size();
    arma::mat ar(n+4, n+4);
    ar.zeros(n+4,n+4);
    this->distanceMatrix(rbf_centers, stencil, irbf, ar.memptr(), 3);

    // Fill the polynomial part
    for (int i=0; i < n; i++) {
        ar(n, i) = 1.0;
        ar(n+1, i) = rbf_centers[stencil[i]].x();
        ar(n+2, i) = rbf_centers[stencil[i]].y();
        ar(n+3, i) = rbf_centers[stencil[i]].z();
        ar(i, n) = 1.0;
        ar(i, n+1) = rbf_centers[stencil[i]].x();
        ar(i, n+2) = rbf_centers[stencil[i]].y();
        ar(i, n+3) = rbf_centers[stencil[i]].z();
    }

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

    n = n + 1; // deriv(constant) = 0
    n = n + 3; // deriv(linear function) = constant

    // mat NewMat(memspace, rowdim, coldim, reuseMemSpace?)
    mat U;
    mat V;
    mat s;

    // Will there be memory leak? If I use svd multiple times with different size arrays,
    // how is memory allocated or reallocated or released?

    svd(U,s,V,ar);

    printf("cond number(%d): %g\n", irbf, s(0)/s(n-1));

    double s1 = s(0);
    nb_eig=0;
    double svd_tol = 1.e-7;

    for (int i=0; i < n; i++) {
        //printf("s(i)= %21.14f\n", s(i));
        if ((s(i) / s1) < svd_tol) {
            break;
        }
        nb_eig++;
    }

    // construct pseudo matrix (\sum_i u_i s v_i^{T})
    //printf("nb_eig= %d, n= %d\n", nb_eig, n);

    for (int i=nb_eig; i < n; i++) {
        s(i) = 0.0;
    }

    return st_center;
}
#endif

//----------------------------------------------------------------------
void Derivative::distanceMatrix(vector<Vec3>& rbf_centers, vector<int>& stencil,int irbf, double* distance_matrix, int nrows, int ncols, int dim_num)
{
    Vec3& c = rbf_centers[irbf];
    int n = stencil.size();
    //printf("stencil size= %d\n", n);

    //printf("stencil size(%d): n= %d\n", irbf, n);
    //printf("n= %d\n", n);

    //arma::mat &ar = *distance_matrix;
    // mat NewMat(memspace, rowdim, coldim, reuseMemSpace?)
    arma::mat ar(distance_matrix,nrows,ncols,false);


    IRBF rbf(epsilon, dim_num);

    //mat ar(n,n);
    // Derivative of a constant should be zero
    // Derivative of a linear function should be constant
    //
    if ((ar.n_cols > n)||(ar.n_rows > n)) {
        ar.submat(0,0,n-1,n-1) = zeros<mat>(n,n);
    } else {
        ar.zeros(n,n);
    }

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

    // stencil includes the point itself
    for (int i=0; i < n; i++) {
        Vec3& xiv = rbf_centers[stencil[i]];
        for (int j=0; j < n; j++) {
            // rbf centered at xi
            Vec3& xjv = rbf_centers[stencil[j]];
            ar(i,j) = rbf(xiv, xjv);
        }}
 //   ar.print("INSIDE DMATRIX");
}

//----------------------------------------------------------------------
void Derivative::computeWeightsSVD(vector<Vec3>& rbf_centers, vector<int>&
                                   stencil, int irbf, const char* choice)
{
    printf("Computing Weights for Stencil %d\n", irbf);

    int st_center = -1;

    // which stencil point is irbf
    for (int i=0; i < stencil.size(); i++) {
        if (irbf == stencil[i]) {
            st_center = i;
            break;
        }
    }
    if (st_center == -1) {
        printf("inconsistency with global rbf map (stencil should contain center: %d)\n", irbf);
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
    double rad = 1.1;              // rad should also be proportional to (1/avg_stencil_radius)
    double eps = 1.1; // * var_eps[irbf]; // variable epsilon (for 300 pts)
    //double eps = 1.5 * var_eps[irbf]; // variable epsilon (for 1000 pts)

    //printf("var_eps[%d]= %f\n", irbf, var_eps[irbf]);

    IRBF rbf(eps, 3);
    Stencils sten(&rbf, rad, eps, &xd, choice);
    //arma::mat rd2 = sten.computeDistMatrix2(xd,xd);

    int N = 128; // Why can't I increase N?
    arma::mat fd_coeffs = sten.execute(N);

#if DEBUG
    char label[256];
    sprintf(label, "%s Derivative Coefficients =", choice);
    fd_coeffs.print(label);
#endif
    //exit(0);

    // There should be a better way of doing this

    //printf("choice= %s\n", choice);

    if (strcmp(choice, "lapl") == 0) {
        if (lapl_weights[irbf] == NULL) {
            lapl_weights[irbf] = new double[stencil.size()];
        }
        for (int j = 0; j < stencil.size(); j++) {
            lapl_weights[irbf][j] = fd_coeffs(j);
        }
        //lapl_weights[irbf] = fd_coeffs;
    } else if (strcmp(choice, "x") == 0) {
        //printf("irbf= %d\n", irbf);
        //x_weights[irbf] = fd_coeffs;
        if (x_weights[irbf] == NULL) {
            x_weights[irbf] = new double[stencil.size()];
        }
        for (int j = 0; j < stencil.size(); j++) {
            x_weights[irbf][j] = fd_coeffs(j);
        }
    } else if (strcmp(choice, "y") == 0) {
        //y_weights[irbf] = fd_coeffs;
        if (y_weights[irbf] == NULL) {
            y_weights[irbf] = new double[stencil.size()];
        }
        for (int j = 0; j < stencil.size(); j++) {
            y_weights[irbf][j] = fd_coeffs(j);
        }
    } else if (strcmp(choice, "z") == 0) {
        //z_weights[irbf] = fd_coeffs;
        if (z_weights[irbf] == NULL) {
            z_weights[irbf] = new double[stencil.size()];
        }
        for (int j = 0; j < stencil.size(); j++) {
            z_weights[irbf][j] = fd_coeffs(j);
        }
    } else {
        printf("not covered\n");
    }
    //printf("-----------------\n");
}
//----------------------------------------------------------------------
int Derivative::computeWeights(vector<Vec3>& rbf_centers, vector<int>& stencil, int irbf, int dim_num)
{
    //IRBF rbf(epsilon, dim_num);
    int n = stencil.size();
    int np = 1+dim_num; // +3 for the x,y,z monomials

    // +4 to get the monomial parts

    arma::rowvec bx, by, bz, blapl, br;

    bx.zeros(n+np); // extra lines to enforce d/dx(constant)=0, d/dx(linear fct) is constant
    by.zeros(n+np);
    bz.zeros(n+np);
    br.zeros(n+np);
    blapl.zeros(n+np);

    // stencil includes the point itself
    //printf("IRBF = %d, centered: %d\n", irbf, stencil[irbf]);

    Vec3& x0v = rbf_centers[stencil[0]];
    //printf("CenterSize:%d\tStencilSize: %d\n", rbf_centers.size(), stencil.size());
   // x0v.print("x0v = ");

    rbfs[irbf].resize(n);
    for (int j=0; j < n; j++) {

        rbfs[irbf][j] = new IRBF(epsilon, dim_num);

        // We want to evaluate every basis function. Each stencil node xjv has
        // its own basis function and we evaluate them with the distance to the
        // stencil center node x0v. This is B_j(||x0v - xjv||)
        // NOTE: when all basis functions are the same (eqn and support) then
        // B_c(||x0v - xjv||) = B_j(||x0v - xjv||).
        IRBF* rbf = rbfs[irbf][j];

        // printf("%d\t%d\n", j, stencil[j]);
        Vec3& xjv = rbf_centers[stencil[j]];
        //xjv.print("xjv = ");
        bx(j) = rbf->xderiv(xjv, x0v);
        by(j) = rbf->yderiv(xjv, x0v);
        bz(j) = rbf->zderiv(xjv, x0v);
        br(j) = rbf->radialderiv(xjv, x0v);
        blapl(j) = rbf->lapl_deriv(xjv, x0v);
       // printf("blapl(%d)= %f [lapl(%f, %f)]\n", j, blapl(j), (x0v-xjv).magnitude(), epsilon);
       // (x0v-xjv).print("lapl(x)");
    }

    if (np > 1) {
        // monomial terms. To get these, apply the operator to [ 1  x   y  z ]
        // d/dx = [ 0 1 0 0 ]
        bx(n)   = 0.0;
        bx(n+1) = 1.; // (if =1, then sum x(i) w(i) = 1 (DO NOT KNOW WHY)
        bx(n+2) = 0.0;
        if(dim_num == 3) {
            bx(n+3) = 0.0;
        }

        by(n)   = 0.0;
        by(n+1) = 0.;
        by(n+2) = 1.0;
        if(dim_num == 3) {
            by(n+3) = 0.0;
        }

        bz(n)   = 0.0;
        bz(n+1) = 0.0;
        bz(n+2) = 0.0;
        if(dim_num == 3) {
            bz(n+3) = 1.0;
        }

        // laplacian of constant and linear functions is zero
        blapl(n) = 0.0;
        blapl(n+1) = 0.0;
        blapl(n+2) = 0.0;
        if(dim_num == 3) {
            blapl(n+3) = 0.0;
        }

        // d/dr = ((x/r) d/dx + (y/r) d/dy + (z/r) d/dz)  => [ 0 x/r y/r z/r ]
        // NOTE: we apply d/dr to each: 1, x, y, z
        // (x/r) dx/dx + (y/r) dx/dy + (z/r) dx/dz = x/r
        double rp = x0v.magnitude();
        br(n)   = 0.0;
        br(n+1) = x0v.x() / rp;
        br(n+2) = x0v.y() / rp;
        if(np == 4) {
            br(n+3) = x0v.z() / rp;
        }
    }

    // Generate a distance matrix and find the SVD of it.
    // n+4 = 1 + dim(3) for x,y,z
    arma::mat d_matrix(n+np, n+np);
    d_matrix.zeros(n+np,n+np);
    this->distanceMatrix(rbf_centers, stencil, irbf, d_matrix.memptr(), d_matrix.n_rows, d_matrix.n_cols, dim_num);

    // Fill the polynomial part
    for (int i=0; i < n; i++) {
        d_matrix(n, i) = 1.0;
        d_matrix(i, n) = 1.0;
    }
    if (np > 1) {
        for (int i=0; i < n; i++) {
            d_matrix(n+1, i) = rbf_centers[stencil[i]].x();
            d_matrix(i, n+1) = rbf_centers[stencil[i]].x();

            d_matrix(n+2, i) = rbf_centers[stencil[i]].y();
            d_matrix(i, n+2) = rbf_centers[stencil[i]].y();

            if (np == 4) {
                d_matrix(n+3, i) = rbf_centers[stencil[i]].z();
                d_matrix(i, n+3) = rbf_centers[stencil[i]].z();
            }
        }
    }

   //  d_matrix.print("DISTANCE MATRIX: ");
  // d_matrix.print("DISTANCE MATRIX: ");
   // blapl.print("BLAPL: ");
    // TODO: find a backslash "/" equivalent to matlab's which allow us to avoid
    // computing and storing the full inverse matrix.
    //arma::mat Ainv = inv(d_matrix);

    // Remember: b*(A^-1) = (b*(A^-1))^T = (A^-T) * b^T = (A^-1) * b^T
    // because A is symmetric. Rather than compute full inverse we leverage
    // the solver for increased efficiency
    arma::mat weights_x = arma::solve(d_matrix, trans(bx)); //bx*Ainv;
    arma::mat weights_y = arma::solve(d_matrix, trans(by)); //by*Ainv;
    arma::mat weights_z = arma::solve(d_matrix, trans(bz)); //bz*Ainv;
    arma::mat weights_lapl = arma::solve(d_matrix, trans(blapl)); //blapl*Ainv;

//X
    if (this->x_weights[irbf] == NULL) {
        this->x_weights[irbf] = new double[n+np];
    }
    for (int j = 0; j < n+np; j++) {
        this->x_weights[irbf][j] = weights_x[j];
    }

// Y
    if (this->y_weights[irbf] == NULL) {
        this->y_weights[irbf] = new double[n+np];
    }
    for (int j = 0; j < n+np; j++) {
        this->y_weights[irbf][j] = weights_y[j];
    }

// Z
    if (this->z_weights[irbf] == NULL) {
        this->z_weights[irbf] = new double[n+np];
    }
    for (int j = 0; j < n+np; j++) {
        this->z_weights[irbf][j] = weights_z[j];
    }

    // Laplacian
    if (this->lapl_weights[irbf] == NULL) {
        this->lapl_weights[irbf] = new double[n+np];
    }
    double sum_nodes_only = 0.;
    double sum_nodes_and_monomials = 0.;
    for (int j = 0; j < n; j++) {
        this->lapl_weights[irbf][j] = weights_lapl[j];
        sum_nodes_only += weights_lapl[j];
    }
    sum_nodes_and_monomials = sum_nodes_only;
    for (int j = n; j < n+np; j++) {
        this->lapl_weights[irbf][j] = weights_lapl[j];
        sum_nodes_and_monomials += weights_lapl[j];
    }
    if (this->debug_mode) {
        cout << "(" << irbf << ") ";
        weights_lapl.print("lapl_weights");
        cout << "Sum of Stencil Node Weights: " << sum_nodes_only << endl;
        cout << "Sum of Node and Monomial Weights: " << sum_nodes_and_monomials << endl;
        if (sum_nodes_only > 1e-7) {
            cout << "WARNING! SUM OF WEIGHTS FOR LAPL NODES IS NOT ZERO: " << sum_nodes_only << endl;
            exit(EXIT_FAILURE);
        }
    }


#if 0
    double sum_r = 0.;
    double sum_l = 0.;
    for (int is = 0; is < n; is++) {
        sum_r += this->r_weights[irbf][is];
        sum_l += this->lapl_weights[irbf][is];
    }
#if 0
    if (sum_r > 1e-7) {
        cout << "WARNING! SUM OF WEIGHTS FOR R IS NOT ZERO: " << sum_r << endl;
        exit(EXIT_FAILURE);
    }
#endif
    if (sum_l > 1e-7) {
        cout << "WARNING! SUM OF WEIGHTS FOR LAPL IS NOT ZERO: " << sum_l << endl;
        exit(EXIT_FAILURE);
    }
#if 0
    //br.print("br = ");
   // d_matrix.print("Distance Matrix =");

    blapl.print("A_l");
    this->lapl_weights[irbf].print("Laplacian Weights = ");

    br.print("A_r");
    this->r_weights[irbf].print("R Weights = ");

    cout << "LAPL WEIGHT SUM: " << sum_l << endl;
    cout << "R WEIGHT SUM: " << sum_r << endl;

#endif

    bx.reset();
    by.reset();
    bz.reset();
    br.reset();
    blapl.reset();
#endif

    return stencil.size();
    //printf("DONE COMPUTING WEIGHTS: %d\n", irbf);
}
//----------------------------------------------------------------------
void Derivative::computeWeightsSVD_Direct(vector<Vec3>& rbf_centers, vector<int>& stencil, int irbf)
{
    int nb_eig = stencil.size();
    IRBF rbf(epsilon, 3);
    int n = stencil.size();

    arma::rowvec bx, by, bz, blapl, br;

    bx.zeros(n+4); // extra lines to enforce d/dx(constant)=0, d/dx(linear fct) is constant
    by.zeros(n+4);
    bz.zeros(n+4);
    blapl.zeros(n+4);

    // stencil includes the point itself
    //printf("IRBF = %d, centered: %d\n", irbf, stencil[irbf]);

    Vec3& x0v = rbf_centers[stencil[0]];
    //printf("CenterSize:%d\tStencilSize: %d\n", rbf_centers.size(), stencil.size());
    //x0v.print("x0v = ");
    for (int j=0; j < n; j++) {
        // printf("%d\t%d\n", j, stencil[j]);
        Vec3& xjv = rbf_centers[stencil[j]];
        //xjv.print("xjv = ");
        bx(j) = rbf.xderiv(x0v, xjv);
        by(j) = rbf.yderiv(x0v, xjv);
        bz(j) = rbf.zderiv(x0v, xjv);
        blapl(j) = rbf.lapl_deriv(x0v, xjv);
        //printf("blapl(%d)= %f\n", j, blapl(j));
    }

    // I am not convinced of these lines
    bx(n)   = 0.0;
    bx(n+1) = 1.;//1.0; (if =1, then sum x(i) w(i) = 1 (DO NOT KNOW WHY)
    bx(n+2) = 0.0;
    bx(n+3) = 0.0;

    by(n)   = 0.0;
    by(n+1) = 0.; // 1.0;
    by(n+2) = 1.0;
    by(n+3) = 0.0;

    bz(n)   = 0.0;
    bz(n+1) = 0.0;
    bz(n+2) = 0.0;
    bz(n+3) = 1.0;

    // laplacian of constant and linear functions is zero
    blapl(n) = 0.0;
    blapl(n+1) = 0.0;
    blapl(n+2) = 0.0;
    blapl(n+3) = 0.0;

    bx.print("bx = ");

    // Generate a distance matrix and find the SVD of it.

  //  n = n + 1; // deriv of constant is zero
  //  n = n + 3; // deriv of linear function is exact

    arma::mat ar(n+4, n+4);
    ar.zeros(n+4,n+4);
    this->distanceMatrix(rbf_centers, stencil, irbf, ar.memptr(), ar.n_rows, ar.n_cols, 3);

    // Fill the polynomial part
    for (int i=0; i < n; i++) {
        ar(n, i) = 1.0;
        ar(n+1, i) = rbf_centers[stencil[i]].x();
        ar(n+2, i) = rbf_centers[stencil[i]].y();
        ar(n+3, i) = rbf_centers[stencil[i]].z();
        ar(i, n) = 1.0;
        ar(i, n+1) = rbf_centers[stencil[i]].x();
        ar(i, n+2) = rbf_centers[stencil[i]].y();
        ar(i, n+3) = rbf_centers[stencil[i]].z();
    }

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

    n = n + 1; // deriv(constant) = 0
    n = n + 3; // deriv(linear function) = constant

    arma::mat U;
    arma::mat V;
    arma::colvec s;

    // Will there be memory leak? If I use svd multiple times with different size arrays,
    // how is memory allocated or reallocated or released?

    svd(U,s,V,ar);

    printf("cond number(%d): %g\n", irbf, s(0)/s(n-1));

    double s1 = s(0);
    nb_eig=0;
    double svd_tol = 1.e-7;

    for (int i=0; i < n; i++) {
        //printf("s(i)= %21.14f\n", s(i));
        if ((s(i) / s1) < svd_tol) {
            break;
        }
        nb_eig++;
    }

    // construct pseudo matrix (\sum_i u_i s v_i^{T})
    //printf("nb_eig= %d, n= %d\n", nb_eig, n);

    for (int i=nb_eig; i < n; i++) {
        s(i) = 0.0;
    }

    colvec wwx(n);
    colvec wwy(n);
    colvec wwz(n);
    colvec wwlapl(n);

    wwx.zeros();
    wwy.zeros();
    wwz.zeros();
    wwlapl.zeros();

    // TEMP WAY TO COMPUTE WEIGHTS
    for (int i=0; i < nb_eig; i++) {
        wwx = wwx +       dot(trans(U.col(i)),bx   )*V.col(i) / s(i);
        wwy = wwy +       dot(trans(U.col(i)),by   )*V.col(i) / s(i);
        wwz = wwz +       dot(trans(U.col(i)),bz   )*V.col(i) / s(i);
        wwlapl = wwlapl + dot(trans(U.col(i)),blapl)*V.col(i) / s(i);
    }

#if 0
    x_weights[irbf]    = wwx;
    y_weights[irbf]    = wwy;
    z_weights[irbf]	   = wwz;
    lapl_weights[irbf] = wwlapl;
#else
    if (this->x_weights[irbf] == NULL) {
        this->x_weights[irbf] = new double[stencil.size()];
    }
    for (int j = 0; j < stencil.size(); j++) {
        this->x_weights[irbf][j] = wwx[j];
    }

    if (this->y_weights[irbf] == NULL) {
        this->y_weights[irbf] = new double[stencil.size()];
    }
    for (int j = 0; j < stencil.size(); j++) {
        this->y_weights[irbf][j] = wwy[j];
    }

    if (this->z_weights[irbf] == NULL) {
        this->z_weights[irbf] = new double[stencil.size()];
    }
    for (int j = 0; j < stencil.size(); j++) {
        this->z_weights[irbf][j] = wwz[j];
    }

    if (this->lapl_weights[irbf] == NULL) {
        this->lapl_weights[irbf] = new double[stencil.size()];
    }
    for (int j = 0; j < stencil.size(); j++) {
        this->lapl_weights[irbf][j] = wwlapl[j];
    }
#endif

#if 0
    double wx_sum = 0.;
    double wy_sum = 0.;
    double wxx_sum = 0.;
    double wxy_sum = 0.;
    for (int i=0; i < (n-3); i++) {
        wx_sum += wwx[i];
        wy_sum += wwy[i];
        wxx_sum += wwx[i]*rbf_centers[st[i]].x();
        wxy_sum += wwy[i]*rbf_centers[st[i]].y();
    }
    printf("sum(weights, xy)= %f, %f\n", wx_sum, wy_sum);
    printf("sum(w*x, w*y)= %f, %f\n", wxx_sum, wxy_sum);
#endif

    bx.reset();
    by.reset();
    bz.reset();
    blapl.reset();
    printf("DONE COMPUTING WEIGHTS: %d\n", irbf);
}
//----------------------------------------------------------------------
Derivative::AF& Derivative::solve(AF& l, AF& b)
{
    const int* dims = l.getDims();
    int n = dims[0];

    AF* y = new AF(n,1);
    AF& yr = *y;

    /* b1 = l11 y1
	   b2 = l21*y1 + l22*y2;
	   b3 = l31*y1 + l32*y2 + l33*z3;
    */

    // solve  l y = b     // lower triangular

    for (int i=0; i < n; i++) {
        yr(i) = b(i);
        for (int j=0; j < i; j++) {
            yr(i) -= l(i,j)*yr(j);
        }
        yr(i) /= l(i,i);
    }

#if 0
    // CORRECT
    b.print("b");
    AF& xx = matmul(l, yr);
    xx.print("b = l*y");
    //exit(0);
#endif

    // solve l^T x = y    // upper triangular
    AF* x = new AF(n,1);
    AF& xr = *x;

#if 0
    yr.print("yr");
    l.print("l");
#endif

    /* y2 = r22*x2;
	   y1 = r12*x2 + r11*x1;
	   y0 = r02*x2 + r01*x1 + r00*x0;
	*/

    // use r(i,j) matrix, then sent r(i,j) <-- l(j,i)

    // STILL INCORRECT on 2x2

    for (int i=n-1; i >= 0; i--) {
        xr(i) = yr(i);
        for (int j=n-1; j > i; j--) {
            xr(i) -= l(j,i)*xr(j);
        }
        xr(i) = xr(i) / l(i,i);
    }

    //xr.print("xr");

    delete y;
    return xr;
}
//----------------------------------------------------------------------
Derivative::AF& Derivative::matmul(AF& arr, AF& x)
{
    const int* dims = arr.getDims();
    int n = dims[0];
    int m = x.getDims()[1];

    AF* b = new AF(n,m);
    AF& br = *b;
    br.setTo(0.);

    for (int k=0; k < m; k++) {
	for (int j=0; j < n; j++) {
            for (int i=0; i < n; i++) {
                br(j,k) += arr(j,i)*x(i,k);
            }
	}}

    return br;
}
//----------------------------------------------------------------------
// derivative array already allocated
void Derivative::computeDeriv(DerType which, vector<double>& u, vector<double>& deriv)
{
    // printf("computeDeriv, u= %d, deriv= %d\n", &u[0], &deriv[0]);
    //  printf("computeDeriv, u.size= %d, deriv.size= %d\n", u.size(), deriv.size());
    computeDeriv(which, &u[0], &deriv[0], deriv.size());
}
//----------------------------------------------------------------------
void Derivative::computeDeriv(DerType which, double* u, double* deriv, int npts)
{

    cout << "COMPUTING DERIVATIVE: ";
    vector<double*>* weights_p;

    switch(which) {
    case X:
        weights_p = &x_weights;
        //printf("weights_p= %d\n", weights_p);
        //exit(0);
        cout << "X" << endl;
        break;

    case Y:
        weights_p = &y_weights;
        cout << "Y" << endl;
        break;

    case Z:
        //vector<mat>& weights = z_weights;
        weights_p = &z_weights;
        cout << "Z" << endl;
        break;

    case LAPL:
        weights_p = &lapl_weights;
        cout << "LAPL" << endl;
        break;

    default:
        cout << "???" << endl;
        printf("Wrong derivative choice\n");
        printf("Should not happen\n");
        exit(EXIT_FAILURE);
    }

    vector<double*>& weights = *weights_p;

    double der;
#if 0
    //for (int i=0; i < 1600; i++) {
    for (int i=1325; i < 1330; i++) {
        vector<int>& v = stencil[i];
        printf("stencil[%d]\n", i);
        for (int s=0; s < v.size(); s++) {
            printf("%d ", v[s]);
        }
        printf("\n");
    }
    //exit(0);
#endif


// DEBUGGING STATEMENTS
#if 0
    printf("Weights size: %d\n", (int)weights.size());
    printf("Stencils size: %d\n", (int)stencil.size());
#endif
    for (int i=0; i < stencil.size(); i++) {
        double* w = weights[i];
        vector<int>& st = stencil[i];
        //printf("nb el in weight[%d]: %d, in stencil[%d]: %d\n", i, w.n_elem, i, st.size());
        //printf("i=%d, w[0] = %f\n", i, w[0]);
        der = 0.0;
        int n = st.size();
#if 0
        cout << "STENCIL " << i << "(" << n << "): " << endl;
#endif
        //printf("(%d) stencil size: %d\n", i, n);
        for (int s=0; s < n; s++) {
#if 0
            printf("\tw[%d]= %f * ", s, w[s]);
            printf("st[%d]= %d\t\t%f  * %f\n", s, st[s], w[s], u[st[s]]);
#endif
            der += w[s] * u[st[s]]; // SOMETHING WRONG!
        }
        deriv[i] = der;
    }
}
//----------------------------------------------------------------------
double Derivative::computeEig()
{
    vector<double*>* weights_p;
    weights_p = &lapl_weights;  // <<<< lapl_weights
    vector<double*>& weights = *weights_p;

#if 1
    // compute eigenvalues of derivative operator
    int sz = weights.size();
    printf("sz= %d\n", sz);
    printf("weights.size= %d\n", (int) weights.size());

    mat eigmat(sz, sz);
    eigmat.zeros();

    for (int i=nb_bnd; i < sz; i++) {
        double* w = weights[i];
        vector<int>& st = stencil[i];
        for (int j=0; j < st.size(); j++) {
            eigmat(i,st[j])  = w[j];
            //		printf ("eigmat(%d, st[%d]) = w[%d] = %g\n", i, j, j, eigmat(i,st[j]));
        }
    }
    for (int i=0; i < nb_bnd; i++) {
        eigmat(i,i) = 1.0;
    }

    cx_colvec eigval;
    cx_mat eigvec;
    eig_gen(eigval, eigvec, eigmat);
    //eigval.print("eigval");

    int count=0;
    double max_neg_eig = abs(real(eigval(0)));
    double min_neg_eig = abs(real(eigval(0)));

    printf("min abs(real(eig)) (among negative reals): %f\n", min_neg_eig);
    printf("max abs(real(eig)) (among negative reals): %f\n", max_neg_eig);
    printf("sz= %d, nb_bnd= %d\n", sz, nb_bnd);

    // Compute number of unstable modes
    // Also compute the largest and smallest (in magnitude) eigenvalue

    for (int i=0; i < (sz-nb_bnd); i++) {
        double e = real(eigval(i));
        if (e > 0.) {
            count++;
        } else {
            if (abs(e) > max_neg_eig) {
                max_neg_eig = abs(e);
            }
            if (abs(e) < min_neg_eig) {
                min_neg_eig = abs(e);
            }
        }
    }
    printf("nb unstable eigenvalues: %d\n", count);
    printf("min abs(real(eig)) (among negative reals): %f\n", min_neg_eig);
    printf("max abs(real(eig)) (among negative reals): %f\n", max_neg_eig);

    if (count > 0) {
        printf("unstable Laplace operator\n");
        exit(0);
    }
    //exit(0);
#endif

    return max_neg_eig;
}
//----------------------------------------------------------------------
double Derivative::minimum(vector<double>& vec)
{
    double min = 1.e10;

    for (int i=0; i < vec.size(); i++) {
        if (vec[i] < min) {
            min = vec[i];

        }
    }
    return min;
}
//----------------------------------------------------------------------
