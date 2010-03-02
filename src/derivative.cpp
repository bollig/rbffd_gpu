#include <stdlib.h>
#include <math.h>
#include "derivative.h"
#include "rbf_gaussian.h"
#include "rbf_mq.h"
#include "contour_svd.h"
#include <armadillo>
#include "stencils.h"

using namespace std;
using namespace arma;

//typedef RBF_Gaussian IRBF;
typedef RBF_MQ IRBF;

//----------------------------------------------------------------------
//Derivative::Derivative(int nb_rbfs) : arr(1,1,1), maxint(maxint = (1 << 31) - 1)
Derivative::Derivative(vector<Vec3>& rbf_centers_, vector<vector<int> >& stencil_, int nb_bnd_) : 
     rbf_centers(rbf_centers_), stencil(stencil_), maxint((1 << 31) - 1.)
{
	this->nb_bnd = nb_bnd_;
	this->nb_rbfs = rbf_centers.size();
	//this->nb_rbfs = stencil.size();
	Up = new mat();
	Vp = new mat();
	sp = new colvec();
	x_weights.resize(nb_rbfs);
	y_weights.resize(nb_rbfs);
	lapl_weights.resize(nb_rbfs);

	// derivative depends strongly on epsilon!
	// there must be a range of epsilon for which the derivative is approx. constant!
	//epsilon = .1; // laplacian returns zero (related to SVD perhaps)
	//epsilon = 3.; // Pretty accurate
	epsilon = 2.; // Pretty accurate
	//epsilon = 0.5; // Pretty accurate

	printf("nb_rbfs= %d\n", nb_rbfs); // ok
}
//----------------------------------------------------------------------
Derivative::~Derivative()
{
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
//----------------------------------------------------------------------
// return stencil weights for node irbf (global numbering)
// memory for weights must be assigned
int Derivative::distanceMatrix(vector<Vec3>& rbf_centers, vector<int>& stencil, 
    int irbf, int& nb_eig)
{
	Vec3& c = rbf_centers[irbf];
	vector<int>& st= stencil;
	int n = st.size();
	//printf("stencil size= %d\n", n);

	//printf("stencil size(%d): n= %d\n", irbf, n);
	//printf("n= %d\n", n);

	IRBF rbf(epsilon);

	//mat ar(n,n);
	// Derivative of a constant should be zero
	// Derivative of a linear function should be constant
	mat ar(n+3,n+3);

	int st_center = -1;

	// which stencil point is irbf
	for (int i=0; i < n; i++) {
		if (irbf == stencil[i]) {
			st_center = i;
			break;
		}
	}
	if (st_center == -1) {
		printf("inconsistency with global rbf %d\n", irbf);
		exit(0);
	}

	// stencil includes the point itself
	for (int i=0; i < n; i++) {
		Vec3& xiv = rbf_centers[st[i]];
	for (int j=0; j < n; j++) {
		// rbf
		Vec3& xjv = rbf_centers[st[j]];
		ar(i,j) = rbf(xiv, xjv);
	}}

	// last row and column of ar (distance matrix) should be 1
	for (int i=0; i < n; i++) {
		ar(n, i) = 1.0;
		ar(n+1, i) = rbf_centers[st[i]].x();
		ar(n+2, i) = rbf_centers[st[i]].y();
		ar(i, n) = 1.0;
		ar(i, n+1) = rbf_centers[st[i]].x();
		ar(i, n+2) = rbf_centers[st[i]].y();
	}
	for (int j=0; j < 3; j++) {
	for (int i=0; i < 3; i++) {
		ar(n+i, n+j) = 0.;
	}}
	n = n + 1; // deriv(constant) = 0
	n = n + 2; // deriv(linear function) = constant

	mat& U = *Up;
	mat& V = *Vp;
	colvec& s = *sp;

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
//----------------------------------------------------------------------
void Derivative::computeWeightsSVD(vector<Vec3>& rbf_centers, vector<int>&
stencil, int irbf, const char* choice)
{
	//printf("stencil %d\n", irbf);

	int st_center = -1;

	// which stencil point is irbf
	for (int i=0; i < stencil.size(); i++) {
		if (irbf == stencil[i]) {
			st_center = i;
			break;
		}
	}
	if (st_center == -1) {
		printf("inconsistency with global rbf %d\n", irbf);
		exit(0);
	}

	//printf("st_center= %d\n", st_center);
	if (st_center != 0) {
		printf("st_center should be the first element of the stencil!\n");
		exit(0);
	}

	// estimate radius for contour-svd method
	// distance matrix: each entry is the square of the internode distance
	arma::mat xd(stencil.size(), 2);
	for (int i=0; i < stencil.size(); i++) {
		Vec3& rc = rbf_centers[stencil[i]];
		xd(i,0) = rc[0];
		xd(i,1) = rc[1];
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

	var_eps[irbf] = 1.; // works
	var_eps[irbf] *= .07; // TEMP Does not work

	double rad = 1.1;              // rad should also be proportional to (1/avg_stencil_radius)
	double eps = 1.1; // * var_eps[irbf]; // variable epsilon (for 300 pts)
	//double eps = 1.5 * var_eps[irbf]; // variable epsilon (for 1000 pts)

	//printf("var_eps[%d]= %f\n", irbf, var_eps[irbf]);
	
	IRBF rbf(eps);
	Stencils sten(&rbf, rad, eps, &xd, choice);
    //arma::mat rd2 = sten.computeDistMatrix2(xd,xd);

	int N = 128; // Why can't I increase N?
	arma::mat fd_coeffs = sten.execute(N);

	char label[256]; 
	sprintf(label, "%s Derivative Coefficients =", choice); 
	fd_coeffs.print(label);
	//exit(0);

	// There should be a better way of doing this

	//printf("choice= %s\n", choice);

	if (strcmp(choice, "lapl") == 0) {
		lapl_weights[irbf] = fd_coeffs;
	} else if (strcmp(choice, "x") == 0) {
		//printf("irbf= %d\n", irbf);
		x_weights[irbf] = fd_coeffs;
	} else if (strcmp(choice, "y") == 0) {
		y_weights[irbf] = fd_coeffs;
	} else {
		printf("not covered\n");
	}
	//printf("-----------------\n");
}
//----------------------------------------------------------------------
void Derivative::computeWeights(vector<Vec3>& rbf_centers, vector<int>& stencil, 
    	int irbf)
{
	int nb_eig;

	int st_center = distanceMatrix(rbf_centers, stencil, irbf, nb_eig);
	vector<int>& st= stencil;
	int n = st.size();
	IRBF rbf(epsilon);

	// Phi'(st_center)   (should choose the line that corresponds to the center RBF)

	bx.zeros(n+3); // extra lines to enforce d/dx(constant)=0, d/dx(linear fct) is constant
	by.zeros(n+3);
	blapl.zeros(n+3);

	// stencil includes the point itself
	Vec3& x0v = rbf_centers[st[st_center]];
	for (int j=0; j < n; j++) {
		Vec3& xjv = rbf_centers[st[j]];
		bx(j) = rbf.xderiv(x0v, xjv);
		by(j) = rbf.yderiv(x0v, xjv);
		blapl(j) = rbf.lapl_deriv(x0v, xjv);
		//printf("blapl(%d)= %f\n", j, blapl(j));
	}
	//exit(0);

	// I am not convinced of these lines
	bx(n)   = 0.0;
	bx(n+1) = 1.;//1.0; (if =1, then sum x(i) w(i) = 1 (DO NOT KNOW WHY)
	bx(n+2) = 0.0;
	by(n)   = 0.0;
	by(n+1) = 0.; // 1.0;
	by(n+2) = 1.0;
	// laplacian of constant and linear functions is zero
	blapl(n) = 0.0;
	blapl(n+1) = 0.0;
	blapl(n+2) = 0.0;

	n = n + 1; // deriv of constant is zero
	n = n + 2; // deriv of linear function is exact

	colvec wwx(n);
	colvec wwy(n);
	colvec wwlapl(n);

	wwx.zeros();
	wwy.zeros();
	wwlapl.zeros();

	mat& U = *Up;
	mat& V = *Vp;
	mat& s = *sp;

	// TEMP WAY TO COMPUTE WEIGHTS
	for (int i=0; i < nb_eig; i++) {
		wwx = wwx +       dot(trans(U.col(i)),bx   )*V.col(i) / s(i); 
		wwy = wwy +       dot(trans(U.col(i)),by   )*V.col(i) / s(i); 
		wwlapl = wwlapl + dot(trans(U.col(i)),blapl)*V.col(i) / s(i); 
	}
	x_weights[irbf]    = wwx;
	y_weights[irbf]    = wwy;
	lapl_weights[irbf] = wwlapl;

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


	U.reset();
	V.reset();
	s.reset();

	bx.reset();
	by.reset();
	blapl.reset();
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
	//printf("computeDeriv, u= %d, deriv= %d\n", &u[0], &deriv[0]);
	//printf("computeDeriv, u.size= %d, deriv.size= %d\n", u.size(), deriv.size());
	computeDeriv(which, &u[0], &deriv[0], deriv.size());
}
//----------------------------------------------------------------------
void Derivative::computeDeriv(DerType which, double* u, double* deriv, int npts)
{
	
	cout << "COMPUTING DERIVATIVE: ";
	vector<mat>* weights_p;

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

	vector<mat>& weights = *weights_p;

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
	printf("Weights size: %d\n", (int)weights.size());
	printf("Stencils size: %d\n", (int)stencil.size());
	for (int i=0; i < stencil.size(); i++) {
		mat& w = weights[i];
		vector<int>& st = stencil[i];
		//printf("nb el in weight[%d]: %d, in stencil[%d]: %d\n", i, w.n_elem, i, st.size());
		//printf("i=%d, w[0] = %f\n", i, w[0]);
		der = 0.0;
		int n = st.size(); 
		cout << "STENCIL " << i << ": " << endl;
		//printf("(%d) stencil size: %d\n", i, n);
		for (int s=0; s < n; s++) {
			printf("\tw[%d]= %f * ", s, w[s]);
			printf("st[%d]= %d\t\t%f  * %f\n", s, st[s], w[s], u[st[s]]);
			der += w[s] * u[st[s]]; // SOMETHING WRONG!
		}
		deriv[i] = der;
	}
}
//----------------------------------------------------------------------
double Derivative::computeEig()
{
	vector<mat>* weights_p;
	weights_p = &lapl_weights;  // <<<< lapl_weights
	vector<mat>& weights = *weights_p;

	#if 1
	// compute eigenvalues of derivative operator
	int sz = weights.size();
	printf("sz= %d\n", sz);
	printf("weights.size= %d\n", (int) weights.size());

	mat eigmat(sz, sz);
	eigmat.zeros();

	for (int i=nb_bnd; i < sz; i++) {
		mat& w = weights[i];
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
