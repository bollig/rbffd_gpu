#ifndef _RBF_H_
#define _RBF_H_

#include <math.h>
#include <armadillo>
#include <complex>
#include <valarray>
#include <ArrayT.h>
#include <Vec3.h>
#include <CVec3.h>

// RBF function. 
// I might create subclass for different rbf functions

// Aug. 15, 2009
// Add return of 2D matrix values, using Armadillo (could have use Array3D.h, but not for now)

// Gaussian RBF
// Theorically positive definite

typedef std::complex<double> CMPLX;

class RBF {
protected:
	double eps;
	double eps2;
	CMPLX ceps;
	CMPLX ceps2;

public:
	RBF(double epsilon) : eps(epsilon) {
		eps2 = eps*eps;
		ceps = CMPLX(eps, 0.);
		ceps2 = ceps*ceps;
	}
	RBF(CMPLX epsilon) : ceps(epsilon) {
		ceps2 = ceps*ceps;
		eps = 100.; // values way larger than is possible 
		eps2 = eps*eps; 
	}
	~RBF() {};

	double operator()(const Vec3& x, const Vec3& xi) {
		return eval(x,xi);
	}

	double operator()(const Vec3& x) {
		return eval(x);
	}

	CMPLX operator()(const CVec3& x) {
		return eval(x);
	}

	// x is the distance, argument of rbf
	CMPLX operator()(CMPLX x) {
		return eval(x);
	}

	// x is the distance, argument of rbf
	double operator()(double x) {
		return eval(x);
	}

	// aug 15, 2005, input: 2D array of distances squared, output: 2D array
	// Distances squared for compatibility with Grady's SVD program. I might change this later
	arma::mat operator()(const arma::mat& arr) {
		int nr = arr.n_rows;
		int nc = arr.n_cols;
		arma::mat values(nr, nc);
		for (int j=0; j < nc; j++) { 
		for (int i=0; i < nr; i++) { 
			//printf("sqrt= %f\n", sqrt(arr(i,j)));
			values(i,j) = eval(sqrt(arr(i,j)));
		}}
		return values;
	}

	arma::cx_mat operator()(const arma::cx_mat& arr) {
		int nr = arr.n_rows;
		int nc = arr.n_cols;
		arma::cx_mat values(nr, nc);
		for (int j=0; j < nc; j++) { 
		for (int i=0; i < nr; i++) { 
			//printf("sqrt= %f\n", sqrt(arr(i,j)));
			values(i,j) = eval(std::sqrt(arr(i,j)));
		}}
		return values;
	}

	virtual double eval(const Vec3& x, const Vec3& xi) = 0;
	virtual double eval(const Vec3& x) = 0;
	virtual CMPLX eval(const CVec3& x) = 0;
	virtual double eval(const double x) = 0;
	virtual CMPLX eval(const CMPLX x) = 0;

	virtual double xderiv(const Vec3& xvec, const Vec3& xi) = 0;
	virtual double yderiv(const Vec3& xvec, const Vec3& xi) = 0;
	virtual double zderiv(const Vec3& xvec, const Vec3& xi) = 0;
        virtual double rderiv(const Vec3& xvec, const Vec3& xi) = 0;

	virtual double xderiv(const Vec3& xvec)  = 0;
	virtual CMPLX  xderiv(const CVec3& xvec) = 0;
	virtual double yderiv(const Vec3& xvec)  = 0;
	virtual CMPLX  yderiv(const CVec3& xvec) = 0;
        virtual double zderiv(const Vec3& xvec)  = 0;
        virtual CMPLX  zderiv(const CVec3& xvec) = 0;

	virtual double xxderiv(const Vec3& xvec, const Vec3& xi) = 0;
	virtual double yyderiv(const Vec3& xvec, const Vec3& xi) {
		return(0.);
	}

	//virtual double xderiv(const double x) = 0;
	//virtual CMPLX  xderiv(const CMPLX x)  = 0;
	//virtual double yderiv(const double x) = 0;
	//virtual CMPLX  yderiv(const CMPLX x)  = 0;

	virtual double xyderiv(const Vec3& xvec, const Vec3& xi) = 0;

	virtual double lapl_deriv(const Vec3& xvec, const Vec3& xi) = 0;
	virtual double lapl_deriv(const Vec3& xvec) = 0;
	virtual double lapl_deriv(const double x)   = 0;
	virtual CMPLX  lapl_deriv(const CMPLX x)    = 0;

	// aug 15, 2009, input: 2D array of distances squared, output: 2D array
	// Distances squared for compatibility with Grady's SVD program. I might change this later
	arma::mat lapl_deriv(const arma::mat& arr) {
		int nr = arr.n_rows;
		int nc = arr.n_cols;
		arma::mat values(nr, nc);
		for (int j=0; j < nc; j++) { 
		for (int i=0; i < nr; i++) { 
			values(i,j) = lapl_deriv(sqrt(arr(i,j)));
		}}
		return values;
	}

	arma::mat lapl_deriv(const ArrayT<Vec3>& arr) {
		const int* dims = arr.getDims();
		arma::mat values(dims[0], dims[1]);
		for (int j=0; j < dims[1]; j++) { 
		for (int i=0; i < dims[0]; i++) { 
			values(i,j) = lapl_deriv(sqrt(arr(i,j).square()));
		}}
		return values;
	}

	arma::cx_mat lapl_deriv(const ArrayT<CVec3>& arr) {
		const int* dims = arr.getDims();
		arma::cx_mat values(dims[0], dims[1]);
		for (int j=0; j < dims[1]; j++) { 
		for (int i=0; i < dims[0]; i++) { 
			values(i,j) = lapl_deriv(CMPLX(arr(i,j).magnitude()));
		}}
		return values;
	}


	// array of square of distances
	arma::cx_mat lapl_deriv(const arma::cx_mat& arr) {
		int nr = arr.n_rows;
		int nc = arr.n_cols;
		arma::cx_mat values(nr, nc);
		for (int j=0; j < nc; j++) { 
		for (int i=0; i < nr; i++) { 
			values(i,j) = lapl_deriv(CMPLX(sqrt(arr(i,j))));
		}}
		return values;
	}

	arma::mat xderiv(const ArrayT<Vec3>& arr) {
		const int* dims = arr.getDims();
		int nr = dims[0];
		int nc = dims[1];
		arma::mat values(nr, nc);
		for (int j=0; j < nc; j++) { 
		for (int i=0; i < nr; i++) { 
			values(i,j) = xderiv(arr(i,j));
		}}
		return values;
	}

	arma::mat yderiv(const ArrayT<Vec3>& arr) {
		const int* dims = arr.getDims();
		int nr = dims[0];
		int nc = dims[1];
		arma::mat values(nr, nc);
		for (int j=0; j < nc; j++) { 
		for (int i=0; i < nr; i++) { 
			values(i,j) = yderiv(arr(i,j));
		}}
		return values;
	}

        arma::mat zderiv(const ArrayT<Vec3>& arr) {
                const int* dims = arr.getDims();
                int nr = dims[0];
                int nc = dims[1];
                arma::mat values(nr, nc);
                for (int j=0; j < nc; j++) {
                for (int i=0; i < nr; i++) {
                        values(i,j) = zderiv(arr(i,j));
                }}
                return values;
        }


	arma::cx_mat xderiv(const ArrayT<CVec3>& arr) {
		//printf("inside xderiv ArrayT<CVec3>\n");
		const int* dims = arr.getDims();
		arma::cx_mat values(dims[0], dims[1]);
		for (int j=0; j < dims[1]; j++) { 
		for (int i=0; i < dims[0]; i++) { 
			CVec3 v = arr(i,j);
			values(i,j) = xderiv(v);
		}}
		return values;
	}

	arma::cx_mat yderiv(const ArrayT<CVec3>& arr) {
		const int* dims = arr.getDims();
		arma::cx_mat values(dims[0], dims[1]);
		for (int j=0; j < dims[1]; j++) { 
		for (int i=0; i < dims[0]; i++) { 
			CVec3 v = arr(i,j);
			values(i,j) = yderiv(v);
		}}
		return values;
	}

        arma::cx_mat zderiv(const ArrayT<CVec3>& arr) {
                const int* dims = arr.getDims();
                arma::cx_mat values(dims[0], dims[1]);
                for (int j=0; j < dims[1]; j++) {
                for (int i=0; i < dims[0]; i++) {
                        CVec3 v = arr(i,j);
                        values(i,j) = zderiv(v);
                }}
                return values;
        }

	void setEpsilon(double e) { 
		eps = e; 
		eps2 = eps*eps; 
		ceps = CMPLX(eps, 0.);
		ceps2 = ceps*ceps;
	}
	void setEpsilon(CMPLX e) { 
		ceps = e; 
		ceps2 = ceps*ceps; 
		/*printf("ceps= %f, %f\n", real(ceps), imag(ceps)); */
	}
};

#endif
