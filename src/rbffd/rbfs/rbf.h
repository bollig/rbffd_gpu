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
    int dim;

public:
    RBF(double epsilon, int dim_num) : eps(epsilon), dim(dim_num) {
        eps2 = eps*eps;
        ceps = CMPLX(eps, 0.);
	ceps2 = CMPLX(5.5, 0.);
    }
    RBF(CMPLX epsilon, int dim_num) : ceps(epsilon), dim(dim_num) {
	ceps2 = CMPLX(6.3, 0.);
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
        //ceps2 = ceps*ceps;
	ceps2 = CMPLX(2.6, 0.);	
    }
    void setEpsilon(CMPLX e) {
        ceps = e;
        ceps2 = e*e;
	CMPLX ceps2alt = CMPLX(e.real()*e.real(), e.imag()*e.imag());
	if (ceps2 != ceps2alt) {
		std::cout << "ERROR: complex arithmetic is not functioning\n"; 
		exit(EXIT_FAILURE); 
	}
        /*printf("ceps= %f, %f\n", real(ceps), imag(ceps)); */
    }

protected:
    inline double Power(double a, double b) { return pow(a, b); }
    inline double Sqrt(double a) { return sqrt(a); }
    inline double Sin(double a) { return sin(a); }
    inline double Cos(double a) { return cos(a); }

        public: // PURE VIRTUAL REQUIRE OVERRIDES

            virtual double eval(const Vec3& x, const Vec3& xi) = 0;
            virtual double eval(const Vec3& x) = 0;
            virtual CMPLX eval(const CVec3& x) = 0;
            virtual double eval(const double x) = 0;
            virtual CMPLX eval(const CMPLX x) = 0;

            // First derivative:  d Phi(r) / dr
            virtual double first_deriv(const Vec3& x) = 0;
            // First derivative:  d Phi(r) / dr
            virtual double second_deriv(const Vec3& x) = 0;

            // radial derivative dPhi/dr(x,y,z) = x/r * d/dx  + y/r * d/dy + z/r * d/dz
            virtual double radialderiv(const Vec3& xvec, const Vec3& xi) = 0;

            virtual double xderiv(const Vec3& xvec, const Vec3& xi) = 0;
            virtual double yderiv(const Vec3& xvec, const Vec3& xi) = 0;
            virtual double zderiv(const Vec3& xvec, const Vec3& xi) = 0;

            virtual double xderiv(const Vec3& xvec)  = 0;
            virtual CMPLX  xderiv(const CVec3& xvec) = 0;
            virtual double yderiv(const Vec3& xvec)  = 0;
            virtual CMPLX  yderiv(const CVec3& xvec) = 0;
            virtual double zderiv(const Vec3& xvec)  = 0;
            virtual CMPLX  zderiv(const CVec3& xvec) = 0;

            virtual double xxderiv(const Vec3& xvec, const Vec3& xi) = 0;
            virtual double yyderiv(const Vec3& xvec, const Vec3& xi) = 0;
            virtual double xyderiv(const Vec3& xvec, const Vec3& xi) = 0;

            virtual double lapl_deriv(const Vec3& xvec, const Vec3& xi) = 0;
            virtual double lapl_deriv(const Vec3& xvec) = 0;
            virtual double lapl_deriv(const double x)   = 0;
            virtual CMPLX  lapl_deriv(const CMPLX x)    = 0;
        };

#endif
