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
            ceps2 = ceps*ceps;
        }
        RBF(CMPLX epsilon, int dim_num) : ceps(epsilon), dim(dim_num) {
            ceps2 = ceps*ceps;
            eps = real(ceps);
            eps2 = eps*eps;
        }
        virtual ~RBF() {};

        // Assume x and x_center are just POINTS (not distances)
        double operator()(const Vec3& x, const Vec3& x_center) {
            return eval(x,x_center);
        }

        // Assume x_minus_xj = x-x_center (i.e., a POINT/VECTOR)
        double operator()(const Vec3& x_minus_xj) {
            return eval(x_minus_xj);
        }

        // Assume x = x-x_center
        CMPLX operator()(const CVec3& x_minus_xj) {
            return eval(x_minus_xj);
        }

        // Assume d = ||x-x_center||_2
        CMPLX operator()(CMPLX d) {
            return eval(d);
        }

        // Assume d = ||x-x_center||_2
        double operator()(double d) {
            return eval(d);
        }

        //------------------------------------------------------------------

        // aug 15, 2005, input: 2D array of distances squared, output: 2D array
        // Distances squared for compatibility with Grady's SVD program. I might change this later
        arma::mat operator()(const arma::mat& arr) {
            int nr = arr.n_rows;
            int nc = arr.n_cols;
            arma::mat values(nr, nc);
            for (int j=0; j < nc; j++) {
                for (int i=0; i < nr; i++) {
                    //printf("sqrt= %f\n", sqrt(arr(i,j)));
                    values(i,j) = eval( sqrt(arr(i,j)) );
                }
            }
            return values;
        }

        arma::cx_mat operator()(const arma::cx_mat& arr) {
            int nr = arr.n_rows;
            int nc = arr.n_cols;
            arma::cx_mat values(nr, nc);
            for (int j=0; j < nc; j++) {
                for (int i=0; i < nr; i++) {
                    //printf("sqrt= %f\n", sqrt(arr(i,j)));
                    values(i,j) = eval( std::sqrt(arr(i,j)) );
                }}
            return values;
        }

        arma::cx_mat operator()(const ArrayT<CVec3>& arr) {
            const int* dims = arr.getDims();
            arma::cx_mat values(dims[0], dims[1]);
            for (int j=0; j < dims[1]; j++) {
                for (int i=0; i < dims[0]; i++) {
                    values(i,j) = eval(arr(i,j));
                }
            }
            return values;
        }


        //------------------------------------------------------------------

        // aug 15, 2009, input: 2D array of distances squared, output: 2D array
        // Distances squared for compatibility with Grady's SVD program. I might change this later
#if 0
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
#endif 

        // These Vec3 are actually (x-xj)
        // NOTE: not magnitude, but actually directional separation
        arma::mat lapl_deriv(const ArrayT<Vec3>& arr) {
            const int* dims = arr.getDims();
            arma::mat values(dims[0], dims[1]);
            for (int j=0; j < dims[1]; j++) {
                for (int i=0; i < dims[0]; i++) {
                    values(i,j) = lapl_deriv(arr(i,j));
                }
            }
            return values;
        }

        arma::cx_mat lapl_deriv(const ArrayT<CVec3>& arr) {
            const int* dims = arr.getDims();
            arma::cx_mat values(dims[0], dims[1]);
            for (int j=0; j < dims[1]; j++) {
                for (int i=0; i < dims[0]; i++) {
                    values(i,j) = lapl_deriv(arr(i,j));
                }
            }
            return values;
        }

#if 0
        // array of square of distances
        arma::cx_mat lapl_deriv(const arma::cx_mat& arr) {
            int nr = arr.n_rows;
            int nc = arr.n_cols;
            arma::cx_mat values(nr, nc);
            for (int j=0; j < nc; j++) {
                for (int i=0; i < nr; i++) {
                    values(i,j) = lapl_deriv(arr(i,j));
                }}
            return values;
        }
#endif 
        //------------------------------------------------------------------

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

        //------------------------------------------------------------------

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

        //------------------------------------------------------------------

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

        //------------------------------------------------------------------

        void setEpsilon(double e) {
            eps = e;
            eps2 = eps*eps;
            ceps = CMPLX(eps, 0.);
            ceps2 = ceps*ceps;
            std::cout << "INFO: eps is real number" << std::endl;
        }

        void setEpsilon(CMPLX e) {
            eps = e.real(); 
            eps2 = e.real()*e.real(); 
            ceps = e;
            ceps2 = e*e;
        }

        //------------------------------------------------------------------
        
    protected:
        inline double Power(double a, double b) { return pow(a, b); }
        inline double Sqrt(double a) { return sqrt(a); }
        inline double Sin(double a) { return sin(a); }
        inline double Cos(double a) { return cos(a); }

    public: // PURE VIRTUAL REQUIRE OVERRIDES

        // NOTE: for all of these, x_center is the CENTER point of the RBF
        //  x is the parameter for the RBF evaluation

        //------------------------------------------------------------------
        // Evaluations: 
        double eval(const Vec3& x, const Vec3& xi) { this->eval(x-xi); }
        virtual double eval(const Vec3& x) = 0;
        virtual CMPLX eval(const CVec3& x) = 0;
        virtual double eval(const double x) = 0;
        virtual CMPLX eval(const CMPLX x) = 0;

        //------------------------------------------------------------------
        // First Derivatives: 
        double xderiv(const Vec3& xvec, const Vec3& x_center) { this->xderiv(xvec-x_center); }
        double yderiv(const Vec3& xvec, const Vec3& x_center) { this->yderiv(xvec-x_center); }
        double zderiv(const Vec3& xvec, const Vec3& x_center) { this->zderiv(xvec-x_center); }
        //------------------------------------------------------------------
        virtual double xderiv(const Vec3& xvec)  = 0;
        virtual CMPLX  xderiv(const CVec3& xvec) = 0;
        //------------------------------------------------------------------
        virtual double yderiv(const Vec3& xvec)  = 0;
        virtual CMPLX  yderiv(const CVec3& xvec) = 0;
        //------------------------------------------------------------------
        virtual double zderiv(const Vec3& xvec)  = 0;
        virtual CMPLX  zderiv(const CVec3& xvec) = 0;

        //------------------------------------------------------------------
        // Second derivatives
#if 0
        virtual double xxderiv(const Vec3& xvec, const Vec3& x_center) = 0;
        virtual double yyderiv(const Vec3& xvec, const Vec3& x_center) = 0;
        virtual double xyderiv(const Vec3& xvec, const Vec3& x_center) = 0;
#endif    
        //------------------------------------------------------------------
        // Laplacians:
        virtual double lapl_deriv(const Vec3& xvec, const Vec3& x_center) { this->lapl_deriv(xvec-x_center); }
        virtual double lapl_deriv(const Vec3& xvec) {
            switch (dim) 
            {
                case 1: 
                    return this->lapl_deriv1D(xvec);
                    break; 
                case 2: 
                    return this->lapl_deriv2D(xvec);
                    break;
                case 3: 
                    return this->lapl_deriv3D(xvec);
                    break; 
                default: 
                    printf("error: unsupported rbf.lapl_deriv dimension\n");
                    exit(EXIT_FAILURE); 
            }
            return -1;
        }

        virtual double lapl_deriv1D(const Vec3& xvec) = 0;
        virtual double lapl_deriv2D(const Vec3& xvec) = 0;
        virtual double lapl_deriv3D(const Vec3& xvec) = 0;

        virtual CMPLX  lapl_deriv(const CVec3& x) {
            switch (dim) 
            {
                case 1: 
                    return this->lapl_deriv1D(x);
                    break; 
                case 2: 
                    return this->lapl_deriv2D(x);
                    break;
                case 3: 
                    return this->lapl_deriv3D(x);
                    break; 
                default: 
                    printf("error: unsupported rbf.lapl_deriv dimension\n");
                    exit(EXIT_FAILURE); 
            }
            return -1;
        }
        
        virtual CMPLX  lapl_deriv1D(const CVec3& x) = 0;
        virtual CMPLX  lapl_deriv2D(const CVec3& x) = 0;
        virtual CMPLX  lapl_deriv3D(const CVec3& x) = 0;
};

#endif
