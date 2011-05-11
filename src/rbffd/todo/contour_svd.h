#ifndef CONTOURSVD
#define CONTOURSVD

#include <ArrayT.h>
#include <CVec3.h>
#include <armadillo>
#include <string>
#include "rbffd/rbfs/rbf.h"
#include "rbffd/rbfs/rbf_mq.h"

class ContourSVD
{
private:
	int N;
	int NUM_AGREEING_COEFFS;
	double COEFF_COMPARISON_TOL;
	double COEFF_MAGNITUDE_TOL;
	int FALSE;
	int TRUE;
	std::string DEBUG;
	int numCoeffs;
	RBF* rbf;
	arma::rowvec ep;
	arma::mat rr0, rrd;
	ArrayT<Vec3>* rrdvec;
	arma::cx_mat crr0, crrd;
	ArrayT<CVec3>* crrdvec;
	arma::mat rd2;
	arma::cx_mat crd2;
	double rad;
	double pi;
	CMPLX iota;
	arma::mat fd_coeffs;
	//ArrayT<double> mm;
	const char* choice;

//----------------------------------------------------------------------
// relates to computation of derivative coefficients
public:
	ContourSVD();
	ContourSVD(RBF* rbf, arma::mat& rd2, arma::rowvec& ep, arma::mat& rr0, arma::mat& rrd, 
		ArrayT<Vec3>& rrdvec, double rad);
        // rbf = RBF choice class (i.e., RBF_MQ for multiquadric)
        // td2 = Squared distance matrix (||x-xi||^2)
        // ep = RBF support parameter divided by stencil radius (eps/rad)
        // rr0 = Squared distance matrix scaled by the radius squared
        // rrd = Squared distance vector for separation between stencil nodes and stencil center Vec(||x-xi||^2)
        // rrdvec = Vec3 vectors representing separation between stencil nodes and the stencil center (x-xi) and scaled by the radius
        //              NOTE: this is NOT the same as rrd_norm (as in Vec(||x-xi||^2)); it is Vec(Vec3(x-xi));
        // rad = {???} typically passed 1.
        ContourSVD(RBF* rbf, arma::mat& rd2, double ep, arma::mat& rr0, arma::mat& rrd, ArrayT<Vec3>& rrdvec, double rad);
	void init(RBF* rbf, arma::mat& rd2, arma::rowvec& ep, arma::mat& rr0, arma::mat& rrd, ArrayT<Vec3>& rrdvec, double rad);
	void execute(int N);
	void getPoles();

	arma::mat    rbffdapp(double eps, arma::mat& rd, arma::mat& re);
	arma::cx_mat rbffdapp(arma::cx_double eps, arma::cx_mat& rd, arma::cx_mat& re);
	arma::mat    rbffdapp(double eps, arma::mat& rd, ArrayT<Vec3>& re, const char* choice);

	arma::cx_mat rbffdapp(CMPLX eps, arma::cx_mat& rd, arma::cx_mat& re, const char* choice);
	arma::cx_mat rbffdapp(CMPLX eps, arma::cx_mat& rd, ArrayT<CVec3>& re, const char* choice);

	arma::rowvec solver(arma::rowvec& A, arma::mat& B);
	arma::cx_rowvec solver(arma::cx_rowvec& A, arma::cx_mat& B);

	template <class T> void getSize(T& matr, int& nr, int& nc);
	template <class T> void printSize(T& cm, char* msg=0);

	void computeLaurentCoeffs(std::vector<arma::cx_mat*> C, double erad, ArrayT<double>& negPows, ArrayT<double>& posPows);

	// 2nd and 3rd argument: only flip sub array and return sub array. 
	// last argument is one part the last array element
	template <class T> std::vector<T*> flipdim3(std::vector<T*>& C, int i1=-1, int i2=-1);
	template <class T> ArrayT<T> flipdim3(ArrayT<T>& C, int i1=-1, int i2=-1);
	template <class T> ArrayT<double> valsAgree(ArrayT<T>& val1, ArrayT<T>& val2, double relError);

	std::vector<double> twoNorm(ArrayT<double>& negPows, std::vector<int>& max_index);

	template<class T> ArrayT<T> hankel(std::vector<T> v);
	arma::mat hankel(std::vector<double> v);
	arma::mat& getFDCoeffs() { return fd_coeffs; }

	void computeCoeffs(double eps);

	template <class T> arma::Mat<T> polyval2(std::vector<T> p, arma::Mat<T> x);
	template <class T> std::vector<T> polyval2(std::vector<T> p, std::vector<T> x);
	template <class T> void print(vector<T> v, const char* msg=0);

	void print(std::vector<CMPLX> v, const char* msg=0);
	void print(arma::cx_mat& v, const char* msg);

	template <class T> void print(arma::Mat<T>& v, const char* msg);
	void print(arma::mat& v, const char* msg);

	void setChoice(const char* choice_) {
		this->choice = (const char*) choice_;
	}
};

#endif
