#include <complex>
#include <vector>
#include "contour_svd.h"
#include "rbf.h"

#include <fftw3.h>

#include <ArrayT.h>
#include <Vec3i.h>

using namespace std;
using namespace arma;

//[fds, poles] = contourSVD(@rbffdapp,rad/rad,ep/rad,N,rd2*rad^2,rd2(1,:)*rad^2,rbf,drbf);
// optional: rd2*rad^2, rd2(1,:)*rad^2, rbf, drbf

ContourSVD::ContourSVD()
{
}
//----------------------------------------------------------------------
ContourSVD::ContourSVD(RBF* rbf, mat& rd2, double ep, mat& rr0, mat& rrd, ArrayT<Vec3>& rrdvec, double rad)
{
    // rad=1 (it is normalized on input

    // wrap scalar ep into a 1x1 matrix
    rowvec eps = zeros<rowvec>(1);
    //printf("r,c= %d, %d\n", eps.n_rows, eps.n_cols); exit(0);
    eps(0) = ep;
    //eps.print("eps, contourSVD, 1 element");
    init(rbf, rd2, eps, rr0, rrd, rrdvec, rad);
}
//----------------------------------------------------------------------
ContourSVD::ContourSVD(RBF* rbf, mat& rd2, rowvec& ep, mat& rr0, mat& rrd, ArrayT<Vec3>& rrdvec, double rad)
{
    init(rbf, rd2, ep, rr0, rrd, rrdvec, rad);
}
//----------------------------------------------------------------------
void ContourSVD::init(RBF* rbf, mat& rd2, rowvec& ep, mat& rr0, mat& rrd, ArrayT<Vec3>& rrdvec_, double rad)
{
    this->rd2 = rd2; // matrix can be changed in calling program without affecting values here
    this->rbf = rbf;
    this->ep  = ep;
    this->rr0 = rr0;
    this->rrd = rrd;
    this->rad = rad;
    rrdvec  = &rrdvec_;

    //printf("rrd size: %d, %d\n", rrd.n_rows, rrd.n_cols);
    //this->rrdvec = new ArrayT<Vec3>(1,9);
    //rrdvec = new ArrayT<Vec3>(rrd.n_rows, rrd.n_cols);
    //rrdvec = new ArrayT<Vec3>(1,9);
    //ArrayT<Vec3> xxx(rrd.n_rows, rrd.n_cols);
    //ArrayT<Vec3>* rrdvec1 = new ArrayT<Vec3>(rrd.n_rows, rrd.n_cols);
    //const int* dims = rrdvec1->getDims();

    this->crd2 = conv_to<cx_mat>::from(rd2);
    this->crr0 = conv_to<cx_mat>::from(rr0);
    this->crrd = conv_to<cx_mat>::from(rrd);

    crrdvec = new ArrayT<CVec3>(rrdvec->getTotPoints());
    Vec3* rp  = rrdvec->getDataPtr();
    CVec3* crp = crrdvec->getDataPtr();

    int npts = crrdvec->getSize();
    //printf("npts= %d\n", npts);
    for (int i=0; i < npts; i++) {
        crp[i] = rp[i];
    }

    const int* dims = rrdvec->getDims();
    //printf("dims= %d, %d, %d\n", dims[0], dims[1], dims[2]);

    const int* cdims = crrdvec->getDims();
    //printf("(crrdec cdims= %d, %d, %d\n", cdims[0], cdims[1], cdims[2]);

    //exit(0);

    NUM_AGREEING_COEFFS = 1;
    COEFF_COMPARISON_TOL = 5.e-3;
    COEFF_MAGNITUDE_TOL = 1.e-13;

    pi = acos(-1.);
    iota = CMPLX(0.,1.);
}
//----------------------------------------------------------------------
void ContourSVD::getPoles()
{
}
//----------------------------------------------------------------------
// function [C, poles] = contourSVD(Cfunc,rad,epsilon,N,varargin)
// C++ Makes variable arguments difficult
void ContourSVD::execute(int N_)
{
    this->N = N_;
    NUM_AGREEING_COEFFS = 1;
    COEFF_COMPARISON_TOL = 5e-3;
    COEFF_MAGNITUDE_TOL = 1e-13;
    FALSE = 0;
    TRUE = !FALSE;
    DEBUG = "DEBUG";

    numCoeffs = N/4+1;   // Actual number of Laurent coefficients

   // printf("inside execute\n");

    // mg, mc: ize of array returned by matlab
    // I need complex arithmetic for contour integration. More complications
    // first argument is epsilon, normalized to unity
    cx_mat cm = rbffdapp(CMPLX(1.,0.), crr0, *crrdvec, choice); //(rad*rad)%rd2, (rad*rad)%matr);
    int mg, mc;
    getSize(cm, mg, mc);
    //printSize(cm, "cm");
    //crr0.print("rr0");

    vector<CMPLX> z(numCoeffs);

    for (int i=0; i < numCoeffs; i++) {
        double theta = (double) i/N;
        z[i] = rad*exp(2.*pi*iota*theta);
        //cout << "z= " << z << endl;
    }

    // Value of z are correct (4-point stencil)

    // I cannot have 3D matrices
    // I'll create an array of 2D matrices, each of size (mg x mc)

    vector<cx_mat*> C(numCoeffs);
    //printSize(*(C[0]), "size(*C)");
    //exit(0);
    //ArrayT<CMPLX> C(mg,mc,numCoeffs);

    for (int i=0; i < numCoeffs; i++) {
        C[i] = new cx_mat(mg,mc);
        //cout << "z= " << z[i] << endl;
        *C[i] = rbffdapp(z[i], crr0, *crrdvec, choice);
        //print(*C[i], "C[i]");
        //printf("BEFORE EXIT\n"); exit(0);
        //(*C[i]).print("c[i]"); // VALUES CANNOT BE CORRECT
    }

    /*
% Extract every other circle value of C. These values will be used to 
% determine whether any singularities of C are enclosed within the contour.
testC = C(:,:,1:2:numCoeffs);
*/
    vector<cx_mat*> testC;
    for (int i=0; i < numCoeffs; i+= 2) {
        testC.push_back(C[i]);
    }
    //printf("testC size: %d\n", testC.size());

    /*
% Compute the Laurent coefficients for entries of C and testC.
[negPows,posPows] = computeLaurentCoeffs(C,rad);
negPowsTest = computeLaurentCoeffs(testC,rad);
*/

    //printf("before computeLaurent\n");
    int M = C.size() - 1;
    ArrayT<double> negPows(mg,mc,M-1);
    ArrayT<double> posPows(mg,mc,M+1);
    computeLaurentCoeffs(C,rad, negPows, posPows);

    //negPows.print("negPows");
    //posPows.print("posPows");
    //exit(0);

    M = testC.size() - 1;
    ArrayT<double> negPowsTest(mg,mc,M-1);
    ArrayT<double> posPowsTest(mg,mc,M+1);
    computeLaurentCoeffs(testC,rad, negPowsTest, posPowsTest); // last argument not required


    /*
% testC is no longer used in the program.
clear testC;
*/

    /*
% Set N to be N/2, since we have reduced the number of laurent coeffs by 1/2.
% Remember only the even terms are present in the expansion of C due to the 
% 4 fold symmetry.
N = N/2;
*/

    N = N / 2;
    //printf("orig_N/2 = %d\n", N);
    //exit(0);

    /*
% Compare the first non-zero negative power Laurent coefficient for each
% column of the C and testC matrices to see if they agree. If they do agree
% then there are singularities enclosed within the circle of radius rad.
% To determine if they agree, sum up the number of non zero coefficients
% that agree for each column of C. If this number is greater than 1 for a
% particular column then we need to compute a denominator for the column.

% Row vector containing the number of coefficients that agree in each
% column
% sum over rows ==> row vector
columnCoeffsCounter = sum(valsAgree(negPows(:,:,1),...
      negPowsTest(:,:,1),COEFF_COMPARISON_TOL) & ...
      abs(negPows(:,:,1)) > COEFF_MAGNITUDE_TOL,1);

*/

    ArrayT<double> result(mg, mc);

    // This trick works since only the first mg*mc elements of both arrays are required
    // Extract
    ArrayT<double> negPows1(negPows.getDataPtr(), mg,mc);
    const int* dims = negPows1.getDims();
    ArrayT<double> negPowsTest1(&negPowsTest(0,0,0), mg,mc);
    result = valsAgree(negPows1, negPowsTest1, COEFF_COMPARISON_TOL);

    urowvec columnCoeffsCounter(mc);

    for (int j=0; j < mc; j++) {
        columnCoeffsCounter(j) = 0.;
        for (int i=0; i < mg; i++) {
            int b = (abs(negPows(i,j)) > COEFF_MAGNITUDE_TOL) ? TRUE : FALSE;
            b = (b == TRUE) && (result(i,j) == TRUE) ? TRUE : FALSE;
            columnCoeffsCounter(j) = columnCoeffsCounter(j) + b;
        }
    }

    //print(columnCoeffsCounter, "columnCoeffsCounter");

    /*
% Row vector indicating which columns of the C matrix need to have a 
% polynomial denominator computed
computeDenomFlag = (columnCoeffsCounter >= NUM_AGREEING_COEFFS);
% find: returns column indices
columnsToComputeDenom = find( computeDenomFlag == 1 );
*/
    urowvec computeDenomFlag(mc);
    for (int i=0; i < mc; i++) {
        computeDenomFlag(i) = (columnCoeffsCounter[i] >= NUM_AGREEING_COEFFS) ? TRUE : FALSE;
    }

    vector<int> columnsToComputeDenom;
    for (int i=0; i < mc; i++) {
        if (columnCoeffsCounter[i] == TRUE) {
            columnsToComputeDenom.push_back(i);
        }
    }

    /*
numColsComputeDenom = length(columnsToComputeDenom);
*/
    int numColsComputeDenom = columnsToComputeDenom.size();

    vector<int> id1;
    vector<double> mx1;
    Mat<double> denom;
    vector<double> vcoefs;

    // Large if statement
    if (numColsComputeDenom > 0) {
        //% Two-norm of the negative power coefficients for each column
        //[mx1,id1] = max(sqrt(sum(negPows(:,:,1:N/2-1).^2,3)),[],1);

        // mx1 IS NOT CORRECT. twoNorm IS NOT CORRECT
        //printf("before twoNorm: N = %d\n", N);
        mx1 = twoNorm(negPows, id1);

        //% Determine which column has the largest two-norm.
        //[mx2,id2] = max(mx1);
        double mx2;

        int id2=0;
        mx2 = mx1[0];

        for (int i=1; i <  mx1.size(); i++) {
            if (mx1[i] > mx2) {
                mx2 = mx1[i];
                id2 = i;
            }
        }

        //% Use the column with the largest two norm to determine the rank of the
        //% Hankel matrix and thus the degree of the denominator of the rational
        //% function.
        //Htilde = hankel(squeeze(negPows(id1(id2),id2,1:N/2-1)));
        //rnk = 2;

        int id12 = id1[id2];
        //printf("id12= %d\n", id12);

        vector<double> negPowsSq;
        for (int k=0; k < (N/2-1); k++) {
            negPowsSq.push_back(negPows(id12, id2, k));
        }
        mat Htilde = hankel(negPowsSq);
        int rnk = 2;

        //Htilde.print("Htilde");
        //exit(0);

#if 0
        % Starting at the upper right corner of the Htilde matrix, extract
   		% larger and large square submatrices and compute their SVD.  Stop doing
   		% this once a "near zero" singular value has been detected.  When this
   		% happens we have found the correct degree of the denominator.
   		while rnk < N/2-1
      		S = svd(Htilde(1:rnk,1:rnk))/mx2;
        if S(rnk) < 1e-5
                break
      		end
      		rnk = rnk + 1;
        end
#endif

   		while (rnk < (N/2-1)) {
            colvec s = svd(Htilde.submat(0,0,rnk-1,rnk-1)) / mx2;
            if (s(rnk-1) < 1.e-5) {
                break;
            }
            rnk = rnk + 1;
        }

#if 0
        if rnk == N/2-1
      		warning('Rational part of expansion, most likely not calculated correctly');
        end
#endif

		if (rnk == (N/2-1)) {
            printf("Rational part of expansion, most likely not calculated correctly\n");
        }

#if 0
        % Now take the SVD of the H matrix to determine the right null vector and
                % the coefficients for the denominator of the rational function.
                [U,S,V] = svd(Htilde(1:rnk,1:rnk-1));
        svdcoefs = U(1:rnk,rnk);
        % Denominator of the rational function
                denom = polyval2(svdcoefs,ep); // ep is epsilon
        else
            denom = 1;
        end
#endif

		mat U, V;
        colvec S;
        mat hsub = Htilde.submat(0,0,rnk-1,rnk-2);
        svd(U,S,V, hsub);
        mat svdcoefs = U.submat(0,rnk-1,rnk-1,rnk-1);
	
        //printSize(svdcoefs, "svdcoefs");
	
        for (int i=0; i < svdcoefs.n_elem; i++) {
            vcoefs.push_back(svdcoefs(i));
        }

        // Denominator of the rational function
        // EVALATED FOR EACH epsilon
        denom = polyval2(vcoefs,ep);

        //printf("APPEARS (GE) to be an error: regarding memory allocation\n");
        //exit(0);

    } else {
        denom = 1. * ones<Mat<double> >(1,ep.n_elem);
    }
    //denom.print("denom"); // seems OK

    //----------------------------------------------------------------------
    if (numColsComputeDenom > 0) {
        vector<CMPLX> cx_vcoefs;
        for (int i=0; i < vcoefs.size(); i++) {
            cx_vcoefs.push_back(vcoefs[i]);
        }

        vector<CMPLX> res = polyval2(cx_vcoefs,z);
        //print(res,"res_polyval2"); // OK

        //printf("mg,mc= %d, %d\n", mg, mc);
        //printf("numCoeffs= %d\n", numCoeffs);
        //printSize(*(C[0]), "size(*C)");

	//vector<cx_mat*> C(numCoeffs);
        //printf("size(C) = %d\n", C.size());
        //printSize(*(C[0]), "*C[0].size");

        //ArrayT<CMPLX> coeffs(mg,mc,numCoeffs);

        for (int j=0; j < mc; j++) {
            for (int i=0; i < mg; i++) {
                for (int k=0; k < numCoeffs; k++) {
                    //coeffs(i,j,k) = (*C[k])(i,j)*res[k];
                    (*C[k])(i,j) = (*C[k])(i,j)*res[k];
		}
            }}
	//coeffs.printcx("coeffs"); // OK
        //exit(0);


	// I expected M and numCoefss to be equal) (numCoefs = N/4+1)

	M = C.size()-1;
	ArrayT<double> negPows2(mg,mc,M-1);
	ArrayT<double> posPows2(mg,mc,M+1);
        computeLaurentCoeffs(C,rad, negPows2, posPows2);

        // I'd like: posPows.resize(n,m,k)
        // if a dimension is smaller, drop the value. If larger, either initialize to zero or leave uninitialized
        ArrayT<double> posPows3(mg,mc,2*(numCoeffs-1));
        //posPows.print("before posPows");
        posPows.resize(mg,mc,2*(numCoeffs-1));
        //posPows.print("posPows");
        //exit(0);


        //posPows3.setTo(0.);

        for (int j=0; j < mc; j++) {
            for (int i=0; i < mg; i++) {
   		for (int k=0; k < numCoeffs; k++) {
                    posPows(i,j,k) = posPows2(i,j,k);
		}
            }}
        //for (int i=0; i < numCoeffs; i++) {}
        //for (int i=0; i < numCoeffs; i++) {}
        // ERROR:
        // grady(52292) malloc: *** error for object 0x80e400: incorrect checksum for freed object - object was probably modified after being freed.

        //negPows2.print("negPows2"); // OK
        //posPows2.print("posPows2"); // OK
        //exit(0);
	//ArrayT<double> negPows(mg,mc,M-1);
	//ArrayT<double> posPows(mg,mc,M+1);

        //GE (error somewhere in these lines
#if 1

	//const int* ineg2 = negPows2.getDims();
	//const int* ipos = posPows.getDims();
	//printf("ineg2= %d, %d, %d\n", ineg2[0], ineg2[1], ineg2[2]);
	//printf("ipos= %d, %d, %d\n", ipos[0], ipos[1], ipos[2]);
	//printf("numCoeffs= %d\n", numCoeffs); // 33 as expected
	// 2*(numCoeffs-1) = 2*(32) = 64 (too high in problematic line!!)

	for (int j=0; j < mc; j++) {
            for (int i=0; i < mg; i++) {
		for (int k=numCoeffs; k < 2*(numCoeffs-1); k++) {
#if 0
                    // MUST IMPLEMENT THIS LINE. Last line I believe
                    posPows(:,:,numCoeffs+1:end) = flipdim(negPows2,3);
#endif
                    int l1 = k-numCoeffs;
                    int l2 = M-1-1 - (k-numCoeffs);
                    // k goes to 63, but 3rd dimension of posPows is only 33
                    //printf("k=%d, l2= %d\n", k, l2);
                    // I must increase the size of posPows
                    posPows(i,j,k) = negPows2(i,j,l2);   // PROBLEM WITH THIS LINE
		}
            }}
	// Replace posPows by posPows3


	//posPows.print("gordon last stage\n");
	// ALMOST DONE: Aug. 21, 2009
        //GE
#endif
    }
    // REACHED THIS POINT JUST STILL DEBUG
    //printf("GE 1\n");

    //clear Htilde;

    //% C will contain the entries of the C-matrix at each epsilon value.
    //C = zeros(mg,mc,length(epsilon));
    //printf("ep.n_elem = %d\n", ep.n_elem);
    ArrayT<CMPLX> Ceps(mg,mc,ep.n_elem);
    CMPLX zero(0.,0.);
    Ceps.setTo(zero);

    fd_coeffs.set_size(mg*mc, ep.n_elem);

#if 0
    % Loop through each entry in the C matrix computing the results of the
            % Contour-SVD method.
            for jj = 1:mc
                     for ii = 1:mg
                              C(ii,jj,:) = polyval2(squeeze(posPows(ii,jj,:)),epsilon)./denom;
    end
            end
#endif

            vector<double> vpospows;
    const int* sz = posPows.getDims();
    //printf("3rd dim of posPows = %d\n", sz[2]);
    vpospows.resize(sz[2]);

    //posPows.print("posRows"); // OK

    // ERROR: value of posPows not the same as in the Matlab code
    // MUST FIX THIS
    // But results from computeLaurent are correct (recheck)

    //denom.print("denom");
    //printf("n_elem= %d\n", ep.n_elem);
    //ep.print("epsilon");

    //	printf("sz[2] = %d (> 33?)\n", sz[2]); // always == 33

    for (int jj=0; jj < mc; jj++) {
        for (int ii=0; ii < mg; ii++) {
            for (int k=0; k < sz[2]; k++) {    // SHOULD extend to either size[2] of posPows or posPows3 (HOW TO FIGURE OUT)
                vpospows[k] = posPows(ii,jj,k);  // should sometimes use posPows and sometimes posPows3 (when pole)
            }
            //print(vpospows, "vpospows");
            // arguments must be vector and Mat of same type, in this case double.
            //print(vpospows, "polyval2_vpospows");
            Mat<double> v = polyval2(vpospows, ep);
            //print(v, "v from polyval2");
            //denom.print("denom");
            //printSize(denom, "denom");
            for (int k=0; k < ep.n_elem; k++) {
                Ceps(ii,jj,k) = v[k] / denom(0,k);
                fd_coeffs(ii+jj*mg, k) = v[k] / denom(0,k);
            }
        }
    }
    //Ceps.printcx("Ceps");
    //fd_coeffs.print("fd_coeffs");
    //exit(0);


#if 0
    % If there are any poles then compute there locations.
            if ( numColsComputeDenom > 0 )
                degreeDenom = 2*(rnk-1);
    temp = zeros(1,degreeDenom+1);
    temp(degreeDenom+1:-2:1) = svdcoefs;
    poles = sort(roots(temp));
    else
        poles = [];
    end
#endif

            if (numColsComputeDenom > 0) {
	// Calculate the poles
	// I need a routine to calculate the roots of a polynomial
    }

    //% End contourSVD

    //exit(0);
}
//----------------------------------------------------------------------
mat ContourSVD::rbffdapp(double eps, mat& rd, mat& re)
        // relates to computation of derivative coefficients
{
    //rd.print("rd");
    //re.print("re");
    //printf("rbffdapp::eps= %f\n", eps);

    printf("rbffdapp::should not be in this routine\n");
    exit(0);

    rbf->setEpsilon(eps);
    mat vals = (*rbf)(rd);

    //printf("eps= %f\n", eps);
    vals.print("vals (WARNING! NOT SURE THIS ROUTINE SHOUD BE CALLED)");

    // replace by rbf->xderiv and rbf->yderiv for the appropriate methods
    mat vald = rbf->lapl_deriv(re);

    //rd.print("rd");
    //vald.print("vald");

    int nr = rd.n_rows;
    int nc = rd.n_cols;
    mat a(nr+1, nc+1);

    // enforce that Lapl(constant) = 0

    a.submat(0, 0, nr-1, nc-1) = vals;
    a.submat(0, nc, nr-1, nc) = ones<mat>(nr,1);
    a.submat(nr, 0, nr, nc-1) = ones<mat>(1,nc);
    a(nr, nc) = 0.;

    // I could also enforce that lapl(x) = lapl(y) = 0.
    // This would add two rows and 2 colums to the matrix "a" and
    //      one column to the matrix "fds"

    mat fds(1,nc+1);
    fds.submat(0,0,0,nc-1) = vald;
    fds(0,nc) = 0.;
    rowvec valrow = fds;

    //valrow.print("valrow");
    //a.print("a");
    //fds.print("fds");

    rowvec row = solver(valrow, a);

    return fds;
}
//----------------------------------------------------------------------
cx_mat ContourSVD::rbffdapp(CMPLX eps, cx_mat& rd, cx_mat& re)
        // relates to computation of derivative coefficients
{
    //printf("------------------------\n");
    //printf("inside complex rbff\n");
    //rd.print("rd");

    printf("rbffdapp(2)::should not be in this routine\n");
    exit(0);

    rbf->setEpsilon(eps);
    cx_mat vals = (*rbf)(rd);

    cx_mat vald = rbf->lapl_deriv(rd.row(0));

    int nr = rd.n_rows;
    int nc = rd.n_cols;
    cx_mat a(nr+1, nc+1);

    a.submat(0, 0, nr-1, nc-1) = vals;
    a.submat(0, nc, nr-1, nc) = ones<cx_mat>(nr,1);
    a.submat(nr, 0, nr, nc-1) = ones<cx_mat>(1,nc);
    a(nr, nc) = 0.;

    cx_mat fds(1,nc+1);
    fds.submat(0,0,0,nc-1) = vald;
    fds(0,nc) = 0.;

    cx_rowvec valrow = fds;

    cx_rowvec row = solver(valrow, a);

    return row.cols(0,nc-1);
}
//----------------------------------------------------------------------

mat ContourSVD::rbffdapp(double eps, mat& rd, ArrayT<Vec3>& re, const char* choice)
        // relates to computation of derivative coefficients
{
    //rd.print("rd");
    //re.print("re");
    //printf("double: rbffdapp::eps= %f, choice= %s\n", eps, choice);
    printf("rbffdapp(2)::should not be in this routine\n");
    exit(0);

    rbf->setEpsilon(eps);
    mat vals = (*rbf)(rd);

    //printf("eps= %f\n", eps);
    //vals.print("vals");

    // replace by rbf->xderiv and rbf->yderiv for the appropriate methods
    mat vald;

    if (strcmp(choice, "x") == 0) {
        // a matrix of distances will not work. Rather, I need a matrix of vectors, because I
        // need a direction
        vald = rbf->xderiv(re);
    } else if (strcmp(choice, "y") == 0) {
        vald = rbf->yderiv(re);
    } else if (strcmp(choice, "z") == 0) {
        vald = rbf->zderiv(re);
    } else if (strcmp(choice, "lapl") == 0) {
        vald = rbf->lapl_deriv(re);
    } else {
        printf("mat rbfdapp:: unknown derivative type\n");
        exit(1);
    }

    //rd.print("rd");
    //vald.print("vald");

    int nr = rd.n_rows;
    int nc = rd.n_cols;
    mat a(nr+1, nc+1);

    // enforce that Lapl(constant) = 0

    a.submat(0, 0, nr-1, nc-1) = vals;
    a.submat(0, nc, nr-1, nc) = ones<mat>(nr,1);
    a.submat(nr, 0, nr, nc-1) = ones<mat>(1,nc);
    a(nr, nc) = 0.;

    // I could also enforce that lapl(x) = lapl(y) = 0.
    // This would add two rows and 2 colums to the matrix "a" and
    //      one column to the matrix "fds"

    mat fds(1,nc+1);
    fds.submat(0,0,0,nc-1) = vald;
    fds(0,nc) = 0.;
    rowvec valrow = fds;

    //valrow.print("valrow");
    //a.print("a");
    //fds.print("fds");

    rowvec row = solver(valrow, a);

    return fds;
}
//----------------------------------------------------------------------
cx_mat ContourSVD::rbffdapp(CMPLX eps, cx_mat& rd, ArrayT<CVec3>& re, const char* choice)
        // relates to computation of derivative coefficients
        // choice: 'x', 'y', 'lapl'
{
    //printf("CMPLX: rbffdapp::eps= %f, %f, choice= %s\n", real(eps), imag(eps), choice);
    //exit(0);
    rbf->setEpsilon(eps);
    cx_mat vals = (*rbf)(rd);

    // replace by rbf->xderiv and rbf->yderiv for the appropriate methods
    cx_mat vald;

    if (strcmp(choice, "x") == 0) {
        vald = rbf->xderiv(re);
        //printf("rbff..., x deriv\n");
        //vald.print("vald(x)");
    } else if (strcmp(choice, "y") == 0) {
        vald = rbf->yderiv(re);
        //vald.print("vald(y)");
    } else if (strcmp(choice, "z") == 0) {
        vald = rbf->zderiv(re);
        //vald.print("vald(z)");
    } else if (strcmp(choice, "lapl") == 0) {
        vald = rbf->lapl_deriv(re);
    } else {
        printf("cx_mat rbfdapp:: unknown derivative type\n");
        exit(1);
    }

    // Add constraint: derivative of a constant = 0

    int nr = rd.n_rows;
    int nc = rd.n_cols;
    cx_mat a(nr+1, nc+1);

    a.submat(0, 0, nr-1, nc-1) = vals;
    a.submat(0, nc, nr-1, nc) = ones<cx_mat>(nr,1);
    a.submat(nr, 0, nr, nc-1) = ones<cx_mat>(1,nc);
    a(nr, nc) = 0.;

    cx_mat fds(1,nc+1);
    fds.submat(0,0,0,nc-1) = vald;
    fds(0,nc) = 0.;

    cx_rowvec valrow = fds;

	// TODO (EVAN): accelerate this solve. its many small solves
	// But we could do many small solves at the same time on a GPU
	// Or change this to an iterative solver or something
    cx_rowvec row = solver(valrow, a);

    return row.cols(0,nc-1);
}
//----------------------------------------------------------------------
rowvec ContourSVD::solver(rowvec& A, mat& B)
        // right / operator:  A * inv(B)
{

    mat c = solve(B, trans(A));
    if (!(c.n_cols > 0 && c.n_rows > 0)) {
        printf("\n!!!!!!!!!!!!!!\nERROR! Linear System could not be solved! Possibly singular!\n!!!!!!!!!!!!!!\n\n");
        cx_double dt = det(B);
        printf("det(B)= (%f,%f)\n", real(dt), imag(dt));
        exit(-10);
    }

    return trans(c);
}
//----------------------------------------------------------------------
cx_rowvec ContourSVD::solver(cx_rowvec& A, cx_mat& B)
        // right / operator:  A * inv(B)
{
    cx_mat ac = htrans(A); // htrans?

    cx_mat c = solve(htrans(B), ac);

    if (!(c.n_cols > 0 && c.n_rows > 0)) {
        printf("\n!!!!!!!!!!!!!!\nERROR! Linear System could not be solved! Possibly singular!\n!!!!!!!!!!!!!!\n\n");
        cx_double dt = det(B);
        printf("det(B)= (%f,%f)\n", real(dt), imag(dt));
        exit(-10);
    }

    // Determinant computes as zero, although it cannot be!!
    // cx_double dt = det(B);
    // wrong value of determinant of a complex matrix
    // printf("det(B)= (%f,%f)\n", real(dt), imag(dt));

    return htrans(c); // htrans?
}
//----------------------------------------------------------------------
template <class T>
        void ContourSVD::getSize(T& matr, int& nr, int& nc)
{
    nr = matr.n_rows;
    nc = matr.n_cols;
}
//----------------------------------------------------------------------
template <class T>
        void ContourSVD::printSize(T& matr, char* msg)
{
    int nr = matr.n_rows;
    int nc = matr.n_cols;
    if (msg == 0) {
        printf("size: %d, %d\n", nr, nc);
    } else {
        printf("%s: %d, %d\n", msg, nr, nc);
    }
}
//----------------------------------------------------------------------
void ContourSVD::computeLaurentCoeffs(vector<cx_mat*> C, double erad, ArrayT<double>& negPows, ArrayT<double>& posPows)
        // Should erad be complex or real? ****
{
    int mg,mc,M;
    M = C.size();
    getSize(*C[0], mg, mc);
    //printSize(*C[0], "*C[0]");
    //printf("M= %d\n", M);
    M--;

    vector<CMPLX> mval;
    vector<cx_mat*> mrep(M);
    //cx_mat tm(mg,mc);
    ArrayT<CMPLX> temp(mg,mc,M);

    //temp = (repmat(reshape(exp(4*pi*1i*((0:M-1))/(4*M)),[1 1 M]),...
    //                 [mg mc 1])).*C(:,:,1:M);

    for (int i=0; i < M; i++) {
        double theta = (double) i/M;
        CMPLX a = exp(pi*iota*theta);
        mval.push_back(a);
    }

    for (int k=0; k < M;  k++) {
	for (int j=0; j < mc; j++) {
            for (int i=0; i < mg; i++) {
		temp(i,j,k) = mval[k];
            }}}

    // Multiply temp (element by element) by C
    for (int k=0; k < M;  k++) {
	for (int j=0; j < mc; j++) {
            for (int i=0; i < mg; i++) {
		temp(i,j,k) *= (*C[k])(i,j);
            }}}

    //temp.printcx("temp");exit(0); // OK


#if 0
    % Prepare the matrix for the inverse FFT calculation.
            sC = zeros(mg,mc,M);
    sC(:,:,1) = 0.5*(real(C(:,:,1)) + 1i*real(C(:,:,1)));
    sC(:,:,2:M) = 0.5*(C(:,:,2:M)+conj(flipdim(C(:,:,2:M),3)) + ...
    1i*(temp(:,:,2:M)+conj(flipdim(temp(:,:,2:M),3))));
#endif

    ArrayT<CMPLX> sC(mg,mc,M);

    vector<cx_mat*> Cflip  = flipdim3(C, 1, M);

    ArrayT<CMPLX> tempflip = flipdim3(temp, 1, M);

    for (int j=0; j < mc; j++) {
	for (int i=0; i < mg; i++) {
            sC(i,j,0) = 0.5*(real((*C[0])(i,j)) + iota*real((*C[0])(i,j)));
            for (int k=1; k < M; k++) {
                // WHY IS C zero everywhere
                // 2nd term in first line is not correct!!! DO NOT FOLLOW.
                sC(i,j,k) = 0.5*((*C[k])(i,j) + conj((*Cflip[k-1])(i,j)))
                            + 0.5*iota*(temp(i,j,k) + conj(tempflip(i,j,k-1)));
            }}}

    //sC.printcx("sC");

    // ifft of sC along 3rd dimension

    /*
temp = ifft(sC,[],3);
*/

    // Slow way of doing it
    fftw_plan p;
    for (int j=0; j < mc; j++) {
	for (int i=0; i < mg; i++) {
            CMPLX si[M];
            CMPLX so[M];
            for (int k=0; k < M; k++) {
                si[k] = sC(i,j,k);
            }
            p = fftw_plan_dft_1d(M, reinterpret_cast<fftw_complex*>(&si[0]), reinterpret_cast<fftw_complex*>(&so[0]), FFTW_BACKWARD, FFTW_ESTIMATE);
            // GRADY: with M=32, you can probably do matrix-vector multiplication much faster than FFT,
            // especially on the GPU
            fftw_execute(p);
            double minv = 1. / M;
            for (int k=0; k < M; k++) {
                //printf("temp(0,0,%d)= (%f,%f)\n", k, real(so[k]), imag(so[k]));
                temp(i,j,k) = so[k] * minv;
            }
	}}

    // ifft is 10-11 significant digit accurate with respect to Octave
    //temp.printcx("temp");

    //p = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(sC), reinterpret_cast<fftw_complex*>(sCo), FFTW_BACKWARD, FFTW_ESTIMATE);
    // end result must be multiplied by (-1) to get the result that Grady gets
    fftw_destroy_plan(p);

    // check values of temp() array compared to what Grady gets at (i,j) = (0,0)

#if 0
    printf("M = %d, N = %d\n", M, N);
    for (int k=0; k < M; k++) {
        printf("temp(0,0,%d)= (%f,%f)\n", k, real(temp(0,0,k)), imag(temp(0,0,k)));
    }
    exit(0);
#endif

    /*
temp2 = repmat(real(1/2/M*C(:,:,M+1)),[1 1 M]);
*/
    ArrayT<double> temp2(mg,mc,M);
    for (int k=0; k < M; k++) {
	for (int j=0; j < mc; j++) {
            for (int i=0; i < mg; i++) {
		// off by 1024 = 32*32 from results from matlab *SVD*.m
		// Check values of C[M] (and C[M-1] . Something is wrong.  Fixed. 
		temp2(i,j,k) = (1./(2.*M))*real((*C[M])(i,j));
		//if (i == 0 && j == 0) { printf("temp2(0,0,%d)= %f\n", k, temp2(0,0,k)); }
            }}}

    //temp2.print("temp2");

    /*
% Extract the Laurent coefficients from the FFT results.

coeffs(:,:,1:2:2*M) = real(temp) + temp2;
coeffs(:,:,2:2:2*M) = imag(temp) - temp2;
*/
    ArrayT<double> coeffs = ArrayT<double>(mg,mc,2*M);
    coeffs.setTo(0.);

    // SOMETHING IS WRONG!!! Fixed.
    // temp is correct on its own
    //  temp2 is ok on its own. temp2 appears to be constant
    for (int k=0; k < M; k++) {
	for (int j=0; j < mc; j++) {
            for (int i=0; i < mg; i++) {
		coeffs(i,j,2*k)   = real(temp(i,j,k)) + temp2(i,j,k);
		coeffs(i,j,2*k+1) = imag(temp(i,j,k)) - temp2(i,j,k);
            }}}
    //	coeffs.print("coeffs"); // accurate to 10 signif. digits


    /*
% To get the actual Laurent coefficients we need to scale the values
% from the inverse fft by the appropriate power of the radius
% used to compute the function values around the circle in the 
% complex epsilon plane.
negPows = coeffs(:,:,2:M).*...
      repmat(reshape(erad.^(2*(1:M-1)),[1 1 M-1]),[mg mc 1]);
posPows = coeffs(:,:,[1 2*M:-1:M+1]).*...
      repmat(reshape(1./erad.^(2*(0:M)),[1 1 M+1]),[mg mc 1]);
end
*/

    // ADD print of COMPLEX NUMBERS to ArrayT?
    vector<double> neg_pow, pos_pow;

    for (int k=0; k < M-1; k++) {
        neg_pow.push_back( pow(erad, 2.*(k+1)) );
    }

    for (int k=0; k < M+1; k++) {
        pos_pow.push_back( pow(erad, -2.*k) );
    }

    for (int k=0; k < M-1; k++) {
	for (int i=0; i < mg; i++) {
            for (int j=0; j < mc; j++) {
		negPows(i,j,k) = coeffs(i,j,k+1)*neg_pow[k]; 
            }}}

    //for (int k=0; k < M+1; k++) {
    int k=0;
    for (int i=0; i < mg; i++) {
	for (int j=0; j < mc; j++) {
            posPows(i,j,k) = coeffs(i,j,0)*pos_pow[0];
	}} //}

    for (int k=1; k < M+1; k++) {
	for (int i=0; i < mg; i++) {
            for (int j=0; j < mc; j++) {
		posPows(i,j,k) = coeffs(i,j,2*M-k)*pos_pow[k]; // WRONG?
            }}}

    //posPows.print("posPows");

    // FINISH THIS PART
}
//----------------------------------------------------------------------
template <class T>
        vector<T*> ContourSVD::flipdim3(vector<T*>& C, int i1, int i2)
{
    //printf("enter flipdim3(vector)\n");
    int sz = C.size();
    //printf("C.size= %d\n", sz);
    vector<T*> v;

    //printf("i1,i2= %d, %d\n", i1, i2);

    int k1 = (i1 == -1) ? 0  : i1; // check that k1 is within bounds
    int k2 = (i2 == -1) ? sz : i2; // check that k1 is within bounds

    //printf("flipdim3(vector): k1,k2= %d, %d ", k1, k2);

    v.resize(k2-k1);
    //printf("v.size()= %d\n", v.size());

    for (int k=k1; k < k2; k++) {
        v[k-k1] = C[k1+k2-k-1];
    }

    return v;
}
//----------------------------------------------------------------------
template <class T>
        ArrayT<T> ContourSVD::flipdim3(ArrayT<T>& C, int i1, int i2)
{
    const int* sz = C.getDims();
    //printf("flipdim::sz= %d, %d, %d\n", sz[0], sz[1],sz[2]);

    int k1 = (i1 == -1) ? 0  : i1; // check that k1 is within boundes
    int k2 = (i2 == -1) ? sz[2] : i2; // check that k1 is within boundes

    //printf("flipdim3(ArrayT): k1,k2= %d, %d ", k1, k2);
    ArrayT<T> v(sz[0], sz[1], k2-k1);

    int count=0;
    for (int k=k1; k < k2; k++) {
        count++;
	for (int j=0; j < sz[1]; j++) {
            for (int i=0; i < sz[0]; i++) {
		//printf("i,j,k= %d, %d, %d\n", i, j, k);
		v(i,j,k-k1) = C(i,j,k1+k2-k-1);
            }}}
    //printf("nb times in k loop: %d\n", count);

    return v;
}
//----------------------------------------------------------------------
template <class T>
        ArrayT<double> ContourSVD::valsAgree(ArrayT<T>& val1, ArrayT<T>& val2, double relError)
{
    //val1.print("val1");
    //val2.print("val2");
    /*
function result = valsAgree(val1,val2,relError)

[mv1 nv1] = size(val1);
[mv2 nv2] = size(val2);

% GE: WHAT IF BOTH ARE SCALARS? 

% Check to see if either val1 or val2 is a scalar.
if ( (mv1 == 1) & (nv1 == 1) )
   val1 = repmat(val1,mv2,nv2);
end

if ( (mv2 == 1) & (nv2 == 1) )
   val2 = repmat(val2,mv1,nv1);
end
*/
    int mv1, nv1, mv2, nv2;
    const int* sz1;
    const int* sz2;

    sz1 = val1.getDims();
    sz2 = val2.getDims();

    mv1 = sz1[0];
    nv1 = sz1[1];
    mv2 = sz2[0];
    nv2 = sz2[1];

    // One of the two dimensions must be 1
    //printf("mv1,nv1= %d, %d\n", mv1, nv1);
    //printf("mv2,nv2= %d, %d\n", mv2, nv2);

    if ((mv1 == 1 && nv1 == 1) || (mv2 == 1 && nv2 == 1)) {
        printf("valsAgree: one of the input matrices is a scalar\n");
        printf("not allowed\n");
        exit(1);
    }

    if ((mv1 != mv2) || (nv1 != nv2)) {
        printf("valsAgree: input matrix dimensions do not agree\n");
        exit(1);
    }

    /*
% Compute the relative error.
% max(2D array) returns a row vector (max over each column)
result = abs(val1-val2)./max(abs(val1),abs(val2));
// ERROR unless val1 and val2 are 1D arrays (ASK GRADY)
*/

    //val1.print("val1");
    //val2.print("val2");
    //exit(0);

    ArrayT<double> result(mv1, nv1);
    for (int j=0; j < nv1; j++) {
	for (int i=0; i < mv1; i++) {
            //double tmp = abs(val1(i,j)-val2(i,j)) / max(abs(val1(i,j)),abs(val2(i,j)));
            double tmp = abs(val1(i,j)-val2(i,j)) / max(abs(val1(i,j)),abs(val2(i,j)));
            result(i,j) = tmp;
	}}

    //result.print("gordon_result");

    /*
% Remove any NaN's since these correpond to an entry in val1 and val2 being
% both zero or at least one equal to inf.
nanIndex = find(isnan(result));

% Make all the NaN entries result matrix equal to zero if they come from two
% corresponding entries being equal to zero.
if ( length(nanIndex) > 0 )
   result(nanIndex(find( (val1(nanIndex) == 0) & (val2(nanIndex) == 0) ))) = 0;
end
*/

    /*
result = result < relError;
end
end
*/

    //printf("relError= %g\n", relError);

    for (int j=0; j < nv1; j++) {
	for (int i=0; i < mv1; i++) {
            result(i,j) = (result(i,j) < relError) ? TRUE : FALSE; // TRUE == 1
	}}

    return result;
}
//----------------------------------------------------------------------
template <class T>
        void  repmat(ArrayT<T>& a, int mv1, int mv2)
        //val2 = repmat(val2,mv1,nv1);
{
}
//----------------------------------------------------------------------
#if 0
template <class T>
        vector<int> ContourSVD::find(T in)
        // return indices of umat that are non-zero
{
    // should check that matrix is 1D (1 column)

    vector<int> v;

    for (int i=0; i < in.n_elem; i++) {
        if (in(i)) {
            v.push_back(i);
        }
    }
    return v;
}
#endif
//----------------------------------------------------------------------
#if 0
-- Built-in Function:  squeeze (X)
        Remove singleton dimensions from X and return the result.  Note
        that for compatibility with MATLAB, all objects have a minimum of
        two dimensions and row vectors are left unchanged.

        squeeze is a built-in function

#endif
#if 0
        template <class T>
        void ContourSVD::squeeze(Arrayt<T> arr)
{
    const int* sz = arr.getDims();

    if (sz[0] == 1 && sz[1] == 1) {
    } else if (sz[0] == 1 && sz[1] != 1) {
	else if (sz[0] != 1 && sz[1] == 1) 
	}
        }
#endif
            //----------------------------------------------------------------------
            vector<double> ContourSVD::twoNorm(ArrayT<double>& negPows, vector<int>& max_index)
            {
                //printf("enter twoNorm, N = %d\n", N);
                //% Two-norm of the negative power coefficients for each column
                //[mx1,id1] = max(sqrt(sum(negPows(:,:,1:N/2-1).^2,3)),[],1);

                int nr, nc;
                const int* sz = negPows.getDims();
                nr = sz[0];
                nc = sz[1];
                //printf("nr,nc= %d, %d\n", nr, nc);
                ArrayT<double> res(nr, nc);

                max_index.resize(nc);
                vector<double> mx;
                mx.resize(nc);

                double sum;
                //printf("N/2-1 = %d\n", N/2-1);

                //negPows.print("twoNorm::netPows"); // OK

                // ATTENTION!!! negPows should not be CMPLX, but probably real or integer?

                for (int j=0; j < nc; j++) {
                    for (int i=0; i < nr; i++) {
                        sum = 0.;
                        for (int k=0; k < N/2-1; k++) {
                            double nn = negPows(i,j,k); // CHECK THIS
                            //printf("k=%d, negPows= %f\n", k, nn);
                            sum += nn*nn;
                        }
                        res(i,j) = sqrt(sum);
                        //printf("i,j= %d,%d, sum= %f\n", i,j, sum);
                        //exit(0);
                    }}

                for (int j=0; j < nc; j++) {
                    mx[j] = 0.;
                    for (int i=0; i < nr; i++) {
                        if (res(i,j) > mx[j]) {
                            mx[j] = res(i,j);
                            max_index[j] = i;
                        }
                    }}

                return mx;
            }
            //----------------------------------------------------------------------
            template<class T>
            ArrayT<T> ContourSVD::hankel(vector<T> v)
            {
                int sz = v.size();
                ArrayT<T> hank(sz, sz);

                for (int j=0; j < sz; j++) {
                    for (int i=0; i < sz; i++) {
                        if ((i+j) < sz) {
                            hank(i,j) = (i+j) < sz ? v[i+j] : 0.;
                        }
                    }}

                return hank;
            }
            //----------------------------------------------------------------------
            mat ContourSVD::hankel(vector<double> v)
            {
                int sz = v.size();
                mat hank(sz, sz);

                for (int j=0; j < sz; j++) {
                    for (int i=0; i < sz; i++) {
                        if ((i+j) < sz) {
                            hank(i,j) = (i+j) < sz ? v[i+j] : 0.;
                        }
                    }}

                return hank;
            }
            //----------------------------------------------------------------------
            template <class T>
                    Mat<T> ContourSVD::polyval2(vector<T> p, Mat<T> x)
                    // x: usually array of epsilon (\phi(||\epsilon r||))
            {
#if 0
                function y = polyval2(p,x)
                             [m n] = size(x);
                % Do the computation for general case where x is an array
                        y = zeros(m,n);
                x = x.^2;
                for j=length(p):-1:1
                      y = x.*y + p(j);
                end
                        end
                        % End polyval2
#endif

                        int m, n;
		getSize(x, m, n);
		//printSize(x, "polyval2, size(x)");

		Mat<T> y = zeros<Mat<T> >(m,n); // WRONG???
		x = x % x; // elementwise multiplication

		//x.print("polyval2 x");
		//print(p, "polyval2_p"); exit(0);

		for (int j=p.size(); j > 0; j--) {
		    //printf("p[%d]= %f\n", j, p[j-1]);
                    y = x % y + p[j-1];
		}

		//y.print("polyval2::y");

		return y;
            }
            //----------------------------------------------------------------------
            template <class T>
                    vector<T> ContourSVD::polyval2(vector<T> p, vector<T> x)
            {
		int m;
		m = x.size();

		vector<T> y;
		y.resize(m);

		for (int i=0; i < m; i++) {
                    y[i] = 0.0;
		}

		for (int i=0; i < x.size(); i++) {
                    x[i] = x[i] * x[i]; // elementwise multiplication
		}

		for (int j=p.size(); j > 0; j--) {
                    for (int s=0; s < m; s++) {
                        y[s] = x[s]*y[s] + p[j-1];
                    }
		}

		return y;
            }
            //----------------------------------------------------------------------
            template <class T>
                    void ContourSVD::print(vector<T> v, const char* msg)
            {
                const char *txt = (msg == 0) ? "" : msg;
                for (int i=0; i < v.size(); i++) {
                    printf("%s, vec[%d]= %g\n", msg, i, (double) v[i]);
                }
            }
            //----------------------------------------------------------------------
            void ContourSVD::print(vector<CMPLX> v, const char* msg)
            {
                const char *txt = (msg == 0) ? "" : msg;
                for (int i=0; i < v.size(); i++) {
                    printf("%s, vec[%d]= (%g,%g)\n", msg, i, real(v[i]), imag(v[i]));
                }
            }
            //----------------------------------------------------------------------
            void ContourSVD::print(cx_mat& v, const char* msg)
            {
                const char *txt = (msg == 0) ? "" : msg;
                for (int j=0; j < v.n_cols; j++) {
                    for (int i=0; i < v.n_rows; i++) {
                        printf("%s(%d,%d)= (%21.14g,%21.14g)\n", msg, i, j, real(v(i,j)), imag(v(i,j)));
                    }}
            }
            //----------------------------------------------------------------------
            void ContourSVD::print(mat& v, const char* msg)
            {
                const char *txt = (msg == 0) ? "" : msg;
                for (int j=0; j < v.n_cols; j++) {
                    for (int i=0; i < v.n_rows; i++) {
                        printf("%s(%d,%d)= %21.14g\n", msg, i, j, (double) v(i,j));
                    }}
            }
            //----------------------------------------------------------------------
            template <class T>
                    void ContourSVD::print(Mat<T>& v, const char* msg)
            {
                const char *txt = (msg == 0) ? "" : msg;
                for (int j=0; j < v.n_cols; j++) {
                    for (int i=0; i < v.n_rows; i++) {
                        printf("%s(%d,%d)= %21.14g\n", msg, i, j, (double) v(i,j));
                    }}
            }
            //----------------------------------------------------------------------
