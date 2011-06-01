#ifndef _STENCILS_H_
#define _STENCILS_H_

#include <stdio.h>
#include <armadillo>
#include <math.h>
#include <Vec3.h>
#include "rbffd/rbfs/rbf.h"
#include "contour_svd.h"


// Rewrite of Matlab/Octave version written by Grady Wright. Code obtained August 14, 2009
//----------------------------------------------------------------------
class Stencils
{
    private:
        double rad;
        double h;   // The separation between nodes (usually stencil's avg radius)
        double eps;
        arma::mat rd2; // distance matrix(xd.n_rows,xd.n_rows);
        ArrayT<Vec3> rdvec; // distance matrix(xd.n_rows,xd.n_rows);
        RBF* rbf;
        ContourSVD* svd;
        arma::mat* xd;  // points
        const char* choice;

    public:
        /// choice : "lapl", "x", "y", "z"
        Stencils(RBF* rbf_, double rad_, double h_, double eps_, arma::mat* xd_, 
                const char* choice_) {
            this->choice = choice_;
            this->eps = eps_; // I should not require epsilon at this stage
            this->rad = rad_; // will be overwritten
            this->h = h_;
            this->rbf = rbf_;
            // XD are the SAMPLE points (physical coordinates)
            this->xd = xd_;
            //this->xd->print("Stencil::xd = ");
            // printf("Stencil::rad = %f\t Stencil::eps = %f\n", this->rad, this->eps);
            this->choice = choice;
            //rd2 = new arma::mat(xd->n_rows, xd->n_cols);
            //if (strcmp(choice, "lapl") == 0) {
            // RD2 is a DISTANCE Matrix (without RBF evaluation)
            rd2 = computeDistMatrix2(*xd, *xd); // unnormalized
            double maxStencilDist = sqrt(matMax(rd2));
            //} else {
            // RDVEC is the partial DISTANCE matrix where each entry is a Vec3(x-x_0, y-y0, z-z0)
            rdvec = computeDistMatrixVec(*xd, *xd); // unnormalized
            //rdvec.printcx("RDVEC");
            const int* dims = rdvec.getDims();
            //printf("before rdvec size: %d, %d, %d\n", dims[0], dims[1], dims[2]);
            rdvec.resize(1,dims[1]); // only keep first row
            const int* dims2 = rdvec.getDims();
            //printf("after rdvec size: %d, %d\n", dims2[0], dims2[1]);
            //}
            

            //EFB052611: removed this because my rbffd class seems to have a
            //better choice of rad to get the most out of the stencils. I found
            //that for 41x41 this expression: 
            //              double rad = 1. / ( 0.100 * grid_ref.getStencilRadius(st_indx));  
            // works great. However, I am concerned that my rad depends on
            // physcial domain size (ie. i get different derivative accuracy
            // when only the domain extents change ([-1,1]x[-1,1] is worse than
            // [-10,10]x[-10,10]).  Also, I found that my derviative tests fail
            // on MOST of the reasonable choices I make for rad and eps. I am
            // goign to continue writing the paper and optimizing with teh GPU,
            // but at some point (probably this summer) ill have have to come
            // back and drastically improve the accuracy of the solution.
#if 0
            if (rad > 1./maxStencilDist) {
                printf("Error: The radius (%1.3e) is too large, it needs to be < %1.3e (1/%f)\n", rad_, 1./maxStencilDist, maxStencilDist);
                printf("Setting radius to 0.9*%1.3e (= %1.3e)\n", 1./maxStencilDist,  0.9/maxStencilDist);
                this->rad = 0.9 / maxStencilDist;
            }
#endif 
            // I need a way to compute the radius automatically

            arma::rowvec eps_norm(1); 
            eps_norm(0) = eps / rad;  // normalized
//            eps_norm.print("EPS");
            //printf("rad, eps= %f, %f\n", this->rad, eps);

            arma::mat matr = rd2.row(0); 
            arma::mat rr0_norm = rd2 * (rad*rad); // normalized

            ArrayT<Vec3> rdvec_norm = rdvec * rad;  // each element is only proportional to r
            const int* dims1 = rdvec_norm.getDims();
            //printf("dims1(rdvec_norm)= %d, %d, %d\n", dims1[0], dims1[1], dims1[2]);
            //exit(0);

            arma::mat rrd_norm = matr * (rad*rad); // normalized
            // Constructor 3/3 for contoursvd
            // rbf = this.rbf = RBF choice class (i.e., RBF_MQ for multiquadric)
            // td2 = this.rd2 = Squared distance matrix (||x-xi||^2)
            // ep = this.eps_norm = RBF support parameter divided by stencil radius (eps/rad)
            // rr0 = this.rr0_norm = Squared distance matrix scaled by the radius squared
            // rrd = this.rrd_norm = Squared distance vector for separation
            //          between stencil nodes and stencil center Vec(||x-xi||^2)
            // rrdvec_ = this.rdvec_norm = Vec3 vectors representing separation
            //      between stencil nodes and the stencil center (x-xi) and scaled
            //      by the radius
            //      NOTE: this is NOT the same as rrd_norm (as in
            //      Vec(||x-xi||^2)); it is Vec(Vec3(x-xi));
            // rad = this.rad/this.rad = 1. (presumably because all other
            //      parameters were scaled by radius) {????}
            //svd = new ContourSVD(rbf, rd2, eps_norm, rr0_norm, rrd_norm, rad/rad);
            //EFB (original): 052311

            //rdvec_norm.printcx("RDVEC_NORM: ");
            svd = new ContourSVD(rbf, rd2, eps_norm, rr0_norm, rrd_norm, rdvec_norm, rad/rad);
            //    svd = new ContourSVD(rbf, rd2, eps_norm, rr0_norm, rrd_norm, rdvec_norm, rad/rad);
        }

        void setRad(double rad) { this->rad = rad; }
        void setEps(double eps) { this->eps = eps; }
        arma::mat computeDistMatrix2(arma::mat& x1, arma::mat& x2);
        ArrayT<Vec3> computeDistMatrixVec(arma::mat& x1, arma::mat& x2);

        arma::mat execute(int N) { 
            // TODO: no z coeffs output here.
            svd->setChoice(choice);
            svd->execute(N); 
            arma::mat coefs = svd->getFDCoeffs();                 
            //coefs /= coefs(1);
            // the only 2nd order operator
            if (strcmp(choice, "lapl") == 0) { 
                // EFB052311:
                // I talked to gordon about this scaling. The weights SHOULD be (-4, 1, 1, 1, 1) 
                // but they also need to be scaled by 1/h^2 (i.e., second centered formula). 
                // Now, if rad == 1/avg_stencil_radius ~= 1/h  (~= is approx.
                // equal) then this is correct. Just double check that RAD is
                // ALWAYS 1/h! If we choose rad too small then we might
                // underscale the weights. Not sure how much impact that will
                // have. If we over scale, since h is small we end up
                // increasing the magnitude of our weights very high and then
                // our derivs become inaccurate.: 
                // EFB052411 (want to play with rad)
                return coefs*rad*rad;
            } else {
                return coefs*rad;
            }
        }
        arma::mat computeCoefs(double eps);

        double matMax(arma::mat& x1) {
            const double mx = arma::max(arma::max(x1));
            return mx;
        }
};
#endif
//----------------------------------------------------------------------
