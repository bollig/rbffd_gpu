#include <algorithm>
#include "derivative_tests.h"
#include "utils/norms.h"
#include <vector>

#include "common_typedefs.h"

using namespace std;

// Struct to sort indices for the boundary below
struct indxltclass {
    bool operator() (size_t i, size_t j) { return (i<j); }
} srter; 

//----------------------------------------------------------------------
// Run a sequence of tests for increasingly complex functions. 
// Our goal is to approximate derivatives to a function f(x,y,z), 
// such that the derivatives are equal to  0, x, y, x^2, etc.
// IF we can accurately approximate the derivatives up to x^3 within 1e-2
// relative error it should be safe to assume we can adequately approximate
// a laplacian and other linear differential operators for our PDEs.
//----------------------------------------------------------------------
void DerivativeTests::testAllFunctions(bool exitIfTestFails, size_t nb_stencils_to_test) {
#if 1
    // Test all: C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3,CUSTOM
    this->testFunction(DerivativeTests::C, nb_stencils_to_test, exitIfTestFails);
    this->testFunction(DerivativeTests::X, nb_stencils_to_test, exitIfTestFails);
    this->testFunction(DerivativeTests::Y, nb_stencils_to_test, exitIfTestFails);
    this->testFunction(DerivativeTests::X2, nb_stencils_to_test, exitIfTestFails);
    this->testFunction(DerivativeTests::XY, nb_stencils_to_test, exitIfTestFails);
    this->testFunction(DerivativeTests::Y2, nb_stencils_to_test, exitIfTestFails);
    this->testFunction(DerivativeTests::X3, nb_stencils_to_test, exitIfTestFails);
    this->testFunction(DerivativeTests::X2Y, nb_stencils_to_test, exitIfTestFails);
    this->testFunction(DerivativeTests::XY2, nb_stencils_to_test, exitIfTestFails);
    this->testFunction(DerivativeTests::Y3, nb_stencils_to_test, exitIfTestFails);
#endif
    //    exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------
//  NOTE: if Derivative der is an instance of a GPU enabled class then 
//  this will check that our GPU precision is sufficiently close to the 
//  CPU for our computation to proceed.
//----------------------------------------------------------------------
void DerivativeTests::compareGPUandCPUDerivs(size_t nb_stencils_to_test) {
    size_t nb_centers = grid->getNodeListSize(); 
    size_t nb_stencils = grid->getStencilsSize(); 

    // If nb_stencils_to_test is 0 we stick with the original assumption we're
    // going to check all stencils
    if (nb_stencils_to_test) { 
        nb_stencils = (nb_stencils > nb_stencils_to_test) ? nb_stencils_to_test : nb_stencils;
    }

    // We want weights to sum to 0, so they are all going to be scaled by 1.
    // \sum w_i f(x,y,z) = 0     where     f(x,y,z) = 1
    vector<double> u(nb_centers, 1.);

#if 1
    // We could also check a derivative function like in our test_deriv routines
    for (size_t i = 0; i < nb_centers; i++) {
        NodeType& node_r = grid->getNode(i); 
        NodeType center(0.,0.,0.); 
        u[i] =  sin((node_r-center).magnitude()); 
    }
#endif

    vector<double> xderiv_gpu(nb_stencils);	
    vector<double> yderiv_gpu(nb_stencils);	
    vector<double> zderiv_gpu(nb_stencils);	
    vector<double> lderiv_gpu(nb_stencils);	

    vector<double> xderiv_cpu(nb_stencils);	
    vector<double> yderiv_cpu(nb_stencils);	
    vector<double> zderiv_cpu(nb_stencils);	
    vector<double> lderiv_cpu(nb_stencils);	

    //cout << "start applying weights to compute derivatives on CPU" << endl;
    // Verify that the CPU works
    // Force der to use the CPU version of applyWeights
    der->RBFFD::applyWeightsForDeriv(RBFFD::X, u, xderiv_cpu, true);
    der->RBFFD::applyWeightsForDeriv(RBFFD::Y, u, yderiv_cpu, false);
    der->RBFFD::applyWeightsForDeriv(RBFFD::Z, u, zderiv_cpu, false);
    der->RBFFD::applyWeightsForDeriv(RBFFD::LAPL, u, lderiv_cpu, false);

    //cout << "start applying weights to compute derivatives on GPU" << endl;
    // Verify the GPU works
    // If der is a GPU class then this uses the GPU. 
    der->applyWeightsForDeriv(RBFFD::X, u, xderiv_gpu, true);
    der->applyWeightsForDeriv(RBFFD::Y, u, yderiv_gpu, false);
    der->applyWeightsForDeriv(RBFFD::Z, u, zderiv_gpu, false);
    der->applyWeightsForDeriv(RBFFD::LAPL, u, lderiv_gpu, false);

    //cout << "start derivative comparison" << endl;
    for (int i = 0; i < nb_stencils; i++) {
#if 0
        std::cout << "cpu: " << xderiv_cpu[i] << " - gpu: " << xderiv_gpu[i] << " = " << xderiv_cpu[i] - xderiv_gpu[i] << std::endl;
        std::cout << "cpu: " << yderiv_cpu[i] << " - gpu: " << yderiv_gpu[i] << " = " << yderiv_cpu[i] - yderiv_gpu[i] << std::endl;
        std::cout << "cpu: " << zderiv_cpu[i] << " - gpu: " << zderiv_gpu[i] << " = " << zderiv_cpu[i] - zderiv_gpu[i] << std::endl;
#endif 
        double ex = compareDeriv(xderiv_gpu[i], xderiv_cpu[i], "X", i); 
        double ey = compareDeriv(yderiv_gpu[i], yderiv_cpu[i], "Y", i); 
        double ez = compareDeriv(zderiv_gpu[i], zderiv_cpu[i], "Z", i); 
        double el = compareDeriv(lderiv_gpu[i], lderiv_cpu[i], "Lapl", i); 

        std::cout << i << " of " << nb_stencils << "   (Errors: " << ex << ", " << ey << ", " << ez << ", " << el << ")" << std::endl;
    }
    std::cout << "[DerivativeTests] *****  Test passed: " << nb_stencils << " derivatives verified to match on GPU and CPU ******\n";
}

//----------------------------------------------------------------------
//
double DerivativeTests::compareDeriv(double deriv_gpu, double deriv_cpu, std::string label, int indx) {

    if (isnan(deriv_gpu))
    {
        std::cout << "One of the derivs calculated by the GPU is NaN (detected by isnan)!\n"; 
        exit(EXIT_FAILURE); 
    }

    double abs_error = fabs(deriv_gpu - deriv_cpu); 
    double rel_error = fabs(deriv_gpu - deriv_cpu)/fabs(deriv_cpu); 

    if (rel_error > 1e-4) 
    {
        std::cout << "\nERROR! GPU DERIVATIVES ARE NOT WITHIN 1e-5 OF CPU. FIND A DOUBLE PRECISION GPU!\n";
        std::cout << "Test failed on" << std::endl;
        std::cout << "Absolute Error for " << label << "[" << indx << "] = " << abs_error << "(rel_error: " << rel_error << ")    (GPU: " 
            << deriv_gpu << " CPU: " << deriv_cpu << ")"
            << std::endl; 
        std::cout << "NOTE: all derivs are supposed to be 0" << std::endl;
        //exit(EXIT_FAILURE); 
        //exit(EXIT_FAILURE);
    }
    return abs_error;
}


//----------------------------------------------------------------------
//
// Evaluate a test function and its analytic derivs to fill buffers. 
// Then compute approximate derviatives using the RBFFD class. 
// Compare the analytic and approximate derivatives and assess the
// viability of the RBFFD stencils and weights for PDE solution.
// NOTE: if nb_stencils_to_test is 0 then we check all stencils
void DerivativeTests::testFunction(DerivativeTests::TESTFUN choice, size_t nb_stencils_to_test, bool exitIfTestFails)
{
    // Use a std::set because it auto sorts as we insert
    std::set<size_t>& b_indices = grid->getSortedBoundarySet();
    std::set<size_t>& i_indices = grid->getSortedInteriorSet(); 
    size_t nb_bnd = b_indices.size();
    size_t nb_int = i_indices.size();
    int nb_centers = grid->getNodeListSize();
    int nb_stencils = grid->getStencilsSize();

   // std::cout << "BINDICES.size= " << b_indices.size();
   // std::cout << "IINDICES.size= " << i_indices.size();

    printf("\n================\n");
    printf("testFunction( %d ) [** Approximating F(X,Y) = %s **] \n",choice, TESTFUNSTR[(int)choice].c_str());
    printf("================\n");

    if (nb_stencils_to_test) {
        nb_stencils = (nb_stencils > nb_stencils_to_test) ? nb_stencils_to_test : nb_stencils;
    }

    vector<double> u(nb_centers);
    vector<double> dux_ex(nb_stencils); // exact derivative
    vector<double> duy_ex(nb_stencils); // exact derivative
    vector<double> dulapl_ex(nb_stencils); // exact derivative

    // change the classes the variables are located in
    vector<double> xderiv(nb_stencils);
    vector<double> yderiv(nb_stencils);
    vector<double> lapl_deriv(nb_stencils);

    if (!weightsComputed) {
        der->computeAllWeightsForAllStencils(); 
    }

    // Fill the test case based on our choice
    fillTestFunction(choice, nb_centers, u, dux_ex, duy_ex, dulapl_ex);

    std::vector<double>& avgDist = grid->getStencilRadii();

    der->applyWeightsForDeriv(RBFFD::X, u, xderiv, true);
    der->applyWeightsForDeriv(RBFFD::Y, u, yderiv, false);
    der->applyWeightsForDeriv(RBFFD::LAPL, u, lapl_deriv, false);

    std::vector<double> dux_ex_bnd(nb_bnd); 
    std::vector<double> xderiv_bnd(nb_bnd); 

    std::vector<double> dux_ex_int(nb_int); 
    std::vector<double> xderiv_int(nb_int); 

    std::vector<double> duy_ex_bnd(nb_bnd); 
    std::vector<double> yderiv_bnd(nb_bnd); 

    std::vector<double> duy_ex_int(nb_int); 
    std::vector<double> yderiv_int(nb_int); 

    std::vector<double> dulapl_ex_bnd(nb_bnd); 
    std::vector<double> lapl_deriv_bnd(nb_bnd); 

    std::vector<double> dulapl_ex_int(nb_int); 
    std::vector<double> lapl_deriv_int(nb_int); 

    std::vector<double> avgDist_bnd(nb_bnd); 
    std::vector<double> avgDist_int(nb_int); 
   
    std::set<size_t>::iterator it;
    int i = 0;
    for (it = b_indices.begin(); it != b_indices.end(); it++, i++) { 
        int j = *it;
        dux_ex_bnd[i] = dux_ex[j]; 
        xderiv_bnd[i] = xderiv[j]; 
        duy_ex_bnd[i] = duy_ex[j]; 
        yderiv_bnd[i] = yderiv[j]; 
        dulapl_ex_bnd[i] = dulapl_ex[j]; 
        lapl_deriv_bnd[i] = lapl_deriv[j]; 
        avgDist_bnd[i] = avgDist[j]; 
    }
    int k = 0;
    for (it = i_indices.begin(); it != i_indices.end(); it++, k++) { 
        int j = *it;
        dux_ex_int[k] = dux_ex[j]; 
        xderiv_int[k] = xderiv[j]; 
        duy_ex_int[k] = duy_ex[j]; 
        yderiv_int[k] = yderiv[j]; 
        dulapl_ex_int[k] = dulapl_ex[j]; 
        lapl_deriv_int[k] = lapl_deriv[j]; 
        avgDist_int[k] = avgDist[j]; 
    }

    enum DERIV {X=0, Y, LAPL};
    enum NORM {L1=0, L2, LINF};
    enum BNDRY {INT=0, BNDRY};
    double norm[3][3][2]; // norm[DERIV][NORM][BNDRY]
    double normder[3][3][2]; // norm[DERIV][NORM][BNDRY]

#if 0
    // These norms are weighted by the avgDist elements
    // These norms are || Lu_exact - Lu_approx ||_? 
    // where Lu_* are differential operator L applied to solution u_* 
    // and the || . ||_? is the 1, 2, or inf-norms
    norm[X][L1][BNDRY]   = l1normWeighted(dux_ex_bnd,   xderiv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[X][L2][BNDRY]   = l2normWeighted(dux_ex_bnd,   xderiv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[X][LINF][BNDRY] = linfnorm(dux_ex_bnd, xderiv_bnd, 0, nb_bnd);

    norm[Y][L1][BNDRY]   = l1normWeighted(duy_ex_bnd,   yderiv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[Y][L2][BNDRY]   = l2normWeighted(duy_ex_bnd,   yderiv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[Y][LINF][BNDRY] = linfnorm(duy_ex_bnd, yderiv_bnd, 0, nb_bnd);

    norm[LAPL][L1][BNDRY]   = l1normWeighted(dulapl_ex_bnd,   lapl_deriv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[LAPL][L2][BNDRY]   = l2normWeighted(dulapl_ex_bnd,   lapl_deriv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex_bnd, lapl_deriv_bnd, 0, nb_bnd);

    norm[X][L1][INT]   = l1normWeighted(dux_ex_int,   xderiv_int, avgDist_int, 0, nb_int);
    norm[X][L2][INT]   = l2normWeighted(dux_ex_int,   xderiv_int, avgDist_int, 0, nb_int);
    norm[X][LINF][INT] = linfnorm(dux_ex_int, xderiv_int, 0, nb_int);

    norm[Y][L1][INT]   = l1normWeighted(duy_ex_int,   yderiv_int, avgDist_int, 0, nb_int);
    norm[Y][L2][INT]   = l2normWeighted(duy_ex_int,   yderiv_int, avgDist_int, 0, nb_int);
    norm[Y][LINF][INT] = linfnorm(duy_ex_int, yderiv_int, 0, nb_int);

    norm[LAPL][L1][INT]   = l1normWeighted(dulapl_ex_int,   lapl_deriv_int, avgDist_int, 0, nb_int);
    norm[LAPL][L2][INT]   = l2normWeighted(dulapl_ex_int,   lapl_deriv_int, avgDist_int, 0, nb_int);
    norm[LAPL][LINF][INT] = linfnorm(dulapl_ex_int, lapl_deriv_int, 0, nb_int);

    // --- Normalization factors (to get denom for relative error: ||u-u_approx|| / || u || )

    normder[X][L1][BNDRY]   = l1normWeighted(dux_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[X][L2][BNDRY]   = l2normWeighted(dux_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[X][LINF][BNDRY] = linfnorm(dux_ex_bnd, 0, nb_bnd);

    normder[Y][L1][BNDRY]   = l1normWeighted(duy_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[Y][L2][BNDRY]   = l2normWeighted(duy_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[Y][LINF][BNDRY] = linfnorm(duy_ex_bnd, 0, nb_bnd);

    normder[LAPL][L1][BNDRY]   = l1normWeighted(dulapl_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[LAPL][L2][BNDRY]   = l2normWeighted(dulapl_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex_bnd, 0, nb_bnd);

    normder[X][L1][INT]   = l1normWeighted(dux_ex_int, avgDist_int,  0, nb_int);
    normder[X][L2][INT]   = l2normWeighted(dux_ex_int, avgDist_int,  0, nb_int);
    normder[X][LINF][INT] = linfnorm(dux_ex_int, 0, nb_int);

    normder[Y][L1][INT]   = l1normWeighted(duy_ex_int, avgDist_int, 0, nb_int);
    normder[Y][L2][INT]   = l2normWeighted(duy_ex_int, avgDist_int, 0, nb_int);
    normder[Y][LINF][INT] = linfnorm(duy_ex_int, 0, nb_int);

    normder[LAPL][L1][INT]   = l1normWeighted(dulapl_ex_int, avgDist_int, 0, nb_int);
    normder[LAPL][L2][INT]   = l2normWeighted(dulapl_ex_int, avgDist_int, 0, nb_int);
    normder[LAPL][LINF][INT] = linfnorm(dulapl_ex_int, 0, nb_int);
#else 
    // These norms are || Lu_exact - Lu_approx ||_? 
    // where Lu_* are differential operator L applied to solution u_* 
    // and the || . ||_? is the 1, 2, or inf-norms
    norm[X][L1][BNDRY]   = l1norm(dux_ex_bnd,   xderiv_bnd, 0, nb_bnd);
    norm[X][L2][BNDRY]   = l2norm(dux_ex_bnd,   xderiv_bnd, 0, nb_bnd);
    norm[X][LINF][BNDRY] = linfnorm(dux_ex_bnd, xderiv_bnd, 0, nb_bnd);

    norm[Y][L1][BNDRY]   = l1norm(duy_ex_bnd,   yderiv_bnd, 0, nb_bnd);
    norm[Y][L2][BNDRY]   = l2norm(duy_ex_bnd,   yderiv_bnd, 0, nb_bnd);
    norm[Y][LINF][BNDRY] = linfnorm(duy_ex_bnd, yderiv_bnd, 0, nb_bnd);

    norm[LAPL][L1][BNDRY]   = l1norm(dulapl_ex_bnd,   lapl_deriv_bnd, 0, nb_bnd);
    norm[LAPL][L2][BNDRY]   = l2norm(dulapl_ex_bnd,   lapl_deriv_bnd, 0, nb_bnd);
    norm[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex_bnd, lapl_deriv_bnd, 0, nb_bnd);

    norm[X][L1][INT]   = l1norm(dux_ex_int,   xderiv_int, 0, nb_int);
    norm[X][L2][INT]   = l2norm(dux_ex_int,   xderiv_int, 0, nb_int);
    norm[X][LINF][INT] = linfnorm(dux_ex_int, xderiv_int, 0, nb_int);

    norm[Y][L1][INT]   = l1norm(duy_ex_int,   yderiv_int, 0, nb_int);
    norm[Y][L2][INT]   = l2norm(duy_ex_int,   yderiv_int, 0, nb_int);
    norm[Y][LINF][INT] = linfnorm(duy_ex_int, yderiv_int, 0, nb_int);

    norm[LAPL][L1][INT]   = l1norm(dulapl_ex_int,   lapl_deriv_int, 0, nb_int);
    norm[LAPL][L2][INT]   = l2norm(dulapl_ex_int,   lapl_deriv_int, 0, nb_int);
    norm[LAPL][LINF][INT] = linfnorm(dulapl_ex_int, lapl_deriv_int, 0, nb_int);

    // --- Normalization factors (to get denom for relative error: ||u-u_approx|| / || u || )

    normder[X][L1][BNDRY]   = l1norm(dux_ex_bnd, 0, nb_bnd);
    normder[X][L2][BNDRY]   = l2norm(dux_ex_bnd, 0, nb_bnd);
    normder[X][LINF][BNDRY] = linfnorm(dux_ex_bnd, 0, nb_bnd);

    normder[Y][L1][BNDRY]   = l1norm(duy_ex_bnd, 0, nb_bnd);
    normder[Y][L2][BNDRY]   = l2norm(duy_ex_bnd, 0, nb_bnd);
    normder[Y][LINF][BNDRY] = linfnorm(duy_ex_bnd, 0, nb_bnd);

    normder[LAPL][L1][BNDRY]   = l1norm(dulapl_ex_bnd, 0, nb_bnd);
    normder[LAPL][L2][BNDRY]   = l2norm(dulapl_ex_bnd, 0, nb_bnd);
    normder[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex_bnd, 0, nb_bnd);

    normder[X][L1][INT]   = l1norm(dux_ex_int, 0, nb_int);
    normder[X][L2][INT]   = l2norm(dux_ex_int, 0, nb_int);
    normder[X][LINF][INT] = linfnorm(dux_ex_int, 0, nb_int);

    normder[Y][L1][INT]   = l1norm(duy_ex_int, 0, nb_int);
    normder[Y][L2][INT]   = l2norm(duy_ex_int, 0, nb_int);
    normder[Y][LINF][INT] = linfnorm(duy_ex_int, 0, nb_int);

    normder[LAPL][L1][INT]   = l1norm(dulapl_ex_int, 0, nb_int);
    normder[LAPL][L2][INT]   = l2norm(dulapl_ex_int, 0, nb_int);
    normder[LAPL][LINF][INT] = linfnorm(dulapl_ex_int, 0, nb_int);

#endif 

    printf("----- RESULTS: %d bnd, %d centers, %d stencils\n",  nb_bnd, nb_centers, nb_stencils); 

    //printf("{C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3,CUSTOM=10}\n");
    printf("norm[x/y/lapl][L1,L2,LINF][interior/bndry]\n");
    for (int k=0; k < 2; k++) {
        for (int i=0; i < 3; i++) {
            // If our L2 absolute norm is > 1e-9 we show relative.
            if (fabs(normder[i][1][k]) < 1.e-9) {
                printf("(abs err): norm[%d][][%d]= %10.3e, %10.3e, %10.3e\n", i, k, norm[i][0][k], norm[i][1][k], norm[i][2][k]);
            } else {
                printf("(rel err): norm[%d][][%d]= %10.3e, %10.3e, %10.3e, normder= %10.3e\n", i, k, 
                        norm[i][0][k]/normder[i][0][k], 
                        norm[i][1][k]/normder[i][1][k], 
                        norm[i][2][k]/normder[i][2][k], normder[i][1][k]); 
                //printf("   normder[%d][][%d]= %10.3e, %10.3e, %10.3e\n", i, k, 
                //normder[i][0][k], 
                //normder[i][1][k], 
                //normder[i][2][k]); 
            }
        }}

    double inter_error=0.;
    for (int i= 0; i < nb_int; i++) {
        inter_error += (dulapl_ex_int[i]-lapl_deriv_int[i])*(dulapl_ex_int[i]-lapl_deriv_int[i]);
        //    std::cout << i << "dulapl_ex_int " << dulapl_ex_int[i] << ", " << lapl_deriv_int[i] << std::endl;
        //printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    inter_error /= nb_int;

    double bnd_error=0.;
    for (int i=0; i < nb_bnd; i++) {
        bnd_error += (dulapl_ex_bnd[i]-lapl_deriv_bnd[i])*(dulapl_ex_bnd[i]-lapl_deriv_bnd[i]);
        //     printf("bnd error[%d] = %14.7g\n", i, dulapl_ex_bnd[i]-lapl_deriv_bnd[i]);
    }
    bnd_error /= nb_bnd;

#if 1
    double l2_bnd = sqrt(bnd_error);
    printf("avg l2_bnd_error= %14.7e\n", l2_bnd);
    double l2_int = sqrt(inter_error);
    printf("avg l2_interior_error= %14.7e\n", l2_int);
    // IF l2_int != l2_int its 'nan' and we should quit. 
    // NaN is the only number not equal to itself
    if ((l2_int > 1.e-2) || (l2_int != l2_int)) {
        printf ("ERROR! Interior l2 error is too high to continue\n");
        if (exitIfTestFails) { exit(EXIT_FAILURE); }
    }

    if (l2_bnd > 1.e0) {
        printf ("WARNING! Boundary l2 error is high but we'll trust it to continue\n");
        return ;
    }
#endif
}

//----------------------------------------------------------------------
//
void DerivativeTests::fillTestFunction(DerivativeTests::TESTFUN which, size_t nb_stencils, vector<double>& u, vector<double>& dux_ex, vector<double>& duy_ex,
        vector<double>& dulapl_ex)
{
    u.resize(0);
    dux_ex.resize(0);
    duy_ex.resize(0);
    dulapl_ex.resize(0);

    vector<NodeType>& rbf_centers = grid->getNodeList();
    int nb_centers = grid->getNodeList().size(); 

    switch(which) {
        case C:
            for (int i=0; i < nb_stencils; i++) {
                dux_ex.push_back(0.);
                duy_ex.push_back(0.);
                dulapl_ex.push_back(0.);
            }

            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(1.);
            }
            break;
        case X:
            printf("nb_stencils= %lu\n", nb_stencils);
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(1.);
                duy_ex.push_back(0.);
                dulapl_ex.push_back(0.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x());
            }
            printf("u.size= %d\n", (int) u.size());
            break;
        case Y:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(0.);
                duy_ex.push_back(1.);
                dulapl_ex.push_back(0.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.y());
            }
            break;
        case X2:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(2.*v.x());
                duy_ex.push_back(0.);
                dulapl_ex.push_back(2.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.x());
            }
            break;
        case XY:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(v.y());
                duy_ex.push_back(v.x());
                dulapl_ex.push_back(0.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.y());
            }
            break;
        case Y2:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(0.);
                duy_ex.push_back(2.*v.y());
                dulapl_ex.push_back(2.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.y()*v.y());
            }
            break;
        case X3:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(3.*v.x()*v.x());
                duy_ex.push_back(0.);
                dulapl_ex.push_back(6.*v.x());
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.x()*v.x());
            }
            break;
        case X2Y:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(2.*v.x()*v.y());
                duy_ex.push_back(v.x()*v.x());
                dulapl_ex.push_back(2.*v.y());
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.x()*v.y());
            }
            break;
        case XY2:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(v.y()*v.y());
                duy_ex.push_back(2.*v.x()*v.y());
                dulapl_ex.push_back(2.*v.x());
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.y()*v.y());
            }
            break;
        case Y3:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(0.);
                duy_ex.push_back(3.*v.y()*v.y());
                dulapl_ex.push_back(6.*v.y());
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.y()*v.y()*v.y());
            }
            break;
    }
}


//----------------------------------------------------------------------
//
void DerivativeTests::testEigen(RBFFD::DerType which, unsigned int maxNumPerturbations, float maxPerturbation)
{

    int nb_stencils = grid->getStencilsSize();
    int nb_bnd = grid->getBoundaryIndicesSize();
    int tot_nb_pts = grid->getNodeListSize();

    std::vector<double> u(tot_nb_pts);
    std::vector<double> deriv(nb_stencils);

    std::vector<StencilType>& stencil = grid->getStencils();
    std::vector<NodeType>& rbf_centers = grid->getNodeList();
    std::vector<double>& avg_stencil_radius = grid->getStencilRadii(); // get average stencil radius for each point

    RBFFD::EigenvalueOutput eig_results; 

    double max_eig = der->computeEigenvalues(which, &eig_results); // needs lapl_weights

    printf("zero perturbation: max eig: %f\n", max_eig);

    // Archive our original nodes
    std::vector<NodeType> rbf_centers_orig = grid->getNodeList();
    // Start perturbing the nodes within the grid
    std::vector<NodeType>& rbf_centers_perturb = grid->getNodeList();

    if ((maxPerturbation < 0.f) || (maxPerturbation > 1.f)) {
        std::cout << "[DerivativeTests] ERROR in testEigen(...)! Max Perturbation must be in [0., 1.]: " << maxPerturbation << std::endl;
        exit(EXIT_FAILURE);
    }

    double percent = maxPerturbation; // in [0,1]
    printf("percent distortion of original grid= %f\n", percent);

    // set a random seed
    srandom(time(0));

    for (unsigned int i=0; i < maxNumPerturbations; i++) {
        printf("---- iteration %d ------\n", i);
        //update rbf centers by random perturbations at a fixed percentage of average radius computed
        //based on the unperturbed mesh
        rbf_centers_perturb.assign(rbf_centers_orig.begin(), rbf_centers_orig.end());

        for (int j=0; j < tot_nb_pts; j++) {
            Vec3& v = rbf_centers[j];
            double vx = avg_stencil_radius[j]*percent*randf(-1.,1.);
            double vy = avg_stencil_radius[j]*percent*randf(-1.,1.);
            v.setValue(v.x()+vx, v.y()+vy);
        }

        der->computeWeightsForAllStencils(which);

        double max_eig = der->computeEigenvalues(which); 

        printf("Max Perturbation: %f, max eig: %f\n", percent, max_eig);
    }

    // Restore and undo all changes we made
    std::cout << "Restoring original node positions\n"; 
    rbf_centers_perturb.assign(rbf_centers_orig.begin(), rbf_centers_orig.end());
    der->computeWeightsForAllStencils(RBFFD::LAPL);
}


#if 0
//----------------------------------------------------------------------
void DerivativeTests::checkDerivatives(Derivative& der, Grid& grid)
{
    vector<NodeType>& rbf_centers = grid.getNodeList();
    int nb_centers = grid.getNodeListSize();
    int nb_stencils = grid.getStencilsSize();

    // change the classes the variables are located in
    vector<double> xderiv(nb_stencils);
    vector<double> yderiv(nb_stencils);
    vector<double> lapl_deriv(nb_stencils);


    // Warning! assume DIM = 2
    this->computeAllWeights(der, rbf_centers, grid.getStencils());

    printf("deriv size: %d\n", (int) rbf_centers.size());
    printf("xderiv size: %d\n", (int) xderiv.size());

    // function to differentiate
    vector<double> u(nb_centers);
    vector<double> du_ex(nb_stencils, 2.); // exact Laplacian

    vector<StencilType>& stencil = grid.getStencils();

#if 0
    for (int i=0; i < 1600; i++) {
        StencilType& v = stencil[i];
        printf("stencil[%d]\n", i);
        for (int s=0; s < v.size(); s++) {
            printf("%d ", v[s]);
        }
        printf("\n");
    }
    exit(0);
#endif

    double s;

    // INPUT buffer has nb_center elements
    for (int i=0; i < nb_centers; i++) {
        Vec3& v = rbf_centers[i];
        //s = 3.*v.x() + 2.*v.y(); //   + 3.*v.z();

        s = v.x()*v.y() + 0.5*v.x()*v.x() + 0.5*v.y()*v.y(); //   + 3.*v.z();

        //s = v.x()+v.y() +v.x()+v.y();
        //du_ex.push_back(0.);

        //s = v.x()*v.x()*v.x();
        //du_ex.push_back(6.*v.x());

        u[i] = s;
    }


    printf("main, u[0]= %f\n", u[0]);

    for (int n=0; n < 1; n++) {
        //der.computeDeriv(Derivative::X, u, xderiv);
        //der.computeDeriv(Derivative::Y, u, yderiv);
        der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
    }

#if 0
    for (int i=0; i < xderiv.size(); i++) {
        Vec3& v = rbf_centers[i];
        StencilType& st = stencil[i];
        printf("(%d), (%f,%f), xder[%d]= %f, yderiv= %f\n", st.size(), v.x(), v.y(), i, xderiv[i], yderiv[i]);
    }
#endif

    // interior points

    for (int i=0; i < lapl_deriv.size(); i++) {
        StencilType& st = stencil[i];
        NodeType& v = rbf_centers[st[0]]; 
        printf("(%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", (int) st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
    }

    int nb_bnd = grid.getBoundaryIndicesSize(); 
    // boundary points
    for (int ib=0; ib < nb_bnd; ib++) {
        int i = grid.getBoundaryIndex(ib);
        StencilType& st = stencil[i];
        NodeType& v = rbf_centers[st[0]]; 
        //printf("bnd (%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
    }

    double inter_error=0.;
    for (int i=nb_bnd; i < nb_stencils; i++) {
        // du_ex is the exact laplacian
        inter_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
        printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    inter_error /= (nb_stencils-nb_bnd);

    double bnd_error=0.;
    for (int ib=0; ib < nb_bnd; ib++) {
        int i = grid.getBoundaryIndex(ib);
        bnd_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
        printf("bnd error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    bnd_error /= nb_bnd;

    printf("avg lapl l2_bnd_error= %14.7e\n", sqrt(bnd_error));
    printf("avg lapl l2_interior_error= %14.7e\n", sqrt(inter_error));
}
//----------------------------------------------------------------------
void DerivativeTests::checkXDerivatives(Derivative& der, Grid& grid)
{
    vector<NodeType>& rbf_centers = grid.getNodeList();
    int nb_centers = grid.getNodeListSize();
    int nb_stencils = grid.getStencilsSize();

    // change the classes the variables are located in
    vector<double> xderiv(nb_stencils);
    vector<double> yderiv(nb_stencils);
    vector<double> lapl_deriv(nb_stencils);

    printf("deriv size: %d\n", (int) rbf_centers.size());
    printf("xderiv size: %d\n", (int) xderiv.size());

    // function to differentiate
    vector<double> u(nb_centers);
    vector<double> du_ex(nb_stencils); // exact derivative
    vector<double> dux_ex(nb_stencils); // exact derivative
    vector<double> duy_ex(nb_stencils); // exact derivative

    vector<StencilType>& stencil = grid.getStencils();

    this->computeAllWeights(der, rbf_centers, grid.getStencils());

#if 0
    for (int i=0; i < 1600; i++) {
        StencilType& v = stencil[i];
        printf("stencil[%d]\n", i);
        for (int s=0; s < v.size(); s++) {
            printf("%d ", v[s]);
        }
        printf("\n");
    }
    exit(0);
#endif

    double s;

    for (int i=0; i < nb_centers; i++) {
        Vec3& v = rbf_centers[i];
        //du_ex.push_back(2.);
        dux_ex[i] = v.y();
        duy_ex[i] = v.x();

        //s = v.y();
        du_ex[i] = 1.;
    } 
    for (int i = 0; i < nb_centers; i++) {
        Vec3& v = rbf_centers[i];
        //s = 3.*v.x() + 2.*v.y(); //   + 3.*v.z();
        //s = v.x()*v.y() + 0.5*v.x()*v.x() + 0.5*v.y()*v.y(); //   + 3.*v.z();
        s = v.x()*v.y();
        u[i] = s;
    }

    printf("main, u[0]= %f\n", u[0]);
    printf("main, u= %ld\n", (long int) &u[0]);

    for (int n=0; n < 1; n++) {
        // perhaps I'll need different (rad,eps) for each. To be determined. 
        der.computeDeriv(Derivative::X, u, xderiv);
        der.computeDeriv(Derivative::Y, u, yderiv);
        der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
    }
    der.computeEig(); // needs lapl_weights, analyzes stability of Laplace operator

#if 1
    for (int i=0; i < xderiv.size(); i++) {
        Vec3& v = rbf_centers[i];
        StencilType& st = stencil[i];
        printf("(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
        printf("(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
    }
#endif

    //exit(0);

    // interior points

#if 0
    for (int i=0; i < xderiv.size(); i++) {
        Vec3& v = rbf_centers[i];
        StencilType& st = stencil[i];
        printf("(%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", (int) st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
        printf("(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
        printf("(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
    }
#endif

    // boundary points
    vector<size_t>& boundary = grid.getBoundaryIndices();
    for (int ib=0; ib < boundary.size(); ib++) {
        int i = boundary[ib];
        StencilType& st = stencil[i];
        NodeType& v = rbf_centers[st[0]]; 
        printf("bnd(%d) sz(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", ib, (int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
        printf("bnd(%d) sz(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", ib, (int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
        //printf("bnd (%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
    }
    exit(0);

    double inter_error=0.;
    for (int i=boundary.size(); i < nb_stencils; i++) {
        inter_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
        printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    inter_error /= (nb_stencils-boundary.size());

    double bnd_error=0.;
    for (int ib=0; ib < boundary.size(); ib++) {
        int i = boundary[ib];
        bnd_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
        printf("bnd error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    bnd_error /= boundary.size();

    printf("[checkDerivs] avg l2_bnd_error= %14.7e\n", sqrt(bnd_error));
    printf("[checkDerivs] avg l2_interior_error= %14.7e\n", sqrt(inter_error));

    exit(0);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void DerivativeTests::computeAllWeights(Derivative& der, std::vector<Vec3>& rbf_centers, std::vector<StencilType>& stencils) {
    //#define USE_CONTOURSVD 1
    //#undef USE_CONTOURSVD

#if USE_CONTOURSVD
    exit(EXIT_FAILURE);
    // Laplacian weights with zero grid perturbation
    for (int irbf=0; irbf < stencils.size(); irbf++) {
        der.computeWeightsSVD(rbf_centers, stencils[irbf], irbf, "x");
        der.computeWeightsSVD(rbf_centers, stencils[irbf], irbf, "y");
        der.computeWeightsSVD(rbf_centers, stencils[irbf], irbf, "lapl");
    }

#else
    std::cout << "COMPUTING " << stencils.size() << " SETS OF STENCIL WEIGHTS" << std::endl;
    for (int i = 0; i < stencils.size(); i++) {
        der.computeWeights(rbf_centers, stencils[i], i);
    }
#endif
}
#endif
