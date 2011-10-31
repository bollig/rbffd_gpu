#ifndef __DERIVATIVE_TESTS_H__
#define __DERIVATIVE_TESTS_H__

#include "rbffd/rbffd.h"
#include "grids/grid_interface.h"
#include "Vec3.h"

#include <string>

class DerivativeTests {
    
    public:
        enum TESTFUN  {C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3};
        std::string TESTFUNSTR[10];

    protected:
        bool weightsComputed;
        RBFFD* der;
        Grid* grid;
        int dim_num;

    public:
        DerivativeTests(int dim, RBFFD* derivative_ptr, Grid* grid_ptr, bool weightsPreComputed=false)
            : weightsComputed(weightsPreComputed), 
            dim_num(dim),
            der(derivative_ptr), grid(grid_ptr)
        { 
            TESTFUNSTR[0] = "0";
            TESTFUNSTR[1] = "X";
            TESTFUNSTR[2] = "Y";
            TESTFUNSTR[3] = "X2";
            TESTFUNSTR[4] = "XY";
            TESTFUNSTR[5] = "Y2";
            TESTFUNSTR[6] = "X3";
            TESTFUNSTR[7] = "X2Y";
            TESTFUNSTR[8] = "XY2";
            TESTFUNSTR[9] = "Y3";

            // Only compute weights if we need to. 
            if (!weightsComputed) {
                der->computeAllWeightsForAllStencils();
                weightsComputed = true;
            }
        };
        ~DerivativeTests() { /*noop*/ }

        //---------- Cleaned -----------
        
        // FIXME: when nb_stencils_to_test != 0 we have a segfault. probably
        // because the subset of stencils we're checkign might NOT have
        // interior nodes, or may over compensate for boundary nodes. 
        void testAllFunctions(bool exitIfTestsFail=true, unsigned int nb_stencils_to_test=0);

        // NOTE: nb_stencils_to_test==0 implies that ALL stencils will be tested. 
        void compareGPUandCPUDerivs(unsigned int nb_stencils_to_test=0);

        void testDerivativeOfFunction(DerivativeTests::TESTFUN choice, unsigned int nb_stencils_to_test=0, bool exitIfTestFails=true);
        
        // Test our interpolation to Franke's test function: 
        //  f(x,y) = (3/4) e^(-(1/4) * ((9x-2)^2 + (9y-2)^2))  
        //            +  (3/4) e^(-(1/49)(9x+1)^2 - (1/10)(9y+1)^2)  
        //            +  (1/2) e^(-(1/4) * ((9x-7)^2 + (9y-3)^2))  
        //            -  (1/5) e^(-(9x-4)^2 - (9y-7)^2)  
        void testInterpolation(unsigned int nb_stencils_to_test=0, bool exitIfTestFails=true);

        void testEigen(RBFFD::DerType which=RBFFD::LAPL, bool exitIfTestFails=true, unsigned int maxNumPerturbations=100, float maxPerturbation=0.05);

        //void testHyperviscosity();

    protected: 
        void fillTestFunction(DerivativeTests::TESTFUN which, unsigned int nb_stencils, unsigned int nb_centers, std::vector<double>& u, std::vector<double>& dux_ex, std::vector<double>& duy_ex, std::vector<double>& dulapl_ex);

        double compareDeriv(double deriv_gpu, double deriv_cpu, std::string label, int indx);

        //---------- TODO --------------

#if 0
        void checkDerivatives(Derivative& der, Grid& grid);
        void checkXDerivatives(Derivative& der, Grid& grid);
        void computeAllWeights(Derivative& der, std::vector<Vec3>& rbf_centers,
                std::vector<StencilType>& stencils);
#endif 
};

#endif
