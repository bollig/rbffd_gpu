#ifndef __DERIVATIVE_TESTS_H__
#define __DERIVATIVE_TESTS_H__

#include "rbffd/rbffd.h"
#include "grids/grid_interface.h"
#include "Vec3.h"

#include <string>
using namespace std;

class DerivativeTests {
    
    public:
        enum TESTFUN  {C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3};
        std::string TESTFUNSTR[10];

    protected:
        bool weightsComputed;
        RBFFD* der;
        Grid* grid;

    public:
        DerivativeTests(RBFFD* derivative_ptr, Grid* grid_ptr, bool weightsPreComputed=false)
            : weightsComputed(weightsPreComputed), 
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
            }
        };
        ~DerivativeTests() { /*noop*/ }

        //---------- Cleaned -----------
        
        void testAllFunctions();

        // NOTE: nb_stencils_to_test==0 implies that ALL stencils will be tested. 
        void compareGPUandCPUDerivs(size_t nb_stencils_to_test=0);
        double compareDeriv(double deriv_gpu, double deriv_cpu, std::string label, int indx);


        //---------- TODO --------------

#if 0
        void checkDerivatives(Derivative& der, Grid& grid);
        void checkXDerivatives(Derivative& der, Grid& grid);

        void testDeriv(DerivativeTests::TESTFUN choice, Derivative& der, Grid&
                grid, std::vector<double> avgDist);

        void testFunction(DerivativeTests::TESTFUN which, Grid& grid,
                vector<double>& u, vector<double>& dux_ex, vector<double>& duy_ex,
                vector<double>& dulapl_ex);

        void testEigen(Derivative& der, Grid& grid);

        void computeAllWeights(Derivative& der, std::vector<Vec3>& rbf_centers,
                std::vector<StencilType>& stencils);
#endif 
};

#endif
