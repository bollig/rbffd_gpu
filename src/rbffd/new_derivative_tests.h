#ifndef __DERIVATIVE_TESTS_H__
#define __DERIVATIVE_TESTS_H__

#include "rbffd/derivative.h"
#include "grids/grid_interface.h"
#include "Vec3.h"

#include <string>
using namespace std;

class DerivativeTests {
public:
    enum TESTFUN  {C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3};
    std::string TESTFUNSTR[10];
public:
    DerivativeTests() { 
		weightsComputed = false; 
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
	};
    ~DerivativeTests() { /*noop*/ }
    void checkDerivatives(Derivative& der, Grid& grid);
    void checkXDerivatives(Derivative& der, Grid& grid);
    void testDeriv(DerivativeTests::TESTFUN choice, Derivative& der, Grid& grid);
    void testFunction(DerivativeTests::TESTFUN which, Grid& grid, vector<double>& u, vector<double>& dux_ex, vector<double>& duy_ex, vector<double>& dulapl_ex);
    void testEigen(Derivative& der, Grid& gridi);
    void testAllFunctions(Derivative& der, Grid& grid);
    void computeAllWeights(Derivative& der, Grid& grid);

protected:
    bool weightsComputed;
    //Derivative* der;
    //Grid* grid;

};

#endif
