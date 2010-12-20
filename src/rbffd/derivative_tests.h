#ifndef __DERIVATIVE_TESTS_H__
#define __DERIVATIVE_TESTS_H__

#include "rbffd/derivative.h"
#include "grids/grid.h"
#include "Vec3.h"

#include <string>
using namespace std;

class DerivativeTests {
public:
    enum TESTFUN  {C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3,CUSTOM};
    std::string TESTFUNSTR[11];
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
		TESTFUNSTR[10] = "CUSTOM";
	};
    ~DerivativeTests() { /*noop*/ }
    void checkDerivatives(Derivative& der, Grid& grid);
    void checkXDerivatives(Derivative& der, Grid& grid);
    void testDeriv(DerivativeTests::TESTFUN choice, Derivative& der, Grid& grid, std::vector<double> avgDist);
    void testFunction(DerivativeTests::TESTFUN which, Grid& grid, vector<double>& u, vector<double>& dux_ex, vector<double>& duy_ex, vector<double>& dulapl_ex);
    void testEigen(Grid& grid, Derivative& der, int stencil_size, int nb_bnd, int tot_nb_pts);
    void testAllFunctions(Derivative& der, Grid& grid);
    void computeAllWeights(Derivative& der, std::vector<Vec3> rbf_centers, std::vector<std::vector<int> > stencils, int nb_stencils);

protected:
    bool weightsComputed;
    Derivative* der;
    Grid* grid;

};

#endif
