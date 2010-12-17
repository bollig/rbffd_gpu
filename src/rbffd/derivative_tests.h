#ifndef __DERIVATIVE_TESTS_H__
#define __DERIVATIVE_TESTS_H__

#include "rbffd/derivative.h"
#include "grids/grid.h"
#include "Vec3.h"

class DerivativeTests {
public:
    enum TESTFUN  {C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3,CUSTOM};
public:
    DerivativeTests() { weightsComputed = false; };
    ~DerivativeTests() { /*noop*/ }
    void checkDerivatives(Derivative& der, Grid& grid);
    void checkXDerivatives(Derivative& der, Grid& grid);
    void testDeriv(DerivativeTests::TESTFUN choice, Derivative& der, Grid& grid, std::vector<double> avgDist);
    void testFunction(DerivativeTests::TESTFUN which, Grid& grid, vector<double>& u, vector<double>& dux_ex, vector<double>& duy_ex, vector<double>& dulapl_ex);
    void testEigen(Grid& grid, int stencil_size, int nb_bnd, int tot_nb_pts);
    void testAllFunctions(Derivative& der, Grid& grid);
    void computeAllWeights(Derivative& der, std::vector<Vec3> rbf_centers, std::vector<std::vector<int> > stencils, int nb_stencils, int dimension);

protected:
    bool weightsComputed;
    Derivative* der;
    Grid* grid;

};

#endif
