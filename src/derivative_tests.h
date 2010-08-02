#ifndef __DERIVATIVE_TESTS_H__
#define __DERIVATIVE_TESTS_H__

#include "derivative.h"
#include "grid.h"

#include "norms.h"

class DerivativeTests {
	public: 
		DerivativeTests(); 
		~DerivativeTests() { /*noop*/ }; 
		void checkDerivatives(Derivative& der, Grid& grid);
		void checkXDerivatives(Derivative& der, Grid& grid);
                void testDeriv(TESTFUN choice, Derivative& der, Grid& grid, std::vector<double> avgDist);
                void testFunction(TESTFUN which, Grid& grid, vector<double>& u, vector<double>& dux_ex, vector<double>& duy_ex, vector<double>& dulapl_ex);
                void testEigen(Grid& grid, int stencil_size, int nb_bnd, int tot_nb_pts);
		

	protected: 
		Derivative* der;
		Grid* grid; 
};

#endif
